# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Nemotron model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
import math
from torch import nn

from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import NemotronConfig

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

# Dima to add the usual RMS Norm
from vllm.model_executor.layers.layernorm import RMSNorm

# Dima in order to add window to local attention
from vllm.config import CacheConfig

# Dima in order to clone CacheConfig if it exists
import copy 

# Dima for debugging purposes
import os
from vllm.logger import init_logger
logger = init_logger(__name__)

from sympy import sympify, lambdify
from sympy.core.sympify import SympifyError

# https://github.com/datamllab/LongLM stuff
GROUP_SIZE =  os.getenv('GROUP_SIZE')
if GROUP_SIZE:
    try:
        GROUP_SIZE = int(GROUP_SIZE)
        logger.info(f'GROUP_SIZE: {GROUP_SIZE}')
    except ValueError:
        logger.info(f'failed to parse GROUP_SIZE: {GROUP_SIZE}, setting to None')
        GROUP_SIZE = None

WINDOW_SIZE = os.getenv('WINDOW_SIZE')
if WINDOW_SIZE:
    try:
        WINDOW_SIZE = int(WINDOW_SIZE)
        logger.info(f'WINDOW_SIZE: {WINDOW_SIZE}')
    except ValueError:
        logger.info(f'failed to parse GROUP_SIZE: {WINDOW_SIZE}, setting to None')
        WINDOW_SIZE = None


# maybe check for a formula here?
MSCALE = os.getenv('MSCALE')
if MSCALE:
    try:  # is it a float?
        MSCALE = float(MSCALE)
        logger.info(f'MSCALE parsed as a float: {MSCALE}')
    except ValueError:   
        logger.info(f'MSCALE is not a float: {MSCALE}')
#         try:
#             MSCALE = sympify(MSCALE)
#             logger.info(f'MSCALE parsed as a sympy formula: {MSCALE}')
#         except SympifyError:
#             logger.error(f'invalid formula {MSCALE}, setting to None')
#             MSCALE = None
#         except Exception as e:
#             logger.info(f'failed to parse MSCALE: {MSCALE}, setting to None')
#             MSCALE = None

# The architecture is pretty similar to Llama, with these changes:
# - There is no gate_proj, just up_proj
# - Normal LayerNorm (with a +1 to the weights) instead of RMSNorm
# - Squared ReLU instead of SwiGLU
# - Adds a partial_rotary_factor to RoPE


def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        return torch.cuda.amp.autocast_mode._cast(
            args, torch.get_autocast_gpu_dtype())


class NemotronLayerNorm1P(nn.LayerNorm):

    def __init__(self,
                 normalized_shape: Union[int, List[int], torch.Size],
                 eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 bias: bool = True,
                 device=None,
                 dtype=None):
        super().__init__(normalized_shape, eps, elementwise_affine, bias,
                         device, dtype) 

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if residual is not None:
            x = x + residual
            residual = x
        args = _cast_if_autocast_enabled(x, self.normalized_shape,
                                         self.weight + 1, self.bias, self.eps)
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.nn.functional.layer_norm(*args)
            return x if residual is None else (x, residual)


class NemotronMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.up_proj = ColumnParallelLinear(input_size=hidden_size,
                                            output_size=intermediate_size,
                                            bias=bias,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.up_proj")
        self.down_proj = RowParallelLinear(input_size=intermediate_size,
                                           output_size=hidden_size,
                                           bias=bias,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.down_proj")
        self.act_fn = get_act_fn(hidden_act)

    def forward(self, x):
        up, _ = self.up_proj(x)
        x = self.act_fn(up)
        x, _ = self.down_proj(x)
        return x


class NemotronAttention(nn.Module):

    def __init__(
        self,
        config: NemotronConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=self.partial_rotary_factor,
        )

        # if GROUP_SIZE and WINDOW_SIZE:
        #     logger.info(f'using a combination of local and group attention')
        #     # does scaling need to be applied equally to local and group attention?
        #     # maybe yes because we are mixing the scores ?
        # 
        #     if cache_config is not None:
        #         logger.info(f'cache_config: {cache_config}')
        #         local_cache_config = copy.deepcopy(cache_config)
        #         cache_config.sliding_window = [-WINDOW_SIZE,0]
        #     else:
        #         # need to check what gpu mem and swap space ought to be set to.. 
        #         local_cache_config = CacheConfig(block_size = 16,
        #                       cache_dtype = "auto",
        #                       sliding_window = [-WINDOW_SIZE,0],
        #                       gpu_memory_utilization = 0.95,
        #                       swap_space = 0)
        #         
        #     # Dima: I hope these two attentions don't double our memory requirement
        #     self.attn_local = Attention(self.num_heads,
        #                       self.head_dim,
        #                       self.scaling,
        #                       num_kv_heads=self.num_kv_heads,
        #                       cache_config=local_cache_config, # Dima
        #                       quant_config=quant_config)
        #     
        #     self.attn_group = Attention(self.num_heads,
        #                       self.head_dim,
        #                       self.scaling,
        #                       num_kv_heads=self.num_kv_heads,
        #                       cache_config=cache_config,
        #                       quant_config=quant_config)
        #     
        # else:
        #     logger.info(f'using regular full attention')
        self.attn = Attention(self.num_heads,
                          self.head_dim,
                          self.scaling,
                          num_kv_heads=self.num_kv_heads,
                          cache_config=cache_config,
                          quant_config=quant_config,
                          prefix=f"{prefix}.attn")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # if GROUP_SIZE and WINDOW_SIZE:

            # likely, this needs to be validated.. 
            # bsz, q_len, _ = hidden_states.size()
            # kv_seq_len = k.shape[-2]

            # the crazy paper implementation
            # query_positions = positions
            # key_positions = positions
            # if q_len != 1:
            #     key_positions = positions 
            # else: 
            #     key_positions = torch.arange(kv_seq_len, dtype=positions.dtype).to(query_positions.device).view(1, kv_seq_len) # only consider bsz=1 for now.

            # below is okay if q_len != 1 but if it is one we need to use key_positions
            # local computation is unchanged with the exception of the window in attn_local
            # q_local, k_local = self.rotary_emb(positions, q, k)
            # logger.info(f'q: {q.shape}, k: {k.shape}, v: {v.shape}, kv_cache: {kv_cache.shape}, positions: {positions.shape}, attn_metadata: {attn_metadata}')

            # do we need to worry about the values v here? probably not?
            # attn_output_local = self.attn_local(q_local, k_local, v, kv_cache, attn_metadata)

            # positions_group = positions // GROUP_SIZE + WINDOW_SIZE - WINDOW_SIZE // GROUP_SIZE # or something like that

            # _re_group_size_2 = 0 if positions.max() < WINDOW_SIZE else WINDOW_SIZE # in case that, the smallest q position, g2-g2//g1 exceed the max position
            # positions_group = positions // GROUP_SIZE + _re_group_size_2 - _re_group_size_2 / GROUP_SIZE

            # group_query_positions = query_positions // GROUP_SIZE + _re_group_size_2 - _re_group_size_2 / GROUP_SIZE
            # group_key_positions = key_positions // GROUP_SIZE
            # done with the crazy paper implementation

            # need to adapt the blow to different query / key positions?
            # q_group, k_group = self.rotary_emb(positions_group, q, k)
            # logger.info(f'q: {q.shape}, k: {k.shape}, v: {v.shape}, kv_cache: {kv_cache.shape}, positions: {positions.shape}, attn_metadata: {attn_metadata}')

 
            # attn_output_group = self.attn_group(q_group, k_group, v, kv_cache, attn_metadata)

            # need to deal with paddings, also normalization?

            # normalize?
            # attn_output_local[:, -true_neighbor_seq_max_length:, ...] = (attn_output_local[:, -true_neighbor_seq_max_length:, ...] * neighbor_softmax_lse)
            # attn_output_group[:, -true_group_seq_max_length:, ...] = (attn_output_group[:, -true_group_seq_max_length:, ...] * group_softmax_lse)

            # need to funky combine the attention score here. local == local, the rest is group.
            # attn_output = torch.empty_like(attn_output_local).copy_(attn_output_local)  # might be slightly faster than clone
            # group_size_2-kv_seq_len ?
            # attn_output[:, WINDOW_SIZE:, ...] += attn_output_group


            # logger.info(f'attn_output: {attn_output.shape}')           
        # else:
        q = self.apply_mscale_if_needed(q, k, positions, MSCALE)
 
        q, k = self.rotary_emb(positions, q, k)
        # if q is not None:
        #     logger.info(f'q: {q.shape}')
        # if k is not None:
        #     logger.info(f'k: {k.shape}')


        # if v is not None:
        #    logger.info(f'v: {v.shape}')
        # if kv_cache is not None:
        #    logger.info(f'kv_cache: {kv_cache.shape}')
        # if positions is not None:
        #    logger.info(f'positions: {positions.shape}')
        # if attn_metadata is not None:
        #    logger.info(f'attn_metadata: {attn_metadata}')
                    
        # logger.info(f'q: {q.shape}, k: {k.shape}, v: {v.shape}, kv_cache: {kv_cache.shape}, positions: {positions.shape}, attn_metadata: {attn_metadata}')
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        # logger.info(f'attn_output: {attn_output.shape}')
        output, _ = self.o_proj(attn_output)
        return output
    
    def apply_mscale_if_needed(self, q, k, positions, mscale) -> torch.Tensor:
        if mscale is None:
            return q
        
        if isinstance(mscale, float): 
            # position - independent scaling factor
            logger.info(f'applying static MSCALE: {mscale}, q shape: {q.shape}, k shape: {k.shape}, positions shape: {positions.shape}')
            # if positions.numel() == 1: 
            #     logger.info(f'positions: {positions.cpu().item()}')

            return q * mscale 

        # mscale is a formula
        logger.info(f'applying dynamic MSCALE: {mscale}, q shape: {q.shape}, k shape: {k.shape}, positions shape: {positions.shape}')
        mscale = mscale.strip().lower()
        compute_dtype = torch.float32

        try:
            # assert mscale == "log", f"mscale not supported: {mscale}"
            # DYNAMIC_SF = 1.0
            # mscale_multiplier = mscale(positions)
            # mscale_multiplier = mscale_multiplier[-q.shape[0]:, ...]
            # return q * mscale_multiplier
            if mscale == "log":
                pos = torch.arange(1, q.shape[0] + 1, dtype=compute_dtype, device=q.device)
                sf = torch.ones_like(pos)
                sf[1:] = torch.log(pos[1:])

            elif mscale.startswith("log"):
                base = float(mscale[3:])
                pos = torch.arange(0, q.shape[0], dtype=compute_dtype, device=q.device)
                # log_base = torch.log(torch.tensor(base, dtype=compute_dtype, device=q.device))
                log_base = math.log(base)
                sf = torch.log(base + pos) / log_base

            else:
                logger.warning(f"mscale formula not supported: {mscale}, falling back to no scaling.")
                sf = torch.ones(q.shape[0], dtype=compute_dtype, device=q.device)
               

        except (ValueError, TypeError) as e:
            logger.warning(f"mscale parsing failed: {e}, falling back to no scaling.")
            sf = torch.ones(q.shape[0], dtype=compute_dtype, device=q.device)

        sf = sf.to(dtype=q.dtype)
        mscale_multiplier = sf.unsqueeze(1)

        return q * mscale_multiplier




class NemotronDecoderLayer(nn.Module):

    def __init__(
        self,
        config: NemotronConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = NemotronAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = NemotronMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )

        # D.R. this should be already validated before
        if getattr(config, "layernorm_type") == 'layernorm1p':
            logger.info(f'layernorm_type: layernorm1p') 
            self.input_layernorm = NemotronLayerNorm1P(config.hidden_size, eps=config.norm_eps)
            self.post_attention_layernorm = NemotronLayerNorm1P(config.hidden_size, eps=config.norm_eps)
        else:
            logger.info(f'layernorm_type: rmsnorm') 
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class NemotronModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: NemotronDecoderLayer(config=config,
                                                cache_config=cache_config,
                                                quant_config=quant_config,
                                                prefix=prefix),
            prefix=f"{prefix}.layers")
        if get_pp_group().is_last_rank:
            if getattr(config, "layernorm_type") == 'layernorm1p':
                logger.info(f'norm_type: layernorm1p') 
                self.norm = NemotronLayerNorm1P(config.hidden_size, eps=config.norm_eps)
            else:
                logger.info(f'layernorm_type: rmsnorm') 
                self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
                
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class NemotronForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj", "o_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        assert isinstance(config, NemotronConfig)

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        self.model = NemotronModel(vllm_config=vllm_config,
                                   prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config else lora_config.lora_vocab_padding_size,
                quant_config=quant_config,
            )
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
        else:
            self.lm_head = PPMissingLayer()

        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(input_ids, positions, kv_caches,
                                  attn_metadata, intermediate_tensors,
                                  inputs_embeds)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
