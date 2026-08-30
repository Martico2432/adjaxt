from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
from enum import StrEnum
import jax
import jax.numpy as jnp
from typing import Callable

@dataclass(frozen=True)
class RMSNormConfig:
    dim: int
    eps: float = 1e-6

@dataclass(frozen=True)
class SwiGLUConfig:
    in_dim: int
    hidden_dim: int

class StandardAttnImplementation(StrEnum):
    CUDNN = "cudnn"
    XLA = "xla"

@dataclass(frozen=True)
class GQAAttnConfig:
    implementation: StandardAttnImplementation
    num_kv_groups: int
    is_causal: bool

def compute_rope_freqs(seq_len: int, head_dim: int, theta: float = 10000.0):
    dim_indices = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (theta ** (dim_indices / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    
    freqs = jnp.concatenate([freqs, freqs], axis=-1)
    
    cos = jnp.cos(freqs)[None, :, None, :]
    sin = jnp.sin(freqs)[None, :, None, :]
    return cos, sin

@dataclass(frozen=True)
class Qwen3AttnConfig:
    gqa_conf: GQAAttnConfig
    q_rms_conf: RMSNormConfig
    k_rms_conf: RMSNormConfig
    d: int
    num_heads: int
    head_dim: int
    rope_theta: float
    cos_table: jax.Array = field(init=False, compare=False, hash=False)
    sin_table: jax.Array = field(init=False, compare=False, hash=False)
    max_position_embeddings: int
    n_layers: int

    def __post_init__(self):
        cos, sin = compute_rope_freqs(
            self.max_position_embeddings, 
            self.head_dim, 
            self.rope_theta
        )
        object.__setattr__(self, "cos_table", cos)
        object.__setattr__(self, "sin_table", sin)

@dataclass(frozen=True)
class Qwen3MLPConfig:
    act_fn: Callable
    in_dim: int
    hidden_dim: int

@dataclass(frozen=True)
class Qwen3MoEBlockConfig:
    mlp_conf: Qwen3MLPConfig
    top_k: int
    num_experts: int
    d_model: int

@dataclass(frozen=True)
class Qwen3MoELayerConfig:
    input_rms_conf: RMSNormConfig
    attn_conf: Qwen3AttnConfig
    post_attn_rms_conf: RMSNormConfig
    moe_block_conf: Qwen3MoEBlockConfig

@dataclass(frozen=True)
class Qwen3MoEModelConfig:
    moe_layer_conf: Qwen3MoELayerConfig
    final_rms_conf: RMSNormConfig
    num_decoder_blocks: int
    vocab_size: int
    d_model: int
    tie_word_embeddings: bool = False