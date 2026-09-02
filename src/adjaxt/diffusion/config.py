from dataclasses import dataclass, field
from typing import Callable
from adjaxt.config import *
import jax
import jax.numpy as jnp

@config_class
class MDLMAttentionConfig:
    mha_conf: MHAAttentionConfig
    qk_rms_conf: RMSNormConfig
    d: int
    num_heads: int
    head_dim: int
    rope_theta: float
    max_position_embeddings: int
    n_layers: int
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)
    cos_table: jax.Array = field(init=False, compare=False, hash=False)
    sin_table: jax.Array = field(init=False, compare=False, hash=False)

    def __post_init__(self):
        cos, sin = compute_rope_freqs(
            self.max_position_embeddings, 
            self.head_dim, 
            self.rope_theta
        )
        object.__setattr__(self, "cos_table", cos)
        object.__setattr__(self, "sin_table", sin)

@config_class
class MDLMDenseMLPConfig:
    in_dim: int
    hidden_dim: int
    act_fn: Callable = jax.nn.silu
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)

@config_class
class AdaLNConfig:
    d: int
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)

@config_class
class MDLMBlockConfig:
    attn_cfg: MDLMAttentionConfig
    mlp_cfg: MDLMDenseMLPConfig
    adaln_cfg: AdaLNConfig
    input_rms_conf: RMSNormConfig
    post_attn_rms_conf: RMSNormConfig
    layers_per_block: int = 4
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)

@config_class
class MDLModelConfig:
    block_cfg: MDLMBlockConfig
    d: int
    vocab_size: int = 32000
    max_insert: int = 4
    scratch_len: int = 8
    canvas_len: int = 20
    mask_token_id: int = 32000
    steps_per_block: int = 2
    diff_blocks_num: int = 6
    final_rms_conf: RMSNormConfig = field(default_factory=RMSNormConfig)
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)