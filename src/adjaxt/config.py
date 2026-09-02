from dataclasses import field
import dataclasses
from typing import Dict, Any, Tuple, Literal
from enum import StrEnum
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from typing import Callable

config_classes = []

def config_class(cls):
    """
    A unified decorator that automatically registers a dataclass as a PyTree,
    intelligently separating JAX arrays from static metadata.
    """
    cls = dataclasses.dataclass(cls)

    def tree_flatten(self):
        dynamic_children = {}
        static_aux_data = {}

        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            # If it's a JAX array, or another registered PyTree (like GQAAttnConfig)
            # it goes into dynamic children so JAX can trace and traverse it.
            if isinstance(val, jax.Array) or hasattr(type(val), "tree_flatten"):
                dynamic_children[f.name] = val
            else:
                # Ints, floats, strings, and PrecisionPolicy go to static metadata
                static_aux_data[f.name] = val

        # JAX requires children to be an iterable, and aux_data to contain enough
        # info to reconstruct the object (we pass the keys and static values)
        children_keys, children_vals = tuple(dynamic_children.keys()), tuple(dynamic_children.values())
        return children_vals, (children_keys, static_aux_data)

    def tree_unflatten(cls_obj, aux_data, children_vals):
        children_keys, static_aux_data = aux_data

        # Zip the dynamic children back with their field names
        reconstructed_children = dict(zip(children_keys, children_vals))

        # Combine dynamic and static kwargs to rebuild the dataclass
        kwargs = {**reconstructed_children, **static_aux_data}

        # Use __new__ to avoid calling __post_init__ again during unflattening,
        # which would regenerate the RoPE tables unnecessarily!
        obj = object.__new__(cls_obj)
        for key, value in kwargs.items():
            object.__setattr__(obj, key, value)
        return obj

    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = classmethod(tree_unflatten)
    cls = register_pytree_node_class(cls)
    config_classes.append(cls)

    return cls

def init_fn_for(cfg_class):
    """Decorator factory to attach an init function/class to a config class."""
    def decorator(cls_or_fn):
        cfg_class.init_fn = staticmethod(cls_or_fn)
        return cls_or_fn
    return decorator

def exec_fn_for(cfg_class):
    """Decorator factory to attach an exec function/class to a config class."""
    def decorator(cls_or_fn):
        cfg_class.exec_fn = staticmethod(cls_or_fn)
        return cls_or_fn
    return decorator

@config_class
class PrecisionPolicy:
    param_dtype: jnp.dtype = jnp.bfloat16    # Storage dtype in memory
    compute_dtype: jnp.dtype = jnp.bfloat16  # Matrix multiplication & attention dtype
    output_dtype: jnp.dtype = jnp.float32    # Residual & loss accumulator dtype

    @property
    def is_mixed(self) -> bool:
        return self.compute_dtype != self.output_dtype

@config_class
class RMSNormConfig:
    dim: int
    eps: float = 1e-6
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)

@config_class
class SwiGLUConfig:
    in_dim: int
    hidden_dim: int
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)

class StandardAttnImplementation(StrEnum):
    CUDNN = "cudnn"
    XLA = "xla"

@config_class
class GQAAttnConfig:
    implementation: StandardAttnImplementation
    num_kv_groups: int
    is_causal: bool
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)

def compute_rope_freqs(seq_len: int, head_dim: int, theta: float = 10000.0):
    dim_indices = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (theta ** (dim_indices / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)

    freqs = jnp.concatenate([freqs, freqs], axis=-1)

    cos = jnp.cos(freqs)[None, :, None, :]
    sin = jnp.sin(freqs)[None, :, None, :]
    return cos, sin

@config_class
class Qwen3AttnConfig:
    gqa_conf: GQAAttnConfig
    q_rms_conf: RMSNormConfig
    k_rms_conf: RMSNormConfig
    d: int
    num_heads: int
    head_dim: int
    rope_theta: float
    max_position_embeddings: int
    n_layers: int

    # 1. Use default_factory to generate a new instance safely
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)

    # 2. Keep these as they are (init=False is correct here)
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
class Qwen3MLPConfig:
    act_fn: Callable
    in_dim: int
    hidden_dim: int
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)

@config_class
class Qwen3MoEBlockConfig:
    mlp_conf: Qwen3MLPConfig
    top_k: int
    num_experts: int
    d_model: int
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)

@config_class
class Qwen3MoELayerConfig:
    input_rms_conf: RMSNormConfig
    attn_conf: Qwen3AttnConfig
    post_attn_rms_conf: RMSNormConfig
    moe_block_conf: Qwen3MoEBlockConfig
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)

@config_class
class Qwen3MoEModelConfig:
    moe_layer_conf: Qwen3MoELayerConfig
    final_rms_conf: RMSNormConfig
    num_decoder_blocks: int
    vocab_size: int
    d_model: int
    tie_word_embeddings: bool = False
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)

@config_class
class MHAAttentionConfig:
    implementation: StandardAttnImplementation
    is_causal: bool
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)

@config_class
class CCEConfig:
    reduction: Literal["mean", "sum", "none"] = "mean"
    ignore_index: int = -100
    shift: bool = True # Does causal shift for NTP

    chunk_size: int = 2048
    soft_cap: int | None = None # Cap logits
    filter_eps: float | None = None

    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)
