"""adjaxt — distributed training with JAX on free compute, for humans."""

from adjaxt.config import (
    RMSNormConfig,
    SwiGLUConfig,
    StandardAttnImplementation,
    GQAAttnConfig,
    Qwen3AttnConfig,
    Qwen3MLPConfig,
    Qwen3MoEBlockConfig,
    Qwen3MoELayerConfig,
    Qwen3MoEModelConfig,
    compute_rope_freqs,
)
from adjaxt import layers, models, model_maps, sharding, export, optim

__version__ = "0.0.1.5"

__all__ = [
    # Configs
    "RMSNormConfig",
    "SwiGLUConfig",
    "StandardAttnImplementation",
    "GQAAttnConfig",
    "Qwen3AttnConfig",
    "Qwen3MLPConfig",
    "Qwen3MoEBlockConfig",
    "Qwen3MoELayerConfig",
    "Qwen3MoEModelConfig",
    "compute_rope_freqs",
    # Submodules
    "layers",
    "models",
    "model_maps",
    "sharding",
    "export",
    "optim"
]