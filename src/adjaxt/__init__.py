"""adjaxt — distributed training with JAX on free compute, for humans."""

from adjaxt.config import (
    RMSNormConfig,
    SwiGLUConfig,
    PrecisionPolicy,
    StandardAttnImplementation,
    GQAAttnConfig,
    Qwen3AttnConfig,
    Qwen3MLPConfig,
    Qwen3MoEBlockConfig,
    Qwen3MoELayerConfig,
    Qwen3MoEModelConfig,
    compute_rope_freqs,
)
from adjaxt import layers, models, model_maps, sharding, export, optim, approx
from adjaxt.approx import WorkerPlan, approx, benchmark_step_throughput
from adjaxt.train import fit, push_fit

__version__ = "0.0.2.10"

__all__ = [
    # Configs
    "RMSNormConfig",
    "SwiGLUConfig",
    "PrecisionPolicy",
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
    "optim",
    "WorkerPlan",
    "approx",
    "benchmark_step_throughput",
    "fit",
    "push_fit",
]