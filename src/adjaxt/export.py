import json
import os
import shutil
import dataclasses
from adjaxt.sharding import save_checkpoint, ModelWeightMap
import adjaxt


def dataclass_to_dict(obj):
    """Recursively serializes dataclasses to dictionaries for config.json."""
    import dataclasses
    
    # Handle JAX dtypes specifically
    if hasattr(obj, "name") and type(obj).__name__ == "dtype":
        return str(obj.name)
        
    if dataclasses.is_dataclass(obj):
        res = {}
        for k, v in obj.__dict__.items():
            if hasattr(v, "shape"):
                continue  # Skip precomputed arrays like cos/sin tables
            res[k] = dataclass_to_dict(v)
        return res
    return obj

def dict_to_dataclass(cls, data: dict):
    import dataclasses
    import jax.numpy as jnp
    
    if not dataclasses.is_dataclass(cls):
        # Convert dtype strings back to actual jnp dtypes
        if isinstance(data, str) and hasattr(jnp, data):
            return getattr(jnp, data)
        return data

    fields = {f.name: f.type for f in dataclasses.fields(cls)}
    init_kwargs = {}
    for k, v in data.items():
        if k in fields:
            ftype = fields[k]
            if dataclasses.is_dataclass(ftype) and isinstance(v, dict):
                init_kwargs[k] = dict_to_dataclass(ftype, v)
            else:
                # Convert string to dtype if the field expects a jnp.dtype
                if isinstance(v, str) and hasattr(jnp, v):
                    init_kwargs[k] = getattr(jnp, v)
                else:
                    init_kwargs[k] = v
    return cls(**init_kwargs)

def export_arbitrary_model_to_hf(
    model_type: str,
    jax_fn_name: str,
    config_cls_name: str,
    cfg: any,
    weights: dict,
    weight_map: ModelWeightMap,
    dim_sizes: dict,
    save_directory: str,
):
    os.makedirs(save_directory, exist_ok=True)

    config_filename = f"configuration_{model_type}"
    modeling_filename = f"modeling_{model_type}"
    model_class_name = "".join([p.capitalize() for p in model_type.split("_")]) + "Model"
    config_wrapper_name = model_class_name + "Config"

    # =====================================================================
    # 1. Auto-Generate configuration_*.py
    # =====================================================================
    config_code = f"""from transformers import PretrainedConfig

class {config_wrapper_name}(PretrainedConfig):
    model_type = "{model_type}"

    def __init__(self, **kwargs):
        self.arch_config = kwargs.pop("arch_config", {{}})
        super().__init__(**kwargs)
"""
    with open(os.path.join(save_directory, f"{config_filename}.py"), "w", encoding="utf-8") as f:
        f.write(config_code)

    # =====================================================================
    # 2. Auto-Generate modeling_*.py (With Dynamic sys.path Resolution)
    # =====================================================================
    modeling_code = f"""import os
import sys
import dataclasses
from transformers import PreTrainedModel

# Inject local repository directory into sys.path for isolated execution
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from {config_filename} import {config_wrapper_name}

# Resolve bundled adjaxt package
try:
    from .adjaxt import config as cfg_module
    from .adjaxt import layers as layers_module
    from .adjaxt.torch_layers import JAX_TO_TORCH_REGISTRY
except (ImportError, ValueError):
    import adjaxt.config as cfg_module
    import adjaxt.layers as layers_module
    from adjaxt.torch_layers import JAX_TO_TORCH_REGISTRY

def dict_to_dataclass(cls, data: dict):
    if not dataclasses.is_dataclass(cls):
        return data
    fields = {{f.name: f.type for f in dataclasses.fields(cls)}}
    init_kwargs = {{}}
    for k, v in data.items():
        if k in fields:
            ftype = fields[k]
            if dataclasses.is_dataclass(ftype) and isinstance(v, dict):
                init_kwargs[k] = dict_to_dataclass(ftype, v)
            else:
                init_kwargs[k] = v
    return cls(**init_kwargs)

class {model_class_name}(PreTrainedModel):
    config_class = {config_wrapper_name}

    def __init__(self, config: {config_wrapper_name}):
        super().__init__(config)

        jax_fn = getattr(layers_module, "{jax_fn_name}")
        config_cls = getattr(cfg_module, "{config_cls_name}")

        jax_cfg = dict_to_dataclass(config_cls, config.arch_config)
        torch_cls = JAX_TO_TORCH_REGISTRY[jax_fn]

        self.model = torch_cls(jax_cfg)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
"""
    with open(os.path.join(save_directory, f"{modeling_filename}.py"), "w", encoding="utf-8") as f:
        f.write(modeling_code)

    # =====================================================================
    # 3. Save config.json with Remote Code Pointers
    # =====================================================================
    hf_config = {
        "model_type": model_type,
        "arch_config": dataclass_to_dict(cfg),
        "auto_map": {
            "AutoConfig": f"{config_filename}.{config_wrapper_name}",
            "AutoModel": f"{modeling_filename}.{model_class_name}",
            "AutoModelForCausalLM": f"{modeling_filename}.{model_class_name}",
        },
    }
    with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)

    # =====================================================================
    # 4. Export Safetensors
    # =====================================================================
    save_checkpoint(
        weights=weights,
        save_directory=save_directory,
        weight_map=weight_map,
        dim_sizes=dim_sizes,
    )

    # =====================================================================
    # 5. Copy Framework Source Files into Standalone Package
    # =====================================================================
    adjaxt_source_dir = os.path.dirname(adjaxt.__file__)
    adjaxt_target_dir = os.path.join(save_directory, "adjaxt")
    os.makedirs(adjaxt_target_dir, exist_ok=True)

    modules_to_copy = [
        "layers.py",
        "torch_layers.py",
        "config.py",
        "sharding.py",
        "model_maps.py",
        "__init__.py",
    ]

    for module in modules_to_copy:
        source_path = os.path.join(adjaxt_source_dir, module)
        if os.path.exists(source_path):
            shutil.copy(source_path, os.path.join(adjaxt_target_dir, module))