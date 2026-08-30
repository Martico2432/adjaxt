import functools
import glob
import itertools
import json
import operator
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file, save_file

# ==============================================================================
# 1. Spec & Model Definitions
# ==============================================================================

@dataclass
class WeightSpec:
    jax_path: str
    hf_path: str
    transpose: bool = False

@dataclass
class ModelWeightMap:
    specs: List[WeightSpec]

# ==============================================================================
# 2. Path Traversal & Pattern Parsing
# ==============================================================================

def get_nested_val(d: dict, path: List[Union[str, int]]) -> Any:
    return functools.reduce(operator.getitem, path, d)

def set_nested_val(d: dict, path: List[Union[str, int]], val: Any) -> None:
    def traverse_or_create(curr: Any, pair: tuple) -> Any:
        key, next_key = pair
        default_next = [] if isinstance(next_key, int) else {}

        if isinstance(curr, list):
            while len(curr) <= key:
                curr.append(default_next.copy())
            return curr[key]
        return curr.setdefault(key, default_next)

    lookahead_pairs = list(zip(path[:-1], path[1:]))
    target_container = functools.reduce(traverse_or_create, lookahead_pairs, d)
    
    last_key = path[-1]
    if isinstance(target_container, list):
        while len(target_container) <= last_key:
            target_container.append(None)
    target_container[last_key] = val

def parse_path_pattern(pattern: str, **kwargs) -> List[Union[str, int]]:
    formatted = pattern.format(**kwargs)
    return [int(t) if t.isdigit() else t for t in formatted.split(".")]

# ==============================================================================
# 3. Automatic Stack/Dimension Expansion
# ==============================================================================

def _expand_spec(spec: WeightSpec, dim_sizes: Dict[str, int]):
    """
    Finds placeholders shared in jax_path (loop dimensions) vs placeholders 
    only present in hf_path (dimensions that need stacking/unstacking).
    """
    jax_placeholders = set(re.findall(r"\{([a-zA-Z0-9_]+)\}", spec.jax_path))
    hf_placeholders = set(re.findall(r"\{([a-zA-Z0-9_]+)\}", spec.hf_path))

    shared_placeholders = list(jax_placeholders & hf_placeholders)
    stack_placeholders = list(hf_placeholders - jax_placeholders)

    # Validate that all required dimensions exist in dim_sizes
    for p in shared_placeholders + stack_placeholders:
        if p not in dim_sizes:
            raise KeyError(f"Missing dimension limit for '{{{p}}}' in dim_sizes.")

    # 1. Expand standard outer loop/layer combinations
    if not shared_placeholders:
        outer_combos = [{}]
    else:
        shared_ranges = [range(dim_sizes[p]) for p in shared_placeholders]
        outer_combos = [dict(zip(shared_placeholders, idxs)) for idxs in itertools.product(*shared_ranges)]

    # 2. Generate sub-combos for stacked dimensions (e.g., experts)
    if not stack_placeholders:
        stack_combos = [{}]
    else:
        stack_ranges = [range(dim_sizes[p]) for p in stack_placeholders]
        stack_combos = [dict(zip(stack_placeholders, idxs)) for idxs in itertools.product(*stack_ranges)]

    for outer in outer_combos:
        yield spec, outer, stack_combos

# ==============================================================================
# 4. Checkpoint Loading & Saving
# ==============================================================================

def load_checkpoint(
    checkpoint_path: str, 
    weight_map: ModelWeightMap, 
    dim_sizes: Optional[Dict[str, int]] = None
) -> dict:
    dim_sizes = dim_sizes or {}

    if os.path.isdir(checkpoint_path):
        index_file = os.path.join(checkpoint_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                index = json.load(f)
            files = {os.path.join(checkpoint_path, v) for v in index["weight_map"].values()}
        else:
            files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
    else:
        files = [checkpoint_path]

    flat_tensors = {}
    for f in files:
        flat_tensors.update(load_file(f))

    weights = {}
    for spec in weight_map.specs:
        for expanded_spec, outer_kwargs, stack_kwargs_list in _expand_spec(spec, dim_sizes):
            jax_path_tokens = parse_path_pattern(expanded_spec.jax_path, **outer_kwargs)

            # Case A: Multiple HuggingFace keys collapse/stack into one JAX tensor
            if len(stack_kwargs_list) > 1:
                stacked_tensors = []
                for stack_kw in stack_kwargs_list:
                    hf_key = expanded_spec.hf_path.format(**outer_kwargs, **stack_kw)
                    t = flat_tensors[hf_key]
                    if expanded_spec.transpose:
                        t = t.T
                    stacked_tensors.append(t)
                set_nested_val(weights, jax_path_tokens, jnp.stack(stacked_tensors, axis=0))
            
            # Case B: Standard 1-to-1 weight mapping
            else:
                hf_key = expanded_spec.hf_path.format(**outer_kwargs)
                t = flat_tensors[hf_key]
                if expanded_spec.transpose:
                    t = t.T
                set_nested_val(weights, jax_path_tokens, jnp.array(t))

    return weights


def _parse_size_in_bytes(size: Union[int, str]) -> int:
    """Converts human-readable size strings (e.g. '5GB', '500MB') to byte count."""
    if isinstance(size, int):
        return size
    units = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12,
             "KIB": 1 << 10, "MIB": 1 << 20, "GIB": 1 << 30, "TIB": 1 << 40}
    match = re.match(r"^([0-9.]+)\s*([a-zA-Z]+)$", size.strip())
    if not match:
        raise ValueError(f"Invalid max_shard_size format: {size}")
    val, unit = match.groups()
    unit = unit.upper()
    if unit not in units:
        raise ValueError(f"Unknown size unit: {unit}")
    return int(float(val) * units[unit])


def save_checkpoint(
    weights: dict, 
    save_directory: str, 
    weight_map: ModelWeightMap, 
    dim_sizes: Optional[Dict[str, int]] = None,
    max_shard_size: Union[str, int] = "5GB"
) -> None:
    """
    Saves a nested JAX weight dictionary to single or multi-shard safetensors 
    with a HuggingFace-compatible index file.
    """
    dim_sizes = dim_sizes or {}
    max_shard_bytes = _parse_size_in_bytes(max_shard_size)
    os.makedirs(save_directory, exist_ok=True)

    # 1. Flatten and convert JAX weights into numpy dict
    flat_tensors: Dict[str, np.ndarray] = {}
    for spec in weight_map.specs:
        for expanded_spec, outer_kwargs, stack_kwargs_list in _expand_spec(spec, dim_sizes):
            jax_path_tokens = parse_path_pattern(expanded_spec.jax_path, **outer_kwargs)
            val = get_nested_val(weights, jax_path_tokens)

            # Case A: Tensor unstacks across missing dimensions (e.g., experts)
            if len(stack_kwargs_list) > 1:
                for idx, stack_kw in enumerate(stack_kwargs_list):
                    hf_key = expanded_spec.hf_path.format(**outer_kwargs, **stack_kw)
                    t = val[idx]
                    if expanded_spec.transpose:
                        t = t.T
                    flat_tensors[hf_key] = np.asarray(jax.device_get(t))

            # Case B: Standard 1-to-1 weight mapping
            else:
                hf_key = expanded_spec.hf_path.format(**outer_kwargs)
                t = val
                if expanded_spec.transpose:
                    t = t.T
                flat_tensors[hf_key] = np.asarray(jax.device_get(t))

    # 2. Partition tensors into shards based on max byte size
    total_size = sum(t.nbytes for t in flat_tensors.values())
    
    # Save as a single un-sharded file if within size limit
    if total_size <= max_shard_bytes:
        single_path = os.path.join(save_directory, "model.safetensors")
        save_file(flat_tensors, single_path)
        return

    shards: List[Dict[str, np.ndarray]] = []
    current_shard: Dict[str, np.ndarray] = {}
    current_shard_size = 0

    for name, tensor in flat_tensors.items():
        tensor_bytes = tensor.nbytes
        if current_shard and (current_shard_size + tensor_bytes > max_shard_bytes):
            shards.append(current_shard)
            current_shard = {}
            current_shard_size = 0

        current_shard[name] = tensor
        current_shard_size += tensor_bytes

    if current_shard:
        shards.append(current_shard)

    # 3. Write individual shard files and compile the index
    num_shards = len(shards)
    weight_to_file = {}

    for idx, shard in enumerate(shards, start=1):
        filename = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
        file_path = os.path.join(save_directory, filename)
        save_file(shard, file_path)

        for tensor_name in shard.keys():
            weight_to_file[tensor_name] = filename

    # 4. Write model.safetensors.index.json
    index_data = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_to_file
    }
    index_path = os.path.join(save_directory, "model.safetensors.index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)