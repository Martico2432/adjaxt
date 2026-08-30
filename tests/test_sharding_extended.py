import os
import tempfile
import pytest
import numpy as np
import jax.numpy as jnp
from adjaxt.sharding import (
    save_checkpoint,
    load_checkpoint,
    ModelWeightMap,
    WeightSpec,
    _parse_size_in_bytes,
)

def test_parse_size_in_bytes_units():
    assert _parse_size_in_bytes("100B") == 100
    assert _parse_size_in_bytes("2MB") == 2 * 10**6
    assert _parse_size_in_bytes("1GiB") == 1024**3
    with pytest.raises(ValueError):
        _parse_size_in_bytes("100XYZ")

def test_multi_shard_saving_and_loading():
    weight_map = ModelWeightMap(
        specs=[
            WeightSpec("layer_a", "model.layers.a.weight"),
            WeightSpec("layer_b", "model.layers.b.weight"),
        ]
    )
    # Generate ~2000 bytes of data
    weights = {
        "layer_a": jnp.ones((250, 4), dtype=jnp.float32),
        "layer_b": jnp.ones((250, 4), dtype=jnp.float32),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        # Force small shard threshold (1500 bytes) to create 2 shards
        save_checkpoint(
            weights=weights,
            save_directory=tmpdir,
            weight_map=weight_map,
            max_shard_size=1500,
        )

        assert os.path.exists(os.path.join(tmpdir, "model.safetensors.index.json"))
        assert os.path.exists(os.path.join(tmpdir, "model-00001-of-00002.safetensors"))
        assert os.path.exists(os.path.join(tmpdir, "model-00002-of-00002.safetensors"))

        loaded = load_checkpoint(tmpdir, weight_map=weight_map)
        np.testing.assert_allclose(np.array(loaded["layer_a"]), np.array(weights["layer_a"]))
        np.testing.assert_allclose(np.array(loaded["layer_b"]), np.array(weights["layer_b"]))