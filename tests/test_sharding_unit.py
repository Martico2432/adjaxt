import pytest
from adjaxt.sharding import (
    parse_path_pattern,
    get_nested_val,
    set_nested_val,
    _parse_size_in_bytes,
    _expand_spec,
    WeightSpec,
)

def test_path_traversal_and_setting():
    tree = {}
    tokens = parse_path_pattern("decoder_blocks.{i}.attn.q_proj", i=0)
    assert tokens == ["decoder_blocks", 0, "attn", "q_proj"]
    
    set_nested_val(tree, tokens, 42)
    assert tree["decoder_blocks"][0]["attn"]["q_proj"] == 42
    assert get_nested_val(tree, tokens) == 42

def test_parse_size_in_bytes():
    assert _parse_size_in_bytes(1024) == 1024
    assert _parse_size_in_bytes("1KB") == 1000
    assert _parse_size_in_bytes("1KiB") == 1024
    assert _parse_size_in_bytes("5GB") == 5 * 10**9
    with pytest.raises(ValueError):
        _parse_size_in_bytes("invalid_size")

def test_expand_spec():
    spec = WeightSpec(
        jax_path="decoder_blocks.{i}.mlp.experts.w_gate",
        hf_path="model.layers.{i}.mlp.experts.{e}.gate_proj.weight"
    )
    dim_sizes = {"i": 2, "e": 3}
    expansions = list(_expand_spec(spec, dim_sizes))
    assert len(expansions) == 2
    for s, outer, stack_list in expansions:
        assert "i" in outer
        assert len(stack_list) == 3