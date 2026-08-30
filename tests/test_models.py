import pytest
import jax.numpy as jnp
from adjaxt.models import create_qwen3_moe_config, _get_act_fn
from adjaxt.config import StandardAttnImplementation

def test_act_fn_lookup():
    assert _get_act_fn("silu") is not None
    assert _get_act_fn("gelu") is not None
    with pytest.raises(ValueError):
        _get_act_fn("unsupported_act")

def test_create_qwen3_moe_config():
    mock_hf_config = {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "rms_norm_eps": 1e-5,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "vocab_size": 1000,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "tie_word_embeddings": True
    }
    cfg = create_qwen3_moe_config(mock_hf_config, StandardAttnImplementation.XLA)
    assert cfg.vocab_size == 1000
    assert cfg.d_model == 128
    assert cfg.num_decoder_blocks == 2
    assert cfg.tie_word_embeddings is True
    assert cfg.moe_layer_conf.moe_block_conf.num_experts == 4
    assert cfg.moe_layer_conf.attn_conf.cos_table.shape[-1] == 32