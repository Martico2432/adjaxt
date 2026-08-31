import pytest
import math
import numpy as np
import jax
import jax.numpy as jnp
import torch

from adjaxt.config import (
    PrecisionPolicy,
    RMSNormConfig,
    SwiGLUConfig,
    StandardAttnImplementation,
    GQAAttnConfig,
    Qwen3AttnConfig,
    Qwen3MLPConfig,
    Qwen3MoEBlockConfig,
    Qwen3MoELayerConfig,
    Qwen3MoEModelConfig,
)
from adjaxt.layers import (
    rms_norm,
    rms_norm_init,
    swiglu,
    swiglu_init,
    gqa_attn,
    qwen3_attn,
    qwen3_attn_init,
    qwen3_mlp,
    qwen3_mlp_init,
    qwen3_moe_block,
    qwen3_moe_block_init,
    qwen3_moe_layer,
    qwen3_moe_layer_init,
    qwen3_moe_model,
    qwen3_moe_model_init,
)
from adjaxt.torch_layers import to_torch_module, _to_torch

# ===================================================================================
# Helper Fixtures & Utilities
# ===================================================================================

@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)

@pytest.fixture
def f32_policy():
    return PrecisionPolicy(param_dtype=jnp.float32, compute_dtype=jnp.float32, output_dtype=jnp.float32)

def assert_allclose(jax_arr, torch_tensor, rtol=1e-3, atol=1e-3):
    jax_np = np.array(jax_arr, dtype=np.float32)
    torch_np = torch_tensor.detach().cpu().float().numpy()
    np.testing.assert_allclose(jax_np, torch_np, rtol=rtol, atol=atol)

# ===================================================================================
# 1. RMSNorm Test
# ===================================================================================
def test_rms_norm_conversion(rng_key, f32_policy):
    cfg = RMSNormConfig(dim=64, eps=1e-6, precision=f32_policy)
    w_jax = rms_norm_init(rng_key, cfg)
    
    torch_layer = to_torch_module(rms_norm, cfg, weights=w_jax)
    
    k1, _ = jax.random.split(rng_key)
    x_jax = jax.random.normal(k1, (2, 8, cfg.dim), dtype=jnp.float32)
    x_torch = _to_torch(x_jax, dtype=torch.float32)
    
    out_jax = rms_norm(x_jax, w_jax, cfg)
    out_torch = torch_layer(x_torch)
    
    assert_allclose(out_jax, out_torch, rtol=1e-5, atol=1e-5)

# ===================================================================================
# 2. SwiGLU Test
# ===================================================================================
def test_swiglu_conversion(rng_key, f32_policy):
    cfg = SwiGLUConfig(in_dim=32, hidden_dim=64, precision=f32_policy)
    w_jax = swiglu_init(rng_key, cfg)
    
    torch_layer = to_torch_module(swiglu, cfg, weights=w_jax)
    
    k1, _ = jax.random.split(rng_key)
    x_jax = jax.random.normal(k1, (2, 4, cfg.in_dim), dtype=jnp.float32)
    x_torch = _to_torch(x_jax, dtype=torch.float32)
    
    out_jax = swiglu(x_jax, w_jax, cfg)
    out_torch = torch_layer(x_torch)
    
    assert_allclose(out_jax, out_torch, rtol=1e-5, atol=1e-5)

# ===================================================================================
# 3. GQA Attention Test
# ===================================================================================
def test_gqa_attn_conversion(rng_key, f32_policy):
    cfg = GQAAttnConfig(
        implementation=StandardAttnImplementation.XLA,
        num_kv_groups=2,
        is_causal=True,
        precision=f32_policy
    )
    
    batch, seq_len, num_q_heads, num_kv_heads, head_dim = 2, 8, 4, 2, 16
    
    k1, k2, k3 = jax.random.split(rng_key, 3)
    q_jax = jax.random.normal(k1, (batch, seq_len, num_q_heads, head_dim), dtype=jnp.float32)
    k_jax = jax.random.normal(k2, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)
    v_jax = jax.random.normal(k3, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)
    
    q_torch = _to_torch(q_jax, dtype=torch.float32)
    k_torch = _to_torch(k_jax, dtype=torch.float32)
    v_torch = _to_torch(v_jax, dtype=torch.float32)
    
    torch_layer = to_torch_module(gqa_attn, cfg)
    
    out_jax = gqa_attn(q_jax, k_jax, v_jax, cfg)
    out_torch = torch_layer(q_torch, k_torch, v_torch)
    
    assert_allclose(out_jax, out_torch, rtol=1e-4, atol=1e-4)

# ===================================================================================
# 4. Qwen3 Attention Test
# ===================================================================================
def test_qwen3_attn_conversion(rng_key, f32_policy):
    d_model = 64
    head_dim = 16
    seq_len = 8
    n_heads = d_model // head_dim
    
    gqa_conf = GQAAttnConfig(
        implementation=StandardAttnImplementation.XLA,
        num_kv_groups=2,
        is_causal=True,
        precision=f32_policy
    )
    q_rms_conf = RMSNormConfig(dim=head_dim, eps=1e-6, precision=f32_policy)
    k_rms_conf = RMSNormConfig(dim=head_dim, eps=1e-6, precision=f32_policy)
    
    cfg = Qwen3AttnConfig(
        gqa_conf=gqa_conf,
        q_rms_conf=q_rms_conf,
        k_rms_conf=k_rms_conf,
        d=d_model,
        head_dim=head_dim,
        rope_theta=10000.0,
        max_position_embeddings=128,
        n_layers=2,
        num_heads=n_heads,
        precision=f32_policy
    )
    
    w_jax = qwen3_attn_init(rng_key, cfg)
    torch_layer = to_torch_module(qwen3_attn, cfg, weights=w_jax)
    
    k1, _ = jax.random.split(rng_key)
    x_jax = jax.random.normal(k1, (2, seq_len, d_model), dtype=jnp.float32)
    x_torch = _to_torch(x_jax, dtype=torch.float32)
    
    out_jax = qwen3_attn(x_jax, w_jax, cfg)
    out_torch = torch_layer(x_torch)
    
    assert_allclose(out_jax, out_torch, rtol=1e-4, atol=1e-4)

# ===================================================================================
# 5. Qwen3 MoE Block Test
# ===================================================================================
def test_qwen3_moe_block_conversion(rng_key, f32_policy):
    d_model = 32
    hidden_dim = 64
    num_experts = 4
    top_k = 2
    
    mlp_conf = Qwen3MLPConfig(
        act_fn=jax.nn.silu,
        in_dim=d_model,
        hidden_dim=hidden_dim,
        precision=f32_policy
    )
    cfg = Qwen3MoEBlockConfig(
        mlp_conf=mlp_conf,
        top_k=top_k,
        num_experts=num_experts,
        d_model=d_model,
        precision=f32_policy
    )
    
    w_jax = qwen3_moe_block_init(rng_key, cfg)
    torch_layer = to_torch_module(qwen3_moe_block, cfg, weights=w_jax).float()
    
    k1, _ = jax.random.split(rng_key)
    x_jax = jax.random.normal(k1, (2, 4, d_model), dtype=jnp.float32)
    x_torch = _to_torch(x_jax, dtype=torch.float32)
    
    out_jax = qwen3_moe_block(x_jax, w_jax, cfg)
    out_torch = torch_layer(x_torch)
    
    assert_allclose(out_jax, out_torch, rtol=1e-4, atol=1e-4)

# ===================================================================================
# 6. Qwen3 MoE Layer Test
# ===================================================================================
def test_qwen3_moe_layer_conversion(rng_key, f32_policy):
    d_model = 32
    head_dim = 8
    seq_len = 4
    
    gqa_conf = GQAAttnConfig(
        implementation=StandardAttnImplementation.XLA,
        num_kv_groups=2,
        is_causal=True,
        precision=f32_policy
    )
    q_rms_conf = RMSNormConfig(dim=head_dim, eps=1e-6, precision=f32_policy)
    k_rms_conf = RMSNormConfig(dim=head_dim, eps=1e-6, precision=f32_policy)
    
    attn_conf = Qwen3AttnConfig(
        gqa_conf=gqa_conf,
        q_rms_conf=q_rms_conf,
        k_rms_conf=k_rms_conf,
        d=d_model,
        head_dim=head_dim,
        rope_theta=10000.0,
        max_position_embeddings=128,
        n_layers=2,
        num_heads=d_model // head_dim,
        precision=f32_policy
    )
    
    mlp_conf = Qwen3MLPConfig(act_fn=jax.nn.silu, in_dim=d_model, hidden_dim=64, precision=f32_policy)
    moe_block_conf = Qwen3MoEBlockConfig(
        mlp_conf=mlp_conf,
        top_k=2,
        num_experts=4,
        d_model=d_model,
        precision=f32_policy
    )
    
    cfg = Qwen3MoELayerConfig(
        input_rms_conf=RMSNormConfig(dim=d_model, eps=1e-6, precision=f32_policy),
        attn_conf=attn_conf,
        post_attn_rms_conf=RMSNormConfig(dim=d_model, eps=1e-6, precision=f32_policy),
        moe_block_conf=moe_block_conf,
        precision=f32_policy
    )
    
    w_jax = qwen3_moe_layer_init(rng_key, cfg)
    torch_layer = to_torch_module(qwen3_moe_layer, cfg, weights=w_jax).float()
    
    k1, _ = jax.random.split(rng_key)
    x_jax = jax.random.normal(k1, (2, seq_len, d_model), dtype=jnp.float32)
    x_torch = _to_torch(x_jax, dtype=torch.float32)
    
    out_jax = qwen3_moe_layer(x_jax, w_jax, cfg)
    out_torch = torch_layer(x_torch)
    
    assert_allclose(out_jax, out_torch, rtol=1e-4, atol=1e-4)

# ===================================================================================
# 7. Full Qwen3 MoE Model Test
# ===================================================================================
@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_qwen3_moe_model_conversion(rng_key, tie_word_embeddings, f32_policy):
    vocab_size = 128
    d_model = 32
    head_dim = 8
    seq_len = 4
    
    gqa_conf = GQAAttnConfig(
        implementation=StandardAttnImplementation.XLA,
        num_kv_groups=2,
        is_causal=True,
        precision=f32_policy
    )
    q_rms_conf = RMSNormConfig(dim=head_dim, eps=1e-6, precision=f32_policy)
    k_rms_conf = RMSNormConfig(dim=head_dim, eps=1e-6, precision=f32_policy)
    
    attn_conf = Qwen3AttnConfig(
        gqa_conf=gqa_conf,
        q_rms_conf=q_rms_conf,
        k_rms_conf=k_rms_conf,
        d=d_model,
        head_dim=head_dim,
        rope_theta=10000.0,
        max_position_embeddings=128,
        n_layers=2,
        num_heads=d_model // head_dim,
        precision=f32_policy
    )
    
    mlp_conf = Qwen3MLPConfig(act_fn=jax.nn.silu, in_dim=d_model, hidden_dim=64, precision=f32_policy)
    moe_block_conf = Qwen3MoEBlockConfig(
        mlp_conf=mlp_conf,
        top_k=2,
        num_experts=4,
        d_model=d_model,
        precision=f32_policy
    )
    
    moe_layer_conf = Qwen3MoELayerConfig(
        input_rms_conf=RMSNormConfig(dim=d_model, eps=1e-6, precision=f32_policy),
        attn_conf=attn_conf,
        post_attn_rms_conf=RMSNormConfig(dim=d_model, eps=1e-6, precision=f32_policy),
        moe_block_conf=moe_block_conf,
        precision=f32_policy
    )
    
    cfg = Qwen3MoEModelConfig(
        moe_layer_conf=moe_layer_conf,
        final_rms_conf=RMSNormConfig(dim=d_model, eps=1e-6, precision=f32_policy),
        num_decoder_blocks=2,
        vocab_size=vocab_size,
        d_model=d_model,
        tie_word_embeddings=tie_word_embeddings,
        precision=f32_policy
    )
    
    w_jax = qwen3_moe_model_init(rng_key, cfg)
    torch_model = to_torch_module(qwen3_moe_model, cfg, weights=w_jax).float()
    
    tokens = np.random.randint(0, vocab_size, size=(2, seq_len))
    tokens_jax = jnp.array(tokens)
    tokens_torch = torch.from_numpy(tokens).long()
    
    logits_jax = qwen3_moe_model(tokens_jax, w_jax, cfg)
    logits_torch = torch_model(tokens_torch)
    
    assert_allclose(logits_jax, logits_torch, rtol=1e-4, atol=1e-4)