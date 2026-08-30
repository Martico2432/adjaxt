import jax
import jax.numpy as jnp
import pytest
from adjaxt.config import PrecisionPolicy, RMSNormConfig, Qwen3MLPConfig, Qwen3AttnConfig, GQAAttnConfig, StandardAttnImplementation
from adjaxt.layers import rms_norm, rms_norm_init, qwen3_mlp, qwen3_mlp_init

def test_precision_policy_defaults():
    policy = PrecisionPolicy()
    assert policy.param_dtype == jnp.bfloat16
    assert policy.compute_dtype == jnp.bfloat16
    assert policy.output_dtype == jnp.float32
    assert policy.is_mixed

def test_rms_norm_precision_handling():
    policy = PrecisionPolicy(param_dtype=jnp.float32, compute_dtype=jnp.float16, output_dtype=jnp.float32)
    cfg = RMSNormConfig(dim=128, precision=policy)
    
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 16, 128), dtype=jnp.float32)
    w = rms_norm_init(key, cfg)
    
    out = rms_norm(x, w, cfg)
    assert out.dtype == jnp.float32
    assert out.shape == (2, 16, 128)

def test_mlp_precision_handling():
    policy = PrecisionPolicy(param_dtype=jnp.float32, compute_dtype=jnp.float16, output_dtype=jnp.float32)
    cfg = Qwen3MLPConfig(act_fn=jax.nn.silu, in_dim=64, hidden_dim=128, precision=policy)
    
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    
    x = jax.random.normal(k1, (2, 8, 64), dtype=jnp.float32)
    weights = qwen3_mlp_init(k2, cfg)
    
    out = qwen3_mlp(x, weights, cfg)
    assert out.dtype == jnp.float32
    assert out.shape == (2, 8, 64)