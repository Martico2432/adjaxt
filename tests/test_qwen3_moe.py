import os
import tempfile
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.nn.functional as F

from adjaxt.config import (
    RMSNormConfig,
    SwiGLUConfig,
    GQAAttnConfig,
    Qwen3AttnConfig,
    Qwen3MLPConfig,
    Qwen3MoEBlockConfig,
    Qwen3MoELayerConfig,
    Qwen3MoEModelConfig,
    StandardAttnImplementation,
)
from adjaxt.layers import (
    qwen3_mlp,
    qwen3_mlp_init,
    qwen3_moe_block,
    qwen3_moe_block_init,
    qwen3_moe_layer,
    qwen3_moe_layer_init,
    qwen3_moe_model,
    qwen3_moe_model_init,
)
from adjaxt.model_maps import QWEN3_MOE_WEIGHT_MAP
from adjaxt.sharding import save_checkpoint, load_checkpoint
from adjaxt.torch_layers import _to_torch

# ==============================================================================
# PyTorch Reference Implementations
# ==============================================================================

class PyTorchQwen3MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class PyTorchQwen3MoEBlock(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            PyTorchQwen3MLP(d_model, hidden_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        tokens = x.view(-1, self.d_model)
        logits = self.router(tokens)
        probs = F.softmax(logits.float(), dim=-1)
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(tokens)
        for t in range(tokens.size(0)):
            for k in range(self.top_k):
                exp_idx = topk_idx[t, k].item()
                weight = topk_probs[t, k]
                out[t] += weight * self.experts[exp_idx](tokens[t : t + 1]).squeeze(0)
        return out.view(orig_shape)


# ==============================================================================
# 1. Qwen3 MLP Parity Test
# ==============================================================================

@pytest.mark.parametrize("batch_size, seq_len", [(2, 8), (1, 16)])
@pytest.mark.parametrize("in_dim, hidden_dim", [(32, 64), (64, 128)])
def test_qwen3_mlp_matches_pytorch(batch_size, seq_len, in_dim, hidden_dim):
    cfg = Qwen3MLPConfig(act_fn=jax.nn.silu, in_dim=in_dim, hidden_dim=hidden_dim)
    w_jax = qwen3_mlp_init(jax.random.key(42), cfg)

    pt_mlp = PyTorchQwen3MLP(in_dim, hidden_dim)
    with torch.no_grad():
        w_gate_torch = _to_torch(w_jax["w_gate"])
        w_up_torch = _to_torch(w_jax["w_up"])
        w_down_torch = _to_torch(w_jax["w_down"])

        pt_mlp.w_gate.weight.copy_(w_gate_torch.T)
        pt_mlp.w_up.weight.copy_(w_up_torch.T)
        pt_mlp.w_down.weight.copy_(w_down_torch.T)

    x_np = np.random.randn(batch_size, seq_len, in_dim).astype(np.float32)
    jax_out = qwen3_mlp(jnp.array(x_np), w_jax, cfg)
    with torch.no_grad():
        pt_out = pt_mlp(torch.from_numpy(x_np))

    np.testing.assert_allclose(np.array(jax_out), pt_out.numpy(), rtol=1e-5, atol=1e-5)


# ==============================================================================
# 2. Qwen3 MoE Block Parity Test
# ==============================================================================

@pytest.mark.parametrize("d_model, hidden_dim, num_experts, top_k", [
    (32, 64, 4, 2),
    (64, 128, 8, 2),
])
def test_qwen3_moe_block_matches_pytorch(d_model, hidden_dim, num_experts, top_k):
    mlp_cfg = Qwen3MLPConfig(act_fn=jax.nn.silu, in_dim=d_model, hidden_dim=hidden_dim)
    moe_cfg = Qwen3MoEBlockConfig(
        mlp_conf=mlp_cfg,
        top_k=top_k,
        num_experts=num_experts,
        d_model=d_model,
    )
    w_jax = qwen3_moe_block_init(jax.random.key(123), moe_cfg)

    # Cast bf16 init to float32 for clean numerical testing
    w_jax_f32 = {
        "router": w_jax["router"].astype(jnp.float32),
        "experts": {
            k: v.astype(jnp.float32) for k, v in w_jax["experts"].items()
        }
    }

    pt_moe = PyTorchQwen3MoEBlock(d_model, hidden_dim, num_experts, top_k)
    with torch.no_grad():
        pt_moe.router.weight.copy_(torch.from_numpy(np.array(w_jax_f32["router"]).T))
        for e in range(num_experts):
            pt_moe.experts[e].w_gate.weight.copy_(torch.from_numpy(np.array(w_jax_f32["experts"]["w_gate"][e]).T))
            pt_moe.experts[e].w_up.weight.copy_(torch.from_numpy(np.array(w_jax_f32["experts"]["w_up"][e]).T))
            pt_moe.experts[e].w_down.weight.copy_(torch.from_numpy(np.array(w_jax_f32["experts"]["w_down"][e]).T))

    x_np = np.random.randn(2, 4, d_model).astype(np.float32)
    jax_out = qwen3_moe_block(jnp.array(x_np), w_jax_f32, moe_cfg)
    with torch.no_grad():
        pt_out = pt_moe(torch.from_numpy(x_np))

    np.testing.assert_allclose(np.array(jax_out), pt_out.numpy(), rtol=1e-4, atol=1e-4)


# ==============================================================================
# 3. Layer & Model Shape / Invariant Forward Passes
# ==============================================================================

def test_qwen3_moe_layer_execution():
    d_model, head_dim, num_heads, num_kv_heads = 64, 16, 4, 2
    attn_cfg = Qwen3AttnConfig(
        gqa_conf=GQAAttnConfig(StandardAttnImplementation.XLA, num_heads // num_kv_heads, is_causal=True),
        q_rms_conf=RMSNormConfig(dim=head_dim),
        k_rms_conf=RMSNormConfig(dim=head_dim),
        d=d_model,
        head_dim=head_dim,
        rope_theta=10000.0,
        max_position_embeddings=64,
        n_layers=1,
        num_heads=num_heads
    )
    moe_block_cfg = Qwen3MoEBlockConfig(
        mlp_conf=Qwen3MLPConfig(jax.nn.silu, in_dim=d_model, hidden_dim=128),
        top_k=2,
        num_experts=4,
        d_model=d_model,
    )
    layer_cfg = Qwen3MoELayerConfig(
        input_rms_conf=RMSNormConfig(dim=d_model),
        attn_conf=attn_cfg,
        post_attn_rms_conf=RMSNormConfig(dim=d_model),
        moe_block_conf=moe_block_cfg,
    )

    w = qwen3_moe_layer_init(jax.random.key(0), layer_cfg)
    x = jnp.ones((2, 8, d_model), dtype=jnp.float32)
    out = qwen3_moe_layer(x, w, layer_cfg)

    assert out.shape == (2, 8, d_model)
    assert not jnp.isnan(out).any()


@pytest.mark.parametrize("tie_weights", [True, False])
def test_qwen3_moe_model_shapes_and_jit(tie_weights):
    vocab_size, d_model, num_blocks = 100, 64, 2
    head_dim, num_heads, num_kv_heads = 16, 4, 2

    attn_cfg = Qwen3AttnConfig(
        gqa_conf=GQAAttnConfig(StandardAttnImplementation.XLA, num_heads // num_kv_heads, is_causal=True),
        q_rms_conf=RMSNormConfig(dim=head_dim),
        k_rms_conf=RMSNormConfig(dim=head_dim),
        d=d_model,
        head_dim=head_dim,
        rope_theta=10000.0,
        max_position_embeddings=32,
        n_layers=num_blocks,
        num_heads=num_heads
    )
    layer_cfg = Qwen3MoELayerConfig(
        input_rms_conf=RMSNormConfig(dim=d_model),
        attn_conf=attn_cfg,
        post_attn_rms_conf=RMSNormConfig(dim=d_model),
        moe_block_conf=Qwen3MoEBlockConfig(
            mlp_conf=Qwen3MLPConfig(jax.nn.silu, in_dim=d_model, hidden_dim=128),
            top_k=2,
            num_experts=4,
            d_model=d_model,
        ),
    )
    model_cfg = Qwen3MoEModelConfig(
        moe_layer_conf=layer_cfg,
        final_rms_conf=RMSNormConfig(dim=d_model),
        num_decoder_blocks=num_blocks,
        vocab_size=vocab_size,
        d_model=d_model,
        tie_word_embeddings=tie_weights,
    )

    w = qwen3_moe_model_init(jax.random.key(1), model_cfg)
    input_ids = jnp.array([[1, 5, 23, 8], [9, 12, 44, 2]], dtype=jnp.int32)

    # Validate non-jitted & JIT-compiled execution
    logits = qwen3_moe_model(input_ids, w, model_cfg)
    jitted_fn = jax.jit(lambda x: qwen3_moe_model(x, w, model_cfg))
    jitted_logits = jitted_fn(input_ids)

    assert logits.shape == (2, 4, vocab_size)
    assert jnp.allclose(logits, jitted_logits, atol=5e-2, rtol=5e-2)


# ==============================================================================
# 4. Checkpoint Save/Load Roundtrip Test
# ==============================================================================

def test_checkpoint_save_and_load_roundtrip():
    d_model, num_blocks, num_experts, vocab_size = 32, 2, 4, 50
    head_dim = 16

    attn_cfg = Qwen3AttnConfig(
        gqa_conf=GQAAttnConfig(StandardAttnImplementation.XLA, 1, is_causal=True),
        q_rms_conf=RMSNormConfig(dim=head_dim),
        k_rms_conf=RMSNormConfig(dim=head_dim),
        d=d_model,
        head_dim=head_dim,
        rope_theta=10000.0,
        max_position_embeddings=16,
        n_layers=num_blocks,
        num_heads=d_model//head_dim
    )
    layer_cfg = Qwen3MoELayerConfig(
        input_rms_conf=RMSNormConfig(dim=d_model),
        attn_conf=attn_cfg,
        post_attn_rms_conf=RMSNormConfig(dim=d_model),
        moe_block_conf=Qwen3MoEBlockConfig(
            mlp_conf=Qwen3MLPConfig(jax.nn.silu, in_dim=d_model, hidden_dim=64),
            top_k=2,
            num_experts=num_experts,
            d_model=d_model,
        ),
    )
    model_cfg = Qwen3MoEModelConfig(
        moe_layer_conf=layer_cfg,
        final_rms_conf=RMSNormConfig(dim=d_model),
        num_decoder_blocks=num_blocks,
        vocab_size=vocab_size,
        d_model=d_model,
    )

    original_weights = qwen3_moe_model_init(jax.random.key(77), model_cfg)
    dim_sizes = {"i": num_blocks, "e": num_experts}

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_checkpoint(
            weights=original_weights,
            save_directory=tmp_dir,
            weight_map=QWEN3_MOE_WEIGHT_MAP,
            dim_sizes=dim_sizes,
            max_shard_size="1MB",
        )

        loaded_weights = load_checkpoint(
            checkpoint_path=tmp_dir,
            weight_map=QWEN3_MOE_WEIGHT_MAP,
            dim_sizes=dim_sizes,
        )

        # Check PyTree parity across all mapped parameters
        def assert_leaves_match(orig, loaded):
            np.testing.assert_allclose(
                np.asarray(orig, dtype=np.float32),
                np.asarray(loaded, dtype=np.float32),
                rtol=1e-5,
                atol=1e-5,
            )

        jax.tree_util.tree_map(
            assert_leaves_match,
            original_weights["decoder_blocks"],
            loaded_weights["decoder_blocks"],
        )
        assert_leaves_match(original_weights["embeds"], loaded_weights["embeds"])
        assert_leaves_match(original_weights["norm"], loaded_weights["norm"])
        assert_leaves_match(original_weights["lm_head"], loaded_weights["lm_head"])