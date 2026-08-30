import numpy as np
import jax, jax.numpy as jnp
import torch, torch.nn as nn
import torch.nn.functional as F
import pytest
from typing import Optional, Tuple
from dataclasses import dataclass

from adjaxt.layers import rms_norm, rms_norm_init, swiglu, swiglu_init, gqa_attn, qwen3_attn, qwen3_attn_init
from adjaxt.config import RMSNormConfig, SwiGLUConfig, GQAAttnConfig, Qwen3AttnConfig, compute_rope_freqs, StandardAttnImplementation

def torch_rms_norm(x, w, eps):
    """Reference: HF Qwen3RMSNorm, verbatim logic."""
    dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return w * x.to(dtype)


#@pytest.mark.skip()
@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
@pytest.mark.parametrize("shape", [(1, 5, 64), (2, 17, 128)])
@pytest.mark.parametrize("key", [0, 1, 12])
def test_rms_norm_matches_hf(dtype, shape, key):
    rng = np.random.default_rng(key)
    x_np = rng.standard_normal(shape).astype(np.float32)
    cfg = RMSNormConfig(dim=shape[-1], eps=1e-6)  # or however minimal Config construction looks
    #w_np = rng.standard_normal(shape[-1]).astype(np.float32)
    w_np = rms_norm_init(key, cfg)

    if dtype == "bfloat16":
        x_j, w_j = jnp.asarray(x_np, jnp.bfloat16), jnp.asarray(w_np, jnp.float32)
        x_t, w_t = torch.tensor(x_np).bfloat16(), torch.tensor(w_np)
        tol = 1e-2
    else:
        x_j, w_j = jnp.asarray(x_np), jnp.asarray(w_np)
        x_t, w_t = torch.tensor(x_np), torch.tensor(w_np)
        tol = 1e-5

    ours = np.asarray(rms_norm(x_j, w_j, cfg), dtype=np.float32)
    ref = torch_rms_norm(x_t, w_t, cfg.eps).float().numpy()

    np.testing.assert_allclose(ours, ref, rtol=tol, atol=tol)

def torch_swiglu(x, w):
    gate = torch.nn.functional.silu(torch.nn.functional.linear(x, torch.from_numpy(np.asarray(w["w_gate"])).T))
    up = torch.nn.functional.linear(x, torch.from_numpy(np.asarray(w["w_up"])).T)
    return torch.nn.functional.linear(gate * up, torch.from_numpy(np.asarray(w["w_down"]).T))

#@pytest.mark.skip()
#@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
@pytest.mark.parametrize("in_dim", [1, 5, 64, 2, 17, 128])
@pytest.mark.parametrize("hidden_dim", [1, 5, 64, 2, 17, 128])
@pytest.mark.parametrize("k",  [0, 1, 12])
def test_swiglu_matches_hf(in_dim, hidden_dim, k):
    key = jax.random.key(seed=k)
    rng = np.random.default_rng(seed=k)
    x_np = rng.standard_normal(in_dim).astype(np.float32)
    cfg = SwiGLUConfig(in_dim, hidden_dim)
    w_np = swiglu_init(key, cfg)

    ours = np.asarray(swiglu(jnp.array(x_np), w_np, cfg), dtype=np.float32)
    ref = torch_swiglu(torch.from_numpy(x_np), w_np)
    tol = 1e-5

    np.testing.assert_allclose(ours, ref, rtol=tol, atol=tol)

@dataclass(frozen=True)
class JaxGQAAttnConfig:
    implementation: Optional[str] = None
    num_kv_groups: Optional[int] = None
    is_causal: bool = False


@dataclass(frozen=True)
class TorchGQAAttnConfig:
    is_causal: bool = False

def torch_compute_rope_freqs(
    seq_len: int, head_dim: int, theta: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (dim_indices / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    freqs = torch.cat([freqs, freqs], dim=-1)
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(2)
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(2)
    return cos, sin


def torch_gqa_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: Optional[TorchGQAAttnConfig] = None,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if cfg is None:
        cfg = TorchGQAAttnConfig()

    batch, q_len, num_q_heads, head_dim = q.shape
    _, kv_len, num_kv_heads, _ = k.shape
    num_groups = num_q_heads // num_kv_heads

    q_grouped = q.view(
        batch, q_len, num_kv_heads, num_groups, head_dim
    ).permute(0, 2, 3, 1, 4)
    k_grouped = k.view(batch, kv_len, num_kv_heads, 1, head_dim).permute(
        0, 2, 3, 1, 4
    )
    v_grouped = v.view(batch, kv_len, num_kv_heads, 1, head_dim).permute(
        0, 2, 3, 1, 4
    )

    is_causal = cfg.is_causal if attn_mask is None else False

    out = F.scaled_dot_product_attention(
        query=q_grouped,
        key=k_grouped,
        value=v_grouped,
        attn_mask=attn_mask,
        is_causal=is_causal,
    )
    return out.permute(0, 3, 1, 2, 4).reshape(
        batch, q_len, num_q_heads, head_dim
    )

#@pytest.mark.skip()
@pytest.mark.parametrize("seq_len", [16, 64, 128])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_rope_freqs_match(seq_len: int, head_dim: int):
    """Validates that JAX and PyTorch RoPE tables are identical."""
    j_cos, j_sin = compute_rope_freqs(seq_len, head_dim)
    t_cos, t_sin = torch_compute_rope_freqs(seq_len, head_dim)
    np.testing.assert_allclose(
        np.array(j_cos), t_cos.numpy(), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        np.array(j_sin), t_sin.numpy(), rtol=1e-5, atol=1e-6
    )

#@pytest.mark.skip()
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads",
    [
        (8, 2),  # Standard GQA
        (8, 1),  # Multi-Query Attention (MQA)
        (8, 8),  # Multi-Head Attention (MHA)
    ],
)

#@pytest.mark.skip()
@pytest.mark.parametrize("is_causal", [False, True])
def test_gqa_attention_match(
    num_q_heads: int, num_kv_heads: int, is_causal: bool
):
    """Feeds identical numpy data to JAX and PyTorch to verify identical output."""
    np.random.seed(42)
    batch, seq_len, head_dim = 2, 16, 32
    q_np = np.random.randn(batch, seq_len, num_q_heads, head_dim).astype(
        np.float32
    )
    k_np = np.random.randn(batch, seq_len, num_kv_heads, head_dim).astype(
        np.float32
    )
    v_np = np.random.randn(batch, seq_len, num_kv_heads, head_dim).astype(
        np.float32
    )
    # 1. Run JAX implementation
    jax_cfg = JaxGQAAttnConfig(is_causal=is_causal)
    jax_out = gqa_attn(
        jnp.array(q_np), jnp.array(k_np), jnp.array(v_np), cfg=jax_cfg
    )
    # 2. Run PyTorch implementation
    torch_cfg = TorchGQAAttnConfig(is_causal=is_causal)
    torch_out = torch_gqa_attn(
        torch.from_numpy(q_np),
        torch.from_numpy(k_np),
        torch.from_numpy(v_np),
        cfg=torch_cfg,
    )
    # 3. Assert equality across frameworks
    np.testing.assert_allclose(
        np.array(jax_out), torch_out.numpy(), rtol=1e-4, atol=1e-4
    )

class PyTorchRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_pt(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # Shape of q, k: (B, H, S, D); cos, sin: (1, S, 1, D) -> permuted to (1, 1, S, D)
    cos = cos.transpose(1, 2)
    sin = sin.transpose(1, 2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PyTorchQwen3Attention(nn.Module):
    def __init__(self, d: int, num_heads: int, num_kv_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(d, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, d, bias=False)

        self.q_norm = PyTorchRMSNorm(head_dim, eps=eps)
        self.k_norm = PyTorchRMSNorm(head_dim, eps=eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # QK per-head RMSNorm
        q = self.q_norm(q).transpose(1, 2)  # (bsz, num_heads, seq_len, head_dim)
        k = self.k_norm(k).transpose(1, 2)  # (bsz, num_kv_heads, seq_len, head_dim)
        v = v.transpose(1, 2)               # (bsz, num_kv_heads, seq_len, head_dim)

        q, k = apply_rotary_pos_emb_pt(q, k, cos, sin)

        # Grouped Query Attention repeat KV
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled Dot-Product Attention (Causal)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_out)

#@pytest.mark.skip()
@pytest.mark.parametrize("batch_size, seq_len", [(2, 16), (1, 32)])
@pytest.mark.parametrize("d, head_dim, num_heads, num_kv_heads", [
    (128, 32, 4, 2),   # GQA: 4 query heads, 2 KV heads
    (64, 16, 4, 4),    # MHA: 4 query heads, 4 KV heads
])
def test_qwen3_attn_matches_pytorch(batch_size, seq_len, d, head_dim, num_heads, num_kv_heads):
    # Setup configs
    q_rms_conf = RMSNormConfig(dim=head_dim, eps=1e-6)
    k_rms_conf = RMSNormConfig(dim=head_dim, eps=1e-6)
    gqa_conf = GQAAttnConfig(
        implementation=StandardAttnImplementation.XLA,
        num_kv_groups=num_heads // num_kv_heads,
        is_causal=True
    )
    
    jax_cfg = Qwen3AttnConfig(
        gqa_conf=gqa_conf,
        q_rms_conf=q_rms_conf,
        k_rms_conf=k_rms_conf,
        d=d,
        head_dim=head_dim,
        rope_theta=10000.0,
        max_position_embeddings=128,
        n_layers=1,
        num_heads=num_heads
    )

    # Deterministic input states
    np.random.seed(42)
    x_np = np.random.randn(batch_size, seq_len, d).astype(np.float32)

    # Initialize JAX weights

    rkey = jax.random.key(42)
    w_jax = qwen3_attn_init(rkey, jax_cfg)

    #w_jax = {
    #    "q_proj": jnp.array(np.random.randn(d, num_heads * head_dim), dtype=jnp.float32),
    #    "k_proj": jnp.array(np.random.randn(d, num_kv_heads * head_dim), dtype=jnp.float32),
    #    "v_proj": jnp.array(np.random.randn(d, num_kv_heads * head_dim), dtype=jnp.float32),
    #    "o_proj": jnp.array(np.random.randn(num_heads * head_dim, d), dtype=jnp.float32),
    #    "q_norm": jnp.ones((head_dim,), dtype=jnp.float32),
    #    "k_norm": jnp.ones((head_dim,), dtype=jnp.float32)
    #}

    # Initialize PyTorch module & assign identical weights
    pt_layer = PyTorchQwen3Attention(
        d=d, 
        num_heads=num_heads, 
        num_kv_heads=num_kv_heads, 
        head_dim=head_dim, 
        eps=1e-6
    ).to(torch.float32)

    with torch.no_grad():
        # PyTorch linear weights are transposed (out_features, in_features)
        pt_layer.q_proj.weight.copy_(torch.from_numpy(np.array(w_jax["q_proj"].astype(jnp.float32).T)))
        pt_layer.k_proj.weight.copy_(torch.from_numpy(np.array(w_jax["k_proj"].astype(jnp.float32).T)))
        pt_layer.v_proj.weight.copy_(torch.from_numpy(np.array(w_jax["v_proj"].astype(jnp.float32).T)))
        pt_layer.o_proj.weight.copy_(torch.from_numpy(np.array(w_jax["o_proj"].astype(jnp.float32).T)))
        pt_layer.q_norm.weight.copy_(torch.from_numpy(np.array(w_jax["q_norm"].astype(jnp.float32))))
        pt_layer.k_norm.weight.copy_(torch.from_numpy(np.array(w_jax["k_norm"].astype(jnp.float32))))

    # Execute JAX Attention
    x_jax = jnp.array(x_np)
    jax_out = qwen3_attn(x_jax, w_jax, jax_cfg)

    # Execute PyTorch Attention
    x_pt = torch.from_numpy(x_np)
    cos_pt = torch.from_numpy(np.array(jax_cfg.cos_table[:, :seq_len, :, :]))
    sin_pt = torch.from_numpy(np.array(jax_cfg.sin_table[:, :seq_len, :, :]))
    
    with torch.no_grad():
        pt_out = pt_layer(x_pt, cos_pt, sin_pt)

    # Numerical parity verification (allowing for slight floating point precision differences)
    np.testing.assert_allclose(
        np.array(jax_out), 
        pt_out.numpy(), 
        rtol=1e-3, 
        atol=1e-4,
        err_msg="JAX output does not match PyTorch baseline!"
    )

