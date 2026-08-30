import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adjaxt.layers import (
    rms_norm,
    swiglu,
    gqa_attn,
    qwen3_moe_layer,
    qwen3_attn,
    qwen3_mlp,
    qwen3_moe_model,
    qwen3_moe_block,
    rotate_half,
    apply_rot_pos_emb
)

from adjaxt.config import (
    RMSNormConfig,
    SwiGLUConfig,
    GQAAttnConfig,
    Qwen3AttnConfig,
    Qwen3MLPConfig,
    Qwen3MoEBlockConfig,
    Qwen3MoELayerConfig,
    Qwen3MoEModelConfig,
)

JAX_TO_TORCH_REGISTRY = {}

def maps_jax_layer(jax_fn):
    def decorator(torch_cls):
        JAX_TO_TORCH_REGISTRY[jax_fn] = torch_cls
        jax_fn.torch_cls = torch_cls
        return torch_cls
    return decorator


def _to_torch(val, dtype=None):
    if isinstance(val, dict):
        return {k: _to_torch(v, dtype=dtype) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_torch(v, dtype=dtype) for v in val]
    if isinstance(val, torch.Tensor):
        return val.to(dtype) if dtype is not None else val
    
    # Handle JAX/ml_dtypes bfloat16 converting to NumPy
    if hasattr(val, "dtype") and str(val.dtype) == "bfloat16":
        # Casting to float32 natively forces a writable copy
        np_arr = np.array(val, dtype=np.float32)
        t = torch.from_numpy(np_arr)
        return t.to(dtype if dtype is not None else torch.bfloat16)

    # Force a copy for all other types to fix the "not writable" PyTorch warning
    np_arr = np.array(val, copy=True)
    t = torch.from_numpy(np_arr)
    
    # Safely apply dtype only if it is provided
    return t.to(dtype) if dtype is not None else t

# ===================================================================================
# RMS Norm
# ===================================================================================

@maps_jax_layer(rms_norm)
class TorchRMSNorm(nn.Module):
    def __init__(self, cfg: RMSNormConfig, w=None):
        super().__init__()
        self.eps = cfg.eps
        if w is not None:
            self.weight = nn.Parameter(_to_torch(w, dtype=torch.float32))
        else:
            self.weight = nn.Parameter(torch.ones(cfg.dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, w: torch.Tensor = None) -> torch.Tensor:
        weight = self.weight if w is None else w.float()
        x32 = x.float()
        ms = torch.mean(x32.pow(2), dim=-1, keepdim=True)
        return (weight * (x32 * torch.rsqrt(ms + self.eps))).to(x.dtype)

# ===================================================================================
# RoPE
# ===================================================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rot_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

# ===================================================================================
# SwiGLU
# ===================================================================================

@maps_jax_layer(swiglu)
class TorchSwiGLU(nn.Module):
    def __init__(self, cfg: SwiGLUConfig, w: dict = None):
        super().__init__()
        if w is not None:
            self.w_gate = nn.Parameter(_to_torch(w["w_gate"]))
            self.w_up = nn.Parameter(_to_torch(w["w_up"]))
            self.w_down = nn.Parameter(_to_torch(w["w_down"]))
        else:
            self.w_gate = nn.Parameter(torch.randn(cfg.in_dim, cfg.hidden_dim, dtype=torch.float32) * math.sqrt(2.0 / cfg.in_dim))
            self.w_up = nn.Parameter(torch.randn(cfg.in_dim, cfg.hidden_dim, dtype=torch.float32) * math.sqrt(2.0 / cfg.in_dim))
            self.w_down = nn.Parameter(torch.randn(cfg.hidden_dim, cfg.in_dim, dtype=torch.float32) * math.sqrt(2.0 / cfg.hidden_dim))

    def forward(self, x: torch.Tensor, w: dict = None) -> torch.Tensor:
        w_gate = self.w_gate if w is None else w["w_gate"]
        w_up = self.w_up if w is None else w["w_up"]
        w_down = self.w_down if w is None else w["w_down"]
        return (F.silu(x @ w_gate) * (x @ w_up)) @ w_down

# ===================================================================================
# GQA Attention
# ===================================================================================

@maps_jax_layer(gqa_attn)
class TorchGQAAttn(nn.Module):
    def __init__(self, cfg: GQAAttnConfig, w=None):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        target_dtype = q.dtype
        k = k.to(target_dtype)
        v = v.to(target_dtype)

        _, _, num_q_heads, _ = q.shape
        _, _, num_kv_heads, _ = k.shape

        num_groups = num_q_heads // num_kv_heads
        if num_groups > 1:
            k = k.repeat_interleave(num_groups, dim=2)
            v = v.repeat_interleave(num_groups, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        causal_flag = self.cfg.is_causal if attn_mask is None else False
        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            is_causal=causal_flag,
        )
        return out.transpose(1, 2)

# ===================================================================================
# Qwen3 Attention
# ===================================================================================

@maps_jax_layer(qwen3_attn)
class TorchQwen3Attn(nn.Module):
    def __init__(self, cfg: Qwen3AttnConfig, w: dict = None):
        super().__init__()
        self.cfg = cfg
        self.q_rms = TorchRMSNorm(cfg.q_rms_conf)
        self.k_rms = TorchRMSNorm(cfg.k_rms_conf)
        self.gqa = TorchGQAAttn(cfg.gqa_conf)

        self.register_buffer("cos_table", _to_torch(cfg.cos_table, dtype=torch.float32), persistent=False)
        self.register_buffer("sin_table", _to_torch(cfg.sin_table, dtype=torch.float32), persistent=False)

        q_dim = cfg.d
        kv_dim = cfg.d // cfg.gqa_conf.num_kv_groups

        if w is not None:
            self.q_proj = nn.Parameter(_to_torch(w["q_proj"]))
            self.k_proj = nn.Parameter(_to_torch(w["k_proj"]))
            self.v_proj = nn.Parameter(_to_torch(w["v_proj"]))
            self.o_proj = nn.Parameter(_to_torch(w["o_proj"]))
            self.q_norm = nn.Parameter(_to_torch(w["q_norm"], dtype=torch.float32))
            self.k_norm = nn.Parameter(_to_torch(w["k_norm"], dtype=torch.float32))
        else:
            res = 1.0 / math.sqrt(2 * cfg.n_layers)
            self.q_proj = nn.Parameter(torch.randn(cfg.d, q_dim, dtype=torch.bfloat16) / math.sqrt(cfg.d))
            self.k_proj = nn.Parameter(torch.randn(cfg.d, kv_dim, dtype=torch.bfloat16) / math.sqrt(cfg.d))
            self.v_proj = nn.Parameter(torch.randn(cfg.d, kv_dim, dtype=torch.bfloat16) / math.sqrt(cfg.d))
            self.o_proj = nn.Parameter(torch.randn(q_dim, cfg.d, dtype=torch.bfloat16) * res / math.sqrt(q_dim))
            self.q_norm = nn.Parameter(torch.ones(cfg.head_dim, dtype=torch.float32))
            self.k_norm = nn.Parameter(torch.ones(cfg.head_dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, w: dict = None) -> torch.Tensor:
        seq_len = x.shape[1]
        input_shape = x.shape[:-1]
        
        q_heads = self.cfg.d // self.cfg.head_dim
        kv_heads = q_heads // self.cfg.gqa_conf.num_kv_groups
        
        q_shape = (*input_shape, q_heads, self.cfg.head_dim)
        kv_shape = (*input_shape, kv_heads, self.cfg.head_dim)

        q_proj = self.q_proj if w is None else w["q_proj"]
        k_proj = self.k_proj if w is None else w["k_proj"]
        v_proj = self.v_proj if w is None else w["v_proj"]
        o_proj = self.o_proj if w is None else w["o_proj"]
        q_norm_w = self.q_norm if w is None else w["q_norm"]
        k_norm_w = self.k_norm if w is None else w["k_norm"]

        q = self.q_rms((x @ q_proj).view(q_shape), q_norm_w)
        k = self.k_rms((x @ k_proj).view(kv_shape), k_norm_w)
        v = (x @ v_proj).view(kv_shape)

        cos = self.cos_table[:, :seq_len, :, :]
        sin = self.sin_table[:, :seq_len, :, :]
        q, k = apply_rot_pos_emb(q, k, cos, sin)

        attn_out = self.gqa(q, k, v)
        attn_out = attn_out.reshape(*input_shape, -1)
        return attn_out @ o_proj

# ===================================================================================
# Qwen3 MLP
# ===================================================================================

@maps_jax_layer(qwen3_mlp)
class TorchQwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3MLPConfig, w: dict = None):
        super().__init__()
        self.cfg = cfg
        self.act_fn = getattr(cfg, "act_fn", F.silu)
        if w is not None:
            self.w_gate = nn.Parameter(_to_torch(w["w_gate"]))
            self.w_up = nn.Parameter(_to_torch(w["w_up"]))
            self.w_down = nn.Parameter(_to_torch(w["w_down"]))
        else:
            self.w_gate = nn.Parameter(torch.randn(cfg.in_dim, cfg.hidden_dim, dtype=torch.float32) * math.sqrt(2.0 / cfg.in_dim))
            self.w_up = nn.Parameter(torch.randn(cfg.in_dim, cfg.hidden_dim, dtype=torch.float32) * math.sqrt(2.0 / cfg.in_dim))
            self.w_down = nn.Parameter(torch.randn(cfg.hidden_dim, cfg.in_dim, dtype=torch.float32) * math.sqrt(2.0 / cfg.hidden_dim))

    def forward(self, x: torch.Tensor, w: dict = None) -> torch.Tensor:
        w_gate = self.w_gate if w is None else w["w_gate"]
        w_up = self.w_up if w is None else w["w_up"]
        w_down = self.w_down if w is None else w["w_down"]
        return (self.act_fn(x @ w_gate) * (x @ w_up)) @ w_down

# ===================================================================================
# Qwen3 MoE Block
# ===================================================================================

@maps_jax_layer(qwen3_moe_block)
class TorchQwen3MoEBlock(nn.Module):
    def __init__(self, cfg: Qwen3MoEBlockConfig, w: dict = None):
        super().__init__()
        self.cfg = cfg
        self.act_fn = getattr(cfg.mlp_conf, "act_fn", F.silu)
        if w is not None:
            self.router = nn.Parameter(_to_torch(w["router"]))
            self.experts_w_gate = nn.Parameter(_to_torch(w["experts"]["w_gate"]))
            self.experts_w_up = nn.Parameter(_to_torch(w["experts"]["w_up"]))
            self.experts_w_down = nn.Parameter(_to_torch(w["experts"]["w_down"]))
        else:
            self.router = nn.Parameter(torch.randn(cfg.d_model, cfg.num_experts, dtype=torch.bfloat16) * (cfg.d_model ** -0.5))
            self.experts_w_gate = nn.Parameter(torch.randn(cfg.num_experts, cfg.mlp_conf.in_dim, cfg.mlp_conf.hidden_dim, dtype=torch.bfloat16) * math.sqrt(2.0 / cfg.mlp_conf.in_dim))
            self.experts_w_up = nn.Parameter(torch.randn(cfg.num_experts, cfg.mlp_conf.in_dim, cfg.mlp_conf.hidden_dim, dtype=torch.bfloat16) * math.sqrt(2.0 / cfg.mlp_conf.in_dim))
            self.experts_w_down = nn.Parameter(torch.randn(cfg.num_experts, cfg.mlp_conf.hidden_dim, cfg.mlp_conf.in_dim, dtype=torch.bfloat16) * math.sqrt(2.0 / cfg.mlp_conf.hidden_dim))

    def forward(self, x: torch.Tensor, w: dict = None) -> torch.Tensor:
        router = self.router if w is None else w["router"]
        w_gate = self.experts_w_gate if w is None else w["experts"]["w_gate"]
        w_up = self.experts_w_up if w is None else w["experts"]["w_up"]
        w_down = self.experts_w_down if w is None else w["experts"]["w_down"]

        tokens = x.reshape(-1, x.shape[-1])
        router_logits = tokens.to(router.dtype) @ router
        probs = F.softmax(router_logits.float(), dim=-1)
        topk_probs, topk_idx = torch.topk(probs, self.cfg.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        weights = torch.zeros_like(probs).scatter_(1, topk_idx, topk_probs.to(probs.dtype))

        # Ensure tokens match expert weight precision
        expert_tokens = tokens.to(w_gate.dtype)
        \
        #gate_out = self.act_fn(torch.einsum("td,edh->eth", expert_tokens, w_gate))
        #up_out = torch.einsum("td,edh->eth", expert_tokens, w_up)
        #expert_out = torch.einsum("eth,ehd->etd", gate_out * up_out, w_down)

        # Replace einsum with matmul (broadcasting handles the 'E' experts dimension)
        gate_out = F.silu(torch.matmul(tokens.unsqueeze(0), w_gate)) # (E, T, H)
        up_out   = torch.matmul(tokens.unsqueeze(0), w_up)           # (E, T, H)
        expert_out = torch.matmul(gate_out * up_out, w_down)         # (E, T, D)

        # Replace final einsum with batched matmul
        weights_exp = weights.unsqueeze(1).to(x.dtype)               # (T, 1, E)
        expert_out_perm = expert_out.permute(1, 0, 2).to(x.dtype)    # (T, E, D)
        
        out = torch.matmul(weights_exp, expert_out_perm).squeeze(1)  # (T, D)
        return out.reshape(x.shape)

# ===================================================================================
# Qwen3 MoE Layer
# ===================================================================================

@maps_jax_layer(qwen3_moe_layer)
class TorchQwen3MoELayer(nn.Module):
    def __init__(self, cfg: Qwen3MoELayerConfig, w: dict = None):
        super().__init__()
        self.cfg = cfg
        self.input_layernorm = TorchRMSNorm(cfg.input_rms_conf, w["input_layernorm"] if w else None)
        self.attn = TorchQwen3Attn(cfg.attn_conf, w["attn"] if w else None)
        self.post_attn_layernorm = TorchRMSNorm(cfg.post_attn_rms_conf, w["post_attn_layernorm"] if w else None)
        self.mlp = TorchQwen3MoEBlock(cfg.moe_block_conf, w["mlp"] if w else None)

    def forward(self, x: torch.Tensor, w: dict = None) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x, None if w is None else w["input_layernorm"])
        x = self.attn(x, None if w is None else w["attn"])
        x = x + residual
        residual = x
        x = self.post_attn_layernorm(x, None if w is None else w["post_attn_layernorm"])
        x = self.mlp(x, None if w is None else w["mlp"])
        return residual + x

# ===================================================================================
# Qwen3 MoE Model
# ===================================================================================

@maps_jax_layer(qwen3_moe_model)
class TorchQwen3MoEModel(nn.Module):
    def __init__(self, cfg: Qwen3MoEModelConfig, w: dict = None):
        super().__init__()
        self.cfg = cfg

        if w is not None:
            self.embeds = nn.Parameter(_to_torch(w["embeds"]))
            self.decoder_blocks = nn.ModuleList([
                TorchQwen3MoELayer(cfg.moe_layer_conf, lw) for lw in w["decoder_blocks"]
            ])
            self.norm = TorchRMSNorm(cfg.final_rms_conf, w["norm"])
            if cfg.tie_word_embeddings:
                self.lm_head = None
            else:
                self.lm_head = nn.Parameter(_to_torch(w["lm_head"]))
        else:
            self.embeds = nn.Parameter(torch.randn(cfg.vocab_size, cfg.d_model, dtype=torch.bfloat16) * (cfg.d_model ** -0.5))
            self.decoder_blocks = nn.ModuleList([
                TorchQwen3MoELayer(cfg.moe_layer_conf) for _ in range(cfg.num_decoder_blocks)
            ])
            self.norm = TorchRMSNorm(cfg.final_rms_conf)
            if cfg.tie_word_embeddings:
                self.lm_head = None
            else:
                self.lm_head = nn.Parameter(torch.randn(cfg.d_model, cfg.vocab_size, dtype=torch.bfloat16) * (cfg.d_model ** -0.5))

    def forward(self, x: torch.Tensor, w: dict = None) -> torch.Tensor:
        embeds = self.embeds if w is None else w["embeds"]
        norm_w = None if w is None else w["norm"]

        hidden = embeds[x]
        for i, block in enumerate(self.decoder_blocks):
            lw = None if w is None else w["decoder_blocks"][i]
            hidden = block(hidden, lw)

        hidden = self.norm(hidden, norm_w)
        
        if self.cfg.tie_word_embeddings:
            return F.linear(hidden, embeds)
        
        lm_head = self.lm_head if w is None else w["lm_head"]
        return hidden @ lm_head

def to_torch_module(jax_fn, cfg, weights=None):
    if jax_fn not in JAX_TO_TORCH_REGISTRY:
        raise ValueError(f"No PyTorch layer registered for JAX function: {jax_fn.__name__}")
    return JAX_TO_TORCH_REGISTRY[jax_fn](cfg, w=weights)