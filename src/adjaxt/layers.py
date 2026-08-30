import jax
import jax.numpy as jnp
import math
from adjaxt.config import *

# ===================================================================================
# RMS Norm
# ===================================================================================

def rms_norm(x: jax.Array, w: jax.Array, cfg: RMSNormConfig) -> jax.Array:
    """
    Args:
        x:   [..., D]
        w:   [D]
    Returns:
        out: [..., D]
    """
    x32 = x.astype(jnp.float32)
    ms = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return (w * (x32 * jax.lax.rsqrt(ms + cfg.eps))).astype(x.dtype)

def rms_norm_init(key, cfg: RMSNormConfig, dtype=jnp.float32) -> jax.Array:
    return jnp.ones((cfg.dim,), dtype=dtype)

# ===================================================================================
# RoPE
# ===================================================================================

def rotate_half(x: jax.Array) -> jax.Array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)

def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    return x * cos + rotate_half(x) * sin

def apply_rot_pos_emb(q: jax.Array, k: jax.Array, cos: jax.Array, sin: jax.Array):
    cos = cos.astype(q.dtype)
    sin = sin.astype(q.dtype)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

# ===================================================================================
# SwiGLU
# ===================================================================================

def swiglu(x: jax.Array, w: dict, cfg: SwiGLUConfig) -> jax.Array:
    return (jax.nn.silu(x @ w["w_gate"]) * (x @ w["w_up"])) @ w["w_down"]

def swiglu_init(key, cfg: SwiGLUConfig) -> dict:
    k1, k2, k3 = jax.random.split(key, 3)
    def kaiming(k, shape):
        return jax.random.normal(k, shape, dtype=jnp.float32) * jnp.sqrt(2.0 / shape[0])
    return {
        "w_gate": kaiming(k1, (cfg.in_dim, cfg.hidden_dim)),
        "w_up": kaiming(k2, (cfg.in_dim, cfg.hidden_dim)),
        "w_down": kaiming(k3, (cfg.hidden_dim, cfg.in_dim))
    }

# ===================================================================================
# GQA Attention
# ===================================================================================

def repeat_kv(hidden_states: jax.Array, n_rep: int) -> jax.Array:
    if n_rep == 1:
        return hidden_states
    return jnp.repeat(hidden_states, n_rep, axis=2)

def gqa_attn(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cfg: GQAAttnConfig,
    attn_mask: jax.Array = None
) -> jax.Array:
    target_dtype = q.dtype
    k = k.astype(target_dtype)
    v = v.astype(target_dtype)

    _, _, num_q_heads, _ = q.shape
    _, _, num_kv_heads, _ = k.shape

    num_groups = num_q_heads // num_kv_heads
    if num_groups > 1:
        k = repeat_kv(k, num_groups)
        v = repeat_kv(v, num_groups)

    causal_flag = cfg.is_causal if attn_mask is None else False

    out = jax.nn.dot_product_attention(
        query=q,
        key=k,
        value=v,
        mask=attn_mask,
        is_causal=causal_flag,
        implementation=cfg.implementation,
    )
    return out

# ===================================================================================
# Qwen3 Attention
# ===================================================================================

def qwen3_attn(
    x: jax.Array,
    w: dict,
    cfg: Qwen3AttnConfig
):
    input_shape = x.shape[:-1]
    seq_len = x.shape[1]
    
    # Explicit head counts to avoid shape inference bugs
    q_heads = cfg.num_heads
    kv_heads = q_heads // cfg.gqa_conf.num_kv_groups
    
    q_shape = (*input_shape, q_heads, cfg.head_dim)
    kv_shape = (*input_shape, kv_heads, cfg.head_dim)
    
    q = rms_norm((x @ w["q_proj"]).reshape(q_shape), w["q_norm"], cfg.q_rms_conf)
    k = rms_norm((x @ w["k_proj"]).reshape(kv_shape), w["k_norm"], cfg.k_rms_conf)
    v = (x @ w["v_proj"]).reshape(kv_shape)
    
    cos = cfg.cos_table[:, :seq_len, :, :]
    sin = cfg.sin_table[:, :seq_len, :, :]
    q, k = apply_rot_pos_emb(q, k, cos, sin)
    
    attn_out = gqa_attn(q, k, v, cfg.gqa_conf)
    attn_out = attn_out.reshape(*input_shape, -1)
    return attn_out @ w["o_proj"]

def qwen3_attn_init(key, cfg: Qwen3AttnConfig) -> dict:
    k1, k2, k3, k4 = jax.random.split(key, 4)
    res = 1.0 / math.sqrt(2 * cfg.n_layers)
    
    q_dim = cfg.num_heads * cfg.head_dim
    kv_dim = (cfg.num_heads // cfg.gqa_conf.num_kv_groups) * cfg.head_dim
    
    return {
        "q_proj": jax.random.normal(k1, (cfg.d, q_dim), jnp.bfloat16) / math.sqrt(cfg.d),
        "k_proj": jax.random.normal(k2, (cfg.d, kv_dim), jnp.bfloat16) / math.sqrt(cfg.d),
        "v_proj": jax.random.normal(k3, (cfg.d, kv_dim), jnp.bfloat16) / math.sqrt(cfg.d),
        "o_proj": jax.random.normal(k4, (q_dim, cfg.d), jnp.bfloat16) * res / math.sqrt(q_dim),
        "q_norm": jnp.ones((cfg.head_dim,), dtype=jnp.float32),
        "k_norm": jnp.ones((cfg.head_dim,), dtype=jnp.float32),
    }

# ===================================================================================
# Qwen3 MLP
# ===================================================================================

def qwen3_mlp(x: jax.Array, w: dict, cfg: Qwen3MLPConfig) -> jax.Array:
    act_fn = getattr(cfg, "act_fn", jax.nn.silu)
    return (act_fn(x @ w["w_gate"]) * (x @ w["w_up"])) @ w["w_down"]

def qwen3_mlp_init(key, cfg: Qwen3MLPConfig, dtype=jnp.bfloat16) -> dict:
    k1, k2, k3 = jax.random.split(key, 3)
    def kaiming(k, shape):
        return (jax.random.normal(k, shape, dtype=jnp.float32) * jnp.sqrt(2.0 / shape[0])).astype(dtype)
    return {
        "w_gate": kaiming(k1, (cfg.in_dim, cfg.hidden_dim)),
        "w_up": kaiming(k2, (cfg.in_dim, cfg.hidden_dim)),
        "w_down": kaiming(k3, (cfg.hidden_dim, cfg.in_dim)),
    }

# ===================================================================================
# Qwen3 MoE Block
# ===================================================================================

def qwen3_moe_block(x: jax.Array, w: dict, cfg: Qwen3MoEBlockConfig) -> jax.Array:
    tokens = x.reshape(-1, x.shape[-1])                     # (T, d)
    router_logits = tokens @ w["router"]                    # (T, E)
    probs = jax.nn.softmax(router_logits.astype(jnp.float32), -1)
    topk_probs, topk_idx = jax.lax.top_k(probs, cfg.top_k)      # (T, k)
    topk_probs = topk_probs / topk_probs.sum(-1, keepdims=True)  # (T, k)

    weights = jnp.zeros_like(probs).at[
        jnp.arange(tokens.shape[0])[:, None], topk_idx
    ].set(topk_probs)

    expert_out = jax.vmap(
        lambda we: qwen3_mlp(tokens, we, cfg.mlp_conf)
    )(w["experts"])                                          # (E, T, d)
    out = jnp.einsum("te,etd->td", weights.astype(x.dtype), expert_out)
    return out.reshape(x.shape)

def qwen3_moe_block_init(key, cfg: Qwen3MoEBlockConfig) -> dict:
    k_router, k_experts = jax.random.split(key, 2)
    expert_keys = jax.random.split(k_experts, cfg.num_experts)
    return {
        "router": jax.random.normal(k_router, (cfg.d_model, cfg.num_experts), dtype=jnp.bfloat16) * (cfg.d_model ** -0.5),
        "experts": jax.vmap(lambda k: qwen3_mlp_init(k, cfg.mlp_conf))(expert_keys),
    }

# ===================================================================================
# Qwen3 MoE Layer
# ===================================================================================

def qwen3_moe_layer(x: jax.Array, w: dict, cfg: Qwen3MoELayerConfig) -> jax.Array:
    residual = x
    x = rms_norm(x, w["input_layernorm"], cfg.input_rms_conf)
    x = qwen3_attn(x, w["attn"], cfg.attn_conf)
    x = x + residual
    residual = x
    x = rms_norm(x, w["post_attn_layernorm"], cfg.post_attn_rms_conf)
    x = qwen3_moe_block(x, w["mlp"], cfg.moe_block_conf)
    return residual + x

def qwen3_moe_layer_init(key, cfg: Qwen3MoELayerConfig) -> dict:
    attn_key, mlp_key, k3, k4 = jax.random.split(key, 4)
    return {
        "attn": qwen3_attn_init(attn_key, cfg.attn_conf),
        "mlp": qwen3_moe_block_init(mlp_key, cfg.moe_block_conf),
        "input_layernorm": rms_norm_init(k3, cfg.input_rms_conf),
        "post_attn_layernorm": rms_norm_init(k4, cfg.post_attn_rms_conf)
    }

# ===================================================================================
# Qwen3 MoE Model
# ===================================================================================

def qwen3_moe_model(x: jax.Array, w: dict, cfg: Qwen3MoEModelConfig) -> jax.Array:
    x = w["embeds"][x]
    for l in w["decoder_blocks"]:
        x = qwen3_moe_layer(x, l, cfg.moe_layer_conf)
    x = rms_norm(x, w["norm"], cfg.final_rms_conf)
    return x @ w["lm_head"]

def qwen3_moe_model_init(key, cfg: Qwen3MoEModelConfig) -> dict:
    k_embed, k_layers, k_norm, k_head = jax.random.split(key, 4)
    embeds = jax.random.normal(k_embed, shape=(cfg.vocab_size, cfg.d_model), dtype=jnp.bfloat16) * (
        cfg.d_model ** -0.5
    )
    layer_keys = jax.random.split(k_layers, cfg.num_decoder_blocks)
    decoder_blocks = [
        qwen3_moe_layer_init(k, cfg.moe_layer_conf) for k in layer_keys
    ]
    norm = rms_norm_init(k_norm, cfg.final_rms_conf)
    lm_head = embeds.T if cfg.tie_word_embeddings else (
        jax.random.normal(k_head, shape=(cfg.d_model, cfg.vocab_size), dtype=jnp.bfloat16) * (cfg.d_model ** -0.5)
    )

    return {
        "embeds": embeds,
        "decoder_blocks": decoder_blocks,
        "norm": norm,
        "lm_head": lm_head,
    }