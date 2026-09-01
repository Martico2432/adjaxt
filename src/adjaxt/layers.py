import jax
import jax.numpy as jnp
import math
from adjaxt.config import *
import optax

# ===================================================================================
# RMS Norm
# ===================================================================================

@exec_fn_for(RMSNormConfig)
def rms_norm(x: jax.Array, w: jax.Array, cfg: RMSNormConfig) -> jax.Array:
    # Norms should compute in float32 for numerical stability
    x32 = x.astype(jnp.float32)
    w32 = w.astype(jnp.float32)
    
    ms = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    normed = w32 * (x32 * jax.lax.rsqrt(ms + cfg.eps))
    
    return normed.astype(cfg.precision.output_dtype)

@init_fn_for(RMSNormConfig)
def rms_norm_init(key, cfg: RMSNormConfig) -> jax.Array:
    return jnp.ones((cfg.dim,), dtype=cfg.precision.param_dtype)

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
    # Align RoPE tables to the exact compute precision of Q and K
    cos = cos.astype(q.dtype)
    sin = sin.astype(q.dtype)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

# ===================================================================================
# SwiGLU
# ===================================================================================

@exec_fn_for(SwiGLUConfig)
def swiglu(x: jax.Array, w: dict, cfg: SwiGLUConfig) -> jax.Array:
    c_dtype = cfg.precision.compute_dtype
    x_c = x.astype(c_dtype)
    w_gate = w["w_gate"].astype(c_dtype)
    w_up = w["w_up"].astype(c_dtype)
    w_down = w["w_down"].astype(c_dtype)
    
    act = jax.nn.silu(x_c @ w_gate) * (x_c @ w_up)
    out = act @ w_down
    
    return out.astype(cfg.precision.output_dtype)

@init_fn_for(SwiGLUConfig)
def swiglu_init(key, cfg: SwiGLUConfig) -> dict:
    k1, k2, k3 = jax.random.split(key, 3)
    def kaiming(k, shape):
        return jax.random.normal(k, shape, dtype=cfg.precision.param_dtype) * jnp.sqrt(2.0 / shape[0])
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

@exec_fn_for(GQAAttnConfig)
def gqa_attn(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cfg: GQAAttnConfig,
    attn_mask: jax.Array = None
) -> jax.Array:
    # q, k, v should already be safely inside compute_dtype
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

@exec_fn_for(Qwen3AttnConfig)
def qwen3_attn(
    x: jax.Array,
    w: dict,
    cfg: Qwen3AttnConfig
):
    c_dtype = cfg.precision.compute_dtype
    
    input_shape = x.shape[:-1]
    seq_len = x.shape[1]
    
    q_heads = cfg.num_heads
    kv_heads = q_heads // cfg.gqa_conf.num_kv_groups
    
    q_shape = (*input_shape, q_heads, cfg.head_dim)
    kv_shape = (*input_shape, kv_heads, cfg.head_dim)
    
    # Cast to compute precision
    x_c = x.astype(c_dtype)
    q_proj = w["q_proj"].astype(c_dtype)
    k_proj = w["k_proj"].astype(c_dtype)
    v_proj = w["v_proj"].astype(c_dtype)
    o_proj = w["o_proj"].astype(c_dtype)
    
    q_raw = (x_c @ q_proj).reshape(q_shape)
    k_raw = (x_c @ k_proj).reshape(kv_shape)
    v = (x_c @ v_proj).reshape(kv_shape)
    
    # RMS norm usually returns output_dtype, so cast back down for Attention
    q = rms_norm(q_raw, w["q_norm"], cfg.q_rms_conf).astype(c_dtype)
    k = rms_norm(k_raw, w["k_norm"], cfg.k_rms_conf).astype(c_dtype)
    
    cos = cfg.cos_table[:, :seq_len, :, :]
    sin = cfg.sin_table[:, :seq_len, :, :]
    q, k = apply_rot_pos_emb(q, k, cos, sin)
    
    attn_out = gqa_attn(q, k, v, cfg.gqa_conf)
    attn_out = attn_out.reshape(*input_shape, -1)
    
    out = attn_out @ o_proj
    return out.astype(cfg.precision.output_dtype)

@init_fn_for(Qwen3AttnConfig)
def qwen3_attn_init(key, cfg: Qwen3AttnConfig) -> dict:
    k1, k2, k3, k4 = jax.random.split(key, 4)
    res = 1.0 / math.sqrt(2 * cfg.n_layers)
    
    q_dim = cfg.num_heads * cfg.head_dim
    kv_dim = (cfg.num_heads // cfg.gqa_conf.num_kv_groups) * cfg.head_dim
    p_dtype = cfg.precision.param_dtype
    
    return {
        "q_proj": jax.random.normal(k1, (cfg.d, q_dim), dtype=p_dtype) / math.sqrt(cfg.d),
        "k_proj": jax.random.normal(k2, (cfg.d, kv_dim), dtype=p_dtype) / math.sqrt(cfg.d),
        "v_proj": jax.random.normal(k3, (cfg.d, kv_dim), dtype=p_dtype) / math.sqrt(cfg.d),
        "o_proj": jax.random.normal(k4, (q_dim, cfg.d), dtype=p_dtype) * res / math.sqrt(q_dim),
        "q_norm": jnp.ones((cfg.head_dim,), dtype=p_dtype),
        "k_norm": jnp.ones((cfg.head_dim,), dtype=p_dtype),
    }

# ===================================================================================
# Qwen3 MLP
# ===================================================================================

@exec_fn_for(Qwen3MLPConfig)
def qwen3_mlp(x: jax.Array, w: dict, cfg: Qwen3MLPConfig) -> jax.Array:
    c_dtype = cfg.precision.compute_dtype
    x_c = x.astype(c_dtype)
    w_gate = w["w_gate"].astype(c_dtype)
    w_up = w["w_up"].astype(c_dtype)
    w_down = w["w_down"].astype(c_dtype)
    
    act_fn = getattr(cfg, "act_fn", jax.nn.silu)
    act = act_fn(x_c @ w_gate) * (x_c @ w_up)
    out = act @ w_down
    
    return out.astype(cfg.precision.output_dtype)

@init_fn_for(Qwen3MLPConfig)
def qwen3_mlp_init(key, cfg: Qwen3MLPConfig) -> dict:
    k1, k2, k3 = jax.random.split(key, 3)
    def kaiming(k, shape):
        return jax.random.normal(k, shape, dtype=cfg.precision.param_dtype) * jnp.sqrt(2.0 / shape[0])
    return {
        "w_gate": kaiming(k1, (cfg.in_dim, cfg.hidden_dim)),
        "w_up": kaiming(k2, (cfg.in_dim, cfg.hidden_dim)),
        "w_down": kaiming(k3, (cfg.hidden_dim, cfg.in_dim)),
    }

# ===================================================================================
# Qwen3 MoE Block
# ===================================================================================

@exec_fn_for(Qwen3MoEBlockConfig)
def qwen3_moe_block(x: jax.Array, w: dict, cfg: Qwen3MoEBlockConfig) -> jax.Array:
    tokens = x.reshape(-1, x.shape[-1])                     # (T, d)
    
    # Routing always happens in float32 for stability
    router_logits = tokens.astype(jnp.float32) @ w["router"].astype(jnp.float32)
    probs = jax.nn.softmax(router_logits, -1)
    topk_probs, topk_idx = jax.lax.top_k(probs, cfg.top_k)      
    topk_probs = topk_probs / topk_probs.sum(-1, keepdims=True)  

    weights = jnp.zeros_like(probs).at[
        jnp.arange(tokens.shape[0])[:, None], topk_idx
    ].set(topk_probs)

    # expert_out manages its own precision boundaries internally (via qwen3_mlp)
    expert_out = jax.vmap(
        lambda we: qwen3_mlp(tokens, we, cfg.mlp_conf)
    )(w["experts"])                                          
    
    out_dtype = cfg.precision.output_dtype
    out = jnp.einsum("te,etd->td", weights.astype(out_dtype), expert_out.astype(out_dtype))
    
    return out.reshape(x.shape)

@init_fn_for(Qwen3MoEBlockConfig)
def qwen3_moe_block_init(key, cfg: Qwen3MoEBlockConfig) -> dict:
    k_router, k_experts = jax.random.split(key, 2)
    expert_keys = jax.random.split(k_experts, cfg.num_experts)
    return {
        "router": jax.random.normal(k_router, (cfg.d_model, cfg.num_experts), dtype=cfg.precision.param_dtype) * (cfg.d_model ** -0.5),
        "experts": jax.vmap(lambda k: qwen3_mlp_init(k, cfg.mlp_conf))(expert_keys),
    }

# ===================================================================================
# Qwen3 MoE Layer
# ===================================================================================

@exec_fn_for(Qwen3MoELayerConfig)
def qwen3_moe_layer(x: jax.Array, w: dict, cfg: Qwen3MoELayerConfig) -> jax.Array:
    # x is guaranteed to be in output_dtype (float32) throughout the residual stream
    residual = x
    x = rms_norm(x, w["input_layernorm"], cfg.input_rms_conf)
    x = qwen3_attn(x, w["attn"], cfg.attn_conf)
    x = x + residual
    
    residual = x
    x = rms_norm(x, w["post_attn_layernorm"], cfg.post_attn_rms_conf)
    x = qwen3_moe_block(x, w["mlp"], cfg.moe_block_conf)
    
    return residual + x

@init_fn_for(Qwen3MoELayerConfig)
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

@exec_fn_for(Qwen3MoEModelConfig)
def qwen3_moe_model(x: jax.Array, w: dict, cfg: Qwen3MoEModelConfig) -> jax.Array:
    x = w["embeds"][x].astype(cfg.precision.output_dtype)
    
    for l in w["decoder_blocks"]:
        x = qwen3_moe_layer(x, l, cfg.moe_layer_conf)
        
    x = rms_norm(x, w["norm"], cfg.final_rms_conf)
    
    # LM Head Matmul
    c_dtype = cfg.precision.compute_dtype
    x_c = x.astype(c_dtype)
    lm_head = w["lm_head"].astype(c_dtype)
    
    logits = x_c @ lm_head
    return logits.astype(jnp.float32) # Always return loss inputs in f32

@init_fn_for(Qwen3MoEModelConfig)
def qwen3_moe_model_init(key, cfg: Qwen3MoEModelConfig) -> dict:
    k_embed, k_layers, k_norm, k_head = jax.random.split(key, 4)
    p_dtype = cfg.precision.param_dtype
    
    embeds = jax.random.normal(k_embed, shape=(cfg.vocab_size, cfg.d_model), dtype=p_dtype) * (
        cfg.d_model ** -0.5
    )
    layer_keys = jax.random.split(k_layers, cfg.num_decoder_blocks)
    decoder_blocks = [
        qwen3_moe_layer_init(k, cfg.moe_layer_conf) for k in layer_keys
    ]
    norm = rms_norm_init(k_norm, cfg.final_rms_conf)
    lm_head = embeds.T if cfg.tie_word_embeddings else (
        jax.random.normal(k_head, shape=(cfg.d_model, cfg.vocab_size), dtype=p_dtype) * (cfg.d_model ** -0.5)
    )

    return {
        "embeds": embeds,
        "decoder_blocks": decoder_blocks,
        "norm": norm,
        "lm_head": lm_head,
    }

def chunked_cross_entropy_loss(
    hidden: jax.Array,
    embed_table: jax.Array,
    labels: jax.Array,
    chunk_size: int = 512,
    compute_dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
    """Computes cross-entropy loss in chunks with gradient checkpointing and dynamic padding."""
    # Align and shift for causal LM
    shift_hidden = hidden[:, :-1, :]
    shift_labels = labels[:, 1:]

    b, t, d = shift_hidden.shape
    total_tokens = b * t
    
    h_flat = shift_hidden.reshape(total_tokens, d)
    targets_flat = shift_labels.reshape(total_tokens)
    
    # Calculate padding needed to reach the next multiple of chunk_size
    pad_len = (chunk_size - (total_tokens % chunk_size)) % chunk_size
    
    # Pad arrays with zeros, and create a mask to ignore the padding
    if pad_len > 0:
        h_flat = jnp.pad(h_flat, ((0, pad_len), (0, 0)))
        targets_flat = jnp.pad(targets_flat, (0, pad_len))
        mask_flat = jnp.pad(jnp.ones(total_tokens), (0, pad_len))
    else:
        mask_flat = jnp.ones(total_tokens)
        
    num_chunks = (total_tokens + pad_len) // chunk_size
    h_chunks = h_flat.reshape(num_chunks, chunk_size, d)
    t_chunks = targets_flat.reshape(num_chunks, chunk_size)
    m_chunks = mask_flat.reshape(num_chunks, chunk_size)

    @jax.remat
    def _chunk_loss(h_c, t_c, m_c):
        # Dynamically cast using the specified compute precision
        logits = jnp.matmul(
            h_c.astype(compute_dtype),
            embed_table.astype(compute_dtype).T,
            preferred_element_type=jnp.float32,
        )
        loss_c = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=t_c)
        # Multiply by mask so padded indices contribute exactly 0.0 to the loss sum
        return jnp.sum(loss_c * m_c)

    def _scan_step(acc_loss, xs):
        return acc_loss + _chunk_loss(xs[0], xs[1], xs[2]), None

    total_loss, _ = jax.lax.scan(_scan_step, 0.0, (h_chunks, t_chunks, m_chunks))
    
    # Divide only by the actual number of tokens
    return total_loss / total_tokens