from adjaxt.diffusion.config import *
from adjaxt import layers
import math
import jax
import jax.numpy as jnp

def timestep_embedding(timesteps: jax.Array, dim: int, max_period: int = 10000) -> jax.Array:
    """
    timesteps: (B,) continuous t in [0, 1]
    returns: (B, dim)
    """
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
    args = timesteps[:, None] * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.pad(embedding, [[0, 0], [0, 1]])
    return embedding

def mask_batch_for_block(
    key: jax.Array, 
    token_ids: jax.Array, 
    mask_token_id: int, 
    t_min: float, 
    t_max: float
):
    key_t, key_mask = jax.random.split(key)
    B, L = token_ids.shape

    t = jax.random.uniform(key_t, shape=(B, 1), minval=t_min, maxval=t_max)
    mask_prob = t
    random_probs = jax.random.uniform(key_mask, shape=(B, L))
    mask_indices = random_probs < mask_prob
    corrupted_ids = jnp.where(mask_indices, mask_token_id, token_ids)
    
    return corrupted_ids, t.squeeze(-1), mask_indices

def make_canvas_rope_indices(context_len: int, scratch_len: int, canvas_len: int) -> jax.Array:
    """
    Assigns continuous absolute positions to Context, Scratchpad, and Canvas.
    """
    ctx_pos = jnp.arange(0, context_len)
    scratch_pos = jnp.arange(context_len, context_len + scratch_len)
    canvas_pos = jnp.arange(context_len + scratch_len, context_len + scratch_len + canvas_len)
    return jnp.concatenate([ctx_pos, scratch_pos, canvas_pos], axis=0)

def modulate(x: jax.Array, shift: jax.Array, scale: jax.Array) -> jax.Array:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]

def apply_levenshtein_deletions(
    seq: jax.Array,           # (B, L)
    del_logits: jax.Array,    # (B, L, 2)
    is_editable: jax.Array,   # (B, L) bool
    pad_token_id: int
) -> jax.Array:
    B, L = seq.shape
    # Class 1 = DELETE, Class 0 = KEEP
    should_delete = (del_logits[..., 1] > del_logits[..., 0]) & is_editable
    keep = ~should_delete
    
    # Destination index for each surviving token (0, 1, 2, ...)
    dest_idx = jnp.cumsum(keep.astype(jnp.int32), axis=-1) - 1
    safe_dest = jnp.where(keep, dest_idx, L)  # Send deleted tokens to dummy index L
    
    # Static buffer with dummy slot L to absorb deleted tokens without collision
    compacted = jnp.full((B, L + 1), pad_token_id, dtype=seq.dtype)
    b_idx = jnp.arange(B)[:, None]
    compacted = compacted.at[b_idx, safe_dest].set(seq)
    
    return compacted[:, :L]

def apply_levenshtein_insertions(
    seq: jax.Array,           # (B, L)
    ins_logits: jax.Array,    # (B, L - 1, max_insert + 1)
    is_editable: jax.Array,   # (B, L) bool
    mask_token_id: int
) -> jax.Array:
    B, L = seq.shape
    ins_counts = jnp.argmax(ins_logits, axis=-1)  # (B, L - 1)
    
    # Only allow insertions inside editable boundaries
    editable_boundary = is_editable[:, :-1] & is_editable[:, 1:]
    ins_counts = jnp.where(editable_boundary, ins_counts, 0)
    
    # Total inserted tokens before index i
    cum_ins = jnp.pad(jnp.cumsum(ins_counts, axis=-1), ((0, 0), (1, 0)))  # (B, L)
    new_pos = jnp.arange(L)[None, :] + cum_ins                           # (B, L)
    
    # Target buffer initialized to [MASK]
    expanded = jnp.full((B, L + 1), mask_token_id, dtype=seq.dtype)
    safe_pos = jnp.where(new_pos < L, new_pos, L)
    
    b_idx = jnp.arange(B)[:, None]
    # new_pos is strictly monotonically increasing -> zero scatter collisions
    expanded = expanded.at[b_idx, safe_pos].set(seq)
    
    return expanded[:, :L]

def apply_scratchpad_expansion(
    seq: jax.Array,           # (B, L)
    ins_logits: jax.Array,    # (B, L - 1, max_insert + 1)
    mask_token_id: int,
    scratch_start: int,
    scratch_len: int
) -> jax.Array:
    """
    Uses highest-entropy insertion predictions to populate the scratchpad
    with [MASK] slots for refinement in the next diffusion cycle.
    """
    # Number of insertions predicted per boundary
    ins_counts = jnp.argmax(ins_logits, axis=-1)  # (B, L - 1)
    has_insertions = (ins_counts > 0)
    
    # If the canvas requires new tokens, ensure scratchpad slots are active [MASK]
    scratch_indices = jnp.arange(scratch_start, scratch_start + scratch_len)
    seq = seq.at[:, scratch_indices].set(
        jnp.where(jnp.any(has_insertions, axis=-1, keepdims=True), mask_token_id, seq[:, scratch_indices])
    )
    return seq

@exec_fn_for(MDLMAttentionConfig)
def mdlm_attn_fwd(
    x: jax.Array, 
    w: dict, 
    conf: MDLMAttentionConfig, 
    pos_ids: jax.Array = None
) -> jax.Array:
    c_dtype = conf.precision.compute_dtype
    input_shape = x.shape[:-1]
    seq_len = x.shape[1]
    qkv_heads = conf.num_heads
    qkv_shape = (*input_shape, qkv_heads, conf.head_dim)
    x_c = x.astype(c_dtype)

    q_proj = w["q_proj"].astype(c_dtype)
    k_proj = w["k_proj"].astype(c_dtype)
    v_proj = w["v_proj"].astype(c_dtype)
    o_proj = w["o_proj"].astype(c_dtype)

    q_raw = (x_c @ q_proj).reshape(qkv_shape)
    k_raw = (x_c @ k_proj).reshape(qkv_shape)
    v = (x_c @ v_proj).reshape(qkv_shape)

    # Use qk_rms_conf defined in MDLMAttentionConfig
    q = layers.rms_norm(q_raw, w["q_norm"], conf.qk_rms_conf).astype(c_dtype)
    k = layers.rms_norm(k_raw, w["k_norm"], conf.qk_rms_conf).astype(c_dtype)

    if pos_ids is not None:
        cos = conf.cos_table[:, pos_ids, :, :]
        sin = conf.sin_table[:, pos_ids, :, :]
    else:
        cos = conf.cos_table[:, :seq_len, :, :]
        sin = conf.sin_table[:, :seq_len, :, :]
        
    q, k = layers.apply_rot_pos_emb(q, k, cos, sin)    

    attn_out = layers.mha_attn(q, k, v, conf.mha_conf)
    attn_out = attn_out.reshape(*input_shape, -1)
        
    out = attn_out @ o_proj
    return out.astype(conf.precision.output_dtype)

@init_fn_for(MDLMAttentionConfig)
def mdlm_attn_init(key: jax.Array, cfg: MDLMAttentionConfig) -> dict:
    k1, k2, k3, k4 = jax.random.split(key, 4)
    res = 1.0 / math.sqrt(2 * cfg.n_layers)
    
    q_dim = cfg.num_heads * cfg.head_dim
    kv_dim = q_dim  # 1:1 KV ratio for standard MHA
    p_dtype = cfg.precision.param_dtype
    
    return {
        "q_proj": jax.random.normal(k1, (cfg.d, q_dim), dtype=p_dtype) / math.sqrt(cfg.d),
        "k_proj": jax.random.normal(k2, (cfg.d, kv_dim), dtype=p_dtype) / math.sqrt(cfg.d),
        "v_proj": jax.random.normal(k3, (cfg.d, kv_dim), dtype=p_dtype) / math.sqrt(cfg.d),
        "o_proj": jax.random.normal(k4, (q_dim, cfg.d), dtype=p_dtype) * res / math.sqrt(q_dim),
        "q_norm": jnp.ones((cfg.head_dim,), dtype=p_dtype),
        "k_norm": jnp.ones((cfg.head_dim,), dtype=p_dtype),
    }

@exec_fn_for(MDLMDenseMLPConfig)
def mdlm_dense_fwd(x: jax.Array, w: dict, cfg: MDLMDenseMLPConfig) -> jax.Array:
    c_dtype = cfg.precision.compute_dtype
    x_c = x.astype(c_dtype)
    w_gate = w["w_gate"].astype(c_dtype)
    w_up = w["w_up"].astype(c_dtype)
    w_down = w["w_down"].astype(c_dtype)
    
    act_fn = getattr(cfg, "act_fn", jax.nn.silu)
    act = act_fn(x_c @ w_gate) * (x_c @ w_up)
    out = act @ w_down
    
    return out.astype(cfg.precision.output_dtype)

@init_fn_for(MDLMDenseMLPConfig)
def mdlm_dense_init(key: jax.Array, cfg: MDLMDenseMLPConfig) -> dict:
    k1, k2, k3 = jax.random.split(key, 3)
    def kaiming(k, shape):
        return jax.random.normal(k, shape, dtype=cfg.precision.param_dtype) * jnp.sqrt(2.0 / shape[0])
    return {
        "w_gate": kaiming(k1, (cfg.in_dim, cfg.hidden_dim)),
        "w_up": kaiming(k2, (cfg.in_dim, cfg.hidden_dim)),
        "w_down": kaiming(k3, (cfg.hidden_dim, cfg.in_dim)),
    }

@exec_fn_for(AdaLNConfig)
def adaln_forward(t_emb: jax.Array, w: dict, cfg: AdaLNConfig) -> jax.Array:
    c_dtype = cfg.precision.compute_dtype
    return (t_emb.astype(c_dtype) @ w["adaln_w"].astype(c_dtype)) + w["adaln_b"].astype(c_dtype)

@init_fn_for(AdaLNConfig)
def adaln_init(key: jax.Array, cfg: AdaLNConfig) -> dict:
    p_dtype = cfg.precision.param_dtype
    d = cfg.d
    return {
        "adaln_w": jnp.zeros((d, 6 * d), dtype=p_dtype),
        "adaln_b": jnp.zeros((6 * d,), dtype=p_dtype),
    }

@init_fn_for(MDLMBlockConfig)
def mdlm_block_init(key: jax.Array, cfg: MDLMBlockConfig) -> dict:
    sub_keys = jax.random.split(key, cfg.layers_per_block)
    layers_dict = {}
    for i in range(cfg.layers_per_block):
        k_attn, k_mlp, k_rms1, k_rms2, k_adaln = jax.random.split(sub_keys[i], 5)
        layers_dict[f"layer_{i}"] = {
            "attn": mdlm_attn_init(k_attn, cfg.attn_cfg),
            "mlp": mdlm_dense_init(k_mlp, cfg.mlp_cfg),
            "adaln": adaln_init(k_adaln, cfg.adaln_cfg),
            "rms_norm1": layers.rms_norm_init(k_rms1, cfg.input_rms_conf),
            "rms_norm2": layers.rms_norm_init(k_rms2, cfg.post_attn_rms_conf),
        }
    return layers_dict

@exec_fn_for(MDLMBlockConfig)
def mdlm_block_fwd(
    x: jax.Array, 
    t_emb: jax.Array, 
    w: dict, 
    cfg: MDLMBlockConfig, 
    pos_ids: jax.Array = None
) -> jax.Array:
    c_dtype = cfg.precision.compute_dtype
    for i in range(cfg.layers_per_block):
        layer_w = w[f"layer_{i}"]
        x_c = x.astype(c_dtype)
        
        # 1. AdaLN Modulation
        adaln_out = adaln_forward(t_emb, layer_w["adaln"], cfg.adaln_cfg)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(adaln_out, 6, axis=-1)

        # 2. Attention
        norm1_x = layers.rms_norm(x_c, layer_w["rms_norm1"], cfg.input_rms_conf)
        norm1_mod = modulate(norm1_x, shift_msa, scale_msa)
        attn_out = mdlm_attn_fwd(norm1_mod, layer_w["attn"], cfg.attn_cfg, pos_ids=pos_ids)
        x = x + (gate_msa[:, None, :] * attn_out).astype(cfg.precision.output_dtype)

        # 3. MLP
        norm2_x = layers.rms_norm(x.astype(c_dtype), layer_w["rms_norm2"], cfg.post_attn_rms_conf)
        norm2_mod = modulate(norm2_x, shift_mlp, scale_mlp)
        mlp_out = mdlm_dense_fwd(norm2_mod, layer_w["mlp"], cfg.mlp_cfg)
        x = x + (gate_mlp[:, None, :] * mlp_out).astype(cfg.precision.output_dtype)

    return x

def get_block_noise_bounds(block_idx: int, num_blocks: int):
    step = 1.0 / num_blocks
    t_min = 1.0 - (block_idx + 1) * step
    t_max = 1.0 - block_idx * step
    return float(t_min), float(t_max)

@init_fn_for(MDLModelConfig)
def mdl_model_init(key: jax.Array, cfg: MDLModelConfig) -> dict:
    k_emb, k_t1, k_t2, k_del, k_ins, k_norm, k_blocks = jax.random.split(key, 7)
    p_dtype = cfg.precision.param_dtype
    d = cfg.d
    
    block_keys = jax.random.split(k_blocks, cfg.diff_blocks_num)
    return {
        "token_emb": jax.random.normal(k_emb, (cfg.vocab_size, d), dtype=p_dtype) * 0.02,
        "t_w1": jax.random.normal(k_t1, (d, d), dtype=p_dtype) * math.sqrt(2.0 / d),
        "t_w2": jax.random.normal(k_t2, (d, d), dtype=p_dtype) * math.sqrt(2.0 / d),
        "del_head": jax.random.normal(k_del, (d, 2), dtype=p_dtype) * 0.02,
        "ins_head": jax.random.normal(k_ins, (2 * d, cfg.max_insert + 1), dtype=p_dtype) * 0.02,
        "blocks": [mdlm_block_init(block_keys[i], cfg.block_cfg) for i in range(cfg.diff_blocks_num)],
        "final_norm": layers.rms_norm_init(k_norm, cfg.final_rms_conf)
    }

def forward_network(
    seq_ids: jax.Array,
    t_val: jax.Array,
    params: dict,
    cfg: MDLModelConfig,
    block_idx: int,
    pos_ids: jax.Array = None
):
    d = cfg.d
    c_dtype = cfg.precision.compute_dtype
    
    w_emb = params["token_emb"].astype(c_dtype)
    h = w_emb[seq_ids] * math.sqrt(d)
    
    t_raw = timestep_embedding(t_val, d).astype(c_dtype)
    t_emb = jax.nn.silu(t_raw @ params["t_w1"].astype(c_dtype)) @ params["t_w2"].astype(c_dtype)
    
    h = mdlm_block_fwd(h, t_emb, params["blocks"][block_idx], cfg.block_cfg, pos_ids=pos_ids)
    h = layers.rms_norm(h.astype(c_dtype), params["final_norm"], cfg.final_rms_conf)
    
    token_logits = h @ w_emb.T
    del_logits = h @ params["del_head"].astype(c_dtype)
    
    consecutive = jnp.concatenate([h[:, :-1, :], h[:, 1:, :]], axis=-1)
    ins_logits = consecutive @ params["ins_head"].astype(c_dtype)
    
    return token_logits, del_logits, ins_logits

@exec_fn_for(MDLModelConfig)
def mdl_model_fwd(
    key: jax.Array,
    prefix_ids: jax.Array,
    params: dict,
    cfg: MDLModelConfig
) -> jax.Array:
    B = prefix_ids.shape[0]
    C_prev = prefix_ids.shape[1]
    active_len = cfg.scratch_len + cfg.canvas_len
    pad_token_id = 0
    
    # 1. Initialize scratchpad and canvas with [MASK]
    active_init = jnp.full((B, active_len), cfg.mask_token_id, dtype=jnp.int32)
    seq = jnp.concatenate([prefix_ids, active_init], axis=1)
    is_editable = jnp.zeros_like(seq, dtype=jnp.bool_).at[:, C_prev:].set(True)
    
    pos_ids = make_canvas_rope_indices(C_prev, cfg.scratch_len, cfg.canvas_len)
    total_blocks = cfg.diff_blocks_num
    
    # 2. Diffusion Drafting Pass (Block 0 -> Block B-1)
    for b_idx in range(total_blocks):
        t_min, t_max = get_block_noise_bounds(b_idx, total_blocks)
        t_steps = [t_max - i * (t_max - t_min) / (cfg.steps_per_block - 1) for i in range(cfg.steps_per_block)]
        
        for i in range(len(t_steps) - 1):
            t_curr = t_steps[i]
            t_next = t_steps[i + 1]
            
            key, k_sample, k_mask = jax.random.split(key, 3)
            t_tensor = jnp.full((B,), t_curr)
            
            token_logits, _, _ = forward_network(
                seq, t_tensor, params, cfg, block_idx=b_idx, pos_ids=pos_ids
            )
            
            pred_tokens = jax.random.categorical(k_sample, token_logits, axis=-1)
            seq_cand = jnp.where((seq == cfg.mask_token_id) & is_editable, pred_tokens, seq)
            
            # Stochastic re-masking
            remask_prob = t_next / (t_curr + 1e-8)
            random_draw = jax.random.uniform(k_mask, shape=seq.shape)
            should_remask = (random_draw < remask_prob) & is_editable
            
            seq = jnp.where((t_next > 0.0) & should_remask, cfg.mask_token_id, seq_cand)

    # 3. Levenshtein Edit Cycle (Clean Drafting Phase at t = 0.0)
    t_zero = jnp.zeros((B,))
    fine_block_idx = total_blocks - 1

    # Pass A: Evaluate draft sequence to obtain deletion and insertion logits
    _, del_logits, ins_logits = forward_network(
        seq, t_zero, params, cfg, block_idx=fine_block_idx, pos_ids=pos_ids
    )

    # Pass B: Delete unwanted / erroneous tokens
    seq = apply_levenshtein_deletions(seq, del_logits, is_editable, pad_token_id=cfg.mask_token_id)

    # Pass C: Insert [MASK] placeholders where ins_logits requests insertions
    seq = apply_levenshtein_insertions(seq, ins_logits, is_editable, mask_token_id=cfg.mask_token_id)

    # Pass D: Denoise newly inserted [MASK] tokens with the language model head
    key, k_infill = jax.random.split(key)
    token_logits, _, _ = forward_network(
        seq, t_zero, params, cfg, block_idx=fine_block_idx, pos_ids=pos_ids
    )
    infilled_tokens = jax.random.categorical(k_infill, token_logits, axis=-1)
    seq = jnp.where((seq == cfg.mask_token_id) & is_editable, infilled_tokens, seq)

    # 4. Extract Finalized Canvas
    canvas_start = C_prev + cfg.scratch_len
    clean_canvas_tokens = seq[:, canvas_start : canvas_start + cfg.canvas_len]
    return clean_canvas_tokens

