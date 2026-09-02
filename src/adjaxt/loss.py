import jax
import jax.numpy as jnp
from adjaxt.config import CCEConfig, exec_fn_for


def _softcap_logits(logits: jax.Array, softcap: float) -> jax.Array:
    return softcap * jnp.tanh(logits / softcap)


@exec_fn_for(CCEConfig)
def exec_cce_autograd(
    cfg: CCEConfig,
    params: dict[str, jax.Array],
    hidden_states: jax.Array,
    weight: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    hidden_states = hidden_states.astype(cfg.precision.compute_dtype)
    weight = weight.astype(cfg.precision.compute_dtype)

    if cfg.shift:
        hidden_states = hidden_states[:, :-1, :]
        labels = labels[:, 1:]

    flat_h = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_labels = labels.reshape(-1)
    vocab_size = weight.shape[0]

    num_chunks = (vocab_size + cfg.chunk_size - 1) // cfg.chunk_size
    pad_size = num_chunks * cfg.chunk_size - vocab_size
    if pad_size > 0:
        weight_padded = jnp.pad(
            weight, ((0, pad_size), (0, 0)), constant_values=0.0
        )
    else:
        weight_padded = weight

    def compute_chunk_lse(carry, chunk_idx):
        start_idx = chunk_idx * cfg.chunk_size
        weight_chunk = jax.lax.dynamic_slice_in_dim(
            weight_padded, start_index=start_idx, slice_size=cfg.chunk_size, axis=0
        )
        logits_chunk = jnp.matmul(
            flat_h, weight_chunk.T, preferred_element_type=cfg.precision.output_dtype
        )

        if cfg.soft_cap is not None:
            logits_chunk = _softcap_logits(logits_chunk, cfg.soft_cap)

        chunk_indices = start_idx + jnp.arange(cfg.chunk_size)
        mask = chunk_indices < vocab_size
        logits_chunk = jnp.where(mask[None, :], logits_chunk, -1e9)

        chunk_max = jnp.max(logits_chunk, axis=-1, keepdims=True)
        chunk_sumexp = jnp.sum(jnp.exp(logits_chunk - chunk_max), axis=-1, keepdims=True)
        return carry, (chunk_max, chunk_sumexp)

    checkpointed_chunk_fn = jax.checkpoint(compute_chunk_lse)

    _, (chunk_maxes, chunk_sumexps) = jax.lax.scan(
        checkpointed_chunk_fn, None, jnp.arange(num_chunks)
    )
    global_max = jnp.max(chunk_maxes, axis=0)
    global_sumexp = jnp.sum(
        chunk_sumexps * jnp.exp(chunk_maxes - global_max), axis=0
    )

    lse = (global_max + jnp.log(jnp.maximum(global_sumexp, 1e-10))).squeeze(-1)

    safe_labels = jnp.where(flat_labels == cfg.ignore_index, 0, flat_labels)
    target_weights = jnp.take(weight, safe_labels, axis=0)
    target_logits = jnp.sum(
        flat_h * target_weights, axis=-1, dtype=cfg.precision.output_dtype
    )

    if cfg.soft_cap is not None:
        target_logits = _softcap_logits(target_logits, cfg.soft_cap)

    loss = lse - target_logits
    valid_mask = flat_labels != cfg.ignore_index
    loss = jnp.where(valid_mask, loss, 0.0)

    if cfg.reduction == "mean":
        return loss.sum() / jnp.maximum(valid_mask.sum(), 1)
    elif cfg.reduction == "sum":
        return loss.sum()

    return loss
