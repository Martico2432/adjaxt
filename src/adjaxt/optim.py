from typing import Union, Callable
import optax
import jax
import jax.numpy as jnp

def newton_schulz_iteration(g: jax.Array, steps: int = 5, eps: float = 1e-7) -> jax.Array:
    """
    Applies quintic Newton-Schulz iteration to orthogonalize matrices:
    X_{k+1} = X_k (3.4445 * I - 4.7750 * A + 2.0315 * A^2), where A = X_k^T X_k.
    """
    orig_shape = g.shape
    if g.ndim > 2:
        g = g.reshape(g.shape[0], -1)

    # Ensure shape is (rows, cols) where rows >= cols for standard iteration
    transposed = False
    if g.shape[0] < g.shape[1]:
        g = g.T
        transposed = True

    m, n = g.shape
    # Scale matrix for stable quintic iteration
    norm = jnp.linalg.norm(g) + eps
    x = g / norm

    # Quintic polynomial coefficients
    a, b, c = 3.4445, -4.7750, 2.0315
    eye = jnp.eye(n, dtype=x.dtype)

    for _ in range(steps):
        a_mat = x.T @ x
        x = x @ (a * eye + b * a_mat + c * (a_mat @ a_mat))

    # Scale updates to match root-mean-square aspect ratio
    x = x * jnp.sqrt(jnp.maximum(1.0, m / n))

    if transposed:
        x = x.T

    return x.reshape(orig_shape)

def scale_by_muon(ns_steps: int = 5):
    """Optax transformation applying Muon orthogonalization to gradient matrices."""
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        new_updates = jax.tree.map(
            lambda g: newton_schulz_iteration(g, steps=ns_steps) if g.ndim >= 2 else g, 
            updates
        )
        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)

def get_param_labels(params):
    return jax.tree.map(
        lambda p: "muon" if p.ndim >= 2 else "adamw",
        params
    )

def create_hybrid_muon_adamw(
    learning_rate: Union[float, Callable[[int], float]], 
    muon_momentum: float = 0.95, 
    adam_b1: float = 0.9, 
    adam_b2: float = 0.999, 
    weight_decay: float = 0.01,
    ns_steps: int = 5
):
    # Momentum MUST be accumulated before orthogonalization
    muon_chain = optax.chain(
        optax.trace(decay=muon_momentum, nesterov=False),
        scale_by_muon(ns_steps=ns_steps),
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_learning_rate(learning_rate)
    )
    
    adamw_chain = optax.adamw(
        learning_rate=learning_rate,
        b1=adam_b1,
        b2=adam_b2,
        weight_decay=weight_decay
    )
    
    return optax.multi_transform(
        transforms={"muon": muon_chain, "adamw": adamw_chain},
        param_labels=get_param_labels
    )