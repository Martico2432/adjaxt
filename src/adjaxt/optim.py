"""
Adjaxt Fused Optimizers
High-performance hybrid Muon (fused quintic Newton-Schulz) + AdamW optimizer.
"""

from functools import partial
from typing import NamedTuple, Optional, Union
import jax
import jax.numpy as jnp
import optax


# =========================================================================
# 1. Fused Quintic Newton-Schulz Iteration (Polar Decomposition)
# =========================================================================
@partial(jax.jit, static_argnames=("steps",))
def zeropower_via_newtonschulz5(
    G: jax.Array,
    steps: int = 5,
    eps: float = 1e-7,
) -> jax.Array:
    """
    Newton-Schulz iteration (quintic polynomial) for polar decomposition in Muon.
    Scales by sqrt(max(M/N, N/M)) to ensure uniform spectral energy across shapes.
    """
    if G.ndim != 2:
        return G

    a, b, c = (3.4445, -4.7750, 2.0315)
    orig_dtype = G.dtype
    M, N = G.shape
    transposed = False

    # Scale by aspect ratio: max(M, N) / min(M, N)
    scale = jnp.sqrt(jnp.maximum(float(M) / float(N), float(N) / float(M)))

    # Compute in float32 for high precision and stability
    X = G.astype(jnp.float32)
    if M > N:
        X = X.T
        transposed = True

    # Normalize by Frobenius norm so spectral radius < sqrt(3)
    norm = jnp.linalg.norm(X) + eps
    X = X / norm

    # Unrolled Newton-Schulz steps
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return (X * scale).astype(orig_dtype)


# Alias for backward compatibility
newton_schulz_iteration = zeropower_via_newtonschulz5


# =========================================================================
# 2. Muon Gradient Transformation
# =========================================================================
class MuonState(NamedTuple):
    count: jax.Array
    momentum: optax.Updates


def muon(
    learning_rate: Union[float, optax.Schedule] = 6e-4,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    weight_decay: float = 0.01,
) -> optax.GradientTransformation:
    """Standard Muon optimizer transformation applying polar orthogonalization."""
    def init_fn(params):
        return MuonState(
            count=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree.map(lambda p: jnp.zeros_like(p), params),
        )

    def update_fn(updates, state, params=None):
        count = state.count + 1
        lr = learning_rate(count) if callable(learning_rate) else learning_rate

        # 1. Update momentum PyTree
        new_momentum = jax.tree.map(
            lambda g, m: momentum * m + (1.0 - momentum) * g if g.ndim == 2 else m,
            updates,
            state.momentum,
        )

        # 2. Compute update PyTree
        def _calc_update(g, m, p):
            if g.ndim != 2:
                return g

            grad = (1.0 - momentum) * g + momentum * m if nesterov else m
            v = zeropower_via_newtonschulz5(grad, steps=ns_steps)
            u = lr * v

            if p is not None and weight_decay > 0.0:
                u = u + (lr * weight_decay) * p

            return u

        if params is not None:
            new_updates = jax.tree.map(_calc_update, updates, new_momentum, params)
        else:
            new_updates = jax.tree.map(lambda g, m: _calc_update(g, m, None), updates, new_momentum)

        return new_updates, MuonState(count=count, momentum=new_momentum)

    return optax.GradientTransformation(init_fn, update_fn)


# =========================================================================
# 3. Hybrid Optimizer (Muon on 2D weights + AdamW on 1D/embeddings)
# =========================================================================
def get_param_labels(params):
    def _label(path, val):
        str_path = "/".join(str(p.key if hasattr(p, "key") else p) for p in path).lower()
        # Route embeddings, 1D params (norms/biases), AND output heads to AdamW
        if any(k in str_path for k in ("embed", "head")) or getattr(val, "ndim", 0) < 2:
            return "adamw"
        return "muon"

    return jax.tree_util.tree_map_with_path(_label, params)

def create_hybrid_muon_adamw(
    learning_rate: float = 6e-4,
    adamw_lr: Optional[float] = None,
    muon_momentum: float = 0.95,
    adamw_b1: float = 0.9,
    adamw_b2: float = 0.95,
    adamw_eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> optax.GradientTransformation:
    """Partitions 2D weights to Muon and 1D vectors/embeddings to AdamW."""
    if adamw_lr is None:
        adamw_lr = learning_rate * 0.5

    muon_opt = muon(
        learning_rate=learning_rate,
        momentum=muon_momentum,
        weight_decay=weight_decay,
    )
    adamw_opt = optax.adamw(
        learning_rate=adamw_lr,
        b1=adamw_b1,
        b2=adamw_b2,
        eps=adamw_eps,
        weight_decay=weight_decay,
    )

    return optax.multi_transform(
        transforms={"muon": muon_opt, "adamw": adamw_opt},
        param_labels=get_param_labels,
    )


# =========================================================================
# 4. Builder Dispatcher
# =========================================================================
def build_optimizer_from_config(config: dict) -> optax.GradientTransformation:
    """Builds the optimizer directly from train_config.json specifications."""
    opt_type = str(config.get("optimizer_type", "muon")).lower()
    lr = float(config.get("inner_lr", 6e-4))
    wd = float(config.get("weight_decay", 0.01))

    if opt_type in ("muon", "hybrid", "hybrid_muon"):
        return create_hybrid_muon_adamw(learning_rate=lr, weight_decay=wd)

    return optax.adamw(learning_rate=lr, weight_decay=wd)