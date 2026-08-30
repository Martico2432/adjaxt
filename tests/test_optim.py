import pytest
import jax
import jax.numpy as jnp
import optax
import numpy as np

from adjaxt.optim import (
    newton_schulz_iteration,
    get_param_labels,
    create_hybrid_muon_adamw,
)


def test_newton_schulz_shape_and_orthonormal_input():
    """Validates exact orthogonal preservation and multi-dimensional reshaping."""
    key = jax.random.PRNGKey(42)

    # 1. Test exact orthogonal input preservation (Q from QR decomposition)
    raw = jax.random.normal(key, (32, 64))
    q, _ = jnp.linalg.qr(raw.T)
    q_ortho = q.T[:32, :]  # (32, 64) with orthonormal rows

    x_ortho = newton_schulz_iteration(q_ortho, steps=5)
    assert x_ortho.shape == q_ortho.shape

    # Rows are strictly orthonormal, so off-diagonals must be ~0
    cov = x_ortho @ x_ortho.T
    off_diag = cov - jnp.diag(jnp.diag(cov))
    assert jnp.all(jnp.abs(off_diag) < 1e-4), "Off-diagonals on orthogonal inputs must be ~0."

    # 2. Test >2D tensor handling
    tensor_3d = jax.random.normal(key, (4, 16, 32))
    out_3d = newton_schulz_iteration(tensor_3d, steps=5)
    assert out_3d.shape == tensor_3d.shape


def test_newton_schulz_singular_value_compression():
    """Validates that Newton-Schulz compresses singular values and condition number."""
    key = jax.random.PRNGKey(123)
    g = jax.random.normal(key, (32, 64))

    # Calculate initial singular value spread
    s_initial = jnp.linalg.svd(g, compute_uv=False)
    cond_initial = s_initial[0] / s_initial[-1]

    x = newton_schulz_iteration(g, steps=5)
    assert x.shape == g.shape

    # Calculate post-iteration singular values
    s_post = jnp.linalg.svd(x, compute_uv=False)
    cond_post = s_post[0] / s_post[-1]

    # Condition number should drop significantly towards 1
    assert cond_post < cond_initial
    assert cond_post < 2.0, f"Condition number should be compressed near 1, got {cond_post:.2f}"

    # Singular values should be tightly bounded around the aspect ratio scale sqrt(64/32) = 1.414
    expected_scale = np.sqrt(64 / 32)
    assert jnp.all(s_post > 0.5 * expected_scale)
    assert jnp.all(s_post < 1.5 * expected_scale)


def test_get_param_labels():
    """Ensures parameter dimensionality correctly maps to 'muon' or 'adamw'."""
    params = {
        "attention": {
            "kernel": jnp.zeros((128, 128)),  # >= 2D -> muon
            "bias": jnp.zeros((128,)),        # < 2D -> adamw
        },
        "layernorm": {
            "scale": jnp.zeros((128,)),       # < 2D -> adamw
        },
    }
    labels = get_param_labels(params)

    assert labels["attention"]["kernel"] == "muon"
    assert labels["attention"]["bias"] == "adamw"
    assert labels["layernorm"]["scale"] == "adamw"


def test_hybrid_optimizer_init_and_update():
    """Verifies that the hybrid optimizer initializes and updates without shape errors."""
    params = {
        "kernel": jnp.ones((16, 16)),
        "bias": jnp.zeros((16,)),
    }

    grads = jax.tree.map(lambda p: jnp.ones_like(p) * 0.1, params)

    optimizer = create_hybrid_muon_adamw(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    updates, new_state = optimizer.update(grads, opt_state, params)

    assert updates["kernel"].shape == params["kernel"].shape
    assert updates["bias"].shape == params["bias"].shape
    assert not jnp.array_equal(updates["kernel"], grads["kernel"])
    assert not jnp.array_equal(updates["bias"], grads["bias"])