import pytest
import jax
import jax.numpy as jnp
import optax
import numpy as np

# Assuming your code is in optim.py
from adjaxt.optim import (
    newton_schulz_iteration,
    get_param_labels,
    create_hybrid_muon_adamw
)

def test_newton_schulz_iteration():
    """Validates orthogonalization and shape preservation of Newton-Schulz."""
    key = jax.random.PRNGKey(42)
    
    # Generate a random 2D matrix
    g = jax.random.normal(key, (32, 64))
    x = newton_schulz_iteration(g, steps=15)
    
    # Ensure shape is completely preserved
    assert x.shape == g.shape
    
    # Check if the rows are roughly orthogonalized (X @ X.T should be proportional to Identity)
    cov = x @ x.T
    diag = jnp.diag(cov)
    off_diag = cov - jnp.diag(diag)
    
    # With float32, precision isn't perfect, but off-diagonals should be near zero
    assert jnp.all(jnp.abs(off_diag) < 1e-2), "Off-diagonals should be close to zero."

def test_get_param_labels():
    """Ensures parameter dimensionality correctly maps to 'muon' or 'adamw'."""
    params = {
        "attention": {
            "kernel": jnp.zeros((128, 128)),  # >= 2D -> muon
            "bias": jnp.zeros((128,))         # < 2D -> adamw
        },
        "layernorm": {
            "scale": jnp.zeros((128,))        # < 2D -> adamw
        }
    }
    labels = get_param_labels(params)
    
    assert labels["attention"]["kernel"] == "muon"
    assert labels["attention"]["bias"] == "adamw"
    assert labels["layernorm"]["scale"] == "adamw"

def test_hybrid_optimizer_init_and_update():
    """Verifies that the hybrid optimizer initializes and updates without shape errors."""
    params = {
        "kernel": jnp.ones((16, 16)),
        "bias": jnp.zeros((16,))
    }
    
    # Generate dummy gradients
    grads = jax.tree.map(lambda p: jnp.ones_like(p) * 0.1, params)
    
    optimizer = create_hybrid_muon_adamw(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    
    # Execute a single update step
    updates, new_state = optimizer.update(grads, opt_state, params)
    
    # Assert updates map precisely to the original parameter shapes
    assert updates["kernel"].shape == params["kernel"].shape
    assert updates["bias"].shape == params["bias"].shape
    
    # Ensure gradients were actively modified by the optimizer routines
    assert not jnp.array_equal(updates["kernel"], grads["kernel"])
    assert not jnp.array_equal(updates["bias"], grads["bias"])