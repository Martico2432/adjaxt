"""Shared test infrastructure. Grows on demand — don't pre-populate."""
import numpy as np
import pytest

# Tolerance calibration from the Qwen3-30B logit match (see examples/qwen3_moe_match.py)
TOL_FP32 = 1e-5
TOL_BF16 = 1e-2

SEED = 0


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)