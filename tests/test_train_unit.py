import os
import json
import tempfile
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import optax
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from adjaxt.train import (
    dict_to_dataclass,
    _extract_callable_source,
    build_loss_and_step_fn,
    build_optimizer_from_config,
    start_heartbeat_thread,
    mark_chunks_completed,
    wait_for_global_sync,
    fetch_train_manifest,
)
from adjaxt.config import RMSNormConfig


@dataclass
class DummyChildConfig:
    dim: int


@dataclass
class DummyParentConfig:
    child: DummyChildConfig
    name: str


def test_dict_to_dataclass():
    raw_dict = {"child": {"dim": 64}, "name": "test_model"}
    cfg = dict_to_dataclass(DummyParentConfig, raw_dict)
    assert isinstance(cfg, DummyParentConfig)
    assert isinstance(cfg.child, DummyChildConfig)
    assert cfg.child.dim == 64
    assert cfg.name == "test_model"


def test_extract_callable_source():
    def sample_fn(x):
        return x * 2

    source = _extract_callable_source(sample_fn)
    assert "def sample_fn(x):" in source
    assert "return x * 2" in source


def test_build_loss_and_step_fn():
    vocab_size = 8
    
    # Forward function returning 3D logits (batch, seq_len, vocab_size)
    def dummy_forward(input_ids, params):
        return jnp.tile(params["w"][None, None, :], (*input_ids.shape, 1))

    params = {"w": jnp.ones((vocab_size,), dtype=jnp.float32)}
    optimizer = optax.adamw(learning_rate=1e-3)
    step_fn = build_loss_and_step_fn(dummy_forward, optimizer)

    batch = {
        "input_ids": jnp.ones((2, 4), dtype=jnp.int32),
        "labels": jnp.zeros((2, 4), dtype=jnp.int32),
    }

    opt_state = optimizer.init(params)
    new_params, new_opt_state, loss = step_fn(params, opt_state, batch)

    assert loss is not None
    assert new_params["w"].shape == params["w"].shape
    assert not jnp.isnan(loss)


def test_build_optimizer_from_config():
    opt_adam = build_optimizer_from_config({"optimizer_type": "adamw", "inner_lr": 1e-4})
    opt_muon = build_optimizer_from_config({"optimizer_type": "muon", "inner_lr": 1e-3})
    assert isinstance(opt_adam, optax.GradientTransformation)
    assert isinstance(opt_muon, optax.GradientTransformation)


def test_start_heartbeat_thread():
    mock_api = MagicMock()
    stop_event = start_heartbeat_thread(mock_api, "repo/test", "worker_0", interval=1)
    assert not stop_event.is_set()
    stop_event.set()
    assert stop_event.is_set()


def test_mark_chunks_completed():
    mock_api = MagicMock()
    mark_chunks_completed(mock_api, "repo/test", ["chunk_00001"], "worker_0")
    assert mock_api.create_commit.called


@patch("adjaxt.train.hf_hub_download")
def test_wait_for_global_sync(mock_download, tmp_path):
    state_file = tmp_path / "sync_state.json"
    state_file.write_text(json.dumps({"round": 2}))
    mock_download.return_value = str(state_file)

    mock_api = MagicMock()
    next_round = wait_for_global_sync(mock_api, "repo/test", current_round=1, poll_interval=0.01, timeout=2)
    assert next_round == 2


@patch("adjaxt.train.hf_hub_download")
def test_fetch_train_manifest(mock_download, tmp_path):
    manifest_data = {
        "model_config_cls": "RMSNormConfig",
        "model_config": {"dim": 128, "eps": 1e-5},
        "training_config": {"batch_size": 4, "inner_lr": 1e-4},
    }
    cfg_file = tmp_path / "train_config.json"
    cfg_file.write_text(json.dumps(manifest_data))
    mock_download.return_value = str(cfg_file)

    model_cfg, train_cfg = fetch_train_manifest("repo/test")
    assert isinstance(model_cfg, RMSNormConfig)
    assert model_cfg.dim == 128
    assert train_cfg["batch_size"] == 4