import json
import pytest
import jax
import jax.numpy as jnp
import optax
from unittest.mock import MagicMock, patch
from adjaxt.approx import benchmark_step_throughput, claim_chunks_from_ledger, approx, WorkerPlan

def test_benchmark_step_throughput():
    vocab_size = 8

    # Forward function returning 3D logits (batch, seq_len, vocab_size)
    def forward_fn(input_ids, params):
        return jnp.tile(params["w"][None, None, :], (*input_ids.shape, 1))

    params = {"w": jnp.ones((vocab_size,), dtype=jnp.float32)}
    optimizer = optax.adamw(1e-3)
    sample_batch = {
        "input_ids": jnp.ones((2, 4), dtype=jnp.int32),
        "labels": jnp.zeros((2, 4), dtype=jnp.int32),
    }

    steps_per_sec = benchmark_step_throughput(
        forward_fn=forward_fn,
        params=params,
        optimizer=optimizer,
        sample_batch=sample_batch,
        num_warmup=1,
        num_steps=2,
    )
    assert steps_per_sec > 0.0

@patch("adjaxt.approx.hf_hub_download")
def test_claim_chunks_from_ledger(mock_download, tmp_path):
    ledger = {
        "chunk_00000": {"status": "unassigned"},
        "chunk_00001": {"status": "unassigned"},
        "chunk_00002": {"status": "completed"},
    }
    ledger_path = tmp_path / "data_ledger.json"
    ledger_path.write_text(json.dumps(ledger))
    mock_download.return_value = str(ledger_path)

    mock_api = MagicMock()
    claimed = claim_chunks_from_ledger(mock_api, "repo/test", "worker_1", needed_chunks=2)

    assert claimed == ["chunk_00000", "chunk_00001"]
    with open(ledger_path, "r") as f:
        updated = json.load(f)
    assert updated["chunk_00000"]["status"] == "in_progress"
    assert updated["chunk_00000"]["worker"] == "worker_1"
    assert mock_api.upload_file.called

@patch("adjaxt.approx.benchmark_step_throughput", return_value=10.0)
@patch("adjaxt.approx.claim_chunks_from_ledger", return_value=["chunk_00000"])
@patch("adjaxt.approx.HfApi")
def test_approx_orchestration(mock_api_cls, mock_claim, mock_bench, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api

    plan = approx(
        repo_id="repo/test",
        worker_id="worker_42",
        forward_fn=MagicMock(),
        params={},
        optimizer=optax.adamw(1e-3),
        sample_batch={},
        target_sync_interval_sec=10.0,
        chunk_rows=100,
        batch_size=4,
    )
    assert isinstance(plan, WorkerPlan)
    assert plan.worker_id == "worker_42"
    assert plan.h_steps == 100
    assert plan.assigned_chunks == ["chunk_00000"]
    assert mock_api.upload_file.called