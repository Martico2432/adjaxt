import os
import json
import time
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from safetensors.numpy import save_file, load_file
from adjaxt.coordinator import run_coordinator


@patch("adjaxt.coordinator.time.sleep", side_effect=InterruptedError("Stop loop"))
@patch("adjaxt.coordinator.list_repo_files")
@patch("adjaxt.coordinator.hf_hub_download")
@patch("adjaxt.coordinator.HfApi")
def test_run_coordinator_single_round(mock_api_cls, mock_download, mock_list_files, mock_sleep, tmp_path):
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api

    # 1. Prepare sync state
    sync_file = tmp_path / "sync_state.json"
    sync_file.write_text(json.dumps({"round": 0}))

    # 2. Prepare global weights
    global_weights = {"w": np.array([10.0, 10.0], dtype=np.float32)}
    global_file = tmp_path / "global_weights.safetensors"
    save_file(global_weights, str(global_file))

    # 3. Prepare worker pseudo-gradients
    w1_grad = {"w": np.array([2.0, 2.0], dtype=np.float32)}
    w2_grad = {"w": np.array([4.0, 4.0], dtype=np.float32)}
    w1_file = tmp_path / "w1.safetensors"
    w2_file = tmp_path / "w2.safetensors"
    save_file(w1_grad, str(w1_file))
    save_file(w2_grad, str(w2_file))

    mock_list_files.return_value = [
        "worker_updates/worker_1__round_0__steps_100.safetensors",
        "worker_updates/worker_2__round_0__steps_100.safetensors",
    ]

    def download_router(repo_id, filename, **kwargs):
        if filename == "sync_state.json":
            return str(sync_file)
        if filename == "global_weights.safetensors":
            return str(global_file)
        if "worker_1" in filename:
            return str(w1_file)
        if "worker_2" in filename:
            return str(w2_file)
        return str(tmp_path / filename)

    mock_download.side_effect = download_router

    # Run coordinator (aborts on sleep via side_effect)
    with pytest.raises(InterruptedError):
        run_coordinator(
            repo_id="repo/test",
            min_workers_per_round=2,
            outer_lr=0.5,
            outer_momentum=0.0,
            poll_interval=1,
        )

    # Verify upload was called for both updated weights and new sync state
    uploaded_targets = [call.kwargs.get("path_in_repo") for call in mock_api.upload_file.call_args_list]
    assert "global_weights.safetensors" in uploaded_targets
    assert "sync_state.json" in uploaded_targets