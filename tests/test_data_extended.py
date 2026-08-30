import os
import pytest
import numpy as np
from datasets import Dataset
from unittest.mock import MagicMock, patch
from adjaxt.data import jax_dataloader, pretokenize_and_upload

def test_jax_dataloader_drop_last_error():
    ds = Dataset.from_dict({"input_ids": [[1, 2], [3, 4]]})
    with pytest.raises(ValueError, match="Dataset size .* is smaller than batch_size"):
        next(jax_dataloader(ds, batch_size=8, drop_last=True))

@patch("adjaxt.data.HfApi")
def test_pretokenize_and_upload(mock_api_cls, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api

    data = {"input_ids": [[1, 2, 3, 4, 5, 6, 7, 8] for _ in range(20)]}
    ds = Dataset.from_dict(data)

    pretokenize_and_upload(
        dataset=ds,
        seq_len=8,
        chunk_size=10,
        repo_id="repo/test",
    )
    assert mock_api.upload_file.called
    assert os.path.exists(tmp_path / "temp_chunks" / "data_ledger.json")