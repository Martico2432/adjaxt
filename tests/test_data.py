import numpy as np
import pytest
from datasets import Dataset
import jax.numpy as jnp
from adjaxt.data import pack_sequences, jax_dataloader, datamix, curriculum_jax_dataloader

def test_pack_sequences():
    raw_data = {"input_ids": [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]}
    ds = Dataset.from_dict(raw_data)
    packer = pack_sequences(seq_len=4, token_col="input_ids")
    packed_ds = packer(ds)
    
    assert len(packed_ds) == 2
    assert packed_ds[0]["input_ids"] == [1, 2, 3, 4]
    assert packed_ds[1]["input_ids"] == [5, 6, 7, 8]

def test_jax_dataloader_batching():
    data = {"input_ids": np.arange(20).reshape(10, 2)}
    ds = Dataset.from_dict(data)
    loader = jax_dataloader(ds, batch_size=4, drop_last=True, shuffle=False)
    
    batch = next(loader)
    assert isinstance(batch["input_ids"], jnp.ndarray)
    assert batch["input_ids"].shape == (4, 2)
    assert np.array_equal(batch["input_ids"][:2], [[0, 1], [2, 3]])

def test_datamix_and_curriculum_loader():
    ds1 = Dataset.from_dict({"val": [1, 2, 3, 4]})
    ds2 = Dataset.from_dict({"val": [10, 20, 30, 40]})
    mixed_stages = datamix([[ds1], [ds2]])
    
    loader = curriculum_jax_dataloader(
        mixed_stages=mixed_stages,
        stage_transitions=[2, 4],
        batch_size=2
    )
    
    batches = [next(loader) for _ in range(4)]
    assert len(batches) == 4
    for b in batches:
        assert b["val"].shape == (2,)