import numpy as np
import jax.numpy as jnp
from typing import List, Optional, Callable, Iterator, Dict
from datasets import Dataset, interleave_datasets
from itertools import chain
import json
import math
import os
from huggingface_hub import HfApi

DataStage = Callable[[Dataset], Dataset]

def pack_sequences(seq_len: int, token_col: str = "input_ids") -> DataStage:
    """Concatenates all tokenized sequences and chunks them into exact static lengths."""
    def stage(ds: Dataset) -> Dataset:
        def group_texts(examples):
            # O(N) flattening via itertools.chain
            concatenated = list(chain.from_iterable(examples[token_col]))
            total_length = len(concatenated)

            if total_length >= seq_len:
                total_length = (total_length // seq_len) * seq_len

            return {
                token_col: [
                    concatenated[i : i + seq_len]
                    for i in range(0, total_length, seq_len)
                ]
            }

        return ds.map(
            group_texts,
            batched=True,
            remove_columns=ds.column_names,
            desc=f"Packing sequences to length {seq_len}",
        )

    return stage


def jax_dataloader(
    dataset,
    batch_size: int,
    seed: int = 42,
    drop_last: bool = True,
    shuffle: bool = True,
) -> Iterator[Dict[str, jnp.ndarray]]:
    """Endless batch generator yielding batches formatted as JAX device arrays."""
    num_rows = len(dataset)
    if drop_last and num_rows < batch_size:
        raise ValueError(
            f"Dataset size ({num_rows}) is smaller than batch_size ({batch_size}) with drop_last=True."
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(num_rows)

    while True:
        if shuffle:
            rng.shuffle(indices)

        for i in range(0, num_rows, batch_size):
            batch_idx = indices[i : i + batch_size]
            if drop_last and len(batch_idx) < batch_size:
                continue

            batch_dict = dataset[batch_idx.tolist()]
            yield {
                k: jnp.array(np.asarray(v))
                for k, v in batch_dict.items()
            }


def pretokenize_and_upload(
    dataset: Dataset,
    seq_len: int,
    chunk_size: int,
    repo_id: str,
    token: Optional[str] = None,
    token_col: str = "input_ids",
) -> None:
    api = HfApi(token=token)
    packed_ds = pack_sequences(seq_len=seq_len, token_col=token_col)(dataset)
    total_rows = len(packed_ds)
    num_chunks = math.ceil(total_rows / chunk_size)
    ledger = {}
    
    os.makedirs("temp_chunks/pretokenized", exist_ok=True)

    for i in range(num_chunks):
        chunk_name = f"chunk_{i:05d}"
        chunk = packed_ds.shard(num_shards=num_chunks, index=i, contiguous=True)
        file_path = f"temp_chunks/pretokenized/{chunk_name}.parquet"
        chunk.to_parquet(file_path)
        ledger[chunk_name] = {
            "status": "unassigned",
            "worker": None,
            "claimed_at": None,
            "rows": len(chunk),
        }

    with open("temp_chunks/data_ledger.json", "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)

    # Single atomic commit for all dataset chunks + ledger
    api.upload_folder(
        folder_path="temp_chunks",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload {num_chunks} pretokenized data chunks and ledger",
    )


def datamix(
    stages: List[List[Dataset]], 
    mix_probabilities: Optional[List[List[float]]] = None,
    seed: int = 42
) -> List[Dataset]:
    mixed_stages = []
    for i, stage_datasets in enumerate(stages):
        if len(stage_datasets) == 1:
            mixed_stages.append(stage_datasets[0])
        else:
            probs = mix_probabilities[i] if mix_probabilities else None
            mixed = interleave_datasets(
                stage_datasets, 
                probabilities=probs, 
                seed=seed,
                stopping_strategy="all_exhausted"
            )
            mixed_stages.append(mixed)
    return mixed_stages


def curriculum_jax_dataloader(
    mixed_stages: List[Dataset],
    stage_transitions: List[int],
    batch_size: int,
    seed: int = 42
) -> Iterator[Dict[str, jnp.ndarray]]:
    if len(mixed_stages) != len(stage_transitions):
        raise ValueError("Must provide a transition step count for every stage.")
    
    rng = np.random.default_rng(seed)
    current_step = 0
    
    for stage_idx, max_steps_for_stage in enumerate(stage_transitions):
        dataset = mixed_stages[stage_idx]
        
        while current_step < max_steps_for_stage:
            indices = np.arange(len(dataset))
            rng.shuffle(indices)
            
            for i in range(0, len(indices), batch_size):
                if current_step >= max_steps_for_stage:
                    break
                    
                batch_indices = indices[i : i + batch_size]
                if len(batch_indices) < batch_size:
                    continue
                    
                batch = dataset[batch_indices.tolist()]
                current_step += 1
                
                yield {k: jnp.array(np.asarray(v)) for k, v in batch.items()}