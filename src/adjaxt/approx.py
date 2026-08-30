from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import json
import time
import jax
import jax.numpy as jnp
import optax
from huggingface_hub import HfApi, hf_hub_download


@dataclass
class WorkerPlan:
    worker_id: str
    h_steps: int
    throughput_steps_per_sec: float
    assigned_chunks: List[str]
    target_sync_interval_sec: float

def default_loss_fn(batch: dict, params: dict, forward_fn: Callable) -> jax.Array:
    """Default cross-entropy loss function."""
    logits = forward_fn(batch["input_ids"], params)
    labels = batch.get("labels", batch["input_ids"])
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=shift_logits,
        labels=shift_labels,
    )
    return jnp.mean(loss)


def benchmark_step_throughput(
    params: dict,
    optimizer: optax.GradientTransformation,
    sample_batch: dict,
    forward_fn: Optional[Callable] = None,
    loss_fn: Optional[Callable] = None,
    use_fp32_sandbox: bool = True,
    num_warmup: int = 2,
    num_steps: int = 5,
) -> float:
    """Benchmarks step throughput using the framework's core step_fn."""
    
    # Import locally to avoid circular dependencies if needed
    from adjaxt.train import build_loss_and_step_fn 

    step_fn = build_loss_and_step_fn(
        forward_fn=forward_fn,
        optimizer=optimizer,
        loss_fn=loss_fn,
        use_fp32_sandbox=use_fp32_sandbox,
    )
    
    opt_state = optimizer.init(params)
    
    # Warmup steps
    for _ in range(num_warmup):
        params, opt_state, loss = step_fn(params, opt_state, sample_batch)
        loss.block_until_ready()

    # Timed benchmarking steps
    start_t = time.perf_counter()
    for _ in range(num_steps):
        params, opt_state, loss = step_fn(params, opt_state, sample_batch)
        loss.block_until_ready()
    elapsed = time.perf_counter() - start_t

    return num_steps / max(elapsed, 1e-6)


def claim_chunks_from_ledger(
    api: HfApi,
    repo_id: str,
    worker_id: str,
    needed_chunks: int,
    token: Optional[str] = None,
) -> List[str]:
    """Claims unassigned data chunks from the Hub repository's data_ledger.json."""
    ledger_path = hf_hub_download(
        repo_id=repo_id,
        filename="data_ledger.json",
        repo_type="dataset",
        token=token,
    )
    with open(ledger_path, "r") as f:
        ledger = json.load(f)

    claimed = []
    for chunk_name, chunk_meta in ledger.items():
        if len(claimed) >= needed_chunks:
            break
        if chunk_meta.get("status") == "unassigned":
            chunk_meta["status"] = "in_progress"
            chunk_meta["worker"] = worker_id
            chunk_meta["claimed_at"] = time.time()
            claimed.append(chunk_name)

    with open(ledger_path, "w") as f:
        json.dump(ledger, f, indent=2)

    api.upload_file(
        path_or_fileobj=ledger_path,
        path_in_repo="data_ledger.json",
        repo_id=repo_id,
        repo_type="dataset",
    )
    return claimed


def approx(
    repo_id: str,
    worker_id: str,
    forward_fn: Callable,
    params: dict,
    optimizer: optax.GradientTransformation,
    sample_batch: dict,
    token: Optional[str] = None,
    target_sync_interval_sec: float = 600.0,
    chunk_rows: int = 1000,
    batch_size: int = 4,
) -> WorkerPlan:
    """
    Profiles local worker throughput, computes target inner steps (H),
    registers the worker profile, and assigns the initial data partition.
    """
    api = HfApi(token=token)

    # 1. Profile worker compute speed
    steps_per_sec = benchmark_step_throughput(
        forward_fn=forward_fn,
        params=params,
        optimizer=optimizer,
        sample_batch=sample_batch,
    )

    # 2. Derive H (number of inner steps feasible within the sync window)
    h_steps = max(1, int(steps_per_sec * target_sync_interval_sec))
    total_tokens_or_rows = h_steps * batch_size
    needed_chunks = max(1, (total_tokens_or_rows + chunk_rows - 1) // chunk_rows)

    # 3. Claim initial data chunks from remote ledger
    assigned_chunks = claim_chunks_from_ledger(
        api=api,
        repo_id=repo_id,
        worker_id=worker_id,
        needed_chunks=needed_chunks,
        token=token,
    )

    # 4. Register worker profile to Hub
    worker_meta = {
        "worker_id": worker_id,
        "throughput_steps_per_sec": steps_per_sec,
        "h_steps": h_steps,
        "assigned_chunks": assigned_chunks,
        "last_registered": time.time(),
    }
    worker_file = f"{worker_id}_plan.json"
    with open(worker_file, "w") as f:
        json.dump(worker_meta, f, indent=2)

    api.upload_file(
        path_or_fileobj=worker_file,
        path_in_repo=f"workers/{worker_id}.json",
        repo_id=repo_id,
        repo_type="dataset",
    )

    return WorkerPlan(
        worker_id=worker_id,
        h_steps=h_steps,
        throughput_steps_per_sec=steps_per_sec,
        assigned_chunks=assigned_chunks,
        target_sync_interval_sec=target_sync_interval_sec,
    )