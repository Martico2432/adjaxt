from typing import Callable, Dict, Any, Optional, Union, Tuple
import os
import time
import json
import threading
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
import optax
from huggingface_hub import HfApi, hf_hub_download
from safetensors.numpy import save_file, load_file
from datasets import load_dataset

from adjaxt.data import jax_dataloader

def start_heartbeat_thread(
    api: HfApi,
    repo_id: str,
    worker_id: str,
    interval: int = 120,
) -> threading.Event:
    stop_event = threading.Event()

    def heartbeat():
        while not stop_event.is_set():
            beat_data = {"worker_id": worker_id, "last_seen": time.time()}
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
                json.dump(beat_data, tmp)
                tmp_path = tmp.name

            try:
                api.upload_file(
                    path_or_fileobj=tmp_path,
                    path_in_repo=f"heartbeats/{worker_id}.json",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
            except Exception:
                pass
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            stop_event.wait(interval)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    return stop_event


def claim_next_chunk(
    api: HfApi,
    repo_id: str,
    worker_id: str,
    token: Optional[str] = None,
    max_retries: int = 5,
) -> Tuple[Optional[str], Optional[str]]:
    """Claims the next available Parquet chunk with retry backoff."""
    for attempt in range(max_retries):
        try:
            ledger_file = hf_hub_download(
                repo_id=repo_id,
                filename="data_ledger.json",
                repo_type="dataset",
                token=token,
            )
            with open(ledger_file, "r", encoding="utf-8") as f:
                ledger = json.load(f)

            target_chunk = None
            for chunk_name, meta in ledger.items():
                if meta.get("status") == "unassigned":
                    meta["status"] = "in_progress"
                    meta["worker"] = worker_id
                    meta["claimed_at"] = time.time()
                    target_chunk = chunk_name
                    break

            if target_chunk is None:
                return None, None

            with open(ledger_file, "w", encoding="utf-8") as f:
                json.dump(ledger, f, indent=2)

            api.upload_file(
                path_or_fileobj=ledger_file,
                path_in_repo="data_ledger.json",
                repo_id=repo_id,
                repo_type="dataset",
            )

            chunk_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"pretokenized/{target_chunk}.parquet",
                repo_type="dataset",
                token=token,
            )
            return target_chunk, chunk_path
        except Exception as e:
            time.sleep(np.random.uniform(1.0, 3.0))

    return None, None


def mark_chunk_completed(
    api: HfApi,
    repo_id: str,
    chunk_name: str,
    worker_id: str,
    token: Optional[str] = None,
):
    """Marks a chunk as completed in the central ledger."""
    try:
        ledger_file = hf_hub_download(
            repo_id=repo_id,
            filename="data_ledger.json",
            repo_type="dataset",
            token=token,
        )
        with open(ledger_file, "r", encoding="utf-8") as f:
            ledger = json.load(f)

        if chunk_name in ledger and ledger[chunk_name].get("worker") == worker_id:
            ledger[chunk_name]["status"] = "completed"
            ledger[chunk_name]["completed_at"] = time.time()

            with open(ledger_file, "w", encoding="utf-8") as f:
                json.dump(ledger, f, indent=2)

            api.upload_file(
                path_or_fileobj=ledger_file,
                path_in_repo="data_ledger.json",
                repo_id=repo_id,
                repo_type="dataset",
            )
    except Exception as e:
        print(f"Warning: Failed to mark chunk {chunk_name} as completed: {e}")


def wait_for_global_sync(
    api: HfApi,
    repo_id: str,
    current_round: int,
    token: Optional[str] = None,
    poll_interval: int = 15,
    timeout: int = 3600,
) -> int:
    start_time = time.time()
    print(f"Waiting for outer aggregation (current round: {current_round})...")

    while time.time() - start_time < timeout:
        try:
            sync_file = hf_hub_download(
                repo_id=repo_id,
                filename="sync_state.json",
                repo_type="dataset",
                token=token,
            )
            with open(sync_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            latest_round = state.get("round", 0)
            if latest_round > current_round:
                print(f"Global sync complete. Advancing to round {latest_round}.")
                return latest_round
        except Exception:
            pass

        time.sleep(poll_interval)

    raise TimeoutError(f"Timed out waiting for global sync round > {current_round}.")


def build_loss_and_step_fn(forward_fn: Callable, optimizer: optax.GradientTransformation):
    @jax.jit
    def step_fn(params: dict, opt_state: Any, batch: dict):
        def loss_fn(p):
            logits = forward_fn(batch["input_ids"], p)
            labels = batch.get("labels", batch["input_ids"])
            
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=shift_logits,
                labels=shift_labels
            )
            return jnp.mean(loss)

        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val

    return step_fn


def fit(
    repo_id: str,
    worker_id: str,
    forward_fn: Callable,
    inner_optimizer: optax.GradientTransformation,
    token: Optional[str] = None,
    inner_steps: Union[int, Callable[[], int]] = 500,  # Supports static or dynamic H
    batch_size: int = 4,
    seed: int = 42,
    heartbeat_interval: int = 120,
):
    api = HfApi(token=token)
    stop_heartbeat = start_heartbeat_thread(api, repo_id, worker_id, interval=heartbeat_interval)
    step_fn = build_loss_and_step_fn(forward_fn, inner_optimizer)

    current_round = 0

    try:
        while True:
            print(f"Fetching global weights for round {current_round}...")
            weights_file = hf_hub_download(
                repo_id=repo_id,
                filename="global_weights.safetensors",
                repo_type="dataset",
                token=token,
            )
            raw_weights = load_file(weights_file)
            global_weights = {k: jnp.array(v) for k, v in raw_weights.items()}
            local_weights = {k: jnp.array(v) for k, v in global_weights.items()}

            chunk_name, chunk_parquet_path = claim_next_chunk(api, repo_id, worker_id, token=token)
            if not chunk_parquet_path:
                print("No more unassigned data chunks available. Stopping.")
                break

            ds = load_dataset("parquet", data_files=chunk_parquet_path, split="train")
            # Rotate seed per outer round to ensure independent data traversals
            loader = jax_dataloader(ds, batch_size=batch_size, seed=seed + current_round, drop_last=True)
            opt_state = inner_optimizer.init(local_weights)

            target_h = inner_steps() if callable(inner_steps) else inner_steps
            steps_completed = 0

            for batch in loader:
                local_weights, opt_state, loss = step_fn(local_weights, opt_state, batch)
                steps_completed += 1
                
                if steps_completed % 50 == 0:
                    print(f"Round {current_round} | Step {steps_completed}/{target_h} | Loss: {loss:.4f}")
                    
                if steps_completed >= target_h:
                    break

            # Mark Parquet chunk as processed
            mark_chunk_completed(api, repo_id, chunk_name, worker_id, token=token)

            # Delta W = W_global - W_local (pseudo-gradient)
            pseudo_grad = jax.tree.map(
                lambda g, l: np.asarray(g - l, dtype=np.float32),
                global_weights,
                local_weights,
            )

            # Metadata filename formatting: worker_round_steps.safetensors
            update_filename = f"{worker_id}_round_{current_round}_steps_{steps_completed}.safetensors"
            with tempfile.TemporaryDirectory() as tmpdir:
                local_update_path = os.path.join(tmpdir, update_filename)
                save_file(pseudo_grad, local_update_path)

                api.upload_file(
                    path_or_fileobj=local_update_path,
                    path_in_repo=f"worker_updates/{update_filename}",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
            print(f"Uploaded update {update_filename}.")

            current_round = wait_for_global_sync(
                api=api,
                repo_id=repo_id,
                current_round=current_round,
                token=token,
            )

    finally:
        stop_heartbeat.set()