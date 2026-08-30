from typing import Callable, Dict, Any, Optional, List, Tuple, Iterator
import os
import time
import json
import threading
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
import optax
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from safetensors.numpy import save_file, load_file
from datasets import load_dataset
import re
import importlib
import sys
import inspect
import textwrap

from adjaxt.data import jax_dataloader
from adjaxt.export import dataclass_to_dict
from adjaxt.sharding import save_checkpoint, ModelWeightMap
import adjaxt.config as cfg_module
from adjaxt.optim import create_hybrid_muon_adamw
from adjaxt.approx import WorkerPlan, benchmark_step_throughput


def load_remote_user_code(
    repo_id: str,
    entrypoint_file: str,
    forward_fn_name: str,
    token: Optional[str] = None,
) -> Callable:
    code_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"code/{entrypoint_file}",
        repo_type="dataset",
        token=token,
    )
    module_name = "adjaxt_remote_model"
    spec = importlib.util.spec_from_file_location(module_name, code_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification from {code_path}")
    user_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = user_module
    spec.loader.exec_module(user_module)
    if not hasattr(user_module, forward_fn_name):
        raise AttributeError(f"Function '{forward_fn_name}' not found in remote code.")
    return getattr(user_module, forward_fn_name)


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
    """Claims an available Parquet chunk using atomic file registrations."""
    for _ in range(max_retries):
        try:
            repo_files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)

            all_chunks = {
                m.group(1)
                for f in repo_files
                if (m := re.match(r"^pretokenized/(chunk_\d+)\.parquet$", f))
            }
            completed_chunks = {
                m.group(1)
                for f in repo_files
                if (m := re.match(r"^completed/(chunk_\d+)\.json$", f))
            }
            claimed_chunks = {
                m.group(1)
                for f in repo_files
                if (m := re.match(r"^claims/(chunk_\d+)__(.+)\.json$", f))
            }

            available = sorted(list(all_chunks - completed_chunks - claimed_chunks))
            if not available:
                return None, None

            candidate_chunk = available[np.random.randint(0, min(len(available), 3))]
            claim_filename = f"{candidate_chunk}__{worker_id}.json"
            claim_data = {
                "chunk": candidate_chunk,
                "worker_id": worker_id,
                "claimed_at": time.time(),
            }

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = os.path.join(tmpdir, claim_filename)
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(claim_data, f)
                api.upload_file(
                    path_or_fileobj=tmp_path,
                    path_in_repo=f"claims/{claim_filename}",
                    repo_id=repo_id,
                    repo_type="dataset",
                )

            time.sleep(1.0)
            updated_files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
            competing_claims = sorted([
                m.group(1)
                for f in updated_files
                if (m := re.match(rf"^claims/{candidate_chunk}__(.+)\.json$", f))
            ])

            if competing_claims and competing_claims[0] == worker_id:
                chunk_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"pretokenized/{candidate_chunk}.parquet",
                    repo_type="dataset",
                    token=token,
                )
                return candidate_chunk, chunk_path
            else:
                try:
                    api.delete_file(
                        path_in_repo=f"claims/{claim_filename}",
                        repo_id=repo_id,
                        repo_type="dataset",
                    )
                except Exception:
                    pass
        except Exception:
            time.sleep(np.random.uniform(1.0, 3.0))

    return None, None


def mark_chunks_completed(
    api: HfApi,
    repo_id: str,
    chunk_names: List[str],
    worker_id: str,
    token: Optional[str] = None,
) -> None:
    """Promotes claimed chunks to completed state."""
    for chunk_name in chunk_names:
        try:
            completion_data = {
                "chunk": chunk_name,
                "worker_id": worker_id,
                "completed_at": time.time(),
            }
            with tempfile.TemporaryDirectory() as tmpdir:
                completed_file = os.path.join(tmpdir, f"{chunk_name}.json")
                with open(completed_file, "w", encoding="utf-8") as f:
                    json.dump(completion_data, f)
                api.upload_file(
                    path_or_fileobj=completed_file,
                    path_in_repo=f"completed/{chunk_name}.json",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
            api.delete_file(
                path_in_repo=f"claims/{chunk_name}__{worker_id}.json",
                repo_id=repo_id,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"Warning: Failed to mark {chunk_name} as completed: {e}")


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
                labels=shift_labels,
            )
            return jnp.mean(loss)

        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val

    return step_fn


def build_optimizer_from_config(train_cfg: dict) -> optax.GradientTransformation:
    opt_type = train_cfg.get("optimizer_type", "adamw")
    lr = float(train_cfg.get("inner_lr", 1e-4))
    if opt_type == "muon":
        return create_hybrid_muon_adamw(learning_rate=lr)
    return optax.adamw(learning_rate=lr)


def dict_to_dataclass(cls: Any, data: dict) -> Any:
    """Recursively hydrates a dataclass hierarchy from a nested dictionary."""
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return data
    fields = {f.name: f.type for f in dataclasses.fields(cls)}
    init_kwargs = {}
    for k, v in data.items():
        if k in fields:
            ftype = fields[k]
            if dataclasses.is_dataclass(ftype) and isinstance(v, dict):
                init_kwargs[k] = dict_to_dataclass(ftype, v)
            else:
                init_kwargs[k] = v
    return cls(**init_kwargs)


def fetch_train_manifest(
    repo_id: str,
    token: Optional[str] = None,
    max_retries: int = 5,
) -> Tuple[Any, Dict[str, Any]]:
    """Polls and constructs model and training configurations from the repository."""
    for _ in range(max_retries):
        try:
            cfg_file = hf_hub_download(
                repo_id=repo_id,
                filename="train_config.json",
                repo_type="dataset",
                token=token,
            )
            with open(cfg_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            config_cls_name = manifest["model_config_cls"]
            config_cls = getattr(cfg_module, config_cls_name)
            model_cfg = dict_to_dataclass(config_cls, manifest["model_config"])
            train_cfg = manifest["training_config"]
            return model_cfg, train_cfg
        except Exception:
            time.sleep(np.random.uniform(1.0, 3.0))
    raise RuntimeError(f"Failed to fetch train_config.json from {repo_id}")


class DynamicChunkStream:
    """Streams batches across multiple Parquet chunks until the step target is met."""

    def __init__(
        self,
        api: HfApi,
        repo_id: str,
        worker_id: str,
        batch_size: int,
        seed: int,
        token: Optional[str] = None,
    ):
        self.api = api
        self.repo_id = repo_id
        self.worker_id = worker_id
        self.batch_size = batch_size
        self.seed = seed
        self.token = token
        self.claimed_chunks: List[str] = []

    def get_batch_iterator(self, target_steps: int) -> Iterator[dict]:
        steps_yielded = 0
        while steps_yielded < target_steps:
            chunk_name, chunk_path = claim_next_chunk(
                self.api, self.repo_id, self.worker_id, token=self.token
            )
            if not chunk_path:
                break
            self.claimed_chunks.append(chunk_name)
            ds = load_dataset("parquet", data_files=chunk_path, split="train")
            loader = jax_dataloader(
                ds,
                batch_size=self.batch_size,
                seed=self.seed + len(self.claimed_chunks),
                drop_last=True,
            )
            for batch in loader:
                yield batch
                steps_yielded += 1
                if steps_yielded >= target_steps:
                    break


def fit(
    repo_id: str,
    worker_id: str,
    forward_fn: Optional[Callable] = None,
    inner_optimizer: Optional[optax.GradientTransformation] = None,
    plan: Optional[WorkerPlan] = None,
    token: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    """
    Zero-config remote worker loop. Automatically measures passive throughput 
    and dynamically rescales inner steps (H) each round.
    """
    api = HfApi(token=token)

    # 1. Fetch Remote Manifest
    print(f"[{worker_id}] Fetching experiment configuration from {repo_id}...")
    manifest_file = hf_hub_download(
        repo_id=repo_id,
        filename="train_config.json",
        repo_type="dataset",
        token=token,
    )
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_cfg = manifest["training_config"]
    batch_size = train_cfg["batch_size"]
    seq_len = train_cfg.get("seq_len", 2048)
    target_sync_sec = float(train_cfg.get("target_sync_interval_sec", 300.0))
    h_min = int(train_cfg.get("min_inner_steps", 50))
    h_max = int(train_cfg.get("max_inner_steps", 1500))
    ema_alpha = float(train_cfg.get("ema_alpha", 0.3))
    base_seed = seed if seed is not None else train_cfg["seed"]
    heartbeat_interval = train_cfg.get("heartbeat_interval", 120)

    # 2. Dynamic Code Import if forward_fn was not provided
    if forward_fn is None:
        print(f"[{worker_id}] Loading forward pass '{manifest['forward_fn_name']}'...")
        forward_fn = load_remote_user_code(
            repo_id=repo_id,
            entrypoint_file=manifest["code_entrypoint"],
            forward_fn_name=manifest["forward_fn_name"],
            token=token,
        )

    # 3. Build Optimizer
    if inner_optimizer is None:
        inner_optimizer = build_optimizer_from_config(train_cfg)

    # 4. Fetch Global Weights
    weights_file = hf_hub_download(
        repo_id=repo_id,
        filename="global_weights.safetensors",
        repo_type="dataset",
        token=token,
    )
    raw_weights = load_file(weights_file)
    global_weights = {k: jnp.array(v) for k, v in raw_weights.items()}
    local_weights = {k: jnp.array(v) for k, v in global_weights.items()}

    # 5. Throughput Profiling & Initial H Calibration
    throughput_ema: Optional[float] = None

    if plan is not None:
        throughput_ema = plan.throughput_steps_per_sec
        raw_h = int(throughput_ema * target_sync_sec)
        current_h = int(np.clip(raw_h, h_min, h_max))
        print(f"\n[{worker_id}] Loaded Manual WorkerPlan:")
        print(f"  * Profiled Throughput:     {plan.throughput_steps_per_sec:.2f} steps/s")
        print(f"  * Target Sync Window:     {target_sync_sec:.1f}s")
        print(f"  * Initial H Steps:        {current_h} (Bounds: [{h_min}, {h_max}])\n")
    else:
        print(f"\n[{worker_id}] No pre-computed WorkerPlan found. Benchmarking hardware throughput...")
        sample_batch = {
            "input_ids": jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
            "labels": jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        }
        measured_throughput = benchmark_step_throughput(
            forward_fn=forward_fn,
            params=local_weights,
            optimizer=inner_optimizer,
            sample_batch=sample_batch,
            num_warmup=2,
            num_steps=5,
        )
        throughput_ema = measured_throughput
        raw_h = int(throughput_ema * target_sync_sec)
        current_h = int(np.clip(raw_h, h_min, h_max))

        print(f"[{worker_id}] Hardware Calibration Completed:")
        print(f"  * Measured Throughput:     {measured_throughput:.2f} steps/s")
        print(f"  * Target Sync Window:     {target_sync_sec:.1f}s")
        print(f"  * Calibrated Initial H:   {current_h} steps (Bounds: [{h_min}, {h_max}])\n")

    stop_heartbeat = start_heartbeat_thread(api, repo_id, worker_id, interval=heartbeat_interval)
    step_fn = build_loss_and_step_fn(forward_fn, inner_optimizer)
    current_round = 0

    try:
        while True:
            if current_round > 0:
                raw_h = int(throughput_ema * target_sync_sec)
                current_h = int(np.clip(raw_h, h_min, h_max))
                print(
                    f"\n[{worker_id}] Round {current_round} Target H: {current_h} steps "
                    f"(Throughput EMA: {throughput_ema:.2f} st/s)"
                )

            # Re-fetch global weights on subsequent rounds
            if current_round > 0:
                weights_file = hf_hub_download(
                    repo_id=repo_id,
                    filename="global_weights.safetensors",
                    repo_type="dataset",
                    token=token,
                )
                raw_weights = load_file(weights_file)
                global_weights = {k: jnp.array(v) for k, v in raw_weights.items()}
                local_weights = {k: jnp.array(v) for k, v in global_weights.items()}

            opt_state = inner_optimizer.init(local_weights)

            # Stream Batches & Measure Compute Time
            stream = DynamicChunkStream(
                api=api,
                repo_id=repo_id,
                worker_id=worker_id,
                batch_size=batch_size,
                seed=base_seed + current_round,
                token=token,
            )
            steps_completed = 0
            start_compute_time = time.perf_counter()

            for batch in stream.get_batch_iterator(target_steps=current_h):
                local_weights, opt_state, loss = step_fn(local_weights, opt_state, batch)
                steps_completed += 1
                if steps_completed % 50 == 0 or steps_completed == current_h:
                    print(f"[{worker_id}] Round {current_round} | Step {steps_completed}/{current_h} | Loss: {loss:.4f}")

            loss.block_until_ready()
            compute_elapsed = time.perf_counter() - start_compute_time

            if steps_completed == 0:
                print(f"[{worker_id}] No data chunks available. Training finished.")
                break

            # Update EMA Throughput passively
            round_throughput = steps_completed / max(compute_elapsed, 1e-4)
            throughput_ema = (ema_alpha * round_throughput) + ((1.0 - ema_alpha) * throughput_ema)
            print(
                f"[{worker_id}] Round {current_round} finished in {compute_elapsed:.1f}s "
                f"({round_throughput:.2f} st/s). Updated EMA: {throughput_ema:.2f} st/s"
            )

            # Mark Processed Chunks
            mark_chunks_completed(api, repo_id, stream.claimed_chunks, worker_id, token=token)

            # Upload Pseudo-Gradient: Delta W = W_global - W_local
            pseudo_grad = jax.tree.map(
                lambda g, l: np.asarray(g - l, dtype=np.float32),
                global_weights,
                local_weights,
            )
            update_filename = f"{worker_id}__round_{current_round}__steps_{steps_completed}.safetensors"
            with tempfile.TemporaryDirectory() as tmpdir:
                local_update_path = os.path.join(tmpdir, update_filename)
                save_file(pseudo_grad, local_update_path)
                api.upload_file(
                    path_or_fileobj=local_update_path,
                    path_in_repo=f"worker_updates/{update_filename}",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
            print(f"[{worker_id}] Uploaded update {update_filename}.")

            # Barrier Wait for Coordinator
            current_round = wait_for_global_sync(
                api=api,
                repo_id=repo_id,
                current_round=current_round,
                token=token,
            )
    finally:
        stop_heartbeat.set()


def _extract_callable_source(fn: Callable) -> str:
    """Extracts the source code of a callable."""
    try:
        source = inspect.getsource(fn)
        return textwrap.dedent(source)
    except (OSError, TypeError) as e:
        raise ValueError(
            f"Unable to inspect source code for '{fn.__name__}'. "
            "Ensure the function is defined in a standard Python file or notebook cell."
        ) from e


def push_fit(
    repo_id: str,
    forward_fn: Callable,
    weights: dict,
    training_cfg: Optional[Dict[str, Any]] = None,
    model_cfg: Optional[Any] = None,
    weight_map: Optional[ModelWeightMap] = None,
    dim_sizes: Optional[Dict[str, int]] = None,
    token: Optional[str] = None,
    private: bool = True,
) -> None:
    """Uploads model definition, hyperparameter manifest, and initial weights to HF Hub."""
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    training_cfg = training_cfg or {}

    fn_name = getattr(forward_fn, "__name__", "custom_forward")
    fn_source = _extract_callable_source(forward_fn)
    entrypoint_code = (
        "import jax\n"
        "import jax.numpy as jnp\n"
        "import optax\n\n"
        f"{fn_source}\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        config_manifest = {
            "forward_fn_name": fn_name,
            "code_entrypoint": "model_entrypoint.py",
            "model_config_cls": type(model_cfg).__name__ if model_cfg else None,
            "model_config": dataclass_to_dict(model_cfg) if model_cfg else {},
            "training_config": {
                "target_sync_interval_sec": training_cfg.get("target_sync_interval_sec", 300.0),
                "min_inner_steps": training_cfg.get("min_inner_steps", 50),
                "max_inner_steps": training_cfg.get("max_inner_steps", 1500),
                "batch_size": training_cfg.get("batch_size", 4),
                "seq_len": training_cfg.get("seq_len", 2048),
                "inner_lr": training_cfg.get("inner_lr", 1e-4),
                "optimizer_type": training_cfg.get("optimizer_type", "adamw"),
                "seed": training_cfg.get("seed", 42),
                "heartbeat_interval": training_cfg.get("heartbeat_interval", 120),
                "ema_alpha": training_cfg.get("ema_alpha", 0.3),
                "extra_args": training_cfg.get("extra_args", {}),
            },
        }
        cfg_path = os.path.join(tmpdir, "train_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config_manifest, f, indent=2)
        api.upload_file(
            path_or_fileobj=cfg_path,
            path_in_repo="train_config.json",
            repo_id=repo_id,
            repo_type="dataset",
        )

        entrypoint_path = os.path.join(tmpdir, "model_entrypoint.py")
        with open(entrypoint_path, "w", encoding="utf-8") as f:
            f.write(entrypoint_code)
        api.upload_file(
            path_or_fileobj=entrypoint_path,
            path_in_repo="code/model_entrypoint.py",
            repo_id=repo_id,
            repo_type="dataset",
        )

        sync_state = {"round": 0, "last_updated": 0.0}
        sync_path = os.path.join(tmpdir, "sync_state.json")
        with open(sync_path, "w", encoding="utf-8") as f:
            json.dump(sync_state, f, indent=2)
        api.upload_file(
            path_or_fileobj=sync_path,
            path_in_repo="sync_state.json",
            repo_id=repo_id,
            repo_type="dataset",
        )

        if weight_map is not None:
            save_checkpoint(
                weights=weights,
                save_directory=tmpdir,
                weight_map=weight_map,
                dim_sizes=dim_sizes,
            )
            ckpt_path = os.path.join(tmpdir, "model.safetensors")
        else:
            flat_weights = {
                k: np.asarray(jax.device_get(v))
                for k, v in weights.items()
            }
            ckpt_path = os.path.join(tmpdir, "global_weights.safetensors")
            save_file(flat_weights, ckpt_path)
        api.upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo="global_weights.safetensors",
            repo_id=repo_id,
            repo_type="dataset",
        )
    print(f"Successfully published DiLoCo experiment to {repo_id}")