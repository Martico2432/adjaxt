import os
import re
import json
import time
import tempfile
import numpy as np
from typing import Optional, Dict
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from safetensors.numpy import save_file, load_file

def run_coordinator(
    repo_id: str,
    min_workers_per_round: int = 2,
    outer_lr: float = 0.7,
    outer_momentum: float = 0.9,
    poll_interval: int = 20,
    token: Optional[str] = None,
):
    """Aggregates worker pseudo-gradients weighted by each worker's inner step count H."""
    api = HfApi(token=token)
    
    # Initialize outer momentum buffer
    velocity: Dict[str, np.ndarray] = {}

    # Parse round state
    sync_file = hf_hub_download(repo_id=repo_id, filename="sync_state.json", repo_type="dataset", token=token)
    with open(sync_file, "r") as f:
        current_round = json.load(f).get("round", 0)

    print(f"Coordinator running for round {current_round}...")

    while True:
        files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
        pattern = re.compile(rf"^worker_updates/(.+)__round_{current_round}__steps_(\d+)\.safetensors$")

        round_updates = []
        for file_path in files:
            match = pattern.match(file_path)
            if match:
                worker_id = match.group(1)
                h_k = int(match.group(2))
                round_updates.append((file_path, worker_id, h_k))

        if len(round_updates) >= min_workers_per_round:
            print(f"Aggregating {len(round_updates)} updates for round {current_round}...")

            total_h = sum(h for _, _, h in round_updates)
            weighted_pseudo_grads = {}

            # Weighted aggregation
            for file_path, _, h_k in round_updates:
                local_path = hf_hub_download(repo_id=repo_id, filename=file_path, repo_type="dataset", token=token)
                grad_dict = load_file(local_path)
                weight_factor = h_k / total_h

                for k, v in grad_dict.items():
                    if k not in weighted_pseudo_grads:
                        weighted_pseudo_grads[k] = v * weight_factor
                    else:
                        weighted_pseudo_grads[k] += v * weight_factor

            # Fetch current global weights
            weights_file = hf_hub_download(repo_id=repo_id, filename="global_weights.safetensors", repo_type="dataset", token=token)
            global_weights = load_file(weights_file)

            # Apply outer Nesterov momentum update
            new_global_weights = {}
            for k, w in global_weights.items():
                g = weighted_pseudo_grads[k]
                v_old = velocity.get(k, np.zeros_like(w))
                v_new = outer_momentum * v_old + g
                velocity[k] = v_new
                
                # Nesterov update step: W_new = W_old - lr * (g + momentum * v_new)
                new_global_weights[k] = w - outer_lr * (g + outer_momentum * v_new)

            # Push updated global weights and advance sync round
            with tempfile.TemporaryDirectory() as tmpdir:
                weights_path = os.path.join(tmpdir, "global_weights.safetensors")
                save_file(new_global_weights, weights_path)
                api.upload_file(
                    path_or_fileobj=weights_path,
                    path_in_repo="global_weights.safetensors",
                    repo_id=repo_id,
                    repo_type="dataset",
                )

                sync_state = {"round": current_round + 1, "last_updated": time.time()}
                sync_path = os.path.join(tmpdir, "sync_state.json")
                with open(sync_path, "w") as f:
                    json.dump(sync_state, f, indent=2)

                api.upload_file(
                    path_or_fileobj=sync_path,
                    path_in_repo="sync_state.json",
                    repo_id=repo_id,
                    repo_type="dataset",
                )

            current_round += 1
            print(f"Round {current_round - 1} complete. Advanced to round {current_round}.")

        time.sleep(poll_interval)