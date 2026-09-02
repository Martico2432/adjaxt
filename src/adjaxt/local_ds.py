import os
import glob
import numpy as np
from huggingface_hub import snapshot_download

def setup_local_dataset(repo_id: str, local_dir: str = "/tmp/dataset"):
    """Downloads shards directly to fast instance NVMe."""
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=["*.npy", "*.bin", "*.parquet"],
        max_workers=8
    )
    return sorted(glob.glob(f"{local_dir}/*.bin"))

class ShardedTokenLoader:
    def __init__(self, file_paths, batch_size: int, seq_len: int):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.current_file_idx = 0
        self._load_shard(0)

    def _load_shard(self, idx):
        # Memory-map the token array (shape: [-1] or [N, L])
        self.data = np.load(self.file_paths[idx], mmap_mode="r").reshape(-1)
        self.num_tokens = len(self.data)
        self.cursor = 0

    def get_batch(self):
        needed = self.batch_size * self.seq_len
        if self.cursor + needed > self.num_tokens:
            self.current_file_idx = (self.current_file_idx + 1) % len(self.file_paths)
            self._load_shard(self.current_file_idx)

        chunk = self.data[self.cursor : self.cursor + needed]
        self.cursor += needed
        return chunk.reshape(self.batch_size, self.seq_len)

class BinShardedLoader:
    def __init__(
        self, 
        data_dir: str, 
        batch_size: int, 
        seq_len: int, 
        dtype: np.dtype = np.uint16
    ):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if not self.files:
            raise FileNotFoundError(f"No .bin files found in {data_dir}")
            
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokens_per_batch = batch_size * seq_len
        self.dtype = dtype
        
        self.file_idx = 0
        self._load_shard(self.file_idx)

    def _load_shard(self, idx: int):
        self.file_idx = idx % len(self.files)
        filepath = self.files[self.file_idx]
        
        # Zero-copy memory map
        self.current_mmap = np.memmap(filepath, dtype=self.dtype, mode="r")
        self.shard_len = len(self.current_mmap)
        self.cursor = 0

    def get_batch(self) -> np.ndarray:
        # Check if current shard has enough remaining tokens
        if self.cursor + self.tokens_per_batch > self.shard_len:
            self._load_shard(self.file_idx + 1)
            
        chunk = self.current_mmap[self.cursor : self.cursor + self.tokens_per_batch]
        self.cursor += self.tokens_per_batch
        
        # JAX requires int32 for array index lookups (token_emb[ids])
        return chunk.reshape(self.batch_size, self.seq_len).astype(np.int32)

