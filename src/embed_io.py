"""Compact, GitHub-friendly persistence for embedding matrices.

Embeddings are stored as float16 and split into shards kept under GitHub's
100 MB per-file push limit, so the precomputed indexes can live in the repo
itself (no Git LFS). Cosine ranking is insensitive to the float16 round-trip.
"""
import os
import glob
import numpy as np

# Keep each shard comfortably under GitHub's 100 MB per-file cap.
SHARD_BYTES = 90 * 1024 * 1024


def _shard_glob(path: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}.part*{ext}"


def save_embeddings(arr: np.ndarray, path: str) -> list:
    """Save `arr` as float16, sharded under SHARD_BYTES. Returns written paths."""
    arr = np.ascontiguousarray(arr, dtype=np.float16)
    base, ext = os.path.splitext(path)

    # Clear any stale single-file or sharded output from a previous run.
    for f in glob.glob(_shard_glob(path)):
        os.remove(f)
    if os.path.exists(path):
        os.remove(path)

    rows = arr.shape[0]
    bytes_per_row = int(np.prod(arr.shape[1:])) * arr.itemsize if rows else 1
    rows_per_shard = max(1, SHARD_BYTES // max(1, bytes_per_row))

    if rows <= rows_per_shard:
        np.save(path, arr)
        return [path]

    files = []
    for idx, start in enumerate(range(0, rows, rows_per_shard)):
        fp = f"{base}.part{idx:03d}{ext}"
        np.save(fp, arr[start:start + rows_per_shard])
        files.append(fp)
    return files


def load_embeddings(path: str, dtype=np.float32) -> np.ndarray:
    """Load embeddings written by `save_embeddings`, single-file or sharded."""
    shards = sorted(glob.glob(_shard_glob(path)))
    if shards:
        arr = np.concatenate([np.load(s) for s in shards], axis=0)
    else:
        arr = np.load(path)
    return arr.astype(dtype, copy=False)


def embeddings_exist(path: str) -> bool:
    return os.path.exists(path) or bool(glob.glob(_shard_glob(path)))
