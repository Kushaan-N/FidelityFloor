"""Idempotent, atomic result persistence (spec §0.5/§0.6).

Every shard writes to a deterministic path derived from config hash + IDs,
skips work whose output already validates, and lands files via tmp+rename.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

REQUIRED_ROLLOUT_KEYS = ("cube_pos", "pusher_pos", "actions", "meta_json")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2, sort_keys=True, default=str).encode())


def atomic_save_npz(path: Path, **arrays) -> None:
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    atomic_write_bytes(path, buf.getvalue())


def npz_valid(path: Path, required_keys=REQUIRED_ROLLOUT_KEYS) -> bool:
    """True iff the file exists, unzips, and carries the required keys."""
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            return all(k in z.files for k in required_keys)
    except Exception:
        return False


def save_rollout(dirpath: Path, result: dict) -> Path:
    """Persist one rollout: raw.npz (+ frames/NNN.png if frames captured)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    meta = dict(result.get("meta", {}))
    frames = result.get("frames") or []
    frame_steps = result.get("frame_steps") or []
    meta["n_frames"] = len(frames)
    meta["frame_steps"] = list(map(int, frame_steps))
    arrays = {
        "cube_pos": np.asarray(result["cube_pos"], dtype=np.float64),
        "cube_quat": np.asarray(result["cube_quat"], dtype=np.float64),
        "pusher_pos": np.asarray(result["pusher_pos"], dtype=np.float64),
        "actions": np.asarray(result["actions"], dtype=np.float64),
        "meta_json": np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8),
    }
    atomic_save_npz(dirpath / "raw.npz", **arrays)
    for step, png_bytes in zip(frame_steps, frames):
        atomic_write_bytes(dirpath / "frames" / f"{int(step):04d}.png", png_bytes)
    return dirpath / "raw.npz"


def load_rollout(dirpath: Path) -> dict:
    with np.load(dirpath / "raw.npz", allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "meta_json"}
        out["meta"] = json.loads(bytes(z["meta_json"]).decode())
    fdir = dirpath / "frames"
    out["frame_paths"] = sorted(fdir.glob("*.png")) if fdir.exists() else []
    return out


def rollout_done(dirpath: Path) -> bool:
    return npz_valid(dirpath / "raw.npz")
