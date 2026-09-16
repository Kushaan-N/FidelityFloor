"""The four degradation axes (spec §1.2).

Action-space axes (miscalibration, dyn_noise, horizon) transform the commanded
action sequence BEFORE execution in the imagination rollout. obs_corruption is a
post-hoc CPU pass over already-rendered pristine frames — no re-simulation.

All functions are pure w.r.t. their RNG argument; seeding lives with the caller.
"""

from __future__ import annotations

import io

import numpy as np

from .config import Condition


def rot2d(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s], [s, c]])


def degrade_actions(actions: np.ndarray, cond: Condition, rng: np.random.Generator) -> np.ndarray:
    """Apply an action-space degradation. actions: (T, 2) planar velocities."""
    a = np.asarray(actions, dtype=np.float64)
    if cond.axis == "identity" or cond.axis == "obs_corruption":
        return a.copy()  # bit-identical path for G2
    p = cond.pdict
    if cond.axis == "miscalibration":
        return (p["scale"] * (rot2d(p["rot_deg"]) @ a.T)).T
    if cond.axis == "dyn_noise":
        return a + rng.normal(0.0, p["sigma"], size=a.shape)
    if cond.axis == "horizon":
        t = max(1, int(round(p["fraction"] * len(a))))
        return a[:t].copy()
    raise ValueError(f"unknown axis: {cond.axis}")


# ------------------------------------------------------------- obs corruption

def corrupt_frame(png_bytes: bytes, cond: Condition, frame_index: int, seed: int) -> bytes:
    """Blur + sensor noise + JPEG compression + temporal flicker, graded.

    Deterministic given (seed, frame_index) so re-runs are bit-identical.
    Returns PNG bytes (downstream pipeline stays format-uniform).
    """
    from PIL import Image, ImageFilter

    if cond.axis != "obs_corruption":
        raise ValueError("corrupt_frame only applies to obs_corruption conditions")
    p = cond.pdict
    rng = np.random.default_rng([seed, frame_index, int(cond.severity)])

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    # 1) blur (generative smearing)
    img = img.filter(ImageFilter.GaussianBlur(radius=p["blur"]))
    arr = np.asarray(img, dtype=np.float64) / 255.0
    # 2) temporal flicker (per-frame brightness gain)
    gain = 1.0 + rng.normal(0.0, p["flicker"])
    arr = arr * gain
    # 3) pixel noise
    arr = arr + rng.normal(0.0, p["noise"], size=arr.shape)
    arr = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    # 4) JPEG round-trip (codec artifacts)
    jb = io.BytesIO()
    img.save(jb, format="JPEG", quality=int(p["jpeg_q"]))
    img = Image.open(jb).convert("RGB")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()
