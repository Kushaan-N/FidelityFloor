"""Utilities and ranking metrics for Tier 2 (spec §1.3).

Two scoring modes:
- state-scored: utility from the (imagined) final state — blind to obs_corruption
  by construction; that asymmetry is part of the result.
- VLM-scored: an API VLM scores the rendered final frame of each imagined rollout
  against the instruction (see vlm.py); ranking metrics are computed identically.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def state_utility(cube_pos_traj: np.ndarray, goal_xy: np.ndarray) -> float:
    """Normalized task progress: (d0 - dT) / d0, clipped to [-1, 1]."""
    p = np.asarray(cube_pos_traj)[:, :2]
    g = np.asarray(goal_xy)
    d0 = float(np.linalg.norm(p[0] - g))
    dT = float(np.linalg.norm(p[-1] - g))
    return float(np.clip((d0 - dT) / max(d0, 1e-9), -1.0, 1.0))


def spearman(u_imagined: np.ndarray, u_gt: np.ndarray) -> float:
    rho = sps.spearmanr(u_imagined, u_gt).statistic
    return float(rho) if np.isfinite(rho) else 0.0


def top1_regret(u_imagined: np.ndarray, u_gt: np.ndarray) -> float:
    """GT utility forgone by picking imagination's top candidate instead of GT's."""
    u_gt = np.asarray(u_gt, dtype=np.float64)
    return float(u_gt.max() - u_gt[int(np.argmax(u_imagined))])


def normalized_regret(u_imagined: np.ndarray, u_gt: np.ndarray) -> float:
    """WorldModelGym-style normalized regret in [0, 1]:
    (u* - u_picked) / (u* - u_min). 0 = picked the best, 1 = picked the worst."""
    u_gt = np.asarray(u_gt, dtype=np.float64)
    span = float(u_gt.max() - u_gt.min())
    if span < 1e-12:
        return 0.0
    return top1_regret(u_imagined, u_gt) / span


def rank_metrics(u_imagined, u_gt) -> dict:
    return {
        "spearman": spearman(u_imagined, u_gt),
        "top1_regret": top1_regret(u_imagined, u_gt),
        "normalized_regret": normalized_regret(u_imagined, u_gt),
    }
