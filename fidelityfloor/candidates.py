"""Tier 2 initial states and candidate action-chunk generation (spec §1.3).

Everything here is pure numpy and fully seeded. A candidate is an open-loop
(T, 2) sequence of planar pusher velocity commands — the SAME sequence is
executed in the GT world and in every degraded imagination.
"""

from __future__ import annotations

import numpy as np

from .degrade import rot2d


def sample_initial_state(cfg: dict, state_id: int) -> dict:
    """Deterministic initial state: cube xy, goal xy, pusher xy."""
    ws = cfg["workspace"]
    rng = np.random.default_rng([cfg["seed"], 7001, state_id])
    r = ws["cube_xy_range"]
    cube = rng.uniform(-r, r, size=2)
    ang_g = rng.uniform(0, 2 * np.pi)
    goal = cube + rng.uniform(ws["goal_dist_min"], ws["goal_dist_max"]) * np.array(
        [np.cos(ang_g), np.sin(ang_g)]
    )
    # pusher roughly opposite the goal, +/- jitter
    jit = np.deg2rad(ws["pusher_angle_jitter_deg"])
    ang_p = ang_g + np.pi + rng.uniform(-jit, jit)
    pusher = cube + rng.uniform(ws["pusher_dist_min"], ws["pusher_dist_max"]) * np.array(
        [np.cos(ang_p), np.sin(ang_p)]
    )
    return {
        "state_id": state_id,
        "cube_xy": cube.tolist(),
        "goal_xy": goal.tolist(),
        "pusher_xy": pusher.tolist(),
    }


def _expert_actions(state: dict, cfg: dict, push_rot_deg: float = 0.0, speed_scale: float = 1.0
                    ) -> np.ndarray:
    """Two-phase open-loop plan from the initial state only:
    phase 1 — drive to a contact point behind the cube; phase 2 — push toward the
    goal for exactly the planned distance, then stop (no overshoot).
    push_rot_deg rotates the push direction (and the contact point with it).
    speed_scale scales the executed velocities while keeping the nominal-plan
    durations, so slow/fast variants under/overshoot — a graded degradation."""
    ro = cfg["rollout"]
    ca = cfg["candidates"]
    ws = cfg["workspace"]
    T, dt = ro["horizon_steps"], ro["control_dt"]
    cube = np.array(state["cube_xy"])
    goal = np.array(state["goal_xy"])
    pusher = np.array(state["pusher_xy"])

    push_dir = goal - cube
    d_goal = float(np.linalg.norm(push_dir))
    push_dir = push_dir / (d_goal + 1e-9)
    push_dir = rot2d(push_rot_deg) @ push_dir
    standoff = ws["cube_half"] + ws["pusher_radius"] + 0.01
    contact = cube - standoff * push_dir

    v_ap = ca["expert_approach_speed"]
    v_push = ca["expert_push_speed"]
    vmax = ws["max_speed"]

    acts = np.zeros((T, 2))
    pos = pusher.copy()
    t = 0
    # phase 1: straight line to the contact point at nominal approach speed
    while t < T:
        delta = contact - pos
        d = np.linalg.norm(delta)
        if d < 1e-4:
            break
        step_v = delta / max(d, 1e-9) * min(v_ap, d / dt)
        acts[t] = step_v
        pos = pos + step_v * dt
        t += 1
    # phase 2: push exactly the goal distance (small margin), then hold still
    n_push = int(np.ceil((d_goal + 0.01) / (v_push * dt)))
    acts[t: t + n_push] = v_push * push_dir
    return np.clip(acts * speed_scale, -vmax, vmax)


def _random_actions(state: dict, cfg: dict, idx: int) -> np.ndarray:
    """Smooth random velocity trajectory (OU process), seeded per (state, idx)."""
    ro, ws = cfg["rollout"], cfg["workspace"]
    T = ro["horizon_steps"]
    rng = np.random.default_rng([cfg["seed"], 7333, state["state_id"], idx])
    v = np.zeros(2)
    acts = np.zeros((T, 2))
    for t in range(T):
        v = 0.85 * v + rng.normal(0.0, 0.06, size=2)
        acts[t] = v
    scale = 0.15 / (np.abs(acts).max() + 1e-9)
    return np.clip(acts * scale * rng.uniform(0.8, 1.6), -ws["max_speed"], ws["max_speed"])


def generate_candidates(state: dict, cfg: dict) -> np.ndarray:
    """(K, T, 2) candidate set: expert + graded perturbations + randoms."""
    ca = cfg["candidates"]
    cands = [_expert_actions(state, cfg)]
    for deg in ca["perturb_rot_deg"]:
        cands.append(_expert_actions(state, cfg, push_rot_deg=deg))
    for s in ca["perturb_speed_scale"]:
        cands.append(_expert_actions(state, cfg, speed_scale=s))
    for i in range(ca["n_random"]):
        cands.append(_random_actions(state, cfg, i))
    out = np.stack(cands)
    assert out.shape[0] == ca["k"], f"K mismatch: {out.shape[0]} != {ca['k']}"
    return out


def spread_ok(gt_utilities: np.ndarray, cfg: dict) -> tuple[bool, dict]:
    """G4 candidate-spread check on one state's GT utilities (K,)."""
    u = np.asarray(gt_utilities, dtype=np.float64)
    expert = float(u[0])
    iqr = float(np.percentile(u, 75) - np.percentile(u, 25))
    thresh = cfg["candidates"]["spread_min_iqr_frac"] * max(expert, 1e-9)
    return bool(iqr > thresh and expert > 0), {"iqr": iqr, "expert_utility": expert,
                                               "threshold": thresh}
