"""Ground truth vs degraded imagination rollouts, and the determinism floor (spec §3).

Both sides live in the same simulator: GT executes the commanded actions
pristinely; "imagination" re-executes from the same initial state with a
degradation applied to the actions (obs_corruption instead corrupts stored GT
frames post-hoc and never touches the sim).
"""

from __future__ import annotations

import time

import numpy as np

from .config import Condition, identity_condition
from .degrade import degrade_actions


def run_rollout(env, state: dict, actions: np.ndarray, cond: Condition, seed: int,
                capture_frames: bool = True) -> dict:
    """Execute `actions` from `state` under degradation `cond`. Returns trajectory
    at control-step resolution plus optionally rendered frames (PNG bytes)."""
    ro = env.cfg["rollout"]
    rng = np.random.default_rng([seed, 101])
    acts = degrade_actions(np.asarray(actions), cond, rng)

    t0 = time.time()
    env.reset_to(state)
    cube_pos, cube_quat, pusher_pos = [], [], []
    frames, frame_steps = [], []

    def record(step: int):
        s = env.get_state()
        cube_pos.append(s["cube_pos"])
        cube_quat.append(s["cube_quat"])
        pusher_pos.append(s["pusher_pos"])
        if capture_frames and (step % ro["frame_every"] == 0 or step == len(acts)):
            frames.append(env.render_frame())
            frame_steps.append(step)

    record(0)
    for t, a in enumerate(acts):
        env.step_control(a)
        record(t + 1)

    return {
        "cube_pos": np.stack(cube_pos),
        "cube_quat": np.stack(cube_quat),
        "pusher_pos": np.stack(pusher_pos),
        "actions": acts,
        "frames": frames,
        "frame_steps": frame_steps,
        "meta": {
            "condition": cond.cid,
            "seed": seed,
            "state": state,
            "n_steps_executed": int(len(acts)),
            "wall_seconds": round(time.time() - t0, 3),
        },
    }


def state_distance(traj_a: dict, traj_b: dict) -> dict:
    """Mean + final state-space distance between two rollouts of the same state.

    Matched timesteps up to the shorter horizon (horizon truncation compares the
    imagined prefix against the same GT prefix; final-state error uses each
    trajectory's own last step — the imagination simply stops early, which is
    the point of that axis)."""
    n = min(len(traj_a["cube_pos"]), len(traj_b["cube_pos"]))
    dc = np.linalg.norm(traj_a["cube_pos"][:n] - traj_b["cube_pos"][:n], axis=1)
    dp = np.linalg.norm(traj_a["pusher_pos"][:n] - traj_b["pusher_pos"][:n], axis=1)
    d_final = float(
        np.linalg.norm(traj_a["cube_pos"][-1] - traj_b["cube_pos"][-1])
        + np.linalg.norm(traj_a["pusher_pos"][-1] - traj_b["pusher_pos"][-1])
    )
    return {
        "mean_matched": float(np.mean(dc + dp)),
        "mean_cube": float(np.mean(dc)),
        "final": d_final,
        "final_cube": float(np.linalg.norm(traj_a["cube_pos"][-1] - traj_b["cube_pos"][-1])),
    }


def measure_determinism_floor(env, cfg: dict, n_states: int, actions_fn) -> dict:
    """§3.1: replay each state's expert actions twice at zero degradation; the
    divergence is the noise floor of the whole study. Reported on every figure."""
    ident = identity_condition()
    per_state = []
    for sid in range(n_states):
        from .candidates import sample_initial_state

        state = sample_initial_state(cfg, sid)
        acts = actions_fn(state, cfg)
        r1 = run_rollout(env, state, acts, ident, seed=cfg["seed"], capture_frames=False)
        r2 = run_rollout(env, state, acts, ident, seed=cfg["seed"], capture_frames=False)
        per_state.append(state_distance(r1, r2))
    floor_mean = float(np.mean([d["mean_matched"] for d in per_state]))
    floor_final = float(np.mean([d["final"] for d in per_state]))
    return {
        "n_states": n_states,
        "floor_mean_matched_m": floor_mean,
        "floor_final_m": floor_final,
        "per_state": per_state,
        "floor_max": float(np.max([d["mean_matched"] for d in per_state])),
    }
