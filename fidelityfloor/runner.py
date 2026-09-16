"""Launcher-agnostic shard workers (spec §2.2). Modal and Unity both call these.

Sharding: Tier 2 by initial state — one shard = GT (K candidates) + all assigned
sim conditions for that state, so the paired structure lives in one shard.
obs_corruption + VLM scoring are separate CPU passes over stored frames.
Every unit of work is skip-if-valid, so shards are preemptible and re-runnable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .candidates import generate_candidates, sample_initial_state, spread_ok
from .config import (
    Condition,
    condition_by_id,
    enumerate_conditions,
    gt_dir,
    identity_condition,
    out_root,
    run_dir,
)
from .io_utils import atomic_write_bytes, atomic_write_json, load_rollout, rollout_done, save_rollout
from .oracle import measure_determinism_floor, run_rollout


def _sim_conditions(cfg: dict, condition_ids: list[str] | None) -> list[Condition]:
    conds = enumerate_conditions(cfg) if condition_ids is None else [
        condition_by_id(cfg, cid) for cid in condition_ids
    ]
    return [c for c in conds if c.cid == "identity" or c.needs_sim]


def run_state_shard(cfg: dict, render_cfg: dict, state_id: int,
                    condition_ids: list[str] | None = None, env=None) -> dict:
    """GT + every sim condition for one initial state. Idempotent per rollout."""
    from .envs import PushEnv

    env = env or PushEnv(cfg, render_cfg)
    state = sample_initial_state(cfg, state_id)
    cands = generate_candidates(state, cfg)
    seed = cfg["seed"]
    report = {"state_id": state_id, "done": [], "skipped": [], "wall_s": {}}

    jobs = [("gt", identity_condition(), gt_dir(cfg, state_id))]
    for c in _sim_conditions(cfg, condition_ids):
        jobs.append(("run", c, run_dir(cfg, c, state_id)))

    for tag, cond, base in jobs:
        t0 = time.time()
        n_done = 0
        for ci in range(len(cands)):
            d = base / f"cand_{ci:02d}"
            if rollout_done(d):
                report["skipped"].append(f"{tag}:{cond.cid}:c{ci}")
                continue
            # Per-(state, candidate) seed so dyn_noise draws are independent but
            # reproducible; identical for GT vs zero-degradation replay.
            r = run_rollout(env, state, cands[ci], cond,
                            seed=int(np.random.default_rng(
                                [seed, state_id, ci]).integers(2**31)),
                            capture_frames=True)
            save_rollout(d, r)
            n_done += 1
        report["done"].append(f"{tag}:{cond.cid}:+{n_done}")
        report["wall_s"][f"{tag}:{cond.cid}"] = round(time.time() - t0, 2)

    # G4 spread check on GT utilities (recorded, gated at analysis time)
    from .scoring import state_utility

    gts = [load_rollout(gt_dir(cfg, state_id) / f"cand_{ci:02d}")
           for ci in range(len(cands))]
    u = [state_utility(r["cube_pos"], np.array(state["goal_xy"])) for r in gts]
    ok, info = spread_ok(np.array(u), cfg)
    report["spread_ok"], report["spread_info"], report["gt_utilities"] = ok, info, u
    atomic_write_json(gt_dir(cfg, state_id) / "shard_report.json", report)
    return report


# ------------------------------------------------------- obs corruption (CPU)

def corrupt_frames_pass(cfg: dict, state_ids: list[int]) -> dict:
    """Generate obs_corruption 'rollouts' by corrupting identity frames. CPU-only,
    no simulation. Copies identity raw.npz (trajectories are identical by
    construction) and writes corrupted frames."""
    from .degrade import corrupt_frame

    k = cfg["candidates"]["k"]
    conds = [c for c in enumerate_conditions(cfg) if c.axis == "obs_corruption"]
    n_new = 0
    for sid in state_ids:
        for cond in conds:
            for ci in range(k):
                src = run_dir(cfg, identity_condition(), sid) / f"cand_{ci:02d}"
                dst = run_dir(cfg, cond, sid) / f"cand_{ci:02d}"
                if not rollout_done(src):
                    continue
                if rollout_done(dst) and (dst / "frames").exists():
                    continue
                dst.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(dst / "raw.npz", (src / "raw.npz").read_bytes())
                for fp in sorted((src / "frames").glob("*.png")):
                    out = corrupt_frame(fp.read_bytes(), cond,
                                        frame_index=int(fp.stem), seed=cfg["seed"] + sid)
                    atomic_write_bytes(dst / "frames" / fp.name, out)
                n_new += 1
    return {"corrupted_rollouts_written": n_new, "state_ids": state_ids}


# ------------------------------------------------------------ VLM scoring (CPU)

def vlm_score_pass(cfg: dict, state_ids: list[int],
                   condition_ids: list[str] | None = None) -> dict:
    """Score the final frame of every (condition, state, candidate) rollout with
    the VLM. Cached by content hash — re-runs are free (G5.5)."""
    from .vlm import VLMClient

    k = cfg["candidates"]["k"]
    client = VLMClient(cfg)
    conds = enumerate_conditions(cfg) if condition_ids is None else [
        condition_by_id(cfg, cid) for cid in condition_ids
    ]
    repeats_n = cfg["vlm"]["score_repeats_subset"]
    scores: dict[str, dict] = {}
    missing = []
    for cond in conds:
        for sid in state_ids:
            svec, svec_rep = [], []
            for ci in range(k):
                d = run_dir(cfg, cond, sid) / f"cand_{ci:02d}"
                frames = sorted((d / "frames").glob("*.png"))
                if not rollout_done(d) or not frames:
                    missing.append(f"{cond.cid}:{sid}:c{ci}")
                    svec.append(np.nan)
                    continue
                png = frames[-1].read_bytes()  # final frame
                svec.append(float(client.score_outcome(png)["score"]))
                if sid < repeats_n:
                    svec_rep.append(float(client.score_outcome(png, repeat=1)["score"]))
            entry = {"scores": svec}
            if svec_rep:
                entry["scores_repeat"] = svec_rep
            scores[f"{cond.cid}|{sid}"] = entry
    out = {"scores": scores, "missing": missing, "billed_calls": client.n_billed_calls,
           "model": client.model}
    atomic_write_json(out_root(cfg) / "tables" / "vlm_scores.json", out)
    return out


def load_vlm_scores(cfg: dict) -> dict | None:
    p = out_root(cfg) / "tables" / "vlm_scores.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text())["scores"]
    out = {}
    for key, entry in raw.items():
        cid, sid = key.rsplit("|", 1)
        out[(cid, int(sid))] = np.array(entry["scores"], dtype=np.float64)
    return out


# ------------------------------------------------------------------- smoke test

def smoke_test(cfg: dict, render_cfg: dict) -> dict:
    """P0/G0 payload on ONE worker: scene builds, EGL/Vulkan renders, one rollout
    logs, one degradation applies, determinism mini-floor. VLM round-trip is a
    separate CPU function (needs the API secret, not the GPU)."""
    from .envs import PushEnv

    t0 = time.time()
    env = PushEnv(cfg, render_cfg)
    t_boot = time.time() - t0

    state = sample_initial_state(cfg, 0)
    cands = generate_candidates(state, cfg)

    t0 = time.time()
    gt = run_rollout(env, state, cands[0], identity_condition(), seed=1, capture_frames=True)
    t_render_rollout = time.time() - t0

    t0 = time.time()
    _ = run_rollout(env, state, cands[0], identity_condition(), seed=1, capture_frames=False)
    t_physics_rollout = time.time() - t0

    mis = condition_by_id(cfg, "miscalibration-2")
    deg = run_rollout(env, state, cands[0], mis, seed=1, capture_frames=False)

    from .oracle import state_distance

    floor = measure_determinism_floor(env, cfg, n_states=3,
                                      actions_fn=lambda s, c: generate_candidates(s, c)[0])
    d_deg = state_distance(gt, deg)

    smoke_dir = out_root(cfg) / "smoke"
    save_rollout(smoke_dir / "gt_rollout", gt)
    report = {
        "sim_app_boot_s": round(t_boot, 1),
        "rollout_with_render_s": round(t_render_rollout, 2),
        "rollout_physics_only_s": round(t_physics_rollout, 2),
        "frame_bytes": len(gt["frames"][0]) if gt["frames"] else 0,
        "n_frames": len(gt["frames"]),
        "determinism_floor_3states": {k: v for k, v in floor.items() if k != "per_state"},
        "miscalibration2_induced_error": d_deg,
        "gt_final_cube": np.asarray(gt["cube_pos"])[-1].tolist(),
        "state": state,
    }
    atomic_write_json(smoke_dir / "smoke_report.json", report)
    return report


def determinism_pass(cfg: dict, render_cfg: dict) -> dict:
    """Full §3.1 floor over cfg['determinism']['n_states'] states."""
    from .envs import PushEnv

    env = PushEnv(cfg, render_cfg)
    floor = measure_determinism_floor(
        env, cfg, n_states=cfg["determinism"]["n_states"],
        actions_fn=lambda s, c: generate_candidates(s, c)[0],
    )
    atomic_write_json(out_root(cfg) / "tables" / "determinism_floor.json", floor)
    return floor
