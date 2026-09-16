"""Statistics: induced-error measurement, paired bootstrap, matched-error analysis
(spec §3.2, §7). Everything regenerates from raw arrays on disk — no state.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import Condition, enumerate_conditions, gt_dir, identity_condition, out_root, run_dir
from .io_utils import load_rollout, rollout_done
from .oracle import state_distance
from .scoring import rank_metrics, state_utility


# ------------------------------------------------------------ loading the grid

def load_state_results(cfg: dict, cond: Condition, state_id: int, k: int) -> dict | None:
    """Load all K rollouts of one (condition, state), or None if incomplete."""
    base = run_dir(cfg, cond, state_id)
    rollouts = []
    for ci in range(k):
        d = base / f"cand_{ci:02d}"
        if not rollout_done(d):
            return None
        rollouts.append(load_rollout(d))
    return {"rollouts": rollouts, "dir": base}


def load_gt_results(cfg: dict, state_id: int, k: int) -> dict | None:
    base = gt_dir(cfg, state_id)
    rollouts = []
    for ci in range(k):
        d = base / f"cand_{ci:02d}"
        if not rollout_done(d):
            return None
        rollouts.append(load_rollout(d))
    return {"rollouts": rollouts, "dir": base}


# --------------------------------------------------------------- induced error

def induced_error(cfg: dict, cond: Condition, state_ids: list[int]) -> dict:
    """§3.2: mean state-space distance imagined-vs-GT over paired (state, candidate).
    This is the x-axis of every demand curve."""
    k = cfg["candidates"]["k"]
    mean_matched, final = [], []
    for sid in state_ids:
        gt = load_gt_results(cfg, sid, k)
        im = load_state_results(cfg, cond, sid, k)
        if gt is None or im is None:
            continue
        for g, m in zip(gt["rollouts"], im["rollouts"]):
            d = state_distance(g, m)
            mean_matched.append(d["mean_matched"])
            final.append(d["final"])
    em = float(np.mean(mean_matched)) if mean_matched else np.nan
    ef = float(np.mean(final)) if final else np.nan
    return {
        "condition": cond.cid,
        "n_pairs": len(mean_matched),
        "err_mean_matched": em,
        "err_final": ef,
        # x-axis metric: horizon's imagined trajectory is an identical PREFIX of
        # GT (deterministic sim), so its matched-step error is 0 by construction;
        # its induced error is where the imagination STOPS vs where GT ends.
        "err_xaxis": ef if cond.axis == "horizon" else em,
        "err_std_over_pairs": float(np.std(mean_matched)) if mean_matched else np.nan,
    }


# --------------------------------------------------- per-condition utility table

def condition_table(cfg: dict, state_ids: list[int], vlm_scores: dict | None = None) -> list[dict]:
    """One row per (condition, state): ranking metrics for both scoring modes.

    vlm_scores: optional {(cid, state_id): np.ndarray(K,)} of VLM 0-10 scores.
    obs_corruption's state-scored metrics equal identity's by construction (the
    sim path is untouched) — kept in the table, flagged with state_scored_valid.
    """
    k = cfg["candidates"]["k"]
    rows = []
    for cond in enumerate_conditions(cfg):
        src = identity_condition() if cond.axis == "obs_corruption" else cond
        for sid in state_ids:
            gt = load_gt_results(cfg, sid, k)
            im = load_state_results(cfg, src, sid, k)
            if gt is None or im is None:
                continue
            goal = np.array(json.loads(json.dumps(
                gt["rollouts"][0]["meta"]["state"]))["goal_xy"])
            u_gt = np.array([state_utility(r["cube_pos"], goal) for r in gt["rollouts"]])
            u_im = np.array([state_utility(r["cube_pos"], goal) for r in im["rollouts"]])
            row = {
                "condition": cond.cid, "axis": cond.axis, "severity": cond.severity,
                "state_id": sid,
                "u_gt": u_gt.tolist(), "u_state_scored": u_im.tolist(),
                "state_scored_valid": cond.axis != "obs_corruption",
                **{f"state_{k2}": v for k2, v in rank_metrics(u_im, u_gt).items()},
            }
            if vlm_scores and (cond.cid, sid) in vlm_scores:
                v = np.asarray(vlm_scores[(cond.cid, sid)], dtype=np.float64)
                row.update({f"vlm_{k2}": val for k2, val in rank_metrics(v, u_gt).items()})
                row["u_vlm_scored"] = v.tolist()
            rows.append(row)
    return rows


# -------------------------------------------------------------------- bootstrap

def bootstrap_ci(values_per_state: np.ndarray, cfg: dict, seed_tag: int = 0) -> dict:
    """Percentile bootstrap over states (the pairing unit). values: (n_states,)."""
    v = np.asarray(values_per_state, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    rng = np.random.default_rng([20260916, 42, seed_tag])
    n_boot = cfg["stats"]["bootstrap_resamples"]
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return {"mean": float(v.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n": int(len(v))}


def paired_diff_ci(a_per_state: np.ndarray, b_per_state: np.ndarray, cfg: dict) -> dict:
    """Bootstrap CI on the paired per-state difference a - b."""
    d = np.asarray(a_per_state, dtype=np.float64) - np.asarray(b_per_state, dtype=np.float64)
    out = bootstrap_ci(d, cfg, seed_tag=7)
    out["significant"] = bool(out["lo"] > 0 or out["hi"] < 0)
    return out


# ------------------------------------------------- matched-error comparison (§7)

def matched_error_comparison(cfg: dict, rows: list[dict], errors: dict[str, dict],
                             metric: str = "state_spearman") -> dict:
    """Figure 2: systematic (miscalibration) vs stochastic (dyn_noise) at matched
    measured error, via linear interpolation of each axis's error->utility curve
    onto a common error grid, with per-state paired differences at the nearest
    severity pairs."""
    def axis_curve(axis: str):
        sevs = sorted({r["severity"] for r in rows if r["axis"] == axis})
        errs, utils, per_state = [], [], []
        for s in sevs:
            cid = f"{axis}-{s}"
            rr = [r for r in rows if r["condition"] == cid]
            errs.append(errors[cid]["err_xaxis"])
            utils.append(float(np.mean([r[metric] for r in rr])))
            per_state.append({r["state_id"]: r[metric] for r in rr})
        return np.array(errs), np.array(utils), per_state

    e_m, u_m, ps_m = axis_curve("miscalibration")
    e_n, u_n, ps_n = axis_curve("dyn_noise")
    if len(e_m) == 0 or len(e_n) == 0:
        return {"overlap": False, "note": "miscalibration/dyn_noise not yet run"}
    lo = max(e_m.min(), e_n.min())
    hi = min(e_m.max(), e_n.max())
    if hi <= lo:
        return {"overlap": False,
                "note": "no overlapping induced-error range; adjust severities"}
    grid = np.linspace(lo, hi, 25)
    interp_m = np.interp(grid, e_m, u_m)
    interp_n = np.interp(grid, e_n, u_n)

    # Paired test at the closest-matched severity pair
    pair = min(((i, j) for i in range(len(e_m)) for j in range(len(e_n))),
               key=lambda ij: abs(e_m[ij[0]] - e_n[ij[1]]))
    common_states = sorted(set(ps_m[pair[0]]) & set(ps_n[pair[1]]))
    a = np.array([ps_m[pair[0]][s] for s in common_states])
    b = np.array([ps_n[pair[1]][s] for s in common_states])
    return {
        "overlap": True, "metric": metric,
        "error_grid": grid.tolist(),
        "utility_miscalibration": interp_m.tolist(),
        "utility_dyn_noise": interp_n.tolist(),
        "mean_gap_on_grid": float(np.mean(interp_n - interp_m)),
        "matched_pair": {"miscalibration_sev": pair[0] + 1, "dyn_noise_sev": pair[1] + 1,
                         "err_m": float(e_m[pair[0]]), "err_n": float(e_n[pair[1]])},
        "paired_diff_noise_minus_misc": paired_diff_ci(b, a, cfg),
    }


def summarize(cfg: dict, state_ids: list[int], vlm_scores=None) -> dict:
    """Full analysis pass from raw arrays -> one JSON summary (tables regenerate
    from this; figures.py consumes it)."""
    rows = condition_table(cfg, state_ids, vlm_scores)
    # obs_corruption's induced *state* error is identity's by construction
    # (frames are corrupted; the sim path is untouched) — annotate it as such.
    errors = {}
    for c in enumerate_conditions(cfg):
        src = identity_condition() if c.axis == "obs_corruption" else c
        errors[c.cid] = induced_error(cfg, src, state_ids)
        errors[c.cid]["condition"] = c.cid
        if c.axis == "obs_corruption":
            errors[c.cid]["note"] = "state error = identity (frames corrupted, sim untouched)"
    per_cond = {}
    for c in enumerate_conditions(cfg):
        rr = [r for r in rows if r["condition"] == c.cid]
        entry = {"n_states": len(rr), "induced_error": errors[c.cid]}
        for mode in ("state", "vlm"):
            for m in ("spearman", "top1_regret", "normalized_regret"):
                key = f"{mode}_{m}"
                vals = np.array([r[key] for r in rr if key in r])
                if len(vals):
                    entry[key] = bootstrap_ci(vals, cfg, seed_tag=hash(c.cid + key) % 10000)
        per_cond[c.cid] = entry
    return {
        "rows": rows,
        "induced_errors": errors,
        "per_condition": per_cond,
        "matched_error": matched_error_comparison(cfg, rows, errors),
        "state_ids": state_ids,
    }


def save_summary(cfg: dict, summary: dict, name: str = "summary") -> Path:
    from .io_utils import atomic_write_json

    p = out_root(cfg) / "tables" / f"{name}.json"
    atomic_write_json(p, summary)
    return p
