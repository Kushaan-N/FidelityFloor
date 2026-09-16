"""Automated checks behind the spec's gates (§6). Visual gates (G4) still require
eyes on the contact sheets — these functions cover everything computable.
"""

from __future__ import annotations

import numpy as np

from .config import enumerate_conditions


def g1_perfect_imagination(summary: dict, vlm_present: bool) -> dict:
    """G1: identity condition must recover GT ranking (state Spearman > 0.95,
    top-1 regret ~ 0; VLM-scored Spearman > 0.7)."""
    ident = summary["per_condition"].get("identity", {})
    s = ident.get("state_spearman", {}).get("mean", np.nan)
    r = ident.get("state_top1_regret", {}).get("mean", np.nan)
    out = {
        "state_spearman": s,
        "state_top1_regret": r,
        "state_pass": bool(np.isfinite(s) and s > 0.95 and np.isfinite(r) and r < 0.02),
    }
    if vlm_present:
        v = ident.get("vlm_spearman", {}).get("mean", np.nan)
        out["vlm_spearman"] = v
        out["vlm_pass"] = bool(np.isfinite(v) and v > 0.7)
    out["pass"] = out["state_pass"] and out.get("vlm_pass", True)
    return out


def g2_identity_noop(summary: dict, determinism_floor: float) -> dict:
    """G2: identity's induced error must sit at (within 2x of) the determinism floor."""
    err = summary["induced_errors"].get("identity", {}).get("err_mean_matched", np.nan)
    ok = bool(np.isfinite(err) and err <= max(2.0 * determinism_floor, 1e-6))
    return {"identity_induced_error": err, "determinism_floor": determinism_floor, "pass": ok}


def g3_monotone_error(summary: dict, cfg: dict) -> dict:
    """G3: measured induced error strictly increases with severity, per sim axis."""
    out, all_ok = {}, True
    for axis in ("miscalibration", "dyn_noise", "horizon"):
        errs = []
        for c in enumerate_conditions(cfg):
            if c.axis == axis:
                errs.append((int(c.severity),
                             summary["induced_errors"][c.cid]["err_mean_matched"]))
        errs.sort()
        vals = [e for _, e in errs]
        mono = bool(all(np.isfinite(vals)) and all(b > a for a, b in zip(vals, vals[1:])))
        out[axis] = {"errors_by_severity": vals, "monotone": mono}
        all_ok &= mono
    out["pass"] = all_ok
    return out


def g4_spread(shard_reports: list[dict]) -> dict:
    """G4 (computable half): candidate spread per state."""
    fails = [r["state_id"] for r in shard_reports if not r.get("spread_ok")]
    return {"n_states": len(shard_reports), "failed_states": fails, "pass": not fails}


def determinism_floor_vs_mildest(floor: dict, summary: dict, cfg: dict) -> dict:
    """§3.1 gate: floor < cfg fraction of the mildest condition's induced error."""
    mildest = min(
        summary["induced_errors"][c.cid]["err_mean_matched"]
        for c in enumerate_conditions(cfg) if c.needs_sim
    )
    frac = cfg["determinism"]["floor_max_frac_of_mildest"]
    f = floor["floor_mean_matched_m"]
    return {"floor": f, "mildest_condition_error": float(mildest),
            "threshold": frac * mildest, "pass": bool(f < frac * mildest)}
