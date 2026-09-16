"""End-to-end pipeline logic test on the analytic dynamics stand-in:
degrade -> execute -> utilities -> ranking metrics -> monotonicity, without Isaac.
This is the pre-GPU rehearsal of G1/G3 shapes.
"""

import numpy as np

from fidelityfloor.candidates import generate_candidates, sample_initial_state
from fidelityfloor.config import condition_by_id, identity_condition
from fidelityfloor.degrade import degrade_actions
from fidelityfloor.scoring import rank_metrics, state_utility


def _grid_metrics(cfg, fake_dyn, cond, n_states=8):
    rhos, errs = [], []
    for sid in range(n_states):
        s = sample_initial_state(cfg, sid)
        cands = generate_candidates(s, cfg)
        goal = np.array(s["goal_xy"])
        u_gt, u_im, pair_err = [], [], []
        for ci, a in enumerate(cands):
            gt = fake_dyn.rollout(s, a)
            rng = np.random.default_rng([cfg["seed"], sid, ci])
            im = fake_dyn.rollout(s, degrade_actions(a, cond, rng))
            u_gt.append(state_utility(gt["cube_pos"], goal))
            u_im.append(state_utility(im["cube_pos"], goal))
            n = min(len(gt["cube_pos"]), len(im["cube_pos"]))
            pair_err.append(np.linalg.norm(
                gt["cube_pos"][:n] - im["cube_pos"][:n], axis=1).mean())
        rhos.append(rank_metrics(np.array(u_im), np.array(u_gt))["spearman"])
        errs.append(np.mean(pair_err))
    return float(np.mean(rhos)), float(np.mean(errs))


def test_identity_recovers_gt_ranking(cfg, fake_dyn):
    """G1 shape: perfect imagination == perfect ranking (analytic model is exact)."""
    rho, err = _grid_metrics(cfg, fake_dyn, identity_condition())
    assert rho > 0.999
    assert err < 1e-12


def test_induced_error_monotone_in_severity(cfg, fake_dyn):
    """G3 shape: measured error rises with each severity knob, per axis."""
    for axis in ("miscalibration", "dyn_noise"):
        errs = []
        for sev in ("1", "2", "3"):
            _, e = _grid_metrics(cfg, fake_dyn, condition_by_id(cfg, f"{axis}-{sev}"))
            errs.append(e)
        assert errs[0] < errs[1] < errs[2], f"{axis}: {errs}"


def test_utility_degrades_with_severity(cfg, fake_dyn):
    """Demand-curve shape: ranking quality falls (weakly) as severity rises."""
    for axis in ("miscalibration", "dyn_noise"):
        rhos = [
            _grid_metrics(cfg, fake_dyn, condition_by_id(cfg, f"{axis}-{s}"))[0]
            for s in ("1", "3")
        ]
        assert rhos[1] <= rhos[0] + 0.05, f"{axis}: severity 3 not worse than 1: {rhos}"


def test_horizon_truncation_hurts_ranking(cfg, fake_dyn):
    rho_full, _ = _grid_metrics(cfg, fake_dyn, identity_condition())
    rho_quarter, _ = _grid_metrics(cfg, fake_dyn, condition_by_id(cfg, "horizon-3"))
    assert rho_quarter < rho_full
