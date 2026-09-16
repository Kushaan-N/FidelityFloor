import numpy as np

from fidelityfloor.scoring import (
    normalized_regret,
    rank_metrics,
    spearman,
    state_utility,
    top1_regret,
)
from fidelityfloor.stats import bootstrap_ci, paired_diff_ci


def test_state_utility_bounds():
    goal = np.array([0.2, 0.0])
    # perfect: cube ends on goal
    traj = np.array([[0.0, 0.0, 0.025], [0.2, 0.0, 0.025]])
    assert np.isclose(state_utility(traj, goal), 1.0)
    # regress: cube moves away — clipped at -1
    traj = np.array([[0.1, 0.0, 0.025], [-0.4, 0.0, 0.025]])
    assert state_utility(traj, goal) == -1.0
    # no motion
    traj = np.array([[0.0, 0.0, 0.025], [0.0, 0.0, 0.025]])
    assert np.isclose(state_utility(traj, goal), 0.0)


def test_rank_metrics_perfect_and_inverted():
    u = np.linspace(-0.5, 1.0, 12)
    m = rank_metrics(u, u)
    assert np.isclose(m["spearman"], 1.0)
    assert m["top1_regret"] == 0.0
    assert m["normalized_regret"] == 0.0
    m = rank_metrics(-u, u)
    assert np.isclose(m["spearman"], -1.0)
    assert np.isclose(m["normalized_regret"], 1.0)


def test_top1_regret_partial():
    u_gt = np.array([0.0, 0.5, 1.0])
    u_im = np.array([0.0, 1.0, 0.5])  # picks the middle candidate
    assert np.isclose(top1_regret(u_im, u_gt), 0.5)
    assert np.isclose(normalized_regret(u_im, u_gt), 0.5)


def test_spearman_degenerate_ties():
    assert spearman(np.ones(5), np.ones(5)) == 0.0  # NaN -> 0, not a crash


def test_bootstrap_ci_covers_mean(cfg):
    rng = np.random.default_rng(0)
    v = rng.normal(0.7, 0.1, size=30)
    ci = bootstrap_ci(v, cfg)
    assert ci["lo"] < v.mean() < ci["hi"]
    assert ci["n"] == 30
    # deterministic across calls (seeded bootstrap)
    assert ci == bootstrap_ci(v, cfg)


def test_paired_diff_detects_shift(cfg):
    rng = np.random.default_rng(1)
    base = rng.normal(0.5, 0.2, size=25)
    d = paired_diff_ci(base + 0.15, base, cfg)
    assert d["significant"] and d["lo"] > 0
    d0 = paired_diff_ci(base, base, cfg)
    assert not d0["significant"]
