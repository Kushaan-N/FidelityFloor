import numpy as np

from fidelityfloor.candidates import generate_candidates, sample_initial_state, spread_ok
from fidelityfloor.scoring import state_utility


def test_initial_states_deterministic(cfg):
    a = sample_initial_state(cfg, 3)
    b = sample_initial_state(cfg, 3)
    assert a == b
    assert a != sample_initial_state(cfg, 4)


def test_geometry_constraints(cfg):
    ws = cfg["workspace"]
    for sid in range(20):
        s = sample_initial_state(cfg, sid)
        cube, goal, pusher = map(np.array, (s["cube_xy"], s["goal_xy"], s["pusher_xy"]))
        assert ws["goal_dist_min"] - 1e-9 <= np.linalg.norm(goal - cube) <= ws["goal_dist_max"] + 1e-9
        assert ws["pusher_dist_min"] - 1e-9 <= np.linalg.norm(pusher - cube) <= ws["pusher_dist_max"] + 1e-9


def test_candidate_shapes_and_limits(cfg):
    s = sample_initial_state(cfg, 0)
    c = generate_candidates(s, cfg)
    assert c.shape == (cfg["candidates"]["k"], cfg["rollout"]["horizon_steps"], 2)
    assert np.abs(c).max() <= cfg["workspace"]["max_speed"] + 1e-9
    assert np.array_equal(c, generate_candidates(s, cfg))  # deterministic


def test_expert_beats_random_under_fake_dynamics(cfg, fake_dyn):
    """The expert candidate should push the cube toward the goal far better than
    random candidates in the analytic model — sanity for G4."""
    wins = 0
    n = 10
    for sid in range(n):
        s = sample_initial_state(cfg, sid)
        cands = generate_candidates(s, cfg)
        goal = np.array(s["goal_xy"])
        utils = [state_utility(fake_dyn.rollout(s, a)["cube_pos"], goal) for a in cands]
        expert_u = utils[0]
        rand_us = utils[-cfg["candidates"]["n_random"]:]
        if expert_u > max(rand_us):
            wins += 1
        assert expert_u > 0.3, f"expert too weak on state {sid}: {expert_u:.2f}"
    assert wins >= n - 2


def test_spread_check_under_fake_dynamics(cfg, fake_dyn):
    ok_count = 0
    for sid in range(10):
        s = sample_initial_state(cfg, sid)
        cands = generate_candidates(s, cfg)
        goal = np.array(s["goal_xy"])
        utils = np.array([state_utility(fake_dyn.rollout(s, a)["cube_pos"], goal)
                          for a in cands])
        ok, _ = spread_ok(utils, cfg)
        ok_count += ok
    assert ok_count >= 8  # candidate set produces spread on nearly all states


def test_graded_perturbations_grade_utility(cfg, fake_dyn):
    """Rotating the push direction further should (weakly) reduce utility —
    the ordering signal the whole study rests on."""
    degraded = []
    for sid in range(10):
        s = sample_initial_state(cfg, sid)
        cands = generate_candidates(s, cfg)
        goal = np.array(s["goal_xy"])
        u = [state_utility(fake_dyn.rollout(s, a)["cube_pos"], goal) for a in cands]
        # candidates 1-6 are rotations by +/-10, +/-20, +/-35
        u10 = (u[1] + u[2]) / 2
        u35 = (u[5] + u[6]) / 2
        degraded.append(u10 - u35)
    assert np.mean(degraded) > 0
