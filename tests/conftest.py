import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fidelityfloor.config import load_config, load_render_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def render_cfg():
    return load_render_config()


class FakePushDynamics:
    """Analytic stand-in for PhysX used to unit-test pipeline logic:
    the pusher integrates commanded velocity exactly; the cube is displaced
    when the pusher disk overlaps it (simple projection push)."""

    def __init__(self, cfg):
        self.ws = cfg["workspace"]
        self.dt = cfg["rollout"]["control_dt"]

    def rollout(self, state, actions):
        cube = np.array(state["cube_xy"], dtype=float)
        pusher = np.array(state["pusher_xy"], dtype=float)
        contact = self.ws["cube_half"] + self.ws["pusher_radius"]
        cube_traj, pusher_traj = [cube.copy()], [pusher.copy()]
        for a in actions:
            pusher = pusher + np.asarray(a) * self.dt
            d = cube - pusher
            dist = np.linalg.norm(d)
            if dist < contact:
                cube = pusher + d / max(dist, 1e-9) * contact
            cube_traj.append(cube.copy())
            pusher_traj.append(pusher.copy())
        z = lambda t: np.hstack([np.array(t), np.zeros((len(t), 1))])  # noqa: E731
        return {"cube_pos": z(cube_traj), "pusher_pos": z(pusher_traj)}


@pytest.fixture()
def fake_dyn(cfg):
    return FakePushDynamics(cfg)
