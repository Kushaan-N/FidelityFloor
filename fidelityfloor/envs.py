"""Isaac Sim scene: table + cube + velocity-controlled pusher + goal marker + camera.

Isaac Sim is imported lazily so this module can be *imported* anywhere; actually
constructing PushEnv requires an Isaac container/host with a GPU for rendering.

Design notes (see REPRO_LOG.md deviations):
- No Nucleus/cloud assets: table is a primitive FixedCuboid; lights via USD API.
- CPU PhysX (GPU dynamics off, TGS solver) for replay determinism; the GPU is
  used for RTX rendering only.
- The pusher is a gravity-disabled dynamic cylinder whose linear velocity is set
  every physics substep — effectively a velocity-controlled end effector with
  real PhysX contact against the cube.
"""

from __future__ import annotations

import io

import numpy as np

_SIM_APP = None  # one SimulationApp per process, ever


def start_sim_app(render_cfg: dict):
    """Create (once) and return the headless SimulationApp."""
    global _SIM_APP
    if _SIM_APP is not None:
        return _SIM_APP
    from isaacsim import SimulationApp

    _SIM_APP = SimulationApp(
        {
            "headless": True,
            "width": render_cfg["width"],
            "height": render_cfg["height"],
            "renderer": render_cfg.get("renderer", "RaytracedLighting"),
        }
    )
    return _SIM_APP


class PushEnv:
    CUBE_PATH = "/World/cube"
    PUSHER_PATH = "/World/pusher"
    TABLE_PATH = "/World/table"
    GOAL_PATH = "/World/goal_marker"

    def __init__(self, cfg: dict, render_cfg: dict):
        self.cfg = cfg
        self.render_cfg = render_cfg
        self.app = start_sim_app(render_cfg)

        from isaacsim.core.api import World
        from isaacsim.core.api.objects import (
            DynamicCuboid,
            DynamicCylinder,
            FixedCuboid,
            VisualCylinder,
        )
        from isaacsim.sensors.camera import Camera
        from pxr import PhysxSchema, UsdLux

        ro, ws = cfg["rollout"], cfg["workspace"]
        self.substeps = max(1, int(round(ro["control_dt"] / ro["physics_dt"])))
        self.world = World(
            physics_dt=ro["physics_dt"],
            rendering_dt=ro["physics_dt"],
            stage_units_in_meters=1.0,
            backend="numpy",
        )
        # Determinism-first physics: CPU pipeline, MBP broadphase, TGS solver.
        pc = self.world.get_physics_context()
        pc.enable_gpu_dynamics(False)
        pc.set_broadphase_type("MBP")
        pc.set_solver_type("TGS")
        pc.enable_ccd(True)

        stage = self.world.stage
        # Lights (pure USD, no assets)
        dome = UsdLux.DomeLight.Define(stage, "/World/dome_light")
        dome.CreateIntensityAttr(1000.0)
        sun = UsdLux.DistantLight.Define(stage, "/World/sun")
        sun.CreateIntensityAttr(3000.0)
        sun.AddRotateXYZOp().Set((-35.0, 20.0, 0.0))

        scene = self.world.scene
        self.table = scene.add(
            FixedCuboid(
                prim_path=self.TABLE_PATH, name="table",
                position=np.array([0.0, 0.0, -0.025]),
                scale=np.array([1.2, 1.2, 0.05]),
                color=np.array([0.45, 0.45, 0.48]),
            )
        )
        ch = ws["cube_half"]
        self.cube = scene.add(
            DynamicCuboid(
                prim_path=self.CUBE_PATH, name="cube",
                position=np.array([0.0, 0.0, ch]),
                scale=np.array([2 * ch, 2 * ch, 2 * ch]),
                color=np.array([0.85, 0.1, 0.1]),
                mass=0.1,
            )
        )
        self.pusher_z = ws["pusher_height"] / 2 + 0.001  # 1 mm above the table
        self.pusher = scene.add(
            DynamicCylinder(
                prim_path=self.PUSHER_PATH, name="pusher",
                position=np.array([0.2, 0.2, self.pusher_z]),
                radius=ws["pusher_radius"], height=ws["pusher_height"],
                color=np.array([0.1, 0.2, 0.85]),
                mass=0.5,
            )
        )
        # Gravity off for the pusher: velocity control fully determines its motion.
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(self.PUSHER_PATH))
        rb.CreateDisableGravityAttr(True)
        # Goal marker: visual-only flat disk (no collider -> no physics effect).
        self.goal_marker = scene.add(
            VisualCylinder(
                prim_path=self.GOAL_PATH, name="goal_marker",
                position=np.array([0.3, 0.3, 0.001]),
                radius=0.06, height=0.002,
                color=np.array([0.1, 0.8, 0.2]),
            )
        )

        cam = render_cfg["camera"]
        self.camera = Camera(
            prim_path="/World/camera",
            position=np.array(cam["position"], dtype=np.float64),
            resolution=(render_cfg["width"], render_cfg["height"]),
        )
        self.world.reset()
        self.camera.initialize()
        self.camera.set_focal_length(cam.get("focal_length", 18.0) / 10.0)  # cm units
        self._aim_camera(np.array(cam["position"]), np.array(cam["target"]))

    # ------------------------------------------------------------------ camera
    def _aim_camera(self, position: np.ndarray, target: np.ndarray) -> None:
        """Point the camera at `target` (USD cameras look down -Z)."""
        from isaacsim.core.utils.rotations import euler_angles_to_quat

        fwd = target - position
        fwd = fwd / np.linalg.norm(fwd)
        yaw = np.arctan2(fwd[1], fwd[0])
        pitch = np.arcsin(-fwd[2])
        # Camera convention: rotate so -Z looks along fwd, +Y up-ish.
        quat = euler_angles_to_quat(np.array([0.0, pitch + np.pi / 2, yaw]), degrees=False,
                                    extrinsic=False)
        self.camera.set_world_pose(position=position, orientation=quat)

    def set_camera_pose(self, position: np.ndarray, target: np.ndarray) -> None:
        self._aim_camera(np.asarray(position, dtype=np.float64),
                         np.asarray(target, dtype=np.float64))

    # ------------------------------------------------------------------- reset
    def reset_to(self, state: dict) -> None:
        """Teleport bodies to an initial state and let physics settle."""
        ws, ro = self.cfg["workspace"], self.cfg["rollout"]
        self.world.reset()
        cube_xy = np.array(state["cube_xy"])
        pusher_xy = np.array(state["pusher_xy"])
        goal_xy = np.array(state["goal_xy"])
        self.cube.set_world_pose(
            position=np.array([cube_xy[0], cube_xy[1], ws["cube_half"]]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self.cube.set_linear_velocity(np.zeros(3))
        self.cube.set_angular_velocity(np.zeros(3))
        self.pusher.set_world_pose(
            position=np.array([pusher_xy[0], pusher_xy[1], self.pusher_z]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self.pusher.set_linear_velocity(np.zeros(3))
        self.pusher.set_angular_velocity(np.zeros(3))
        self.goal_marker.set_world_pose(
            position=np.array([goal_xy[0], goal_xy[1], 0.001]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        for _ in range(ro["settle_steps"]):
            self.world.step(render=False)

    # -------------------------------------------------------------------- step
    def step_control(self, action_xy: np.ndarray) -> None:
        """One 10 Hz control step = `substeps` physics substeps at fixed velocity."""
        v = np.array([action_xy[0], action_xy[1], 0.0])
        for _ in range(self.substeps):
            self.pusher.set_linear_velocity(v)
            self.pusher.set_angular_velocity(np.zeros(3))
            self.world.step(render=False)

    def get_state(self) -> dict:
        cp, cq = self.cube.get_world_pose()
        pp, _ = self.pusher.get_world_pose()
        return {"cube_pos": np.asarray(cp), "cube_quat": np.asarray(cq),
                "pusher_pos": np.asarray(pp)}

    # ------------------------------------------------------------------ render
    def render_frame(self) -> bytes:
        """Render the current state and return PNG bytes."""
        from PIL import Image

        for _ in range(self.render_cfg.get("warmup_render_steps", 8)):
            self.world.render()
        rgba = self.camera.get_rgba()
        rgb = np.asarray(rgba)[:, :, :3].astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return buf.getvalue()
