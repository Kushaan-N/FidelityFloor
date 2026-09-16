"""Tier 1 (reduced): programmatic spatial QA with verifiable answers (spec §1.3).

Scenes are seeded arrangements of colored primitives on the table. Questions are
generated from ground-truth state, so every answer is checkable. The VLM receives
the base view plus imagined views along a *scripted* camera orbit; degradations
are obs_corruption on the imagined views, or viewpoint miscalibration (delivered
view rotated by delta from the commanded orbit angle).
"""

from __future__ import annotations

import numpy as np

COLORS = {
    "red": [0.85, 0.1, 0.1],
    "blue": [0.1, 0.2, 0.85],
    "green": [0.1, 0.7, 0.2],
    "yellow": [0.9, 0.85, 0.1],
    "purple": [0.6, 0.15, 0.7],
    "orange": [0.95, 0.55, 0.05],
    "white": [0.95, 0.95, 0.95],
}
SHAPES = ["cube", "sphere", "cylinder"]


def sample_scene(cfg: dict, scene_id: int) -> dict:
    """N distinct-colored primitives at non-overlapping table positions."""
    t1 = cfg["tier1"]
    rng = np.random.default_rng([cfg["seed"], 9001, scene_id])
    n = int(rng.integers(t1["n_objects_min"], t1["n_objects_max"] + 1))
    colors = rng.permutation(list(COLORS.keys()))[:n]
    objs, placed = [], []
    for i in range(n):
        for _ in range(200):
            xy = rng.uniform(-0.28, 0.28, size=2)
            if all(np.linalg.norm(xy - np.array(p)) > 0.11 for p in placed):
                break
        placed.append(xy.tolist())
        objs.append({
            "name": f"{colors[i]} {SHAPES[int(rng.integers(len(SHAPES)))]}",
            "color": colors[i],
            "shape": None,  # filled from name below
            "xy": xy.tolist(),
            "size": float(rng.uniform(0.02, 0.035)),
        })
    for o in objs:
        o["shape"] = o["name"].split()[1]
    return {"scene_id": scene_id, "objects": objs}


def _cam_frame_lr(obj_xy, ref_xy, cam_pos, cam_target) -> str:
    """'left'/'right' of ref as seen from the camera."""
    fwd = np.array(cam_target[:2]) - np.array(cam_pos[:2])
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    left = np.array([-fwd[1], fwd[0]])
    rel = np.array(obj_xy) - np.array(ref_xy)
    return "left" if float(rel @ left) > 0 else "right"


def generate_questions(scene: dict, cfg: dict, render_cfg: dict) -> list[dict]:
    """Deterministic question set per scene. Each: text, choices, gt answer."""
    t1 = cfg["tier1"]
    rng = np.random.default_rng([cfg["seed"], 9002, scene["scene_id"]])
    objs = scene["objects"]
    cam = render_cfg["camera"]
    qs = []
    while len(qs) < t1["questions_per_scene"]:
        kind = rng.choice(["reldir", "nearest", "count"])
        if kind == "reldir" and len(objs) >= 2:
            a, b = rng.choice(len(objs), size=2, replace=False)
            ans = _cam_frame_lr(objs[a]["xy"], objs[b]["xy"], cam["position"], cam["target"])
            qs.append({
                "kind": "reldir",
                "text": f"From the base viewpoint, is the {objs[a]['name']} to the left "
                        f"or to the right of the {objs[b]['name']}?",
                "choices": ["left", "right"],
                "answer": ans,
            })
        elif kind == "nearest" and len(objs) >= 3:
            r = int(rng.integers(len(objs)))
            others = [i for i in range(len(objs)) if i != r]
            dists = [np.linalg.norm(np.array(objs[i]["xy"]) - np.array(objs[r]["xy"]))
                     for i in others]
            order = np.argsort(dists)
            if dists[order[1]] - dists[order[0]] < 0.04:
                continue  # ambiguous — resample
            qs.append({
                "kind": "nearest",
                "text": f"Which object is closest to the {objs[r]['name']}?",
                "choices": [objs[i]["name"] for i in others],
                "answer": objs[others[order[0]]]["name"],
            })
        elif kind == "count":
            r = int(rng.integers(len(objs)))
            radius = 0.15
            cnt = sum(
                1 for i, o in enumerate(objs) if i != r
                and np.linalg.norm(np.array(o["xy"]) - np.array(objs[r]["xy"])) < radius
            )
            qs.append({
                "kind": "count",
                "text": f"How many other objects are within 15 cm of the "
                        f"{objs[r]['name']}?",
                "choices": [str(k) for k in range(len(objs))],
                "answer": str(cnt),
            })
    return qs


def orbit_camera_poses(render_cfg: dict, angles_deg: list[float]) -> list[dict]:
    """Scripted imagined-view camera poses: base pose rotated about the scene center."""
    base = np.array(render_cfg["camera"]["position"], dtype=np.float64)
    target = np.array(render_cfg["camera"]["target"], dtype=np.float64)
    poses = []
    for a in angles_deg:
        r = np.deg2rad(a)
        c, s = np.cos(r), np.sin(r)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        poses.append({"position": (R @ (base - target) + target).tolist(),
                      "target": target.tolist(), "angle_deg": float(a)})
    return poses
