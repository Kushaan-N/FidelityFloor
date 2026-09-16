"""Tier 1 (reduced scope, F2): render scene views, apply view degradations, run
VLM QA, and summarize accuracy against the two anchors (spec §1.3).

Conditions: identity, obs_corruption x3 (on imagined views), viewpoint
miscalibration x2 (delivered view rotated by delta from the commanded orbit).
Anchors: no_imagination (base view only), perfect_imagination (= identity).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import OUTPUT_ROOT, Condition, config_hash, stable_hash
from .io_utils import atomic_write_bytes, atomic_write_json
from .questions import COLORS, generate_questions, orbit_camera_poses, sample_scene


def tier1_root(cfg: dict) -> Path:
    return OUTPUT_ROOT / config_hash(cfg) / "tier1"


def tier1_conditions(cfg: dict) -> list[dict]:
    conds = [{"cid": "identity", "kind": "identity"}]
    for sev, p in cfg["axes"]["obs_corruption"]["severities"].items():
        conds.append({"cid": f"obs_corruption-{sev}", "kind": "obs_corruption",
                      "severity": sev, "params": p})
    for i, d in enumerate(cfg["tier1"]["viewpoint_miscalibration_deg"], start=1):
        conds.append({"cid": f"viewpoint-{i}", "kind": "viewpoint", "delta_deg": float(d)})
    return conds


# ------------------------------------------------------------------ GPU: render

def render_scene_views(cfg: dict, render_cfg: dict, scene_id: int) -> dict:
    """Render base view + commanded orbit views + viewpoint-miscalibrated orbit
    views for one scene. Skip-if-present per PNG."""
    from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, DynamicSphere

    from .envs import PushEnv

    scene = sample_scene(cfg, scene_id)
    sdir = tier1_root(cfg) / f"scene_{scene_id:04d}"
    manifest_p = sdir / "manifest.json"
    if manifest_p.exists():
        return json.loads(manifest_p.read_text())

    env = PushEnv(cfg, render_cfg)
    # Park the Tier-2 actors out of frame; spawn the Tier-1 objects.
    env.cube.set_world_pose(position=np.array([5.0, 5.0, 0.025]))
    env.pusher.set_world_pose(position=np.array([5.0, 5.3, env.pusher_z]))
    env.goal_marker.set_world_pose(position=np.array([5.0, 5.6, 0.001]))

    ctor = {"cube": DynamicCuboid, "sphere": DynamicSphere, "cylinder": DynamicCylinder}
    for i, o in enumerate(scene["objects"]):
        kw = dict(
            prim_path=f"/World/t1_obj_{i}", name=f"t1_obj_{i}",
            position=np.array([o["xy"][0], o["xy"][1], o["size"]]),
            color=np.array(COLORS[o["color"]]),
        )
        if o["shape"] == "cube":
            kw["scale"] = np.array([2 * o["size"]] * 3)
        else:
            kw["radius"] = o["size"]
            if o["shape"] == "cylinder":
                kw["height"] = 2 * o["size"]
        env.world.scene.add(ctor[o["shape"]](**kw))
    for _ in range(cfg["rollout"]["settle_steps"]):
        env.world.step(render=False)

    orbit = cfg["tier1"]["orbit_deg"]
    views = {"base": 0.0, "orbit_pos": orbit, "orbit_neg": -orbit}
    # Viewpoint miscalibration: delivered view differs from commanded by delta
    for i, d in enumerate(cfg["tier1"]["viewpoint_miscalibration_deg"], start=1):
        views[f"orbit_pos_vp{i}"] = orbit + d
        views[f"orbit_neg_vp{i}"] = -(orbit + d)

    files = {}
    for name, angle in views.items():
        p = sdir / f"{name}.png"
        if not p.exists():
            pose = orbit_camera_poses(render_cfg, angles_deg=[angle])[0]
            env.set_camera_pose(np.array(pose["position"]), np.array(pose["target"]))
            atomic_write_bytes(p, env.render_frame())
        files[name] = str(p)

    manifest = {"scene": scene, "views": files,
                "questions": generate_questions(scene, cfg, render_cfg)}
    atomic_write_json(manifest_p, manifest)
    return manifest


# --------------------------------------------------------------- CPU: QA + stats

def _views_for_condition(cfg: dict, manifest: dict, cond: dict, scene_seed: int
                         ) -> list[bytes]:
    """Assemble [base + imagined views] under a Tier-1 condition."""
    from .degrade import corrupt_frame

    def rd(name):
        return Path(manifest["views"][name]).read_bytes()

    base = rd("base")
    if cond["cid"] == "no_imagination":
        return [base]
    if cond["kind"] == "identity":
        return [base, rd("orbit_pos"), rd("orbit_neg")]
    if cond["kind"] == "viewpoint":
        i = cond["cid"].split("-")[1]
        return [base, rd(f"orbit_pos_vp{i}"), rd(f"orbit_neg_vp{i}")]
    if cond["kind"] == "obs_corruption":
        c = Condition("obs_corruption", cond["severity"],
                      tuple(sorted(cond["params"].items())))
        return [base,
                corrupt_frame(rd("orbit_pos"), c, frame_index=1, seed=scene_seed),
                corrupt_frame(rd("orbit_neg"), c, frame_index=2, seed=scene_seed)]
    raise ValueError(cond)


def tier1_qa_pass(cfg: dict, scene_ids: list[int]) -> dict:
    """Ask every question under every condition + anchors; VLM responses cached."""
    from .vlm import VLMClient

    client = VLMClient(cfg)
    conds = tier1_conditions(cfg) + [{"cid": "no_imagination", "kind": "anchor"}]
    results = []
    for sid in scene_ids:
        mp = tier1_root(cfg) / f"scene_{sid:04d}" / "manifest.json"
        if not mp.exists():
            continue
        manifest = json.loads(mp.read_text())
        for cond in conds:
            images = _views_for_condition(cfg, manifest, cond, cfg["seed"] + sid)
            for qi, q in enumerate(manifest["questions"]):
                r = client.answer_question(q["text"], images, q["choices"])
                results.append({
                    "scene_id": sid, "q_index": qi, "kind": q["kind"],
                    "condition": cond["cid"],
                    "answer": r["answer"], "gt": q["answer"],
                    "correct": r["answer"].strip().lower() == q["answer"].strip().lower(),
                })
    out = {"results": results, "billed_calls": client.n_billed_calls}
    atomic_write_json(tier1_root(cfg) / "qa_results.json", out)
    return out


def tier1_summary(cfg: dict) -> dict:
    from .stats import bootstrap_ci

    p = tier1_root(cfg) / "qa_results.json"
    results = json.loads(p.read_text())["results"]
    by_cond = {}
    for cid in sorted({r["condition"] for r in results}):
        rr = [r for r in results if r["condition"] == cid]
        # bootstrap over scenes (the pairing unit)
        per_scene = [np.mean([x["correct"] for x in rr if x["scene_id"] == s])
                     for s in sorted({x["scene_id"] for x in rr})]
        by_cond[cid] = {"accuracy": bootstrap_ci(np.array(per_scene), cfg,
                                                 seed_tag=hash(cid) % 9999),
                        "n_questions": len(rr)}
    summary = {
        "conditions": {k: v for k, v in by_cond.items()
                       if k not in ("no_imagination",)},
        "no_imagination": by_cond.get("no_imagination", {}).get("accuracy"),
        "perfect_imagination": by_cond.get("identity", {}).get("accuracy"),
    }
    atomic_write_json(tier1_root(cfg) / "tier1_summary.json", summary)
    return summary
