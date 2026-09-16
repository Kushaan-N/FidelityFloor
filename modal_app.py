"""FidelityFloor on Modal (spec §0.1, §2).

Staged ramp — never debug at 10x parallelism:
    modal run modal_app.py::p0          # G0: single worker end-to-end + timings
    modal run modal_app.py::determinism # §3.1 floor over 20 states
    modal run modal_app.py::p1          # zero-degradation pilot, 5 states (G1)
    modal run modal_app.py::p2          # dress rehearsal: 3 states x all conditions (G5)
    modal run modal_app.py::f1          # FULL GRID — only after G5 passes
    modal run modal_app.py::analyze     # stats + figures from raw arrays (free to re-run)

Prereqs (one-time):
    modal token new
    modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import modal

ISAAC_VERSION = "5.0.0"
GPU = "L40S"
GPU_USD_PER_HR = 1.95  # spend estimates printed by entrypoints
VOL_MOUNT = "/vol"

app = modal.App("fidelityfloor")
vol = modal.Volume.from_name("fidelityfloor-data", create_if_missing=True)

_vulkan_icd = (
    '{"file_format_version":"1.0.0","ICD":'
    '{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}'
)
_egl_icd = '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}'

isaac_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        # GL/Vulkan/X plumbing for Isaac headless RTX rendering
        "libglvnd0", "libgl1", "libegl1", "libgles2", "libglu1-mesa",
        "libvulkan1", "vulkan-tools",
        "libx11-6", "libxext6", "libxt6", "libxrandr2", "libxinerama1",
        "libxcursor1", "libxi6", "libsm6", "libice6",
        "libglib2.0-0", "libgomp1", "unzip", "wget",
    )
    .env({
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "N",
        "FF_OUTPUT_ROOT": f"{VOL_MOUNT}/outputs",
    })
    .run_commands(
        "mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d",
        f"echo '{_vulkan_icd}' > /usr/share/vulkan/icd.d/nvidia_icd.json",
        f"echo '{_egl_icd}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    )
    .pip_install(
        f"isaacsim[all,extscache]=={ISAAC_VERSION}",
        extra_index_url="https://pypi.nvidia.com",
    )
    .pip_install("numpy<2", "scipy", "pyyaml", "pillow")
    # Late layer (keeps the huge isaacsim layer cached): Modal injects the real
    # NVIDIA ICD at /etc/vulkan/icd.d at runtime. Our baked copy made the same
    # GPU enumerate twice -> ERROR_DEVICE_LOST (P0 run 1). Remove ours and every
    # mesa software ICD, and pin the loader to the injected one.
    .run_commands(
        "rm -f /usr/share/vulkan/icd.d/nvidia_icd.json "
        "/usr/share/vulkan/icd.d/intel_icd.x86_64.json "
        "/usr/share/vulkan/icd.d/intel_hasvk_icd.x86_64.json "
        "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json "
        "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json "
        "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
    )
    .env({
        "VK_DRIVER_FILES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
    })
    .add_local_python_source("fidelityfloor")
    .add_local_dir("configs", remote_path="/root/configs")
)

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "scipy", "pyyaml", "pillow", "matplotlib", "anthropic")
    .env({"FF_OUTPUT_ROOT": f"{VOL_MOUNT}/outputs"})
    .add_local_python_source("fidelityfloor")
    .add_local_dir("configs", remote_path="/root/configs")
)

_retries = modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=10.0)


def _existing_vlm_secrets() -> list:
    """Attach whichever VLM API secrets exist ('gemini-api-key' with GEMINI_API_KEY,
    'anthropic-api-key' with ANTHROPIC_API_KEY), so the sim stages run before any
    key is provisioned. Evaluated locally at deploy time only."""
    if not modal.is_local():
        return []
    found = []
    for name in ("gemini-api-key", "anthropic-api-key"):
        try:
            s = modal.Secret.from_name(name)
            s.hydrate()
            found.append(s)
        except Exception:
            pass
    if not found:
        print("NOTE: no VLM API secret found (gemini-api-key / anthropic-api-key) — "
              "VLM stages will be skipped until one exists. Free option: "
              "https://aistudio.google.com key, then "
              "`modal secret create gemini-api-key GEMINI_API_KEY=...`")
    return found


_vlm_secrets = _existing_vlm_secrets()


def _cfgs():
    from fidelityfloor.config import load_config, load_render_config

    return load_config(), load_render_config()


# =============================================================== GPU functions

@app.function(image=isaac_image, gpu=GPU, timeout=600)
def diag_remote() -> dict:
    """Cheap container diagnostic: driver, Vulkan ICDs, EGL vendors. No Isaac boot."""
    import glob
    import os
    import subprocess

    def run(cmd):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return (r.stdout + r.stderr)[:2500]
        except Exception as e:  # noqa: BLE001
            return f"ERR {e}"

    icds = (glob.glob("/usr/share/vulkan/icd.d/*") + glob.glob("/etc/vulkan/icd.d/*")
            + glob.glob("/usr/local/share/vulkan/icd.d/*"))
    return {
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,driver_version",
                           "--format=csv,noheader"]),
        "vulkan_icds": {p: open(p).read()[:200] for p in icds},
        "egl_vendors": glob.glob("/usr/share/glvnd/egl_vendor.d/*")
        + glob.glob("/etc/glvnd/egl_vendor.d/*"),
        "vulkaninfo_summary": run(["vulkaninfo", "--summary"])[:2500],
        "libcuda": glob.glob("/usr/lib/x86_64-linux-gnu/libcuda*"),
        "vk_env": {k: v for k, v in os.environ.items()
                   if k.startswith(("VK_", "NVIDIA_", "CUDA_", "__EGL"))},
    }


@app.local_entrypoint()
def diag():
    import json

    print(json.dumps(diag_remote.remote(), indent=2))


@app.function(image=isaac_image, gpu=GPU, volumes={VOL_MOUNT: vol},
              timeout=3600, retries=_retries, max_containers=10)
def smoke_remote() -> dict:
    from fidelityfloor.runner import smoke_test

    cfg, rcfg = _cfgs()
    report = smoke_test(cfg, rcfg)
    vol.commit()
    return report


@app.function(image=isaac_image, gpu=GPU, volumes={VOL_MOUNT: vol},
              timeout=3600, retries=_retries)
def determinism_remote() -> dict:
    from fidelityfloor.runner import determinism_pass

    cfg, rcfg = _cfgs()
    floor = determinism_pass(cfg, rcfg)
    vol.commit()
    return {k: v for k, v in floor.items() if k != "per_state"}


@app.function(image=isaac_image, gpu=GPU, volumes={VOL_MOUNT: vol},
              timeout=3600, retries=_retries, max_containers=10)
def state_shard_remote(args: dict) -> dict:
    """One Tier-2 shard: GT + assigned sim conditions for one initial state."""
    import time

    from fidelityfloor.runner import run_state_shard

    cfg, rcfg = _cfgs()
    t0 = time.time()
    report = run_state_shard(cfg, rcfg, state_id=args["state_id"],
                             condition_ids=args.get("condition_ids"))
    report["gpu_wall_s"] = round(time.time() - t0, 1)
    vol.commit()
    return report


@app.function(image=isaac_image, gpu=GPU, volumes={VOL_MOUNT: vol},
              timeout=3600, retries=_retries, max_containers=10)
def tier1_shard_remote(args: dict) -> dict:
    """Tier 1 shard: render base + orbit views for one scene (F2)."""
    from fidelityfloor.tier1 import render_scene_views

    cfg, rcfg = _cfgs()
    out = render_scene_views(cfg, rcfg, scene_id=args["scene_id"])
    vol.commit()
    return out


# =============================================================== CPU functions

@app.function(image=cpu_image, volumes={VOL_MOUNT: vol}, timeout=1800, retries=_retries)
def corrupt_remote(state_ids: list) -> dict:
    from fidelityfloor.runner import corrupt_frames_pass

    cfg, _ = _cfgs()
    out = corrupt_frames_pass(cfg, list(state_ids))
    vol.commit()
    return out


@app.function(image=cpu_image, volumes={VOL_MOUNT: vol}, timeout=7200, secrets=_vlm_secrets)
def vlm_score_remote(state_ids: list) -> dict:
    from fidelityfloor.runner import vlm_score_pass

    cfg, _ = _cfgs()
    out = vlm_score_pass(cfg, list(state_ids))
    vol.commit()
    return {k: v for k, v in out.items() if k != "scores"} | {
        "n_scored": len(out["scores"])
    }


@app.function(image=cpu_image, volumes={VOL_MOUNT: vol}, timeout=1800, secrets=_vlm_secrets)
def vlm_smoke_remote() -> dict:
    """One VLM call round-trip against the smoke-test frame (P0 requirement)."""
    from pathlib import Path

    from fidelityfloor.config import out_root
    from fidelityfloor.vlm import VLMClient, have_api_key

    cfg, _ = _cfgs()
    if not have_api_key(cfg):
        return {"skipped": "no VLM API key for the configured provider — create the "
                           "Modal secret (gemini-api-key / anthropic-api-key) before "
                           "P2's VLM pass"}
    frames = sorted(Path(out_root(cfg) / "smoke" / "gt_rollout" / "frames").glob("*.png"))
    if frames:
        png = frames[-1].read_bytes()
        source = str(frames[-1])
    else:
        # No rendered frames yet: score a synthetic near-success scene so the
        # key + provider + JSON-schema plumbing is verified independently of G0.
        import io

        from PIL import Image, ImageDraw

        img = Image.new("RGB", (640, 480), (115, 115, 122))
        d = ImageDraw.Draw(img)
        d.ellipse([300, 220, 420, 300], fill=(26, 178, 51))   # green goal disk
        d.rectangle([335, 230, 385, 280], fill=(217, 26, 26))  # red cube on it
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()
        source = "synthetic (cube centered on goal; expect a high score)"
    client = VLMClient(cfg)
    r = client.score_outcome(png)
    r["_source"] = source
    vol.commit()
    return r


@app.function(image=cpu_image, volumes={VOL_MOUNT: vol}, timeout=1800)
def analyze_remote(state_ids: list) -> dict:
    """Stats + gates + figures from raw arrays. Free to re-run."""
    import json

    import numpy as np

    from fidelityfloor import gates
    from fidelityfloor.config import out_root
    from fidelityfloor.figures import fig_matched_error, fig_utility_vs_error
    from fidelityfloor.runner import load_vlm_scores
    from fidelityfloor.stats import save_summary, summarize

    cfg, _ = _cfgs()
    vlm_scores = load_vlm_scores(cfg)
    summary = summarize(cfg, list(state_ids), vlm_scores)
    save_summary(cfg, summary)

    floor_path = out_root(cfg) / "tables" / "determinism_floor.json"
    floor = json.loads(floor_path.read_text()) if floor_path.exists() else None
    fdir = out_root(cfg) / "figures"
    figs = []
    for mode in (["state", "vlm"] if vlm_scores else ["state"]):
        for metric in ("spearman", "normalized_regret"):
            p = fig_utility_vs_error(
                summary, fdir, mode=mode, metric=metric,
                determinism_floor=floor["floor_mean_matched_m"] if floor else None)
            if p:
                figs.append(str(p))
    p2 = fig_matched_error(summary, fdir)
    if p2:
        figs.append(str(p2))

    gate_results = {
        "G1": gates.g1_perfect_imagination(summary, vlm_present=vlm_scores is not None),
        "G3": gates.g3_monotone_error(summary, cfg),
    }
    if floor:
        gate_results["G2"] = gates.g2_identity_noop(summary, floor["floor_mean_matched_m"])
        gate_results["floor_vs_mildest"] = gates.determinism_floor_vs_mildest(
            floor, summary, cfg)
    from fidelityfloor.io_utils import atomic_write_json

    atomic_write_json(out_root(cfg) / "tables" / "gates.json", gate_results)
    vol.commit()

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    return _clean({
        "gates": gate_results,
        "per_condition": {
            cid: {k: v for k, v in e.items() if k != "induced_error"}
            for cid, e in summary["per_condition"].items()
        },
        "matched_error": {k: v for k, v in summary["matched_error"].items()
                          if k not in ("error_grid", "utility_miscalibration",
                                       "utility_dyn_noise")},
        "figures": figs,
    })


# ============================================================ local entrypoints

def _print_cost(label: str, gpu_seconds: float):
    print(f"[{label}] GPU time {gpu_seconds/60:.1f} min ≈ "
          f"${gpu_seconds / 3600 * GPU_USD_PER_HR:.2f} at {GPU} rates")


@app.local_entrypoint()
def p0():
    """G0: single-worker smoke test + one VLM round-trip. Run this FIRST."""
    print("== P0 smoke test (one L40S worker) ==")
    r = smoke_remote.remote()
    for k, v in r.items():
        if k != "state":
            print(f"  {k}: {v}")
    print("== VLM round-trip ==")
    v = vlm_smoke_remote.remote()
    print(f"  {v}")
    est = r["rollout_with_render_s"]
    full_rollouts = 30 * 12 * 9  # states x K x (identity + 8 sim conditions incl GT)
    print(f"\nCalibration: full grid ≈ {full_rollouts} sim rollouts "
          f"≈ {full_rollouts * est / 3600:.1f} GPU-hr "
          f"≈ ${full_rollouts * est / 3600 * GPU_USD_PER_HR:.0f} "
          f"(update REPRO_LOG.md §0.7 and recompute the §10 budget)")


@app.local_entrypoint()
def vlm_smoke():
    """One VLM round-trip (synthetic image if no rendered frames on the volume)."""
    print(vlm_smoke_remote.remote())


@app.local_entrypoint()
def determinism():
    """§3.1: replay-determinism floor over 20 states."""
    r = determinism_remote.remote()
    print(r)
    print("Gate: floor must be < 10% of the mildest condition's induced error "
          "(checked automatically in analyze once P2 data exists).")


@app.local_entrypoint()
def p1():
    """Zero-degradation pilot: 5 states, identity only, then G1 check (state mode)."""
    from fidelityfloor.config import load_config

    cfg = load_config()
    n = cfg["tier2"]["n_states_pilot"]
    args = [{"state_id": s, "condition_ids": ["identity"]} for s in range(n)]
    reports = list(state_shard_remote.map(args))
    gpu_s = sum(r.get("gpu_wall_s", 0) for r in reports)
    _print_cost("P1", gpu_s)
    for r in reports:
        print(f"  state {r['state_id']}: spread_ok={r['spread_ok']} {r['spread_info']}")
    print(analyze_remote.remote(list(range(n))))


@app.local_entrypoint()
def p2(with_vlm: bool = True):
    """Dress rehearsal (G5): 3 states x ALL conditions, through stats and figures."""
    from fidelityfloor.config import load_config

    cfg = load_config()
    n = cfg["tier2"]["n_states_rehearsal"]
    sids = list(range(n))
    reports = list(state_shard_remote.map([{"state_id": s} for s in sids]))
    gpu_s = sum(r.get("gpu_wall_s", 0) for r in reports)
    _print_cost("P2 sim", gpu_s)
    print(corrupt_remote.remote(sids))
    if with_vlm:
        print(vlm_score_remote.remote(sids))
    print(analyze_remote.remote(sids))
    print("\nG5 checklist reminders (manual): kill+resume one shard; re-run one shard "
          "and confirm zero new billed VLM calls; LOOK at the contact sheets; "
          "extrapolate $/shard to the full grid before f1.")


@app.local_entrypoint()
def f1(with_vlm: bool = True):
    """FULL Tier-2 grid. Launch exactly once, only after G5 passes."""
    from fidelityfloor.config import load_config

    cfg = load_config()
    n = cfg["tier2"]["n_states_full"]
    sids = list(range(n))
    print(f"Launching {n} shards (max 10 concurrent {GPU} workers)...")
    reports = list(state_shard_remote.map([{"state_id": s} for s in sids]))
    gpu_s = sum(r.get("gpu_wall_s", 0) for r in reports)
    _print_cost("F1 sim", gpu_s)
    bad = [r["state_id"] for r in reports if not r.get("spread_ok")]
    if bad:
        print(f"WARNING: spread check failed for states {bad}")
    print(corrupt_remote.remote(sids))
    if with_vlm:
        print(vlm_score_remote.remote(sids))
    print(analyze_remote.remote(sids))


@app.local_entrypoint()
def analyze(n_states: int = 3):
    """Re-run stats/gates/figures for the first n_states (free, no GPU)."""
    print(analyze_remote.remote(list(range(n_states))))
