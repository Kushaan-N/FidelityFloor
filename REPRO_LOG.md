# REPRO_LOG — FidelityFloor

All times US/Eastern. Every version pin, workaround, and deviation from `spec.md` goes here.

## 2026-09-16

- **15:0x** Repo initialized (empty). Building per spec v1.
- **Environment**: Unity HPC login node (dev box only — no GPU work here). Scratch workspace
  allocated: `/scratch4/workspace/knaskar_umass_edu-fidelityfloor` (30 days, extendable).
  Dev venv at `$WS/venv` (Python 3.12.3, modal 1.5.5, numpy, scipy, pyyaml, pytest).
- **Execution platform**: Modal, GPU `L40S`, preemptible defaults, no region pinning.
  Modal token NOT yet configured (needs interactive `modal token new` by the user).

### Version pins (targets — confirm at P0 and update here)

| Component | Pin | Notes |
|---|---|---|
| Isaac Sim | `isaacsim[all,extscache]==5.0.0` (pip, `pypi.nvidia.com` extra index) | Python 3.11 image. `extscache` bakes extensions into the image so workers don't download at runtime. |
| Isaac Lab | **not used** (deviation, see below) | |
| Python (container) | 3.11 | Isaac Sim 5.0 requirement |
| PhysX | CPU solver (TGS), GPU dynamics disabled | Deliberate: determinism first (§3.1). Scene is 2 rigid bodies; GPU is for RTX rendering only. |
| VLM | `claude-opus-4-8` via Anthropic API (`anthropic` SDK) | Configurable in `configs/conditions.yaml`. Structured JSON output. Responses cached by content hash. |
| Render | 640×480, RTX real-time (`RaytracedLighting`), fixed in `configs/render.yaml` | Final settings chosen at P0 by contact-sheet inspection (G4); update this row then. |

### Deviations from spec

1. **No Isaac Lab.** Spec §2.1/§5 lists Isaac Lab; the v1 scene (table + cube + velocity-
   controlled cylindrical pusher + goal marker + one camera) needs only the Isaac Sim Core
   API. Dropping Isaac Lab removes a large dependency layer from the tail-risk item
   (container build) with no impact on the science: the "EE" is the pusher. A Franka arm
   is a post-deadline extension.
2. **No `add_default_ground_plane()` / Nucleus assets.** The default ground plane pulls
   assets from NVIDIA's cloud. We build the table as a primitive `FixedCuboid` and use a
   `DistantLight` + `DomeLight` created via USD API, so the workers have **zero** runtime
   asset downloads (stronger than the spec's asset-caching mitigation).
3. **Pusher instead of robot arm** (see pin table note). Actions are planar velocity
   commands (vx, vy) at 10 Hz applied to a gravity-disabled dynamic cylinder; the cube
   responds through PhysX contact. All four degradation axes act exactly as specified.

### Open items / to measure at P0 (§0.7)

- [ ] Isaac sec/rollout with rendering; sec/rollout physics-only
- [ ] Container cold-start after image is built
- [ ] VLM $/call and latency (measured from usage fields)
- [ ] PhysX replay-determinism floor over ≥20 states (CPU solver expected ~exact;
      report the measured number regardless)
- [ ] Recompute §10 budget + grid sizes from the above; record here.

## 2026-09-16 (cont.) — P0 debugging

- **P0 run 1 FAILED** (`ERROR_DEVICE_LOST` ~90 s after `app ready`, during first RTX
  render). Root cause via `modal run modal_app.py::diag`: our image baked an NVIDIA
  Vulkan ICD at `/usr/share/vulkan/icd.d/nvidia_icd.json` while Modal injects the real
  one at `/etc/vulkan/icd.d/nvidia_icd.json` — the same L40S enumerated as two Vulkan
  devices (identical deviceUUID), which Isaac explicitly flags as crash-inducing.
  Fix: late image layer removes our ICD + all mesa software ICDs and pins
  `VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json`.
- Container facts (L40S worker): driver **580.95.05**, Vulkan API 1.4.312. Isaac Sim
  5.0.0 pip image built in ~200 s (isaacsim layer), SimulationApp booted headless in
  ~29 s. Warp logged `CUDA error 36` at init (watching — physics is CPU anyway).
- Anthropic key not available yet; VLM secret made optional (functions attach it only
  if the Modal secret exists; `vlm_smoke` returns "skipped"). Free-VLM decision needed
  before P2's VLM pass.
