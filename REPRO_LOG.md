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

## 2026-09-16 (cont. 2) — P0 run 2 and the Modal/Unity pivot

- **P0 run 2 FAILED identically** (`ERROR_DEVICE_LOST` at first RTX submit, ~174 s).
  The duplicate-ICD warning was gone (fix verified), so that was necessary but not
  sufficient. Remaining fingerprints — Warp `cuGetProcAddress`/`cuDeviceGetUuid`
  "not found" on driver 580, plus Vulkan device-lost on first real graphics submit —
  match Modal's gVisor (nvproxy) sandbox intercepting driver calls, not our code.
  App stopped to halt spend (total GPU burn so far: a few dollars at most).
- **Decision (spec §11 mitigation):** validate the identical code path on Unity's
  L40S partition (bare metal, driver-native — Isaac's supported environment) via
  `run_local.py smoke`. If Unity renders, the Modal issue is platform-level; grid
  options become (a) Unity SLURM array at $0 GPU cost, (b) Modal NGC-container
  variant retry. Isaac venv building at
  `/scratch4/workspace/knaskar_umass_edu-fidelityfloor/venv-isaac` (py3.11.7 module).
- VLM provider switched to **Gemini 2.5 Flash** by default (free-tier key; content-hash
  cache unchanged; Anthropic path retained behind `vlm.provider`).

## 2026-09-16 (cont. 3) — Unity G0 debugging + VLM validated

- **Unity L40S renders Isaac fine** (driver 580.173.02, Ubuntu 24.04) — confirms the
  Modal failure is platform-level (gVisor), not our stack. Warp's `cuDeviceGetUuid`
  warning appears on bare metal too → benign, not a gVisor fingerprint after all.
- **P0 bug 2 (blank frames):** hand-rolled camera quaternion aimed at the sky —
  replaced with `set_camera_view`. **P0 bug 3:** Camera default near-clip is 1.0 m,
  which deleted the whole near scene at close poses (the "table floating in sky"
  artifact) — now `set_clipping_range(0.02, 1e6)`.
- **Render settings frozen (G4/P0):** camera (0.85, 0, 0.80) → origin, focal 18,
  dome 350 / sun 1200 — chosen by inspecting a 3-pose contact sweep; objects
  legible and well saturated.
- **P0 bug 4:** pusher cylinder tipped over from contact torques → PhysX
  `lockedRotAxis=7`. Dynamics changed; pre-fix outputs wiped.
- **Modal lesson:** conditionally-attached secrets make functions crash-loop
  ("2 dependencies but 3 object ids") — secrets lists must be identical in local
  and remote evaluation.
- **VLM path VALIDATED end-to-end:** Gemini free key → Modal secret →
  `gemini-3.6-flash` (2.5-flash is retired for new accounts) with robust JSON
  parsing (thought parts, fences, truncation retry). Synthetic cube-on-goal scored
  10/10, latency 3.5 s, ~1.4k tokens/call.
- Timings (warm cache, Unity): kit boot ~14 s, rollout with render ~1.0 s,
  physics-only ~0.55 s. 3-state determinism floor: exactly 0.0 (CPU PhysX bit-exact).

## 2026-09-17 — P2 complete (sim side); gates G1–G5 (except VLM items)

- P2 grid done on Unity: 3 states x 13 conditions x K=12 (492 rollouts incl. reused
  P1 GT/identity; 108 obs-corruption variants generated post-hoc on CPU).
- **Metric fix:** horizon truncation's imagined trajectory is an identical prefix of
  GT (deterministic sim), so matched-step error is 0 by construction; the x-axis now
  uses final-state error for the horizon axis (`err_xaxis`), matched-step error for
  the action axes. Pre-registration unchanged (it already specified this split).
- **Gates:** G1 PASS (identity rho=1.000, regret=0). G2 PASS. G3 PASS (monotone:
  misc .048/.124/.207; noise .028/.061/.121; horizon .022/.104/.242 m). Floor-vs-
  mildest PASS trivially (floor = 0 exactly). G5.6 PASS (same-seed re-run of a wiped
  shard bit-identical). G5.2 PASS (kill at ~15 s left 11/12 rollouts; resume completed
  exactly the missing one). G4 contact sheet inspected: severity 1-2 legible,
  severity 3 at legibility collapse — as intended.
- Draft figs 1-2 rendered. Early n=3 signal: at matched error (~0.12 m), dyn_noise
  ranks WORSE than miscalibration (paired delta = -0.065, CI [-0.103, -0.039]) —
  OPPOSITE of pre-registered P1. n=3; F1 decides.
- Remaining before F1: VLM pass on P2 (blocked on GEMINI_API_KEY file on Unity
  scratch), then G1-VLM bar (>0.7) + G5.5 cache-zero-rebill check.
- Budget status: Modal spend to date only P0 debugging (~$3-5); all sim compute now
  $0 on Unity gpu-preempt. VLM projected $0-5 total.

## 2026-09-17 (cont.) — F1 sim done; VLM scorer redesigned at the G1 gate

- **F1 sim COMPLETE, $0 GPU**: 4,680 rollouts (30 states x 13 conds x K=12 incl.
  obs-corruption variants); 30/30 states pass the G4 spread check. Queue-wait beaten
  by racing an idempotent 30-task array against two consolidated one-job loops.
- **G1-VLM FAILED as designed-for**: 0-10 absolute scoring ceiling on perfect
  imagination rho=0.619 < 0.7 bar (repeat noise |dScore| 0.70, 45% identical).
  Full-grid pass with the weak prompt stopped mid-flight (~$1.5 spent).
- **Scorer bake-off** (P2 frames, ~$0.5): 0-10 score rho 0.619; distance-in-cm
  estimate rho **0.879** (repeat agreement 0.72-0.97); rank-all-12 in one call
  failed (invalid permutations). Production scorer = distance estimate, ranked by
  -distance_cm. Grid rescore relaunched with the winner.

## 2026-09-17 (cont. 2) — F1 state-scored analysis COMPLETE (n=30)

- All gates PASS at n=30. Full per-condition table in tables/summary_f1_state.json.
- **Headline (fig 2):** at matched induced error (misc-2: 0.124 m vs noise-3: 0.118 m),
  Spearman is statistically indistinguishable (Δ=0.016, CI [-0.101, 0.130]) but
  **normalized top-1 regret more than doubles under systematic bias**: 0.586 vs
  0.277, paired Δ(noise−misc) = -0.309, CI [-0.444, -0.169]. Pre-registered P1
  ("bias hurts more") is falsified on rank correlation, confirmed on selection
  regret — bias doesn't scramble the ranking, it moves the argmax.
- Horizon axis is the most forgiving per meter (rho 0.75 at 0.24 m final-state
  error); mild stochastic noise is nearly free (rho 0.88 at 0.025 m).
- VLM-scored column pending: Gemini prepay credits depleted mid-grid; scorer
  redesigned (distance estimate, minimal thinking, 384px) — needs ~$2 top-up.
