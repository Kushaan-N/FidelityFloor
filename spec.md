# FidelityFloor
## How Good Does a World Model Actually Need to Be?
### A controlled fidelity-requirement study using Isaac Sim as a degradable oracle

### Implementation Spec v1 — Modal-first, deadline-driven

(Verbatim copy of the spec handed to the implementing agent, 2026-09-16.
See README.md for the implemented layout; REPRO_LOG.md for deviations.)

> **Positioning.** No model under test, no model's name in the title. The simulator
> itself plays the world model: perfect at first, then degraded along controlled,
> parameterized axes, with downstream decision utility measured at every fidelity
> level. The deliverables are demand curves — utility vs. world-model error, per
> error type — that any researcher's real model can later be placed on as a single
> validation point.

## 0. Instructions for the implementing agent

**0.1** Execution platform is Modal (~10 L40S workers, budget ceiling ~$200 all-in,
preemptible, shard-level idempotence, no reserved instances/region pinning). Unity HPC
L40S is the free fallback for smoke tests; code must be launcher-agnostic.

**0.2** Staged ramp: 1 worker (P0) -> mini-grid dress rehearsal at 2-3 workers taken
through stats and draft figures (P2) -> full 10-wide fan-out (F1) only after the G5
go/no-go passes. Never debug at 10x parallelism; never patch mid-full-grid — fall back
to P2, fix, rerun the rehearsal, relaunch.

**0.3** Tier 2 (action selection) is primary and scored from ground-truth state; Tier 1
(spatial reasoning) runs at reduced scope. Pre-committed; do not invert.

**0.4** Keep REPRO_LOG.md from the first command: pins, workarounds, deviations,
timestamps.

**0.5** Raw-array persistence: per-rollout/per-question .npz with condition ID, seed,
commanded actions, GT trajectory, imagined trajectory, scores, VLM raw responses.
Tables/figures regenerate from raw arrays. Modal Volumes; vol.commit() before return;
per-shard output paths.

**0.6** Determinism and idempotence: seeded everything; deterministic output paths from
config hash; skip-if-valid; .tmp then os.replace(); explicit timeout= on every Modal
function.

**0.7** Calibrate then commit: P0 measures sec/rollout (render and physics-only),
cold-start, VLM $/call + latency, PhysX replay-determinism floor. Recompute the §10
budget and grid sizes before launching at scale.

**0.8** Week-one due diligence (parallel, $0): skim MBPO-line and recent closed-loop
world-model evals; verify nothing does the controlled demand-curve framing; write the
two-paragraph related-work positioning BEFORE the grid. [DONE — docs/related_work.md]

## 1. Thesis

**1.1** The field measures world-model quality constantly but not the demand side: at
what fidelity does imagination start paying for a given decision task, and which error
types destroy its value fastest?

**1.2** Isaac Sim as degradable oracle. GT = pristine execution; imagination = second
rollout from the same state with controlled corruption. Induced error is measurable in
state space — the x-axis of every curve is measured error, not a severity knob.

Four axes: miscalibration (A' = R(theta)*s*A, systematic), dyn_noise (zero-mean
per-step action noise, sigma graded), obs_corruption (post-hoc blur/noise/JPEG/flicker
on rendered imagination frames — reuses existing renders, no re-simulation), horizon
(truncate imagined rollout to a fraction of T). obs_corruption only affects pipelines
that look at imagined frames; state-scored pipelines are blind to it by construction —
that asymmetry is part of the result.

**1.3** Tier 2 (primary): tabletop push-object-to-goal; K=10-12 candidates per state
(expert + graded perturbations + random; spread verified by G4); GT utility from
pristine execution; imagined utility scored (a) from imagined final state, (b) by an
API VLM ranking rendered imagined outcomes. Report Spearman and top-1 regret vs GT
ranking per condition. Tier 1 (reduced): programmatic spatial QA with verifiable
answers; VLM gets base view + imagined views along scripted camera trajectories;
degradations: obs_corruption and viewpoint miscalibration; anchors: no-imagination and
perfect imagination.

**1.4** Headline figures: (1) decision utility vs measured rollout error, per error
type, both scoring modes — where each curve crosses the no-imagination baseline is the
fidelity floor; (2) systematic vs stochastic at matched error magnitude; (3) Tier 1
accuracy vs corruption with both anchors.

**1.5** Non-goals v1: no real world model on the critical path; no training; no agentic
VLM loop; no real robot; one task family per tier; not a leaderboard.

## 2. Modal execution architecture

Container: Isaac Sim via pip in a Modal image (NGC container fallback); pinned; huge
image, build once, iterate via mounted package. Headless EGL verified in-container
(G0). Fixed render settings project-wide chosen at P0 via contact sheet. Assets cached
to a Volume (we avoid cloud assets entirely). Sharding: Tier 2 by initial state (~30
shards, 10 wide); Tier 1 by (scene x condition); obs_corruption generated post-hoc on
CPU; GT computed once per state; VLM calls in a separate async pipeline with responses
cached by (image hash, prompt hash). Hygiene: explicit timeouts, retries with backoff,
vol.commit() before return, budget alarm at $150 spend.

## 3. Ground truth, imagination, determinism floor

**3.1** GT is the recorded pristine execution; zero-degradation imagination is a fresh
rollout, same state, same seed. The divergence is the noise floor — measured at P0 over
>=20 states; must be <10% of the mildest condition's induced error, else force CPU
physics and eat the slowdown. Reported on every figure.

**3.2** Induced error per condition: mean state-space distance between imagined and GT
trajectories (object + EE at matched timesteps, and at horizon). G3 requires it
monotone in each severity knob.

## 4. Experiment sequence

P0 (G0): single-worker smoke — container builds, renders, one rollout, one degradation,
one VLM call; measure §0.7. P1 (G1): zero-degradation pilot, 5 states. P2 (G5): dress
rehearsal, 3 states x 13 conditions x both scoring modes, 2-3 workers, through stats
and draft figures. F1: full grid ~30 states x K~12 x 13 conditions, both modes. F2:
Tier 1 ~100 questions x ~6 conditions + anchors. F3 (post-deadline, optional): one real
model as a validation point.

Deadline tiers if P0 costs exceed plan: Tier 1 halved -> severities 3->2 -> states
30->20. Never cut identity, the determinism floor, or the matched-error comparison.

## 6. Gates

G0 single-worker end-to-end. G1 perfect-imagination ceiling: identity Spearman >0.95,
top-1 regret ~0 state-scored; VLM-scored >0.7 (scorer noise measured and reported).
G2 identity is a no-op (bit-identical transforms; within floor for rollouts).
G3 monotone induced error per knob. G4 candidate spread (IQR > 30% of expert progress)
+ contact sheets — LOOK at them. G5 dress-rehearsal go/no-go: all shards valid and
parseable; one shard deliberately killed and resumed idempotently; draft figures sane;
$/shard extrapolates within budget (Isaac <= ~$60) and wall-clock; VLM cache verified
(rerun => zero new billed calls); same-seed shard rerun reproduces within floor.

## 7. Statistics

Paired everywhere (same states, candidates, seeds across conditions). Bootstrap CIs
(2000 resamples) over states/questions respecting pairing. Matched-error comparison:
(miscalibration, dyn_noise) severity pairs with equal measured error (interpolate),
utility difference tested paired — figure 2 gets the most care. Two VLM sampling
repeats on a subset to bound scorer stochasticity. Regression of utility on measured
error pooled across axes; residuals by axis.

## 8. Pre-registration — see docs/preregistration.md (committed before F1).

## 9. Deliverables and outreach

Artifacts: demand curves (figs 1-3), harness repo (degradable oracle + adapter point
where a real model slots in), raw arrays, REPRO_LOG.md. Outreach after F1: Yuncong Yang
(UMass Embodied AGI — MindJourney/SyncWorld framing; offer SyncWorld as F3); world-model
groups (DINO-WM community, Embodied World Models workshop). Lead with figure 2.

## 10. Budget (recompute at P0)

P0 ~$8; P2 ~$5-8 (x2 runs); F1 ~$45 (20-25 GPU-hr, 10-wide ~2.5h wall); F2 ~$10; VLM
$25-50; rerun reserve ~$30. Total ~$125-155; alarm $150/160; hard ceiling $200.
Wall-clock: P0 half a day; P2 ~1 hr/run; F1 an afternoon or overnight; figures minutes.

## 11. Known failure modes

Isaac-in-Modal tail risk (pip path first, no runtime assets, G0 on one worker, Unity
fallback). PhysX non-determinism (measured first; CPU physics accepted). Render noise
read as corruption (settings fixed at P0). VLM scorer bottleneck (ceiling measured;
state-scored mode carries the headline). Confounded axes (zero-mean vs systematic;
verify error signatures differ in P1). Candidate set too easy/hard (G4 before grid).
VLM cache misses (content-hash keys from day one). Modal spend drift (alarm, timeouts,
reserve). Related-work collision (checked — docs/related_work.md).

## 12. Naming

FidelityFloor provisional; alternatives DemandCurve, OracleDecay, HowGoodEnough. Check
for collisions before anything public.
