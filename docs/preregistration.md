# Pre-registration (spec §8) — committed BEFORE F1 launches

Written 2026-09-16, before any grid data exists. P0–P2 may tune severities (G3
requires monotone induced error) but the predictions below are frozen.

## P1 — Systematic vs stochastic at matched error (Figure 2)

**Prediction:** at matched mean induced state error, `miscalibration` harms
candidate ranking MORE than `dyn_noise` (lower Spearman, higher normalized
regret).

**Reasoning:** a systematic action transform (rotation + scale) biases *every*
candidate's imagined outcome in a correlated direction, re-ordering candidates —
e.g. a +15° rotated world model makes the −10°-perturbed candidate look like the
best aligned one. Zero-mean noise perturbs each candidate's outcome independently
and averages out across K=12 when ranking; it adds variance, not preference
reversal. Selection is a max operator: correlated bias moves the argmax, noise
mostly doesn't (until it swamps the utility gaps).

**Falsifier:** paired Δ(noise−misc) CI at matched error includes 0 or is
negative. Would imply ranking robustness to bias is higher than believed and
calibration papers are targeting the wrong failure mode at these magnitudes.

## P2 — Horizon shape

**Prediction:** a cliff, not gradual decay. Utility holds near the identity
ceiling while the truncated imagination is still long enough to observe whether a
candidate makes contact and moves the cube goal-ward (~50% of T for our expert
plans), then collapses toward the no-imagination baseline below that — at 25% of
T most candidates are still in the approach phase and all imagined outcomes look
alike.

**Falsifier:** approximately linear decay in fraction, which would suggest partial
rollouts carry smoothly accumulating decision-relevant signal.

## P3 — Tier 1 obs_corruption shape

**Prediction:** accuracy degrades slowly with corruption until object legibility
collapses (VLMs are robust to moderate blur/noise/JPEG), then cliffs to the
no-imagination anchor. Severity 1–2 ≈ identity; severity 3 ≈ base-view-only.

**Falsifier:** monotone-graded decline across all severities, or corrupted
imagined views scoring BELOW the no-imagination anchor (imagination actively
misleading — would be the most interesting outcome).

## P4 — Scoring-mode asymmetry (secondary)

**Prediction:** VLM-scored curves sit strictly below state-scored curves at every
condition (scorer noise caps the ceiling — measured at G1), but their *shape*
across the action-space axes matches. obs_corruption harms only the VLM-scored
mode, by construction; we predict measurable harm at severity 2+.

## Metrics locked

Primary: Spearman ρ vs GT ranking; normalized regret (WorldModelGym-compatible).
X-axis: mean matched-step state error (cube+pusher). Pairing unit: initial state.
Bootstrap: 2000 resamples. All as implemented in `stats.py`/`scoring.py` at the
commit that adds this file.
