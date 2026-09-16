# Related work & positioning (spec §0.8) — written 2026-09-16, BEFORE the grid

## Collision check verdict

**No single work does what FidelityFloor proposes**, but the neighborhood is
crowded and three close neighbors appeared in the last ~90 days. The defensible
novelty is the *conjunction* of: (i) causal, per-error-type demand curves with
error magnitude **measured** on the x-axis; (ii) the systematic-vs-stochastic
comparison at matched error magnitude (no prior work found); (iii) a VLM in the
decision loop consuming degraded imagined frames. Drop any leg and the study
collapses into an existing paper.

Must-cite differentiations:

- **World-in-World** (Wang et al., ICLR 2026, arXiv:2510.18135) — established
  closed-loop utility as the metric; evidence is *correlational across model
  zoos*. We are the causal complement: hold everything fixed, turn one knob.
  Frame as "the dose-response function they couldn't measure," never as
  "utility > pixels" (that would read derivative).
- **WorldModelGym** (Reka, 2026) — uses candidate-ranking normalized regret on
  real frozen models. **We adopt their normalized-regret metric verbatim**
  (`scoring.normalized_regret`) for comparability; the metric is not ours to
  claim.
- **WorldSimProbe** (arXiv:2608.09298) — controlled *probing of learned* models
  with a VLM judge; does not dial error synthetically or map error→utility.
  Pre-empt explicitly: "diagnoses learned models' error profiles; we dial error."
- **Palenicek et al.** (ICLR 2023, arXiv:2303.03955 / 2412.20537) — the closest
  classical ancestor: oracle dynamics with interpolated accuracy. Cite as direct
  methodological lineage in state space; we extend to visual world models,
  typed errors, and VLM consumers.
- **Bench-C / VLM-RobustBench** — accuracy-vs-corruption curves for *static* VLM
  QA exist; our obs_corruption novelty is only that corruption lives inside an
  imagined rollout used for a decision. Say so explicitly.
- **Decision-centric position paper** (arXiv:2606.15032) — argues for exactly
  this evaluation philosophy with no experiments; we operationalize it.

## Two-paragraph positioning (paper-ready draft)

The question of how much model error planning can tolerate is as old as
model-based RL: the simulation lemma (Kearns & Singh, 2002) bounds value loss by
one-step model error, MBPO's analysis (Janner et al., 2019) converts empirical
model-generalization estimates into safe rollout horizons, and value-aware and
value-equivalent model learning (Farahmand et al., 2017; Grimm et al., 2020)
formalize the observation — made empirical by the objective-mismatch studies of
Lambert et al. (2020) — that likelihood-style model error is only loosely coupled
to control performance. Empirical follow-ups characterized compounding rollout
error (Lambert et al., 2022; Talvitie, 2017), its corrosive effect on Dyna-style
value updates (Jafferjee et al., 2020; Abbas et al., 2020), and, closest to our
design, Palenicek et al. (2023; 2024) used oracle dynamics models with controlled
accuracy to show that value-expansion methods extract surprisingly little utility
from better models. This literature, however, operates almost entirely in
low-dimensional state space, treats model error as a scalar nuisance rather than
a typed, independently controllable quantity, and predates decision-makers that
consume rendered observations. We port the oracle-controlled-accuracy
experimental design to visual world models, decomposing error into systematic
action miscalibration, stochastic dynamics noise, observation corruption, and
horizon truncation, and measuring each on a common scale.

A rapidly consolidating evaluation literature argues that world models should be
judged by embodied utility rather than pixel fidelity: World-in-World (2025)
benchmarks generative world models in closed loop and finds visual quality does
not guarantee task success; WorldArena (2026), WorldSimBench (2025), and VP2
(Tian et al., 2023) report that perceptual metrics correlate poorly with
downstream control; WorldModelGym (2026) scores frozen world models by normalized
regret on action-sequence ranking; and WorldSimProbe (2026) diagnoses
faithfulness of learned action-conditioned models via controlled input
perturbations. All of these evaluate whatever error profile a given learned model
happens to have, so error magnitude, error type, and model identity remain
confounded — they establish that fidelity metrics and utility decouple, but not
the shape of the function connecting them. In parallel, MindJourney (2025) and
successors (ViSA, 2025; World2VLM, 2026) place world-model imagination inside a
VLM's spatial-reasoning loop, with early evidence that current world models form
an "information bottleneck," yet none varies imagination fidelity as an
independent variable. We close this gap: using a simulator as a degradable
oracle, we causally trace demand curves — decision utility (action-candidate
ranking from state and from VLM-scored rendered frames, plus VLM spatial QA on
imagined views) as a function of measured world-model error, per error type —
including, to our knowledge, the first matched-magnitude comparison of systematic
versus stochastic world-model error for visually grounded, VLM-in-the-loop
decision-making, thereby operationalizing the decision-centric evaluation agenda
recently argued for on position grounds (Yu et al., 2026).

## Full reference list

See the due-diligence sweep (2026-09-16) — 25+ works checked across the recent
world-model-evaluation line, the imagination-assisted-VLM line, and the classical
model-error-vs-planning literature. Key arXiv ids: 2510.18135 (World-in-World),
2509.22642 (WoW), 2602.08971 / 2605.17912 (WorldArena 1/2), 2606.15032 (position),
2608.09298 (WorldSimProbe), 2607.07196 (Admissibility), 2410.18072 (WorldSimBench),
2304.13723 (VP2), 2405.05941 (SIMPLER), 2609.09155 (SyncWorld), 2411.04983
(DINO-WM), 2507.12508 (MindJourney), 2512.05809 (ViSA), 2604.26934 (World2VLM),
1906.08253 (MBPO), 2002.04523 (objective mismatch), 2011.03506 (value equivalence),
2203.09637 (compounding errors), 2303.03955 / 2412.20537 (Palenicek), 2006.04363
(hallucinated value), 2504.01766, 2207.10821 (Rethinking Sim2Real).
