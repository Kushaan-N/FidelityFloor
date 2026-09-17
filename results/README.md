# Results snapshot — F1 Tier-2 grid, state-scored (2026-09-17)

Committed snapshot of the headline outputs. Everything here regenerates from the
raw arrays (`FF_OUTPUT_ROOT` on scratch / the Modal volume) via
`fidelityfloor/stats.py` + `figures.py`; this copy exists so the results survive
scratch expiry and travel with the repo.

## Figures

| File | What it shows |
|---|---|
| `figures/fig1_state_spearman.png` | Demand curves: ranking accuracy (Spearman vs GT) against measured induced error, per error axis. n=30, bootstrap 95% CIs. |
| `figures/fig1_state_normalized_regret.png` | Same x-axis, normalized top-1 regret. |
| `figures/fig2_matched_error.png` | Systematic (miscalibration) vs stochastic (dyn_noise) at matched measured error. |
| `figures/contact_obs_corruption.png` | G4 contact sheet: obs-corruption severity ladder. |

## Tables

- `tables/condition_summary.json` — compact per-condition metrics (the numbers on the figures)
- `tables/summary_f1_state.json` — full per-(condition, state) rows; figures regenerate from this
- `tables/determinism_floor.json` — §3.1 replay floor over 20 states (exactly 0.0 m)
- `tables/vlm_prototype.json` — scorer bake-off raw outputs (0–10 vs distance vs rank-12)

## Headline numbers

- Identity: ρ = 1.000, regret = 0 (G1 ceiling).
- At matched induced error (miscalibration-2: 0.124 m vs dyn_noise-3: 0.118 m):
  Δρ = 0.016, CI [−0.101, 0.130] (n.s.) — but normalized regret **0.586 vs 0.277**,
  paired Δ = −0.309, CI [−0.444, −0.169]. Bias doesn't scramble the ranking; it
  moves the argmax.
- VLM-scored column pending (scorer validated at ρ = 0.879 on identity; grid
  rescore awaiting API credit).
