# FidelityFloor

**How good does a world model actually need to be?** A controlled
fidelity-requirement study using Isaac Sim as a *degradable oracle*: ground truth
is a pristine rollout, "imagination" is a second rollout from the same state with
controlled, parameterized corruption, and downstream decision utility is measured
at every fidelity level. The deliverables are **demand curves** — utility vs.
*measured* world-model error, per error type. Full design: [`spec.md`](spec.md).

## Layout

```
fidelityfloor/        launcher-agnostic core (no Modal imports; Isaac imported lazily)
  envs.py             Isaac scene: table + cube + velocity-controlled pusher + camera
  oracle.py           GT vs degraded rollouts, determinism floor (§3.1)
  degrade.py          the 4 axes: miscalibration / dyn_noise / obs_corruption / horizon
  candidates.py       Tier-2 initial states + K=12 open-loop candidates + spread check
  questions.py        Tier-1 programmatic spatial QA with verifiable answers
  tier1.py            Tier-1 rendering + QA pass (F2)
  scoring.py          utilities, Spearman, top-1 / normalized regret
  vlm.py              Anthropic VLM client, content-hash response cache
  stats.py            induced error (the x-axis), paired bootstrap, matched-error
  gates.py            automated G1/G2/G3/G4 checks
  figures.py          fig1 demand curves, fig2 matched-error, fig3 tier-1, contact sheets
  runner.py           shard workers shared by both launchers
modal_app.py          Modal image/volume/functions + staged entrypoints
run_local.py          same stages on any machine (Unity L40S fallback)
configs/              conditions.yaml (axes, severities, seeds), render.yaml (frozen at P0)
tests/                CPU-only suite incl. analytic-dynamics pipeline rehearsal
```

## Running (staged ramp — never debug at 10x parallelism)

One-time setup:

```bash
pip install modal && modal token new
modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...
```

Then, in order, each gated on the previous (see spec §4/§6):

```bash
modal run modal_app.py::p0            # G0: 1 worker, end-to-end + timing calibration
modal run modal_app.py::determinism   # §3.1 replay floor, 20 states
modal run modal_app.py::p1            # G1: zero-degradation pilot, 5 states
modal run modal_app.py::p2            # G5: dress rehearsal, 3 states x 13 conditions
modal run modal_app.py::f1            # FULL GRID (once, after G5 passes)
modal run modal_app.py::analyze       # stats + gates + figures (free to re-run)
```

Local tests (no GPU): `pytest tests/ -q`.

Outputs land on the Modal Volume `fidelityfloor-data` under a config hash;
tables and figures regenerate from raw arrays. VLM responses are cached by
content hash — re-running the grid re-bills nothing.
