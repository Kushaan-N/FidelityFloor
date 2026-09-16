#!/usr/bin/env python
"""Launcher-agnostic local runner — the free Unity L40S fallback (spec §0.1).

Runs the exact same shard functions as modal_app.py, on whatever machine you're
on (requires an Isaac Sim install + GPU for the sim stages; CPU is fine for
corrupt/vlm/analyze). Point outputs somewhere big first:

    export FF_OUTPUT_ROOT=/scratch4/workspace/<you>-fidelityfloor/outputs

Usage:
    python run_local.py smoke
    python run_local.py determinism
    python run_local.py shard --state-id 0 [--conditions identity,miscalibration-2]
    python run_local.py corrupt --n-states 3
    python run_local.py vlm --n-states 3          (needs ANTHROPIC_API_KEY)
    python run_local.py analyze --n-states 3
"""

from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "determinism", "shard", "corrupt",
                                      "vlm", "analyze"])
    ap.add_argument("--state-id", type=int, default=0)
    ap.add_argument("--n-states", type=int, default=3)
    ap.add_argument("--conditions", type=str, default=None,
                    help="comma-separated condition ids (default: all)")
    args = ap.parse_args()

    from fidelityfloor.config import load_config, load_render_config

    cfg, rcfg = load_config(), load_render_config()
    conds = args.conditions.split(",") if args.conditions else None
    sids = list(range(args.n_states))

    if args.stage == "smoke":
        from fidelityfloor.runner import smoke_test

        out = smoke_test(cfg, rcfg)
    elif args.stage == "determinism":
        from fidelityfloor.runner import determinism_pass

        out = {k: v for k, v in determinism_pass(cfg, rcfg).items() if k != "per_state"}
    elif args.stage == "shard":
        from fidelityfloor.runner import run_state_shard

        out = run_state_shard(cfg, rcfg, args.state_id, conds)
    elif args.stage == "corrupt":
        from fidelityfloor.runner import corrupt_frames_pass

        out = corrupt_frames_pass(cfg, sids)
    elif args.stage == "vlm":
        from fidelityfloor.runner import vlm_score_pass

        out = vlm_score_pass(cfg, sids)
        out = {k: v for k, v in out.items() if k != "scores"}
    else:  # analyze
        from fidelityfloor.runner import load_vlm_scores
        from fidelityfloor.stats import save_summary, summarize

        summary = summarize(cfg, sids, load_vlm_scores(cfg))
        p = save_summary(cfg, summary)
        out = {"summary_path": str(p),
               "per_condition_keys": list(summary["per_condition"].keys())}

    print(json.dumps(out, indent=2, default=str))
    from fidelityfloor.envs import close_sim_app

    close_sim_app()


if __name__ == "__main__":
    main()
