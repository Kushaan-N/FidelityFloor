"""Config loading, condition enumeration, and deterministic hashing/paths."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
# Output root: env var wins (Modal sets it to the Volume mount; Unity sets it to scratch)
OUTPUT_ROOT = Path(os.environ.get("FF_OUTPUT_ROOT", str(REPO_ROOT / "outputs")))


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "conditions.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_render_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "render.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Condition:
    """One cell of the degradation grid."""

    axis: str          # 'identity' | 'miscalibration' | 'dyn_noise' | 'obs_corruption' | 'horizon'
    severity: str      # '0' for identity, else '1'|'2'|'3'
    params: tuple      # sorted (key, value) pairs — hashable

    @property
    def cid(self) -> str:
        return "identity" if self.axis == "identity" else f"{self.axis}-{self.severity}"

    @property
    def pdict(self) -> dict:
        return dict(self.params)

    @property
    def needs_sim(self) -> bool:
        """obs_corruption reuses identity renders — never re-simulated."""
        return self.axis in ("miscalibration", "dyn_noise", "horizon")


def identity_condition() -> Condition:
    return Condition("identity", "0", ())


def enumerate_conditions(cfg: dict) -> list[Condition]:
    """Identity + every (axis, severity). 13 conditions with the default config."""
    conds = [identity_condition()]
    for axis, spec in cfg["axes"].items():
        for sev, params in spec["severities"].items():
            conds.append(Condition(axis, str(sev), tuple(sorted(params.items()))))
    return conds


def condition_by_id(cfg: dict, cid: str) -> Condition:
    for c in enumerate_conditions(cfg):
        if c.cid == cid:
            return c
    raise KeyError(f"unknown condition id: {cid}")


def stable_hash(obj: Any, n: int = 12) -> str:
    """Deterministic short hash of any JSON-serializable object."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:n]


def config_hash(cfg: dict) -> str:
    """Hash of everything that changes simulation results (not VLM/stats settings)."""
    keys = ["seed", "rollout", "workspace", "candidates", "axes"]
    return stable_hash({k: cfg[k] for k in keys})


# ---------------------------------------------------------------- output paths
# Layout (spec §5), all rooted at OUTPUT_ROOT and keyed by config hash so a config
# change can never silently mix with stale results:
#   {root}/{cfg_hash}/gt/{state_id}/raw.npz              + frames/
#   {root}/{cfg_hash}/runs/{condition_id}/{state_id}/raw.npz  + frames/
#   {root}/vlm_cache/{key}.json          (content-addressed; config-independent)
#   {root}/{cfg_hash}/figures/, tables/


def out_root(cfg: dict) -> Path:
    return OUTPUT_ROOT / config_hash(cfg)


def gt_dir(cfg: dict, state_id: int) -> Path:
    return out_root(cfg) / "gt" / f"state_{state_id:04d}"


def run_dir(cfg: dict, cond: Condition, state_id: int) -> Path:
    return out_root(cfg) / "runs" / cond.cid / f"state_{state_id:04d}"


def vlm_cache_dir() -> Path:
    return OUTPUT_ROOT / "vlm_cache"
