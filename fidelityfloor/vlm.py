"""API VLM client with content-addressed response caching (spec §2.2/§11).

Cache key = sha256(model | prompt | all image bytes). Re-running any shard or the
whole grid produces zero new billed calls (gate G5.5). The scorer's own noise is
a measured property: score_repeats_subset states get a second, cache-bypassing
sample tagged repeat=1.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path

from .config import vlm_cache_dir

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}

SCORE_PROMPT = (
    "You are scoring the OUTCOME of a tabletop manipulation attempt. The task was: "
    "push the red cube onto the flat green goal disk on the table. The image shows "
    "the final state after the attempt. Score how successful the attempt was from 0 "
    "to 10: 10 = the red cube is centered on the green disk; 0 = the cube is very far "
    "from the disk or was pushed in the wrong direction. Judge only cube-to-goal "
    "proximity. Respond with JSON."
)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "reason": {"type": "string"}},
    "required": ["answer", "reason"],
    "additionalProperties": False,
}


def _cache_key(model: str, prompt: str, images: list[bytes], repeat: int = 0) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(prompt.encode())
    for im in images:
        h.update(hashlib.sha256(im).digest())
    h.update(str(repeat).encode())
    return h.hexdigest()


class VLMClient:
    def __init__(self, cfg: dict, cache_dir: Path | None = None):
        self.model = cfg["vlm"]["model"]
        self.max_tokens = cfg["vlm"]["max_tokens"]
        self.cache_dir = Path(cache_dir) if cache_dir else vlm_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self.n_billed_calls = 0  # calls that actually hit the API this process

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(max_retries=5)
        return self._client

    # ------------------------------------------------------------------ core
    def _call_json(self, prompt: str, images: list[bytes], schema: dict, repeat: int) -> dict:
        key = _cache_key(self.model, prompt, images, repeat)
        cpath = self.cache_dir / f"{key}.json"
        if cpath.exists():
            try:
                return json.loads(cpath.read_text())
            except Exception:
                pass  # corrupt cache entry -> refetch

        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(im).decode(),
                },
            }
            for im in images
        ]
        content.append({"type": "text", "text": prompt})

        t0 = time.time()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": content}],
        )
        self.n_billed_calls += 1
        text = next(b.text for b in resp.content if b.type == "text")
        out = json.loads(text)
        out["_meta"] = {
            "model": self.model,
            "latency_s": round(time.time() - t0, 3),
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "stop_reason": resp.stop_reason,
        }
        from .io_utils import atomic_write_json

        atomic_write_json(cpath, out)
        return out

    # ------------------------------------------------------------ public API
    def score_outcome(self, final_frame_png: bytes, repeat: int = 0) -> dict:
        """Tier 2: 0-10 success score for one imagined final frame."""
        return self._call_json(SCORE_PROMPT, [final_frame_png], SCORE_SCHEMA, repeat)

    def answer_question(self, question: str, images: list[bytes], choices: list[str],
                        repeat: int = 0) -> dict:
        """Tier 1: multiple-choice spatial QA over base + imagined views."""
        prompt = (
            "You are answering a spatial question about a tabletop scene. The first "
            "image is the base view. Any additional images are extra views of the SAME "
            "scene from different camera angles to help you.\n"
            f"Question: {question}\n"
            f"Answer with exactly one of: {', '.join(choices)}. Respond with JSON."
        )
        return self._call_json(prompt, images, ANSWER_SCHEMA, repeat)


def have_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
