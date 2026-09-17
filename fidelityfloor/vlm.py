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

DIST_SCHEMA = {
    "type": "object",
    "properties": {
        "distance_cm": {"type": "number", "minimum": 0, "maximum": 200},
        "reason": {"type": "string"},
    },
    "required": ["distance_cm", "reason"],
    "additionalProperties": False,
}

DIST_PROMPT = (
    "The image shows a tabletop after a manipulation attempt: a red cube, a blue "
    "cylindrical pusher, and a flat green goal disk (6 cm radius) on a 120 cm square "
    "table. Estimate the distance in centimeters between the CENTER of the red cube "
    "and the CENTER of the green disk. Use the known sizes (cube is 5 cm wide, disk "
    "is 12 cm across, table is 120 cm across) to calibrate your estimate. If the cube "
    "sits on the disk, the distance is small (0-6). Respond with JSON."
)

RANK_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1, "maximum": 12},
            "minItems": 12,
            "maxItems": 12,
        },
        "reason": {"type": "string"},
    },
    "required": ["ranking", "reason"],
    "additionalProperties": False,
}

RANK_PROMPT = (
    "You are shown 12 images, in order (image 1 through image 12). Each shows the "
    "FINAL state of a different attempt at the same task: push the red cube onto the "
    "flat green goal disk. Rank ALL 12 attempts from most successful to least "
    "successful, judging ONLY how close the red cube ended up to the green disk. "
    "Return `ranking` as the image numbers (1-12) ordered best first, worst last — "
    "every number 1-12 exactly once. Respond with JSON."
)


def _cache_key(model: str, prompt: str, images: list[bytes], repeat: int = 0) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(prompt.encode())
    for im in images:
        h.update(hashlib.sha256(im).digest())
    h.update(str(repeat).encode())
    return h.hexdigest()


class VLMClient:
    """Provider-agnostic scorer. cfg['vlm']['provider']: 'anthropic' (default)
    or 'gemini' (free-tier friendly; key from Google AI Studio as GEMINI_API_KEY).
    The cache key includes the model name, so switching providers never mixes
    or invalidates cached responses."""

    def __init__(self, cfg: dict, cache_dir: Path | None = None):
        self.provider = cfg["vlm"].get("provider", "anthropic")
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

        t0 = time.time()
        if self.provider == "gemini":
            out = self._call_gemini(prompt, images, schema)
        else:
            out = self._call_anthropic(prompt, images, schema)
        self.n_billed_calls += 1
        out.setdefault("_meta", {})
        out["_meta"]["model"] = self.model
        out["_meta"]["latency_s"] = round(time.time() - t0, 3)
        from .io_utils import atomic_write_json

        atomic_write_json(cpath, out)
        return out

    def _call_anthropic(self, prompt: str, images: list[bytes], schema: dict) -> dict:
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
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": content}],
        )
        text = next(b.text for b in resp.content if b.type == "text")
        out = json.loads(text)
        out["_meta"] = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "stop_reason": resp.stop_reason,
        }
        return out

    def _call_gemini(self, prompt: str, images: list[bytes], schema: dict) -> dict:
        """Gemini REST (generateContent) with JSON schema output. Retries 429/5xx
        with backoff — the free tier is ~10 RPM, and the content-hash cache means
        every response is only ever paid for (or waited for) once."""
        import urllib.error
        import urllib.request

        api_key = os.environ["GEMINI_API_KEY"]
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent")
        parts = [{"inline_data": {"mime_type": "image/png",
                                  "data": base64.standard_b64encode(im).decode()}}
                 for im in images]
        parts.append({"text": prompt})
        body = json.dumps({
            "contents": [{"parts": parts}],
            "generationConfig": {
                # Gemini 3.x spends output tokens on thinking — generous headroom
                "maxOutputTokens": max(2048, self.max_tokens),
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
            },
        }).encode()
        last_err = None
        rate_waits = 0
        attempt = 0
        while attempt < 6:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json",
                         "x-goog-api-key": api_key})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    resp = json.loads(r.read().decode())
                out = self._parse_gemini_json(resp)
                out["_meta"] = {"usage": resp.get("usageMetadata", {})}
                return out
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}: {e.read()[:300]}"
                if e.code == 429:
                    # per-minute quota: wait it out (up to ~20 min) without
                    # burning generation-retry attempts; daily quota still
                    # fails after that with a clear error
                    rate_waits += 1
                    if rate_waits > 20:
                        raise RuntimeError(
                            f"Gemini quota exhausted (likely daily cap): {last_err}"
                        ) from e
                    time.sleep(62)
                    continue
                if e.code in (500, 503):
                    attempt += 1
                    time.sleep(min(60, 5 * 2 ** attempt))
                    continue
                raise RuntimeError(f"Gemini call failed: {last_err}") from e
            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                # malformed/truncated model output — retry the generation
                last_err = f"parse failure: {e}; resp={str(resp)[:400]}"
                attempt += 1
                time.sleep(2)
                continue
        raise RuntimeError(f"Gemini call failed after retries: {last_err}")

    @staticmethod
    def _parse_gemini_json(resp: dict) -> dict:
        """Extract the JSON object from a Gemini response, tolerating thought
        parts, markdown fences, and prose around the object."""
        import re

        cand = resp["candidates"][0]
        parts = cand.get("content", {}).get("parts", [])
        texts = [p["text"] for p in parts if p.get("text") and not p.get("thought")]
        if not texts:
            raise ValueError(f"no text parts (finishReason={cand.get('finishReason')})")
        text = texts[-1].strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                raise
            return json.loads(m.group(0))

    # ------------------------------------------------------------ public API
    def score_outcome(self, final_frame_png: bytes, repeat: int = 0) -> dict:
        """Tier 2: 0-10 success score for one imagined final frame."""
        return self._call_json(SCORE_PROMPT, [final_frame_png], SCORE_SCHEMA, repeat)

    def estimate_distance(self, final_frame_png: bytes, repeat: int = 0) -> dict:
        """Tier 2 scorer variant A: metric cube-to-goal distance estimate (cm).
        Rank candidates by ascending distance."""
        return self._call_json(DIST_PROMPT, [final_frame_png], DIST_SCHEMA, repeat)

    def rank_outcomes(self, final_frames: list[bytes], repeat: int = 0) -> dict:
        """Tier 2 scorer variant B: one call ranking all K final frames."""
        out = self._call_json(RANK_PROMPT, list(final_frames), RANK_SCHEMA, repeat)
        r = out.get("ranking", [])
        if sorted(r) != list(range(1, len(final_frames) + 1)):
            raise ValueError(f"invalid ranking permutation: {r}")
        return out

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


def have_api_key(cfg: dict | None = None) -> bool:
    provider = (cfg or {}).get("vlm", {}).get("provider", "anthropic")
    var = "GEMINI_API_KEY" if provider == "gemini" else "ANTHROPIC_API_KEY"
    key = os.environ.get(var, "")
    return bool(key) and not key.startswith("placeholder")
