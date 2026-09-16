import json

import numpy as np
import pytest

from fidelityfloor import io_utils
from fidelityfloor.config import config_hash, load_config, stable_hash


def _fake_rollout(T=5):
    return {
        "cube_pos": np.zeros((T, 3)),
        "cube_quat": np.tile([1.0, 0, 0, 0], (T, 1)),
        "pusher_pos": np.ones((T, 3)),
        "actions": np.zeros((T - 1, 2)),
        "frames": [b"\x89PNG-fake-1", b"\x89PNG-fake-2"],
        "frame_steps": [0, 4],
        "meta": {"condition": "identity", "seed": 1},
    }


def test_save_load_roundtrip(tmp_path):
    d = tmp_path / "roll"
    io_utils.save_rollout(d, _fake_rollout())
    assert io_utils.rollout_done(d)
    r = io_utils.load_rollout(d)
    assert r["meta"]["condition"] == "identity"
    assert len(r["frame_paths"]) == 2
    assert np.array_equal(r["pusher_pos"], np.ones((5, 3)))


def test_invalid_npz_detected(tmp_path):
    d = tmp_path / "roll"
    d.mkdir()
    (d / "raw.npz").write_bytes(b"not a zip file")
    assert not io_utils.rollout_done(d)


def test_atomic_write_no_tmp_left(tmp_path):
    p = tmp_path / "a" / "b.json"
    io_utils.atomic_write_json(p, {"x": 1})
    assert json.loads(p.read_text()) == {"x": 1}
    assert list(p.parent.glob("*.tmp*")) == []


def test_config_hash_stability_and_sensitivity():
    cfg = load_config()
    h1 = config_hash(cfg)
    assert h1 == config_hash(load_config())  # stable across loads
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["axes"]["dyn_noise"]["severities"]["2"]["sigma"] = 0.999
    assert config_hash(cfg2) != h1  # sim-relevant change moves outputs
    cfg3 = json.loads(json.dumps(cfg))
    cfg3["vlm"]["model"] = "something-else"
    assert config_hash(cfg3) == h1  # VLM choice does NOT invalidate sim outputs


def test_stable_hash_order_independent():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


# ------------------------------------------------------------------ VLM cache

class _FakeAnthropic:
    """Stands in for anthropic.Anthropic; counts real 'API' calls."""

    calls = 0

    class _Usage:
        input_tokens = 1200
        output_tokens = 30

    def __init__(self):
        self.messages = self

    def create(self, **kw):
        _FakeAnthropic.calls += 1

        class Block:
            type = "text"
            text = json.dumps({"score": 7, "reason": "close to goal"})

        class Resp:
            content = [Block()]
            usage = _FakeAnthropic._Usage()
            stop_reason = "end_turn"

        return Resp()


def test_vlm_cache_zero_rebill(tmp_path, cfg, monkeypatch):
    """G5.5 in miniature: identical (model, prompt, image) never re-bills."""
    from fidelityfloor.vlm import VLMClient

    client = VLMClient(cfg, cache_dir=tmp_path)
    client.provider = "anthropic"  # tests mock the Anthropic client path
    client._client = _FakeAnthropic()
    _FakeAnthropic.calls = 0

    png = b"\x89PNG-image-bytes"
    r1 = client.score_outcome(png)
    assert r1["score"] == 7 and _FakeAnthropic.calls == 1

    # same content -> cache hit, even from a brand-new client instance
    client2 = VLMClient(cfg, cache_dir=tmp_path)
    client2.provider = "anthropic"
    client2._client = _FakeAnthropic()
    r2 = client2.score_outcome(png)
    assert r2["score"] == 7 and _FakeAnthropic.calls == 1
    assert client2.n_billed_calls == 0

    # different image -> new call; repeat tag -> new call (scorer-noise probe)
    client2.score_outcome(b"other-image")
    assert _FakeAnthropic.calls == 2
    client2.score_outcome(png, repeat=1)
    assert _FakeAnthropic.calls == 3


def test_vlm_cache_corrupt_entry_refetched(tmp_path, cfg):
    from fidelityfloor.vlm import VLMClient, _cache_key

    client = VLMClient(cfg, cache_dir=tmp_path)
    client.provider = "anthropic"
    client._client = _FakeAnthropic()
    _FakeAnthropic.calls = 0
    png = b"img"
    from fidelityfloor.vlm import SCORE_PROMPT

    key = _cache_key(cfg["vlm"]["model"], SCORE_PROMPT, [png], 0)
    (tmp_path / f"{key}.json").write_text("{corrupt")
    r = client.score_outcome(png)
    assert r["score"] == 7 and _FakeAnthropic.calls == 1
