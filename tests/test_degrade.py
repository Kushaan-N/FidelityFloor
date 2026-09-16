import io

import numpy as np
import pytest

from fidelityfloor.config import condition_by_id, enumerate_conditions, identity_condition
from fidelityfloor.degrade import corrupt_frame, degrade_actions


def _acts(T=40):
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.1, size=(T, 2))


def test_conditions_enumerate_13(cfg):
    conds = enumerate_conditions(cfg)
    assert len(conds) == 13
    assert conds[0].cid == "identity"


def test_identity_is_bit_identical(cfg):
    """G2: identity transform must be a byte-for-byte no-op."""
    a = _acts()
    out = degrade_actions(a, identity_condition(), np.random.default_rng(1))
    assert np.array_equal(out, a)


def test_miscalibration_systematic_and_deterministic(cfg):
    a = _acts()
    c = condition_by_id(cfg, "miscalibration-2")
    o1 = degrade_actions(a, c, np.random.default_rng(1))
    o2 = degrade_actions(a, c, np.random.default_rng(99))
    assert np.allclose(o1, o2)  # no RNG dependence: systematic
    # norms scale exactly
    assert np.allclose(np.linalg.norm(o1, axis=1),
                       c.pdict["scale"] * np.linalg.norm(a, axis=1))


def test_dyn_noise_zero_mean_and_seeded(cfg):
    a = np.zeros((5000, 2))
    c = condition_by_id(cfg, "dyn_noise-2")
    o = degrade_actions(a, c, np.random.default_rng(7))
    assert abs(o.mean()) < 0.005
    assert np.isclose(o.std(), c.pdict["sigma"], rtol=0.05)
    o2 = degrade_actions(a, c, np.random.default_rng(7))
    assert np.array_equal(o, o2)  # reproducible given the seed


def test_horizon_truncates(cfg):
    a = _acts(40)
    for sev, frac in [("1", 0.75), ("2", 0.5), ("3", 0.25)]:
        c = condition_by_id(cfg, f"horizon-{sev}")
        o = degrade_actions(a, c, np.random.default_rng(0))
        assert len(o) == int(round(frac * 40))
        assert np.array_equal(o, a[: len(o)])


def _dummy_png(w=64, h=48):
    from PIL import Image

    rng = np.random.default_rng(3)
    img = Image.fromarray(rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8))
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def test_corrupt_frame_deterministic_and_graded(cfg):
    png = _dummy_png()
    c1 = condition_by_id(cfg, "obs_corruption-1")
    c3 = condition_by_id(cfg, "obs_corruption-3")
    a = corrupt_frame(png, c1, frame_index=0, seed=42)
    b = corrupt_frame(png, c1, frame_index=0, seed=42)
    assert a == b  # bit-identical on re-run (cache/idempotence requirement)
    assert corrupt_frame(png, c1, frame_index=1, seed=42) != a  # flicker varies by frame

    from PIL import Image

    orig = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=float)
    d1 = np.abs(np.asarray(Image.open(io.BytesIO(a)).convert("RGB"), dtype=float) - orig).mean()
    a3 = corrupt_frame(png, c3, frame_index=0, seed=42)
    d3 = np.abs(np.asarray(Image.open(io.BytesIO(a3)).convert("RGB"), dtype=float) - orig).mean()
    assert d3 > d1 > 0  # severity is graded (G3 analog in pixel space)


def test_corrupt_frame_rejects_wrong_axis(cfg):
    with pytest.raises(ValueError):
        corrupt_frame(_dummy_png(), identity_condition(), 0, 0)
