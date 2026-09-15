"""Tests for processing_config.normalize / coerce (DB-free).

Runs under pytest, or standalone: ``python tests/test_processing_config.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import processing_config as pc


def test_dedup_first_preset_shape():
    out = pc.normalize({"preset": "dedup_first"})
    assert out["preset"] == "dedup_first"
    f = out["features"]
    assert f["thumbnail"] is True
    assert f["sscd_dedup"] is True
    assert f["exif"] is False
    assert f["phash"] is False
    assert f["blur"] is False
    assert f["faces"] is False
    assert f["scene_clip"] is False


def test_sscd_forces_thumbnail():
    # Even if the caller tries to disable thumbnail while SSCD is on, it stays on.
    out = pc.normalize({"features": {"sscd_dedup": True, "thumbnail": False}})
    assert out["features"]["thumbnail"] is True


def test_default_is_full_all_on():
    out = pc.normalize({})
    assert out["preset"] == "full"
    assert all(out["features"][k] for k in pc.FEATURE_KEYS)


def test_custom_label_when_diverging():
    # Full preset minus blur is not a named preset -> custom.
    out = pc.normalize({"preset": "full", "features": {"blur": False}})
    assert out["preset"] == "custom"
    assert out["features"]["blur"] is False


def test_sparse_update_preserves_current_features():
    current = pc.normalize({"preset": "dedup_first"})
    out = pc.normalize({"features": {"faces": True}}, current=current)
    assert out["features"]["faces"] is True
    assert out["features"]["scene_clip"] is False
    assert out["features"]["sscd_dedup"] is True
    assert out["features"]["exif"] is False


def test_unknown_preset_raises():
    try:
        pc.normalize({"preset": "nope"})
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown preset")


def test_unknown_feature_key_raises():
    try:
        pc.normalize({"features": {"bogus": True}})
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown feature key")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
