"""Regression tests for the canonical DINO-X hu16 PNG encoding.

Pins the encode/decode round-trip so the latent 10x scale inconsistency found in
the external adversarial review (Gemini red-team, 2026-09-13) cannot silently
regress. The canonical encoding is::

    encode:  u16 = round(clip(HU, -1000, 4000) * 10) + 32768   (clamped to [0, 65535])
    decode:  HU  = (u16 - 32768) * 0.1

The decode side lives in zoo/data.py, zoo/predict.py, eval_external.py,
finetune_lora.py, phase5_big_run.py, evaluate_panorgan.py, phase5_monitor.py and
is frozen (the shipped model was trained/validated with it). The encode side lives
in scripts/preprocessing/phase2_preprocess_{nifti,lidc_idri}.py.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, rel: str):
    """Import a script module by path (scripts/ is not a package).

    Registers in sys.modules before exec so @dataclass decorators (which resolve
    cls.__module__ via sys.modules) work under importlib loading on Python 3.14.
    """
    import sys
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / rel)
    assert spec is not None and spec.loader is not None, f"cannot load {rel}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The two preprocessing encoders that were fixed to the canonical scale.
ENCODERS = [
    "scripts/preprocessing/phase2_preprocess_nifti.py",
    "scripts/preprocessing/phase2_preprocess_lidc_idri.py",
]


def _canonical_decode(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.float32) - 32768.0) * 0.1


@pytest.mark.parametrize("rel", ENCODERS)
def test_encode_decode_roundtrip(rel):
    """encode -> decode recovers HU to within one decimal of precision."""
    mod = _load_module("enc", rel)
    hu = np.array([-1000.0, -500.0, -100.0, -30.0, 0.0, 40.0, 100.0,
                   500.0, 1000.0, 3000.0, 3276.7], dtype=np.float32)
    rt = _canonical_decode(mod.hu_to_u16(hu))
    assert np.allclose(rt, hu, atol=0.1), f"{rel}: round-trip mismatch {rt} vs {hu}"


@pytest.mark.parametrize("rel", ENCODERS)
def test_encode_no_uint16_overflow(rel):
    """The +4000 clip edge must saturate, not wrap past uint16 max."""
    mod = _load_module("enc", rel)
    u = mod.hu_to_u16(np.array([4000.0], dtype=np.float32))
    assert u.dtype == np.uint16
    assert int(u[0]) == 65535, f"{rel}: +4000 HU should saturate at 65535, got {u[0]}"
    # And must not have wrapped to a small value (the original Gemini finding).
    assert int(u[0]) > 60000


@pytest.mark.parametrize("rel", ENCODERS)
def test_encode_uses_x10_scale(rel):
    """HU=0 -> 32768; HU=1 -> 32778 (x10), NOT 32769 (x1). Guards the regression."""
    mod = _load_module("enc", rel)
    assert int(mod.hu_to_u16(np.array([0.0], dtype=np.float32))[0]) == 32768
    assert int(mod.hu_to_u16(np.array([1.0], dtype=np.float32))[0]) == 32778, (
        f"{rel}: expected x10 scale (1 HU -> +10), got x1 (1 HU -> +1)"
    )


def test_zoo_data_decode_matches_canonical():
    """zoo/data.py's windowed-load decode must be the canonical (u16-32768)*0.1."""
    import inspect
    import zoo.data as zd
    src = inspect.getsource(zd)
    assert "(arr - 32768.0) * 0.1" in src, "zoo/data.py decode drifted from canonical"


def test_zoo_predict_decode_matches_canonical():
    import inspect
    import zoo.predict as zp
    src = inspect.getsource(zp)
    assert "(arr.astype(np.float32) - 32768.0) * 0.1" in src, (
        "zoo/predict.py hu16_png decode drifted from canonical"
    )
