"""Pytest entry point for the LAV-Net self-check script.

`tests/test_lavnet.py` is the runnable self-check (23 checks: smoke, training
correctness, primitive ablation). This wrapper makes `pytest -q` execute it as a
single test and fail on any check.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "test_lavnet.py"


@pytest.fixture(scope="module")
def selfcheck():
    spec = importlib.util.spec_from_file_location("lavnet_selfcheck", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_lavnet_selfcheck_passes(selfcheck, monkeypatch):
    monkeypatch.chdir(ROOT)          # the script adds "." to sys.path
    assert selfcheck.main() == 0, f"failed checks: {selfcheck.FAIL}"
