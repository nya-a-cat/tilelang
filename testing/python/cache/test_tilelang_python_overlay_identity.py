from __future__ import annotations

from hashlib import sha256

import pytest

from tilelang import _build_identity
from tilelang.cache import kernel_cache as kernel_cache_module


@pytest.fixture(autouse=True)
def _clear_overlay_identity_cache():
    _build_identity.get_python_overlay_stamp.cache_clear()
    yield
    _build_identity.get_python_overlay_stamp.cache_clear()


def test_python_overlay_stamp_is_absent_without_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(_build_identity, "_PYTHON_OVERLAY_IDENTITY_PATH", tmp_path / "missing.json")

    assert _build_identity.get_python_overlay_stamp() is None


def test_python_overlay_stamp_hashes_identity_content(monkeypatch, tmp_path):
    identity = b'{"source_sha":"0123456789abcdef"}\n'
    identity_path = tmp_path / "identity.json"
    identity_path.write_bytes(identity)
    monkeypatch.setattr(_build_identity, "_PYTHON_OVERLAY_IDENTITY_PATH", identity_path)

    assert _build_identity.get_python_overlay_stamp() == sha256(identity).hexdigest()


def test_kernel_cache_base_key_includes_overlay_stamp(monkeypatch):
    monkeypatch.setattr(kernel_cache_module, "get_python_overlay_stamp", lambda: "candidate-stamp")
    monkeypatch.setattr(kernel_cache_module.env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")

    assert kernel_cache_module.KernelCache._get_base_key()["python_overlay"] == "candidate-stamp"
