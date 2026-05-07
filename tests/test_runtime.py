from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from rag_flow.runtime import get_torch_device


def _fake_torch(monkeypatch, *, cuda_available: bool) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda_available)),
    )


def test_get_torch_device_can_force_cpu(monkeypatch):
    _fake_torch(monkeypatch, cuda_available=True)

    assert get_torch_device(preferred="cpu") == "cpu"


def test_get_torch_device_can_force_cuda(monkeypatch):
    _fake_torch(monkeypatch, cuda_available=True)

    assert get_torch_device(preferred="cuda") == "cuda"


def test_get_torch_device_errors_when_cuda_is_requested_without_gpu(monkeypatch):
    _fake_torch(monkeypatch, cuda_available=False)

    with pytest.raises(RuntimeError, match="configured to use CUDA"):
        get_torch_device(preferred="cuda")


def test_get_torch_device_auto_falls_back_to_cpu(monkeypatch):
    _fake_torch(monkeypatch, cuda_available=False)

    assert get_torch_device(preferred="auto") == "cpu"
