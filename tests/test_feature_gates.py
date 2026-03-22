"""Tests for runtime feature-gate helpers."""

from __future__ import annotations

import pytest

from xtile.utils.feature_gates import (
    FORCE_MULTIPROCESS_TRANSPORT_ENV,
    forced_multiprocess_transport,
    multiprocess_device_collectives_detail,
    multiprocess_device_collectives_transport_supported,
    multiprocess_device_remote_access_detail,
    multiprocess_device_remote_access_transport_supported,
)


def test_forced_multiprocess_transport_defaults_to_none(monkeypatch) -> None:
    """Empty or auto values should preserve the fallback chain."""
    monkeypatch.delenv(FORCE_MULTIPROCESS_TRANSPORT_ENV, raising=False)
    assert forced_multiprocess_transport() is None

    monkeypatch.setenv(FORCE_MULTIPROCESS_TRANSPORT_ENV, "auto")
    assert forced_multiprocess_transport() is None


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("ctypes_ipc", "ctypes_ipc"),
        ("pytorch_ipc", "pytorch_ipc"),
        ("peer_access_pointer_exchange", "peer_access_pointer_exchange"),
    ],
)
def test_forced_multiprocess_transport_accepts_known_values(
    monkeypatch,
    raw_value: str,
    expected: str,
) -> None:
    """Known transport names should round-trip through the helper."""
    monkeypatch.setenv(FORCE_MULTIPROCESS_TRANSPORT_ENV, raw_value)
    assert forced_multiprocess_transport() == expected


def test_forced_multiprocess_transport_rejects_unknown_value(
    monkeypatch,
) -> None:
    """Unknown transport names must fail fast with a clear error."""
    monkeypatch.setenv(FORCE_MULTIPROCESS_TRANSPORT_ENV, "bogus")
    with pytest.raises(ValueError, match=FORCE_MULTIPROCESS_TRANSPORT_ENV):
        forced_multiprocess_transport()


def test_multiprocess_device_collectives_detail_mentions_controlled_debug() -> None:
    """The gate detail should describe the current conservative policy."""
    detail = multiprocess_device_collectives_detail(
        transport_strategy="ctypes_ipc"
    )
    assert "disabled by default" in detail
    assert "ctypes_ipc" in detail
    assert "controlled" in detail


def test_multiprocess_device_collectives_transport_supported_is_specific() -> None:
    """Only the validated transport should currently be accepted."""
    assert multiprocess_device_collectives_transport_supported("ctypes_ipc") is True
    assert multiprocess_device_collectives_transport_supported("pytorch_ipc") is False
    assert (
        multiprocess_device_collectives_transport_supported(
            "peer_access_pointer_exchange"
        )
        is False
    )


def test_multiprocess_device_remote_access_transport_supported_is_specific() -> None:
    """Minimal Triton remote access must stay aligned with the validated transport set."""
    assert multiprocess_device_remote_access_transport_supported("ctypes_ipc") is True
    assert multiprocess_device_remote_access_transport_supported("pytorch_ipc") is False
    assert (
        multiprocess_device_remote_access_transport_supported(
            "peer_access_pointer_exchange"
        )
        is False
    )


def test_multiprocess_device_remote_access_detail_mentions_operation() -> None:
    """The detail should explain why an unsupported transport is being rejected."""
    detail = multiprocess_device_remote_access_detail(
        transport_strategy="pytorch_ipc",
        operation="xtile.ops.allgather(...)",
    )
    assert "xtile.ops.allgather" in detail
    assert "ctypes_ipc" in detail
    assert "pytorch_ipc" in detail
