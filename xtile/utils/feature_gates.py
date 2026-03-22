"""Feature gates for experimentally unsafe runtime paths."""

from __future__ import annotations

import os

_TRUTHY_VALUES = {"1", "true", "yes", "on"}

MULTIPROCESS_DEVICE_COLLECTIVES_ENV = (
    "XTILE_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES"
)
FORCE_MULTIPROCESS_TRANSPORT_ENV = "XTILE_FORCE_MULTIPROCESS_TRANSPORT"
_MULTIPROCESS_TRANSPORT_VALUES = {
    "ctypes_ipc",
    "pytorch_ipc",
    "peer_access_pointer_exchange",
}
_VALIDATED_MULTIPROCESS_DEVICE_TRANSPORTS = {"ctypes_ipc"}


def multiprocess_device_collectives_enabled() -> bool:
    """Return whether unsafe multiprocess device collectives are explicitly enabled."""
    value = os.getenv(MULTIPROCESS_DEVICE_COLLECTIVES_ENV, "")
    return value.strip().lower() in _TRUTHY_VALUES


def multiprocess_device_collectives_detail(
    *,
    transport_strategy: str | None,
) -> str:
    """Return the current gate explanation for multiprocess device collectives."""
    if not multiprocess_device_collectives_enabled():
        detail = (
            "Multiprocess device collectives are disabled by default because only a "
            "limited 2-GPU experimental matrix has been validated so far; the broader "
            "public/performance contract is not closed yet."
        )
    elif not multiprocess_device_collectives_transport_supported(transport_strategy):
        detail = (
            "Multiprocess device collectives remain transport-sensitive even under the "
            "experimental gate. Real 2-GPU matrix runs currently validate only "
            "transport_strategy='ctypes_ipc'; other transports are not yet safe for "
            "device reduce_scatter."
        )
    else:
        detail = (
            "Multiprocess device collectives are running under an explicit experimental "
            "gate. Current 2-GPU matrix validation passes for "
            "transport_strategy='ctypes_ipc', but the broader public/performance "
            "contract is still not closed."
        )
    if transport_strategy is not None:
        detail += f" Current transport_strategy={transport_strategy!r}."
    detail += (
        f" Set {MULTIPROCESS_DEVICE_COLLECTIVES_ENV}=1 only for controlled "
        "bring-up/debug."
    )
    return detail


def multiprocess_device_remote_access_transport_supported(
    transport_strategy: str | None,
) -> bool:
    """Return whether the transport is validated for Triton device-side remote access."""
    return transport_strategy in _VALIDATED_MULTIPROCESS_DEVICE_TRANSPORTS


def multiprocess_device_remote_access_detail(
    *,
    transport_strategy: str | None,
    operation: str,
) -> str:
    """Return a conservative explanation for Triton device-side remote access."""
    if multiprocess_device_remote_access_transport_supported(transport_strategy):
        detail = (
            f"{operation} relies on Triton device-side remote dereference. "
            "Real 2-GPU minimal remote-load/store diagnostics currently pass for "
            "transport_strategy='ctypes_ipc', but broader world-size/performance "
            "validation is still pending."
        )
    else:
        detail = (
            f"{operation} relies on Triton device-side remote dereference. "
            "Real 2-GPU minimal remote-load/store diagnostics currently validate only "
            "transport_strategy='ctypes_ipc'; other transports are not yet safe."
        )
    if transport_strategy is not None:
        detail += f" Current transport_strategy={transport_strategy!r}."
    return detail


def forced_multiprocess_transport() -> str | None:
    """Return the explicitly forced multiprocess transport, if any.

    The default empty / ``auto`` value means "use the normal fallback chain".
    Any other non-empty value must be one of the known transport strategy names.
    """
    value = os.getenv(FORCE_MULTIPROCESS_TRANSPORT_ENV, "").strip().lower()
    if value in {"", "auto"}:
        return None
    if value not in _MULTIPROCESS_TRANSPORT_VALUES:
        allowed = ", ".join(sorted(_MULTIPROCESS_TRANSPORT_VALUES))
        raise ValueError(
            f"{FORCE_MULTIPROCESS_TRANSPORT_ENV} must be one of "
            f"'auto', {allowed}; got {value!r}"
        )
    return value


def multiprocess_device_collectives_transport_supported(
    transport_strategy: str | None,
) -> bool:
    """Return whether the given transport is currently validated for device collectives."""
    return multiprocess_device_remote_access_transport_supported(transport_strategy)
