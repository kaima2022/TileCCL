# SPDX-License-Identifier: Apache-2.0
"""Feature gates for experimentally unsafe runtime paths."""

from __future__ import annotations

import os

_TRUTHY_VALUES = {"1", "true", "yes", "on"}

MULTIPROCESS_DEVICE_COLLECTIVES_ENV = (
    "TNCC_ENABLE_EXPERIMENTAL_MULTIPROCESS_DEVICE_COLLECTIVES"
)
FORCE_MULTIPROCESS_TRANSPORT_ENV = "TNCC_FORCE_MULTIPROCESS_TRANSPORT"
_MULTIPROCESS_TRANSPORT_VALUES = {
    "ctypes_ipc",
    "pytorch_ipc",
    "peer_access_pointer_exchange",
}
_VALIDATED_MULTIPROCESS_DEVICE_TRANSPORTS = {"ctypes_ipc"}
_VALIDATED_MULTIPROCESS_DEVICE_WORLD_SIZES = {2}


def multiprocess_device_collectives_enabled() -> bool:
    """Return whether unsafe multiprocess device collectives are explicitly enabled."""
    value = os.getenv(MULTIPROCESS_DEVICE_COLLECTIVES_ENV, "")
    return value.strip().lower() in _TRUTHY_VALUES


def multiprocess_device_collectives_detail(
    *,
    transport_strategy: str | None,
    world_size: int | None = None,
) -> str:
    """Return the current gate explanation for multiprocess device collectives."""
    if multiprocess_device_collectives_runtime_supported(
        transport_strategy=transport_strategy,
        world_size=world_size,
    ):
        detail = (
            "Multiprocess device collectives are supported on the currently validated "
            "public surface: world_size=2 with transport_strategy='ctypes_ipc'. "
            "No experimental env gate is required for that surface."
        )
    elif (
        not multiprocess_device_collectives_enabled()
        and multiprocess_device_collectives_transport_supported(transport_strategy)
    ):
        detail = (
            "Multiprocess device collectives outside the validated public surface are "
            "disabled by default. Today only world_size=2 with "
            "transport_strategy='ctypes_ipc' is public-supported; broader world-size "
            "diagnostics still require an explicit opt-in."
        )
    elif not multiprocess_device_collectives_transport_supported(transport_strategy):
        detail = (
            "Multiprocess device collectives remain transport-sensitive even under the "
            "diagnostic gate. Real multiprocess matrix runs currently validate only "
            "transport_strategy='ctypes_ipc'; other transports are not yet safe for "
            "public device collectives."
        )
    else:
        detail = (
            "Multiprocess device collectives are running under an explicit diagnostic "
            "gate outside the validated public surface. Current public support is "
            "limited to world_size=2 with transport_strategy='ctypes_ipc'."
        )
    if transport_strategy is not None:
        detail += f" Current transport_strategy={transport_strategy!r}."
    if world_size is not None:
        detail += f" Current world_size={world_size}."
    if not multiprocess_device_collectives_runtime_supported(
        transport_strategy=transport_strategy,
        world_size=world_size,
    ):
        detail += (
            f" Set {MULTIPROCESS_DEVICE_COLLECTIVES_ENV}=1 only for controlled "
            "bring-up/debug outside the validated public surface."
        )
    return detail


def multiprocess_device_remote_access_transport_supported(
    transport_strategy: str | None,
) -> bool:
    """Return whether the transport is validated for Triton device-side remote access."""
    return transport_strategy in _VALIDATED_MULTIPROCESS_DEVICE_TRANSPORTS


def multiprocess_device_remote_access_runtime_supported(
    *,
    transport_strategy: str | None,
    world_size: int | None,
) -> bool:
    """Return whether the current runtime is within the validated public surface."""
    return (
        multiprocess_device_remote_access_transport_supported(transport_strategy)
        and world_size in _VALIDATED_MULTIPROCESS_DEVICE_WORLD_SIZES
    )


def multiprocess_device_remote_access_detail(
    *,
    transport_strategy: str | None,
    operation: str,
    world_size: int | None = None,
) -> str:
    """Return a conservative explanation for Triton device-side remote access."""
    if multiprocess_device_remote_access_runtime_supported(
        transport_strategy=transport_strategy,
        world_size=world_size,
    ):
        detail = (
            f"{operation} relies on Triton device-side remote dereference. "
            "Real 2-GPU minimal remote-load/store diagnostics currently pass for "
            "transport_strategy='ctypes_ipc'; this is the validated public "
            "multiprocess surface."
        )
    elif multiprocess_device_remote_access_transport_supported(transport_strategy):
        detail = (
            f"{operation} relies on Triton device-side remote dereference. "
            "The transport itself is validated only for the current 2-GPU public "
            "surface; broader world-size usage is not yet public-supported."
        )
    else:
        detail = (
            f"{operation} relies on Triton device-side remote dereference. "
            "Real multiprocess diagnostics currently validate only "
            "transport_strategy='ctypes_ipc'; other transports are not yet safe."
        )
    if transport_strategy is not None:
        detail += f" Current transport_strategy={transport_strategy!r}."
    if world_size is not None:
        detail += f" Current world_size={world_size}."
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


def multiprocess_device_collectives_runtime_supported(
    *,
    transport_strategy: str | None,
    world_size: int | None,
) -> bool:
    """Return whether device collectives are public-supported for this runtime."""
    return multiprocess_device_remote_access_runtime_supported(
        transport_strategy=transport_strategy,
        world_size=world_size,
    )
