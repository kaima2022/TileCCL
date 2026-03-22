"""Shared helpers for benchmark reporting and figure metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_json_payload(path: str | Path) -> dict[str, Any]:
    """Return a structured JSON payload or an empty dict when missing."""
    payload_path = Path(path)
    if not payload_path.exists():
        return {}
    with payload_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        return data
    return {}


def runtime_support_brief(
    payload: dict[str, Any],
    *,
    highlight_ops: Iterable[str] = (),
) -> str | None:
    """Return a compact human-readable runtime-support summary."""
    runtime = payload.get("runtime_support")
    if not isinstance(runtime, dict):
        return None

    context = runtime.get("context")
    if not isinstance(context, dict):
        return None

    parts: list[str] = []
    backend = context.get("backend")
    if backend:
        parts.append(f"backend={backend}")

    world_size = context.get("world_size")
    if isinstance(world_size, int):
        parts.append(f"ws={world_size}")

    has_heap = context.get("has_heap")
    if has_heap:
        heap_mode = context.get("heap_mode")
        if heap_mode:
            parts.append(f"heap={heap_mode}")
        transport = context.get("transport_strategy")
        if transport:
            parts.append(f"transport={transport}")
    else:
        parts.append("heap=none")

    ops = runtime.get("ops")
    if isinstance(ops, dict):
        for op_name in highlight_ops:
            op_status = ops.get(op_name)
            if not isinstance(op_status, dict):
                continue
            state = op_status.get("state")
            if state:
                parts.append(f"{op_name}={state}")

    if not parts:
        return None
    return ", ".join(parts)


def benchmark_footer_text(
    payload: dict[str, Any],
    *,
    source_name: str,
    highlight_ops: Iterable[str] = (),
    include_command: bool = True,
) -> str | None:
    """Return a concise footer string for figures or exported summaries."""
    if not payload:
        return None

    parts = [f"source={source_name}"]
    generated_at = payload.get("generated_at_utc")
    if isinstance(generated_at, str) and generated_at:
        parts.append(f"run={generated_at[:10]}")

    support = runtime_support_brief(payload, highlight_ops=highlight_ops)
    if support:
        parts.append(support)

    command = payload.get("command")
    if include_command and isinstance(command, str) and command:
        parts.append(f"cmd={command}")

    return " | ".join(parts)


def execution_path_brief(
    payload: dict[str, Any],
    *,
    names: Iterable[str],
) -> str | None:
    """Return a compact summary for selected execution-path states."""
    runtime = payload.get("runtime_support")
    if not isinstance(runtime, dict):
        return None

    execution_paths = runtime.get("execution_paths")
    if not isinstance(execution_paths, dict):
        return None

    parts: list[str] = []
    for name in names:
        entry = execution_paths.get(name)
        if not isinstance(entry, dict):
            continue
        state = entry.get("state")
        if state:
            parts.append(f"{name}={state}")
    if not parts:
        return None
    return ", ".join(parts)
