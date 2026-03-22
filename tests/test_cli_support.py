"""Tests for the support-matrix CLI path."""

from __future__ import annotations

import argparse
import json

from xtile.cli import _build_support_context, _format_support_matrix, _handle_support


def test_format_support_matrix_contains_sections(skip_no_gpu, device_info) -> None:
    """Human-readable rendering should include all top-level sections."""
    args = argparse.Namespace(
        backend=device_info.backend,
        world_size=1,
        heap_size_mb=None,
        json=False,
    )
    ctx, cleanup = _build_support_context(args)
    try:
        matrix = ctx.support_matrix()
        rendered = _format_support_matrix(matrix)
        assert "XTile Runtime Support Matrix" in rendered
        assert "Ops:" in rendered
        assert "Contracts:" in rendered
        assert "Execution Paths:" in rendered
        assert "Collectives:" in rendered
        assert "Memory:" in rendered
    finally:
        cleanup()


def test_handle_support_json_output(skip_no_gpu, device_info, capsys) -> None:
    """The JSON mode should emit a machine-readable payload."""
    args = argparse.Namespace(
        backend=device_info.backend,
        world_size=1,
        heap_size_mb=64,
        json=True,
    )
    _handle_support(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["context"]["has_heap"] is True
    assert payload["ops"]["gemm_allscatter"]["state"] == "supported"
    assert payload["ops"]["gemm_allgather"]["state"] == "supported"
    assert payload["ops"]["reduce_scatter"]["state"] == "supported"
    assert payload["execution_paths"]["reduce_scatter.reference"]["state"] == "supported"
    assert payload["execution_paths"]["reduce_scatter.device"]["state"] == "unsupported"
    assert payload["ops"]["gemm_reducescatter"]["state"] == "supported"


def test_build_support_context_multigpu_heap(skip_no_multigpu, device_info) -> None:
    """The support CLI helper should be able to inspect a real multi-GPU heap."""
    args = argparse.Namespace(
        backend=device_info.backend,
        world_size=2,
        heap_size_mb=64,
        json=False,
    )
    ctx, cleanup = _build_support_context(args)
    try:
        matrix = ctx.support_matrix()
        assert matrix.world_size == 2
        assert matrix.has_heap is True
        assert matrix.transport_strategy == "peer_access"
    finally:
        cleanup()
