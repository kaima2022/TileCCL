# SPDX-License-Identifier: Apache-2.0
"""Public smoke tests for package import."""

from __future__ import annotations

import importlib


def test_import_tileccl_package() -> None:
    """The top-level package should import and expose the public entrypoints."""
    tileccl = importlib.import_module("tileccl")

    assert isinstance(tileccl.__version__, str)
    assert callable(tileccl.init)
    assert callable(tileccl.init_local)
    assert importlib.import_module("tileccl.ops") is not None
