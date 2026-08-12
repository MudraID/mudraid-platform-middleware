"""Issue #116 — the middleware version must have ONE source of truth.

``__init__.py`` used to hard-code ``0.1.0`` while ``pyproject.toml`` declared
``0.2.0``. ``__init__`` now reads the installed distribution metadata (source of
truth = pyproject) with a literal fallback for source checkouts; this guards the
fallback against drifting from pyproject again.
"""

from __future__ import annotations

from pathlib import Path

import tomllib

import mudraid_platform_middleware

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_module_version_matches_pyproject():
    declared = tomllib.loads(_PYPROJECT.read_text())["project"]["version"]
    assert mudraid_platform_middleware.__version__ == declared
