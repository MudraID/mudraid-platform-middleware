"""Packaging claims must be true of the built artifact.

A classifier is a promise to a consumer's toolchain. ``Typing :: Typed`` tells
mypy and pyright that this distribution ships inline type information, and PEP
561 says they may only use it when a ``py.typed`` marker is present INSIDE the
installed package. Claiming the classifier without shipping the marker means a
downstream user's type checker silently treats every symbol here as ``Any`` —
the failure is invisible, and it is invisible in the direction that loses
safety.

The marker is asserted where it has to live rather than where it happens to be
convenient: adjacent to ``__init__.py``, so the wheel picks it up as package
data.
"""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 — tomllib entered the stdlib in 3.11.
    # Same parser, pre-stdlib name. Declared in the [dev] extra for 3.10, so
    # the wheel-test lane that installs `[dev]` on every classifier-claimed
    # interpreter can collect this module on all of them; a bare `import
    # tomllib` made collection die on 3.10 before a single test ran.
    import tomli as tomllib  # type: ignore[no-redef]

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PACKAGE_ROOT / "src" / "mudraid_platform_middleware"


def test_the_py_typed_marker_exists_beside_the_package_init() -> None:
    assert (_SRC / "__init__.py").is_file(), "package layout moved; this test needs updating"
    assert (_SRC / "py.typed").is_file(), (
        "pyproject declares 'Typing :: Typed' but no PEP 561 marker ships, so every "
        "consumer's type checker silently sees Any"
    )


def test_the_typed_classifier_and_the_marker_agree() -> None:
    """Either both, or neither. A classifier nobody can act on is worse than
    no classifier, because it is believed."""
    manifest = tomllib.loads((_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    claims_typed = "Typing :: Typed" in manifest["project"]["classifiers"]
    assert claims_typed is (_SRC / "py.typed").is_file()


def test_the_declared_version_is_the_one_the_support_matrix_publishes() -> None:
    """The matrix is the authority on which version is publishable; a manifest
    that has moved past it would publish an artifact nothing declares.

    Two layouts, one guard. In the monorepo this package lives under ``sdks/``
    and the full matrix sits one level above the package root. In the public
    mirror the package root IS the repository root, and mirror-sdk.yml ships a
    trimmed excerpt of the matrix — this package's own name and version —
    alongside the package. Absence in BOTH places is a failure, not a skip:
    the publish run in the public repository is the only run that uploads,
    which makes it exactly the run this guard exists for.
    """
    import json

    for candidate in (
        _PACKAGE_ROOT / "support-matrix.json",  # public mirror layout
        _PACKAGE_ROOT.parent / "support-matrix.json",  # monorepo layout
    ):
        if candidate.is_file():
            matrix = json.loads(candidate.read_text(encoding="utf-8"))
            break
    else:
        raise AssertionError(
            "support-matrix.json exists neither beside the package (mirror layout) nor "
            "one level up (monorepo layout); the publishable-version guard has nothing "
            "to hold the manifest against"
        )

    manifest = tomllib.loads((_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entry = next(p for p in matrix["packages"] if p["name"] == "mudraid-platform-middleware")
    assert manifest["project"]["version"] == entry["version"] == "1.1.0"


def test_this_sdk_does_not_collide_with_the_internal_shared_library() -> None:
    """The collision this package was renamed to remove.

    ``shared/mudraid_middleware`` is an INTERNAL library — correlation IDs,
    error handling, rate limiting — that 14 services pip-install. It declares
    the distribution name ``mudraid-middleware`` and installs a top-level
    ``mudraid_middleware`` package. This SDK declared both of the same names.

    Whichever editable install landed last won in ``site-packages``, so the
    shared library's own test suite had to evict modules and rewrite
    ``sys.path`` in two places to import itself. Publishing this SDK to PyPI
    under that name would have made the ambiguity permanent and external.

    Both halves are asserted, because either one alone still collides: two
    distributions cannot share a PyPI name, and two packages cannot share a
    top-level import name.
    """
    internal_setup = _PACKAGE_ROOT.parents[1] / "shared" / "mudraid_middleware" / "setup.py"
    if not internal_setup.is_file():  # pragma: no cover - repo layout guard
        return
    internal_text = internal_setup.read_text(encoding="utf-8")

    manifest = tomllib.loads((_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    our_dist = manifest["project"]["name"]

    assert 'name="mudraid-middleware"' in internal_text, (
        "the internal library's distribution name moved; this guard needs rechecking "
        "against whatever it is now"
    )
    assert our_dist != "mudraid-middleware", "the distribution names collide again"
    assert _SRC.name != "mudraid_middleware", "the top-level import names collide again"
    internal_pkg = _PACKAGE_ROOT.parents[1] / "shared" / "mudraid_middleware" / "mudraid_middleware"
    assert internal_pkg.is_dir()
