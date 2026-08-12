#!/usr/bin/env python3
"""Assert what the built archives actually contain, before an upload nobody can undo.

WHY THIS EXISTS
===============

Every other gate in publish.yml reads the package through an import. Tests
import from site-packages, lint reads the source tree, twine parses the
metadata. None of them ever opens the archive and looks, so a distribution can
be wrong in two directions while everything stays green:

* **A file that should ship and does not.** ``py.typed`` is the example that
  costs the most and shows the least: without it in the wheel, every downstream
  ``mypy`` run silently treats this package as untyped and stops checking the
  calls into it. Nothing fails. The type checking just stops happening, on the
  customer's side, for a package whose metadata claims ``Typing :: Typed``.
* **A file that ships and should not.** A stray ``.env``, key or credential
  inside a wheel is published to everyone, permanently — PyPI does not allow
  re-uploading a version number even after deletion, so the remedy for either
  mistake is a new version and an advisory, forever.

This runs in the mirrored public repository, so it deliberately uses nothing
but the standard library and nothing outside this package directory.

Usage:  IMPORT_NAME=mudraid python .github/inspect_distribution.py [dist/]
Exit:   0 clean, 1 refusal.
"""

from __future__ import annotations

import os
import re
import sys
import tarfile
import zipfile
from pathlib import Path

#: Patterns that must never appear in a published archive, with the reason each
#: one is here. A bare list would be edited by whoever hit it; a reason is
#: something they have to disagree with first.
FORBIDDEN: list[tuple[str, str]] = [
    (r"(^|/)\.env($|\..*)", "an environment file carries credentials by convention"),
    (r"\.(pem|key|p12|pfx|jks)$", "private key material"),
    (r"(^|/)id_(rsa|dsa|ecdsa|ed25519)$", "an SSH private key"),
    (r"(^|/)\.git($|/)", "repository internals, including full history in .git"),
    (r"(^|/)\.github($|/)", "CI configuration is repository furniture, not library code"),
    (r"(^|/)__pycache__($|/)", "compiled bytecode from the build machine"),
    (r"\.pyc$", "compiled bytecode from the build machine"),
    (r"(^|/)\.venv($|/)", "a virtual environment"),
    (r"(^|/)\.DS_Store$", "editor and OS debris"),
]


def fail(problems: list[str]) -> int:
    print("inspect_distribution: the archives are not what they should be:")
    for problem in problems:
        print(f"  {problem}")
    print(
        "\nPyPI does not allow re-uploading a version number, even after deletion. "
        "Fix the build and cut a new version rather than publishing this one."
    )
    return 1


def wheel_names(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()


def sdist_names(path: Path) -> list[str]:
    with tarfile.open(path, "r:gz") as archive:
        # An sdist is rooted at `<name>-<version>/`. That prefix is not part of
        # what the package declares it ships, so it is stripped before anything
        # is compared — otherwise every expectation below would have to know the
        # version number.
        return [re.sub(r"^[^/]+/", "", name) for name in archive.getnames()]


def main(argv: list[str]) -> int:
    import_name = os.environ.get("IMPORT_NAME", "").strip()
    if not import_name:
        print("inspect_distribution: IMPORT_NAME is not set, so there is nothing to look for.")
        return 1

    dist = Path(argv[0]) if argv else Path("dist")
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))

    problems: list[str] = []

    # Exactly one of each. Two wheels in dist/ means a stale artifact is present
    # and `dist/*` uploads BOTH — the older one silently claiming its version.
    if len(wheels) != 1:
        problems.append(f"expected exactly one wheel in {dist}/, found {[w.name for w in wheels]}")
    if len(sdists) != 1:
        problems.append(f"expected exactly one sdist in {dist}/, found {[s.name for s in sdists]}")
    if problems:
        return fail(problems)

    wheel, sdist = wheel_names(wheels[0]), sdist_names(sdists[0])

    # What must be in the wheel — the thing pip actually installs.
    required_wheel = [
        f"{import_name}/__init__.py",
        # Claimed by the `Typing :: Typed` classifier. Absent, every downstream
        # type check degrades to "untyped" without a word of warning.
        f"{import_name}/py.typed",
    ]
    for entry in required_wheel:
        if entry not in wheel:
            problems.append(f"{wheels[0].name}: missing {entry}")

    if not any(name.endswith(".dist-info/METADATA") for name in wheel):
        problems.append(f"{wheels[0].name}: no .dist-info/METADATA")
    if not any("LICENSE" in name for name in wheel):
        problems.append(
            f"{wheels[0].name}: no LICENSE — the metadata names one, and a wheel that "
            "ships none leaves the installed package with no licence text at all"
        )

    # What must be in the sdist — the thing a distributor rebuilds from.
    for entry in ("pyproject.toml", "README.md", "LICENSE", f"src/{import_name}/__init__.py"):
        if entry not in sdist:
            problems.append(f"{sdists[0].name}: missing {entry}")
    if not any(name.startswith("tests/") for name in sdist):
        problems.append(
            f"{sdists[0].name}: ships no tests/. A distributor rebuilding this package "
            "has nothing to run to decide whether the rebuild worked."
        )

    # What must be in neither.
    for archive_name, names in ((wheels[0].name, wheel), (sdists[0].name, sdist)):
        for pattern, reason in FORBIDDEN:
            hits = sorted(n for n in names if re.search(pattern, n))
            if hits:
                problems.append(f"{archive_name}: ships {hits} — {reason}")

    if problems:
        return fail(problems)

    print(
        f"inspect_distribution: {wheels[0].name} and {sdists[0].name} carry the "
        f"{import_name} package, its typing marker, its licence and its tests, and "
        f"nothing from the {len(FORBIDDEN)} forbidden classes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
