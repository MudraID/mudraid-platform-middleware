"""Parse and validate ``mudraid_scopes.yaml`` at middleware startup.

The schema is the ``MudraidScopesYaml`` component from
``docs/openapi.yaml`` in the MudraID backend repo — that file is the
single source of truth for the wire shape. Any divergence here would
allow a YAML the portal exports to be rejected (or, worse, silently
mis-interpreted) by the middleware, so the validation rules below
match the spec exactly.

Locked rules (don't loosen without an explicit decision):

  - Top level keys ``platform_id``, ``version``, ``routes`` are
    required.
  - ``platform_id`` is a non-empty string.
  - ``version`` is a positive integer (``True`` and ``False`` are
    rejected even though they are ``int`` in Python).
  - ``routes`` is a list; an empty list is permitted (a platform with
    no agent-facing routes is valid — every request would 404 to an
    agent, but that's the developer's choice).
  - Each route has ``method`` (one of the seven HTTP verbs) and
    ``path`` (non-empty, starting with ``/``).
  - Each route has EXACTLY ONE of ``scope`` (non-empty string),
    ``public: true``, or ``skip: true``. Combining them is a bug
    that we refuse to paper over — the developer needs to pick.

Failures raise :class:`MudraIDScopesYamlError`. We never silently
fall back to "allow everything" or "deny everything" on parse
failure — at startup, fail loud; mid-request, the YAML has
already been loaded and validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from mudraid_platform_middleware.exceptions import MudraIDScopesYamlError

# Locked enum from docs/openapi.yaml ``MudraidScopesYaml.routes.method``.
_ALLOWED_METHODS = frozenset(("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"))


@dataclass(frozen=True)
class RouteRule:
    """One row from ``mudraid_scopes.yaml`` after parse + validation.

    Exactly one of ``scope`` / ``public`` / ``skip`` is meaningful per
    row; the loader guarantees this so the middleware can branch
    without re-validating.
    """

    method: str
    path: str
    scope: str | None = None
    public: bool = False
    skip: bool = False


@dataclass(frozen=True)
class ScopesYaml:
    """Parsed, validated contents of ``mudraid_scopes.yaml``.

    Frozen so the middleware can't mutate at runtime — the YAML is
    the source of truth and we want any change to require a process
    restart (M5.2 locked behaviour: load-once at startup).
    """

    platform_id: str
    version: int
    routes: tuple[RouteRule, ...]


def load_scopes_yaml(path: str | Path) -> ScopesYaml:
    """Read, parse, and validate a ``mudraid_scopes.yaml`` file.

    Args:
        path: filesystem path to the YAML file. The middleware passes
            either the integrator-configured path or the default
            ``./mudraid_scopes.yaml`` (resolved against ``os.getcwd()``).

    Returns:
        Frozen :class:`ScopesYaml` ready for handover to the
        route matcher (M5.3) and the middleware dispatch (M5.6).

    Raises:
        MudraIDScopesYamlError: the file is missing, unreadable,
            malformed YAML, or violates the schema. The message
            names the failing line / route index so the developer
            can find it without scrolling.
    """
    yaml_path = Path(path)

    try:
        # ``newline=""`` (the default with ``open(...)``) keeps PyYAML
        # happy on Windows where CRLF would otherwise alter the line
        # numbers it reports for syntax errors. UTF-8 is the only
        # supported encoding — same as every other Python config file
        # in this repo.
        text = yaml_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MudraIDScopesYamlError(
            f"mudraid_scopes.yaml not found at {yaml_path}. "
            "Download it from the MudraID portal and place it at the "
            "configured path (default: project root)."
        ) from exc
    except OSError as exc:
        raise MudraIDScopesYamlError(f"could not read {yaml_path}: {exc}") from exc

    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        line = _yaml_error_line(exc)
        where = f" at line {line}" if line is not None else ""
        raise MudraIDScopesYamlError(f"malformed YAML in {yaml_path}{where}: {exc}") from exc

    return _validate(parsed, source=str(yaml_path))


def _yaml_error_line(exc: yaml.YAMLError) -> int | None:
    """Best-effort extraction of the failing line number from a YAMLError.

    ``problem_mark`` / ``context_mark`` are present on most syntax
    errors but optional in the API; we tolerate either being absent.
    Line numbers in PyYAML are 0-based; we surface the human 1-based
    number that matches what the developer's editor will show.
    """
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    if mark is None:
        return None
    return mark.line + 1


def _validate(parsed: Any, *, source: str) -> ScopesYaml:
    """Walk the parsed YAML and check every constraint from the spec."""
    if not isinstance(parsed, dict):
        raise MudraIDScopesYamlError(
            f"{source}: top-level YAML must be a mapping, got {_typename(parsed)}"
        )

    platform_id = parsed.get("platform_id")
    if not isinstance(platform_id, str) or not platform_id.strip():
        raise MudraIDScopesYamlError(
            f"{source}: 'platform_id' is required and must be a non-empty string"
        )

    version = parsed.get("version")
    # bool is a subclass of int in Python; treat True/False as schema errors.
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise MudraIDScopesYamlError(
            f"{source}: 'version' is required and must be a positive integer"
        )

    raw_routes = parsed.get("routes")
    if raw_routes is None:
        raise MudraIDScopesYamlError(
            f"{source}: 'routes' is required (use an empty list if the "
            "platform has no agent-facing routes yet)"
        )
    if not isinstance(raw_routes, list):
        raise MudraIDScopesYamlError(
            f"{source}: 'routes' must be a list, got {_typename(raw_routes)}"
        )

    routes = tuple(_validate_route(r, idx, source) for idx, r in enumerate(raw_routes))

    return ScopesYaml(
        platform_id=platform_id.strip(),
        version=version,
        routes=routes,
    )


def _validate_route(raw: Any, index: int, source: str) -> RouteRule:
    """Validate one row of the ``routes`` list.

    ``index`` is the 0-based position in the YAML list; surfaced in
    error messages so a developer staring at a 200-line file can jump
    straight to the offending entry.
    """
    where = f"{source} routes[{index}]"

    if not isinstance(raw, dict):
        raise MudraIDScopesYamlError(f"{where}: each route must be a mapping, got {_typename(raw)}")

    method = raw.get("method")
    if not isinstance(method, str) or method not in _ALLOWED_METHODS:
        raise MudraIDScopesYamlError(f"{where}: 'method' must be one of {sorted(_ALLOWED_METHODS)}")

    path = raw.get("path")
    if not isinstance(path, str) or not path.startswith("/") or not path.strip("/"):
        if isinstance(path, str) and path == "/":
            pass  # root path is valid
        else:
            raise MudraIDScopesYamlError(f"{where}: 'path' is required and must start with '/'")

    scope = raw.get("scope")
    public = raw.get("public")
    skip = raw.get("skip")

    if scope is not None and not (isinstance(scope, str) and scope.strip()):
        raise MudraIDScopesYamlError(f"{where}: 'scope' must be a non-empty string if present")
    if public is not None and not isinstance(public, bool):
        raise MudraIDScopesYamlError(f"{where}: 'public' must be a boolean if present")
    if skip is not None and not isinstance(skip, bool):
        raise MudraIDScopesYamlError(f"{where}: 'skip' must be a boolean if present")

    # The locked rule: exactly one mode per row. A scope name PLUS
    # public/skip is ambiguous and almost certainly a mistake.
    modes_chosen = sum(
        (
            scope is not None and bool(scope),
            bool(public),
            bool(skip),
        )
    )
    if modes_chosen == 0:
        raise MudraIDScopesYamlError(
            f"{where}: must specify exactly one of 'scope', "
            "'public: true', or 'skip: true' (got none)"
        )
    if modes_chosen > 1:
        raise MudraIDScopesYamlError(
            f"{where}: must specify exactly one of 'scope', "
            "'public: true', or 'skip: true' (got multiple)"
        )

    return RouteRule(
        method=method,
        path=path,
        scope=scope.strip() if isinstance(scope, str) and scope.strip() else None,
        public=bool(public),
        skip=bool(skip),
    )


def _typename(value: Any) -> str:
    return type(value).__name__
