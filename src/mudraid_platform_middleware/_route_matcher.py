"""Compile and match the routes from ``mudraid_scopes.yaml``.

The middleware's hot path is ``(method, path) → RouteRule | None``.
Each row from the YAML carries a FastAPI-style pattern (e.g.
``/api/v1/items/{id}``) that we need to test against the concrete
path of the incoming request. Patterns get compiled to anchored
regexes once at startup; per-request lookup is then linear over a
small list of pre-compiled rules.

Locked behaviour (don't loosen without an explicit decision):

  - **Method match is exact and case-sensitive.** The YAML schema
    locks methods to the seven uppercase HTTP verbs (M5.2 enforces);
    incoming requests are uppercase by convention. We do not
    normalise either side — a mismatch is a real mismatch.
  - **Path segments do not cross ``/``.** ``{id}`` matches one
    URL segment, not many. ``/items/abc/def`` does NOT match
    ``/items/{id}``. (FastAPI's ``{path:path}`` catch-all is NOT
    supported in v1; the YAML schema doesn't lock its syntax.)
  - **Trailing slashes are significant.** ``/items`` and
    ``/items/`` are different routes. Starlette doesn't auto-
    normalise on the middleware hop either, so this matches what
    a handler downstream would see.
  - **First-match-wins by YAML order.** If two patterns could
    both match a path (e.g. ``/items/new`` listed before
    ``/items/{id}``), the first one in the YAML file wins. This
    lets a platform owner pin literal routes ahead of parametric
    ones by ordering their YAML, without needing a separate
    priority field.
  - **Case-sensitive on path.** URLs are case-sensitive per
    RFC 3986 §6.2.2.1 for the path component.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from mudraid_platform_middleware._yaml_loader import RouteRule

# Matches `{name}` where ``name`` is a Python-identifier-shaped param.
# Patterns like ``{foo-bar}`` are left as literal text in the resulting
# regex — they will never match a real URL but won't crash the matcher,
# which is the safe default for a deliberately-undocumented input.
_PARAM_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _compile_path_pattern(path: str) -> re.Pattern[str]:
    """Convert a FastAPI-style path pattern to an anchored regex.

    ``/items/{id}`` → ``^/items/[^/]+$``
    ``/items/{id}/comments/{cid}`` → ``^/items/[^/]+/comments/[^/]+$``
    ``/health`` → ``^/health$``

    Literal segments are escaped via ``re.escape`` so URL-reserved
    characters (``.``, ``?``, ``+``, etc.) match literally.
    """
    pieces: list[str] = []
    cursor = 0
    for match in _PARAM_RE.finditer(path):
        pieces.append(re.escape(path[cursor : match.start()]))
        # A single URL segment — anything that isn't a slash.
        pieces.append(r"[^/]+")
        cursor = match.end()
    pieces.append(re.escape(path[cursor:]))
    # ``\Z`` and not ``$``: in Python ``$`` also matches immediately BEFORE a
    # trailing newline, so ``^/health$`` matches ``"/health\n"``. That is not
    # hypothetical input — an ASGI server percent-decodes the path into
    # ``scope["path"]``, so a request for ``/health%0a`` arrives here as
    # ``/health\n``. Route identity should not depend on a regex dialect's
    # end-of-line convenience; ``\Z`` is end-of-string and nothing else.
    return re.compile("^" + "".join(pieces) + r"\Z")


@dataclass(frozen=True)
class _CompiledRule:
    method: str
    pattern: re.Pattern[str]
    rule: RouteRule


class RouteMatcher:
    """Match ``(method, path)`` against the YAML's route list."""

    def __init__(self, routes: Iterable[RouteRule]) -> None:
        # Compile once at construction so per-request lookup costs
        # only a linear scan + regex.fullmatch (regex is anchored
        # both ends so this is effectively a literal-prefix test on
        # most rules).
        self._compiled: tuple[_CompiledRule, ...] = tuple(
            _CompiledRule(
                method=r.method,
                pattern=_compile_path_pattern(r.path),
                rule=r,
            )
            for r in routes
        )

    def match(self, method: str, path: str) -> RouteRule | None:
        """Return the first rule whose method *and* compiled regex match.

        ``None`` means no rule applies — the middleware treats that
        as 'route not covered by mudraid_scopes.yaml', which lands as
        a 404 to agents (admin-facing routes shouldn't even exist
        from the agent's perspective; M5.8 implements that
        translation explicitly).
        """
        for compiled in self._compiled:
            if compiled.method == method and compiled.pattern.match(path):
                return compiled.rule
        return None

    def __len__(self) -> int:
        return len(self._compiled)
