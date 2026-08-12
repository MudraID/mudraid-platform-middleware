"""Protected-surface path semantics — the Python half of one canonical rule.

This module is the deliberate mirror of ``kong/plugins/mudraid-enforce/path.lua``.
The two adapters decide whether the entire V2 control loop runs for a request,
and they must decide it the same way: an integrator who moves a workload from
Kong to this middleware should not discover that a different set of routes is
enforced. One canonical rule governs both adapters; this is that rule, stated
once per language and tested against the same table of cases in both.

THE RULE
--------

A request path is protected by a configured prefix when, for ANY spelling of the
request path, the path is *exactly* the prefix or a descendant of it separated
by ``/``::

    prefix "/mcp"  matches  /mcp   /mcp/   /mcp/tools   /mcp//tools
                   misses   /mcpfoo   /mcp-evil   /mcpevil/steal

Two directions compose, and they pull against each other on purpose:

* **Every spelling is considered** (:func:`candidates`) — percent-decoded and
  dot-segment-resolved, alongside the raw path. This can only ADD protection, and
  it is what stops ``/%6dcp/messages`` from skipping the control loop entirely.
* **Each spelling is tested at a segment boundary** (:func:`matches_prefix`).
  This can only REMOVE over-protection, and it is what stops a configured
  ``/mcp`` from also claiming an unrelated ``/mcp-metrics`` route.

Neither alone is correct. A lexical prefix over the raw path — what both
adapters used to do — was wrong in both directions at once.

CONFIGURATION IS READ LITERALLY, REQUESTS GENEROUSLY
----------------------------------------------------

:func:`normalize_prefix` refuses a configured prefix carrying dot segments or
percent-escapes instead of resolving them. A request is hostile input and gets
the benefit of every reading; a configuration file is a statement of intent by
someone who can simply write ``/mcp`` instead of ``/%6dcp``. Accepting both and
guessing is how a surface ends up protecting something its operator never named.
"""

from __future__ import annotations

import re
from urllib.parse import unquote

__all__ = [
    "candidates",
    "is_protected",
    "matches_prefix",
    "normalize_prefix",
    "remove_dot_segments",
    "strip_query",
]

_QUERY_OR_FRAGMENT = re.compile(r"[?#].*$", re.DOTALL)


def strip_query(path: str) -> str:
    """Remove a query string or fragment.

    An ASGI server puts the query in ``scope["query_string"]``, so
    ``request.url.path`` never carries one and on this adapter the call is a
    no-op. It exists because the segment-boundary rule makes a trailing query the
    difference between a match and a miss — ``/mcp?x=1`` is neither ``/mcp`` nor
    a ``/mcp/`` descendant — so anything that ever did hand a query through here
    would be handing through a bypass. The Lua side needs it for the same reason.
    """
    return _QUERY_OR_FRAGMENT.sub("", path)


def _percent_decode(path: str) -> str:
    """Percent-decode exactly once (RFC 3986 §2.4).

    ``%252f`` therefore becomes the literal text ``%2f`` and stops; it does not
    become ``/``. Decoding repeatedly would invent equivalences the routing layer
    does not honour, and the routing layer is the authority on which spellings
    reach the same handler.
    """
    return unquote(path)


def remove_dot_segments(path: str) -> str:
    """Resolve ``.`` and ``..`` segments per RFC 3986 §5.2.4.

    A leading ``..`` cannot escape the root. A trailing ``.`` or ``..`` leaves
    the trailing slash the reference algorithm produces, so ``/a/b/..`` is
    ``/a/`` rather than ``/a``.
    """
    leading_slash = path.startswith("/")
    rest = path[1:] if leading_slash else path
    out: list[str] = []
    trailing_slash = False
    for segment in rest.split("/"):
        if segment == ".":
            trailing_slash = True
        elif segment == "..":
            if out:
                out.pop()
            trailing_slash = True
        else:
            out.append(segment)
            trailing_slash = False
    result = ("/" if leading_slash else "") + "/".join(out)
    if trailing_slash and not result.endswith("/"):
        result += "/"
    return result


def normalize(path: str) -> str:
    """Decode once, then resolve dot segments."""
    return remove_dot_segments(_percent_decode(path))


def candidates(path: str) -> tuple[str, ...]:
    """Every spelling of ``path`` the protected-surface test must consider.

    Always includes the raw path, so this can only widen the set of requests
    classified as protected. Duplicates are collapsed, so an already-normal path
    costs exactly one comparison.
    """
    if not isinstance(path, str) or not path:
        return ()
    path = strip_query(path)
    if not path:
        return ()
    out = [path]
    for variant in (_percent_decode(path), normalize(path)):
        if variant and variant not in out:
            out.append(variant)
    return tuple(out)


def normalize_prefix(prefix: str) -> str:
    """Return the canonical form of a configured prefix.

    Raises:
        ValueError: the prefix is empty, not absolute, carries a query or
            fragment, contains percent-escapes, contains ``.``/``..`` segments,
            or contains an empty ``//`` segment. Each is a prefix whose meaning
            depends on interpretation, and the operator can state it exactly.

    A trailing slash is accepted and stripped: ``/mcp/`` and ``/mcp`` are the
    same surface under the canonical rule, and refusing one spelling of an
    unambiguous intent would be pedantry rather than safety.
    """
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("protected path must be a non-empty string")
    if not prefix.startswith("/"):
        raise ValueError(f"protected path {prefix!r} must be an absolute path beginning with '/'")
    if "?" in prefix or "#" in prefix:
        raise ValueError(
            f"protected path {prefix!r} must be a path only, with no query string or fragment"
        )
    if "%" in prefix:
        raise ValueError(
            f"protected path {prefix!r} must be written literally, without percent-escapes "
            "(write '/mcp', not '/%6dcp')"
        )
    # Trailing slashes carry no meaning under the canonical rule, so they are
    # removed BEFORE the interior checks below. Order matters: the "//" check is
    # about an empty segment in the MIDDLE of a path, and running it first would
    # reject "/mcp///" — repeated trailing slashes, whose intent is not in doubt
    # — with a message about empty segments.
    stripped = prefix.rstrip("/")
    if not stripped:
        return "/"
    segments = stripped.split("/")[1:]
    if any(s in (".", "..") for s in segments):
        raise ValueError(
            f"protected path {prefix!r} must not contain '.' or '..' segments; "
            "write the path it resolves to"
        )
    if any(s == "" for s in segments):
        raise ValueError(f"protected path {prefix!r} must not contain empty segments ('//')")
    return stripped


def matches_prefix(candidate: str, prefix: str) -> bool:
    """Whether one request-path spelling falls under one configured prefix.

    Exact path, or a descendant separated by ``/``. Nothing else — a lexical
    ``startswith`` does not know a path is made of segments, and that is what
    made a configured ``/mcp`` also claim ``/mcpfoo``.
    """
    if not isinstance(candidate, str) or not isinstance(prefix, str) or not prefix:
        return False
    if prefix == "/":
        return candidate.startswith("/")
    # Tolerate an un-normalized prefix reaching here; V2Config normalizes at
    # construction, but this function is also called directly.
    prefix = prefix.rstrip("/")
    if not prefix:
        return candidate.startswith("/")
    return candidate == prefix or candidate.startswith(prefix + "/")


def is_protected(prefixes: tuple[str, ...], path: str) -> bool:
    """Whether ``path`` is on a protected surface, under any of ``prefixes``."""
    spellings = candidates(path)
    return any(matches_prefix(spelling, prefix) for prefix in prefixes for spelling in spellings)
