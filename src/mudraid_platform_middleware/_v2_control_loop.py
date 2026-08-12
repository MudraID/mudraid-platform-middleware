"""V2 enforcement control loop — the adapter-decision semantics, natively.

This module encodes the **portable adapter-decision contract**
(``mudraid.adapter.decision/1``) directly in the middleware, so a platform
running the Python server middleware in ``mode="v2"`` reaches the SAME
normalized outcome as the Kong ``mudraid-enforce`` Lua handler and the
reference runner in ``shared/mudraid_contracts`` — for the same facts.

Why a native copy (and not an import of ``mudraid_contracts``): the middleware
is a distributable SDK. Taking a hard runtime dependency on the internal
``mudraid_contracts`` package would leak the platform's private tooling into
every integrator's environment. So the control-loop **semantics** are mirrored
here, and the middleware's tests mirror the contract's fixture corpus to prove
agreement (see ``tests/unit/test_middleware_v2.py``).

The decision tree order mirrors ``handler.lua:access`` exactly:

  1. reserved ``x-mudraid-*`` request headers are stripped FIRST, before any
     evaluation, and even on requests that will be denied — trusted context is
     never accepted as an input fact;
  2. fail CLOSED when no verified signed bundle is active → ``not_safely_decided``;
  3. method classification — Streamable-HTTP control verbs pass, non-POST denies;
  4. bounded JSON-RPC framing — oversized / unreadable bodies deny, a JSON
     *array* (batch) is rejected wholesale, never partially evaluated;
  5. exact, case-sensitive canonical action resolution — never fuzzy/prefix;
  6. a live ``/decide`` call, required for every protected tool invocation and
     **deny-closed** on timeout / error / unconfigured;
  7. allow — and only then is trusted context injected downstream.

"not safely decided" (no bundle, ``/decide`` timeout/error) is deny-closed —
never optimistically treated as allow.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

#: Reserved request-header prefix stripped before evaluation (contract A03-04).
#: Case-insensitive; mirrors the plugin's default strip prefix. Client-supplied
#: headers under this prefix are NEVER trusted as an input fact.
RESERVED_HEADER_PREFIX = "x-mudraid-"

#: Bounded action-name length (matches the contract's ``MAX_TOOL_NAME_LEN``).
MAX_TOOL_NAME_LEN = 512

#: Streamable-HTTP control verbs. They carry no JSON-RPC request and cannot
#: invoke a tool, so they pass a protected surface without a decision.
_CONTROL_VERBS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS", "DELETE"})

#: Control/discovery JSON-RPC methods allowed to pass a protected surface
#: without carrying a tool invocation.
#:
#: THE CONSERVATIVE SET, matching the Kong plugin's schema default exactly.
#: These three carry no tool invocation and cannot act: ``initialize`` opens the
#: session, ``ping`` is liveness, ``tools/list`` enumerates what exists.
#:
#: ``resources/list`` and ``prompts/list`` are NOT here, and their absence is the
#: point. They enumerate a surface's resources and prompts, which is disclosure,
#: and the two adapters previously disagreed about them — a Kong surface denied
#: them while this middleware allowed them, for identical facts. A "portable
#: adapter-decision contract" cannot quietly contain that, and documenting the
#: divergence only warned readers who found the note. A platform that wants
#: those methods public sets them explicitly, on whichever adapters it runs.
DEFAULT_PUBLIC_METHODS: frozenset[str] = frozenset({"initialize", "ping", "tools/list"})

#: Retained as an explicit alias of the default so existing configuration that
#: named it keeps working and keeps meaning the same thing.
KONG_DEFAULT_PUBLIC_METHODS: frozenset[str] = DEFAULT_PUBLIC_METHODS

# ``/decide`` transport-failure mode → normalized not-safely-decided reason.
# Every one deny-closes; the reason differs by failure mode but the surfaced
# error_code is uniformly ENFORCE_DECIDE_UNAVAILABLE.
_DECIDE_UNAVAILABLE_REASONS: dict[str, str] = {
    "timeout": "deadline_exceeded",
    "error": "authority_source_unavailable",
    "unreachable": "authority_source_unavailable",
    "unconfigured": "adapter_config_stale",
    "credential_unconfigured": "adapter_config_stale",
}

# Reasons that are NOT deny outcomes. A ``/decide`` response that tries to label
# a *deny* with one of these cannot leak an allow/soft outcome through the deny
# path; we fall back to the generic deny reason. This is a self-contained
# safety guard, deliberately NOT a mirror of the full governed reason-code
# registry (the SDK takes no dependency on the internal mudraid_contracts
# package). Extend only with reasons whose canonical outcome is provably not
# "deny".
_NON_DENY_REASONS: frozenset[str] = frozenset(
    {
        "authorized",  # allow
        "adapter_config_stale",  # not_safely_decided
        "deadline_exceeded",  # not_safely_decided
        "authority_source_unavailable",  # not_safely_decided
    }
)

_DECIDE_DENY_DEFAULT_REASON = "policy_rule_denied"


@dataclass(frozen=True, slots=True)
class DecideResult:
    """The outcome of one live ``/decide`` call, as the middleware sees it.

    ``status`` is the failure-aware outcome vocabulary the control loop
    consumes:

      - ``"allow"`` / ``"deny"`` — an authority decision was reached;
      - ``"timeout"`` / ``"error"`` / ``"unreachable"`` / ``"unconfigured"`` /
        ``"credential_unconfigured"`` — the call could not be completed; each is
        deny-closed (``not_safely_decided``), never optimistically allowed.

    A real :class:`DecideClient` translates a network timeout or transport error
    into the matching ``status`` (or simply raises — the middleware wraps the
    call and treats any exception as ``"error"``). ``reason`` carries a
    ``/decide``-supplied deny reason code; ``decision_id`` a correlation id that,
    on allow, is forwarded to the handler as trusted context.
    """

    status: str
    reason: str | None = None
    decision_id: str | None = None


@dataclass(frozen=True, slots=True)
class V2RequestFacts:
    """The normalized facts a single V2 evaluation consumes.

    These mirror the portable contract's decision facts. The middleware
    extracts them from the live request (header scan, bounded body read,
    JSON-RPC parse, action lookup); the decision *tree* itself lives only in
    :func:`evaluate_v2`, so fact extraction and decision logic never drift into
    two competing copies of the policy.
    """

    protected: bool = True
    reserved_headers_presented: tuple[str, ...] = ()
    bundle_active: bool = True
    method: str = "POST"
    body_readable: bool = True
    body_too_large: bool = False
    json_shape: str = "object"  # object | array | scalar | not_json
    jsonrpc: str | None = "2.0"
    rpc_method: str | None = "tools/call"
    tool_name: str | None = None
    action_mapped: bool = False
    action: str | None = None


@dataclass(frozen=True, slots=True)
class V2Decision:
    """The normalized outcome of one V2 evaluation.

    ``outcome`` is the closed ``allow | deny | not_safely_decided`` vocabulary.
    ``error_code`` is the stable agent-facing code surfaced in the structured
    error body; it is ``None`` on an allow/pass-through (the request forwards to
    the handler). Its values mirror the portable contract's adapter codes so an
    agent sees the SAME code from Kong and from this middleware. ``trusted_context``
    is the tuple of ``(name, value)`` headers to inject downstream — populated
    ONLY on an authorized allow, never on a pass-through.
    """

    outcome: str  # allow | deny | not_safely_decided
    reason_code: str
    reason_tier: str  # transport | authorization
    http_status: int
    error_code: str | None
    message: str
    stripped_reserved_headers: tuple[str, ...]
    trusted_context: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def forward(self) -> bool:
        """Whether the request should be forwarded to the wrapped handler."""
        return self.outcome == "allow"


# ---- pure helpers --------------------------------------------------------


def normalize_stripped_headers(
    reserved_headers_presented: tuple[str, ...], *, protected: bool
) -> tuple[str, ...]:
    """Reserved headers stripped from the request before evaluation.

    Case-insensitive prefix match on :data:`RESERVED_HEADER_PREFIX`. Applied on
    every protected request regardless of the eventual outcome (original case is
    preserved in the returned tuple for auditability). On an unprotected surface
    the request passes through untouched, so nothing is stripped.
    """
    if not protected:
        return ()
    return tuple(
        h for h in reserved_headers_presented if h.lower().startswith(RESERVED_HEADER_PREFIX)
    )


def is_reserved_header(name: str) -> bool:
    """True when a header name is a reserved ``x-mudraid-*`` context header."""
    return name.lower().startswith(RESERVED_HEADER_PREFIX)


def valid_tool_name(name: str | None) -> bool:
    """A usable canonical action name: non-empty, within the byte bound."""
    return isinstance(name, str) and 0 < len(name.encode("utf-8")) <= MAX_TOOL_NAME_LEN


def new_decision_id() -> str:
    """A fresh correlation id for a bound allow, forwarded as trusted context."""
    return uuid.uuid4().hex


# ---- decision constructors -----------------------------------------------


def _deny(
    reason_code: str,
    tier: str,
    http_status: int,
    error_code: str,
    message: str,
    stripped: tuple[str, ...],
    *,
    outcome: str = "deny",
) -> V2Decision:
    return V2Decision(
        outcome=outcome,
        reason_code=reason_code,
        reason_tier=tier,
        http_status=http_status,
        error_code=error_code,
        message=message,
        stripped_reserved_headers=stripped,
    )


def _pass_through(reason_code: str, stripped: tuple[str, ...]) -> V2Decision:
    return V2Decision(
        outcome="allow",
        reason_code=reason_code,
        reason_tier="transport",
        http_status=200,
        error_code=None,
        message="",
        stripped_reserved_headers=stripped,
    )


# ---- the control loop ----------------------------------------------------

DecideCallable = Callable[[str], Awaitable[DecideResult]]


async def evaluate_v2(
    facts: V2RequestFacts,
    decide: DecideCallable,
    *,
    public_methods: frozenset[str] | None = None,
) -> V2Decision:
    """Reproduce the reference control loop's outcome for one request.

    ``decide`` is invoked EXACTLY at the ``/decide`` branch (a mapped
    ``tools/call`` on an active bundle) and nowhere else, so a control verb or an
    unmapped action never triggers a live call. Any "not safely decided" state
    is deny-closed, never allow.

    ``public_methods`` is the operator's control/discovery allowlist. ``None``
    selects :data:`DEFAULT_PUBLIC_METHODS`; an EMPTY set is honoured as an empty
    set (every non-``tools/call``, non-``notifications/*`` message denies) rather
    than silently re-expanded to the default — narrowing the allowlist to nothing
    is a legitimate, and the strictest, configuration.
    """
    allowed_methods = DEFAULT_PUBLIC_METHODS if public_methods is None else public_methods
    stripped = normalize_stripped_headers(
        facts.reserved_headers_presented, protected=facts.protected
    )

    # 1. Unprotected surface: pass through untouched (not a bundled surface).
    if not facts.protected:
        return _pass_through("surface_not_protected", stripped)

    # 2. Fail CLOSED: no verified signed bundle active → not_safely_decided.
    if not facts.bundle_active:
        return _deny(
            "adapter_config_stale",
            "authorization",
            503,
            "ENFORCE_NO_VALID_BUNDLE",
            "no verified signed bundle is active; request cannot be safely decided",
            stripped,
            outcome="not_safely_decided",
        )

    # 3. Method classification. Control verbs pass; anything that is neither a
    #    control verb nor POST is denied.
    method = (facts.method or "").upper()
    if method in _CONTROL_VERBS:
        return _pass_through("control_plane_passthrough", stripped)
    if method != "POST":
        return _deny(
            "method_not_allowed",
            "transport",
            405,
            "ENFORCE_METHOD_NOT_ALLOWED",
            "method not allowed on a protected MCP surface",
            stripped,
        )

    # 4. Bounded framing: oversized / unreadable bodies deny, never partial eval.
    if facts.body_too_large:
        return _deny(
            "body_too_large",
            "transport",
            413,
            "ENFORCE_BODY_TOO_LARGE",
            "request body exceeds the bounded framing limit",
            stripped,
        )
    if not facts.body_readable:
        return _deny(
            "body_unreadable",
            "transport",
            400,
            "ENFORCE_BODY_UNREADABLE",
            "request body could not be read",
            stripped,
        )

    # 5. Parse exactly once. Non-object bodies (scalar / non-JSON) are
    #    malformed; a JSON *array* is a batch and rejected wholesale.
    if facts.json_shape == "array":
        return _deny(
            "batch_unsupported",
            "transport",
            400,
            "ENFORCE_BATCH_UNSUPPORTED",
            "JSON-RPC batch requests are not supported",
            stripped,
        )
    if facts.json_shape != "object":
        return _deny(
            "malformed_request",
            "transport",
            400,
            "ENFORCE_MALFORMED_REQUEST",
            "request body is not a single JSON-RPC 2.0 object",
            stripped,
        )
    if facts.jsonrpc != "2.0" or not isinstance(facts.rpc_method, str) or facts.rpc_method == "":
        return _deny(
            "malformed_request",
            "transport",
            400,
            "ENFORCE_MALFORMED_REQUEST",
            "request body is not a single JSON-RPC 2.0 object",
            stripped,
        )

    # 6. Non-tool protocol messages: allowlisted control/discovery + client
    #    notifications pass; everything else on a protected surface denies
    #    rather than slipping through because extraction found no action.
    if facts.rpc_method != "tools/call":
        if facts.rpc_method.startswith("notifications/"):
            return _pass_through("notification_passthrough", stripped)
        if facts.rpc_method in allowed_methods:
            return _pass_through("control_plane_passthrough", stripped)
        return _deny(
            "message_not_allowed",
            "transport",
            403,
            "ENFORCE_MESSAGE_NOT_ALLOWED",
            "JSON-RPC method is not permitted on a protected surface",
            stripped,
        )

    # 7. Exact canonical action resolution — never fuzzy.
    if not valid_tool_name(facts.tool_name):
        return _deny(
            "malformed_request",
            "transport",
            400,
            "ENFORCE_MALFORMED_REQUEST",
            "tools/call params.name is missing or exceeds the action-name bound",
            stripped,
        )
    if not facts.action_mapped:
        return _deny(
            "action_unmapped",
            "authorization",
            403,
            "ENFORCE_ACTION_UNMAPPED",
            "no exact canonical action is mapped for this tool",
            stripped,
        )

    # 8. Live /decide — required for every protected call; deny-closed on
    #    timeout / error / unconfigured.
    action = facts.action or facts.tool_name or ""
    result = await decide(action)
    if result.status == "allow":
        trusted = (
            ("x-mudraid-action-key", action),
            ("x-mudraid-decision-id", result.decision_id or new_decision_id()),
        )
        return V2Decision(
            outcome="allow",
            reason_code="authorized",
            reason_tier="authorization",
            http_status=200,
            error_code=None,
            message="",
            stripped_reserved_headers=stripped,
            trusted_context=trusted,
        )
    if result.status == "deny":
        reason = result.reason or _DECIDE_DENY_DEFAULT_REASON
        # A decide-supplied reason that is not provably a deny code cannot leak
        # an allow/soft outcome through the deny path.
        if reason in _NON_DENY_REASONS:
            reason = _DECIDE_DENY_DEFAULT_REASON
        return _deny(
            reason,
            "authorization",
            403,
            "ENFORCE_DECISION_DENY",
            "the authority denied this action",
            stripped,
        )

    # timeout | error | unreachable | unconfigured | credential_unconfigured
    # → deny-closed 503.
    reason = _DECIDE_UNAVAILABLE_REASONS.get(result.status, "authority_source_unavailable")
    return _deny(
        reason,
        "authorization",
        503,
        "ENFORCE_DECIDE_UNAVAILABLE",
        "the authority could not be reached; request cannot be safely decided",
        stripped,
        outcome="not_safely_decided",
    )
