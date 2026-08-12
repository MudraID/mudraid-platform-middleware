"""Public V2-mode configuration surface for :class:`MudraIDMiddleware`.

``mode="v2"`` runs the V2 enforcement control loop (see ``_v2_control_loop``)
before the wrapped handler, instead of the static V1 route-scope check. V2 mode
is a strict, additive OPT-IN: V1 remains the default and is unchanged.

.. note::

   **A production client now ships.** :class:`HttpDecideClient` implements the
   whole seam — signed-bundle fetch and verification, action-map management, the
   authenticated ``/decide`` call and its refresh lifecycle — so adopting V2 no
   longer means reimplementing security-critical protocol behaviour from these
   docstrings. That was the gap; it is closed.

   **The decision exchange is versioned in both directions.** The adapter
   stamps a named request contract and the authority validates it; the adapter
   refuses any response that does not carry a supported response contract, a
   decision id bound to the request it sent, and a recognised decision. A bare
   ``{"decision": "allow"}`` is not a readable answer.

   **What is still outstanding, and it gates public publication rather than
   staging use:** the decision response is authenticated by the transport only.
   A signed or MAC-authenticated response — bound to the decision id, outcome,
   action, resource and policy version — is required before this is described as
   generally production-ready. Until then V2 is staging-qualified: deployable and
   testable against a real authority, not yet a finished public package.

The two things V2 mode needs that V1 doesn't are supplied here:

  - a :class:`DecideClient` — the seam onto the platform's signed bundle
    (which surfaces are protected / which tool names map to a canonical action)
    and its live ``/decide`` authority. Injecting it keeps the middleware's
    tests hermetic — a fake client needs no network;
  - a small amount of transport policy (which paths are protected, the bounded
    body limit), carried by :class:`V2Config`.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from mudraid_platform_middleware._path import is_protected, normalize_prefix
from mudraid_platform_middleware._v2_control_loop import (
    DEFAULT_PUBLIC_METHODS,
    KONG_DEFAULT_PUBLIC_METHODS,
    DecideResult,
)

__all__ = [
    "DEFAULT_PUBLIC_METHODS",
    "MAX_CORRELATION_ID_LEN",
    "MAX_PATH_LEN",
    "KONG_DEFAULT_PUBLIC_METHODS",
    "DecideClient",
    "DecideContext",
    "DecideResult",
    "V2Config",
]

#: Default bounded-framing limit for a request body (1 MiB). A body larger than
#: this is denied at the framing layer, never partially evaluated.
_DEFAULT_MAX_BODY_BYTES = 1_048_576

#: Default adapter-side deadline for one live ``/decide`` call, in seconds.
#: Mirrors the Kong plugin's ``decide_timeout_ms`` default (2000). This is the
#: middleware's OWN deadline, applied regardless of whether the injected
#: :class:`DecideClient` sets one of its own — see :class:`V2Config`.
_DEFAULT_DECIDE_TIMEOUT_SEC = 2.0

#: Bounds on the request facts carried to the authority.
MAX_PATH_LEN = 2048
MAX_CORRELATION_ID_LEN = 200

#: An HTTP method is a `token` (RFC 9110 §5.6.2). Anything else — a space, a
#: control character, a smuggled newline — is rejected rather than normalized.
_HTTP_TOKEN_RE = re.compile(r"\A[!#$%&'*+\-.^_`|~0-9A-Za-z]+\Z")

#: A correlation id is echoed into headers and logs, so it is held to a
#: conservative charset rather than trusted from the caller.
_CORRELATION_ID_RE = re.compile(r"\A[A-Za-z0-9._:\-]+\Z")


@dataclass(frozen=True, slots=True)
class DecideContext:
    """The live request facts the authority needs beyond the action name.

    WHY THIS EXISTS, because it is not decoration. The authority establishes the
    caller's identity ONLY from the presented protected-action token — a
    ``/decide`` call carrying no credential is rejected ``credential_missing``,
    which the reason-code registry marks ``deny``. A client that could not
    forward the caller's credential would therefore deny every protected request
    while passing its own tests, so it is part of the seam rather than something
    an implementation is left to find for itself.

    NOTHING HERE HAS A DEFAULT, deliberately. A ``http_method="POST"`` or
    ``path="/"`` default would let a caller that forgot to wire the real request
    construct a context that looks valid and describes a request that never
    happened — an integration error that produces confident wrong evidence
    instead of an immediate failure. Absence of a credential is the one thing
    represented explicitly, as ``None``, and it deny-closes downstream.

    WHAT IS DELIBERATELY ABSENT. No agent id, user id or organization id: the
    authority derives identity from the verified credential, and an adapter that
    could assert identity alongside it would be offering a second, weaker path to
    the same claim. No request body either — parameter-level authorization is not
    an approved requirement, and forwarding a body would move customer data
    through the decision path for no decision that reads it.

    Attributes:
        presented_bearer: The request's ``Authorization`` header value, verbatim
            and complete (``"Bearer <token>"``), or ``None`` when the request
            carried none. Canonicalized once here — outer whitespace stripped —
            and forwarded ONLY to the enforcement authority. Never logged.
        http_method: The live request method. Normalized to upper case; a method
            that is not a token per RFC 9110 is rejected.
        path: The live request path. Must be absolute (begin with ``/``) and
            within :data:`MAX_PATH_LEN`; a query string is not accepted here
            because the decision is about the path.
        correlation_id: The inbound correlation id. Validated against a
            conservative charset and length; anything else — including a value
            that would be unusable as a header — is replaced with a freshly
            minted one rather than propagated or dropped, so the exchange is
            always correlatable.
    """

    presented_bearer: str | None
    http_method: str
    path: str
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        bearer = self.presented_bearer
        if bearer is not None:
            if not isinstance(bearer, str):
                raise ValueError("presented_bearer must be a string or None")
            bearer = bearer.strip()
            # An empty or whitespace-only header is the same fact as no header:
            # collapse it here so exactly one representation reaches the
            # authority, rather than two that it has to treat alike.
            bearer = bearer or None
        object.__setattr__(self, "presented_bearer", bearer)

        method = self.http_method
        if not isinstance(method, str) or not _HTTP_TOKEN_RE.match(method):
            raise ValueError(f"http_method is not a valid HTTP method: {method!r}")
        object.__setattr__(self, "http_method", method.upper())

        path = self.path
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError("path must be an absolute path beginning with '/'")
        if len(path) > MAX_PATH_LEN:
            raise ValueError(f"path exceeds {MAX_PATH_LEN} characters")
        if "\n" in path or "\r" in path:
            raise ValueError("path contains a line break")
        object.__setattr__(self, "path", path)

        correlation_id = self.correlation_id
        if (
            not isinstance(correlation_id, str)
            or not correlation_id
            or len(correlation_id) > MAX_CORRELATION_ID_LEN
            or not _CORRELATION_ID_RE.match(correlation_id)
        ):
            correlation_id = uuid.uuid4().hex
        object.__setattr__(self, "correlation_id", correlation_id)


@runtime_checkable
class DecideClient(Protocol):
    """The seam onto the platform's signed bundle and live ``/decide`` authority.

    :class:`HttpDecideClient` is the supported implementation; the seam stays a
    Protocol so tests can inject a fake with no network, and so a platform with
    an unusual transport can supply its own. The three members map 1:1 onto the
    control loop's decision facts:

      - :attr:`bundle_active` — is a verified signed bundle currently active?
        When ``False`` the control loop fails CLOSED (503), never optimistically
        allowing.
      - :meth:`resolve_action` — the EXACT, case-sensitive canonical action a
        ``tools/call`` tool name maps to, or ``None`` when unmapped (→ deny).
        Never fuzzy / prefix / regex.
      - :meth:`decide` — the live authority call for a mapped action. Any
        timeout / transport error must be surfaced as a failure
        :class:`DecideResult` (or raised — the middleware wraps the call and
        deny-closes on any exception).
    """

    @property
    def bundle_active(self) -> bool:
        """Whether a verified signed bundle is currently active."""

    def resolve_action(self, tool_name: str) -> str | None:
        """Exact canonical action for ``tool_name``, or ``None`` if unmapped."""

    async def decide(self, action: str, context: DecideContext) -> DecideResult:
        """Call the live ``/decide`` authority for a mapped canonical action."""


@dataclass(frozen=True)
class V2Config:
    """Configuration selecting and parameterizing V2 mode.

    Args:
        decide_client: The injected :class:`DecideClient` seam. Required.
        protected_paths: Path prefixes whose requests run the V2 control loop.
            ``None`` (default) treats EVERY route as a protected surface. A path
            not covered here passes through untouched (nothing stripped, nothing
            decided) — the equivalent of the contract's unprotected surface.

            Matching is EXACT PATH PLUS ``/``-SEPARATED DESCENDANTS, not a
            lexical prefix: ``/mcp`` covers ``/mcp``, ``/mcp/`` and
            ``/mcp/tools`` and does NOT cover ``/mcpfoo`` or ``/mcp-evil``.
            Every spelling of the request path is tested against it, so an
            encoded or dot-segmented request cannot slip past. Identical to the
            Kong plugin's rule — see :mod:`mudraid_platform_middleware._path`.

            Each entry is validated and normalized at construction; a prefix
            whose meaning depends on interpretation (dot segments,
            percent-escapes, a query string) raises ``ValueError`` there rather
            than resolving to something the operator did not write.
        max_body_bytes: Bounded-framing limit for the request body. A body over
            this (by ``Content-Length`` or actual streamed size) denies with 413.
            The limit is enforced while READING, so a chunked body that never
            declares a ``Content-Length`` is bounded too.
        public_methods: JSON-RPC control/discovery methods allowed to pass a
            protected surface without a tool decision. ``None`` selects
            :data:`~mudraid_platform_middleware._v2_control_loop.DEFAULT_PUBLIC_METHODS`
            (the portable contract's allowlist); an explicit set — INCLUDING an
            empty one — is used verbatim. Pass
            :data:`~mudraid_platform_middleware._v2_control_loop.KONG_DEFAULT_PUBLIC_METHODS`
            to match a Kong surface left on the plugin's narrower schema default;
            the two defaults differ, so anything relying on cross-adapter
            equivalence must set this explicitly on both.
        decide_timeout_sec: The middleware's OWN deadline for one live
            ``/decide`` call. Applied with ``asyncio.wait_for`` around the
            injected client, so an implementation that sets no timeout of its own
            still cannot hold a protected request open indefinitely; the expiry
            is deny-closed as ``timeout``, never an allow. ``None`` disables the
            middleware's deadline and defers wholly to the client — only correct
            when the client is known to bound itself.
    """

    decide_client: DecideClient
    protected_paths: tuple[str, ...] | None = None
    max_body_bytes: int = _DEFAULT_MAX_BODY_BYTES
    public_methods: frozenset[str] | None = None
    decide_timeout_sec: float | None = _DEFAULT_DECIDE_TIMEOUT_SEC

    def __post_init__(self) -> None:
        """Validate and normalize ``protected_paths`` at construction.

        Construction is startup for this middleware — the config object is built
        when the app wires the middleware in, so a bad prefix fails there rather
        than silently protecting the wrong routes for the life of the process.
        The normalized tuple is written back (the dataclass is frozen, hence
        ``object.__setattr__``) so ``is_protected`` never re-derives it per
        request.
        """
        if self.protected_paths is None:
            return
        if isinstance(self.protected_paths, str):
            # A bare string is iterable, so this would silently become one
            # prefix per CHARACTER — including "/", which protects every route
            # on the app. A plausible typo that inverts the configuration.
            raise ValueError(
                "protected_paths must be a tuple of paths, not a single string; "
                f"pass ({self.protected_paths!r},)"
            )
        object.__setattr__(
            self,
            "protected_paths",
            tuple(normalize_prefix(p) for p in self.protected_paths),
        )

    def is_protected(self, path: str) -> bool:
        """Whether ``path`` is a protected surface subject to V2 enforcement."""
        if self.protected_paths is None:
            return True
        return is_protected(self.protected_paths, path)
