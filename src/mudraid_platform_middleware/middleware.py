"""MudraIDMiddleware — Starlette/FastAPI middleware that enforces MudraID scopes.

This is the keystone module. It runs in one of two mutually-exclusive modes,
selected by the ``mode`` constructor argument:

  - **``mode="v1"`` (default)** — the static-YAML route-scope enforcement
    described below. Unchanged; this is the locked V1 contract.
  - **``mode="v2"``** — the portable V2 enforcement control loop
    (``_v2_control_loop``) runs before the handler: reserved ``x-mudraid-*``
    headers are stripped first, the request is classified, and a live
    ``/decide`` call (deny-closed on timeout/error/no-bundle/unmapped) gates the
    handler. See :mod:`mudraid_platform_middleware.v2`. V2 is a strict, additive opt-in —
    a successful V1 JWT check is NEVER inferred as V2 enforcement.

V1 per-request flow:

  1. Match the request's (method, path) against the rules from
     ``mudraid_scopes.yaml`` (loaded lazily on first dispatch).
  2. If no rule matches → 404. Routes not in the YAML are invisible
     to agents — admin endpoints especially.
  3. If the rule is ``public: true`` → forward straight to the
     route handler. (M5.7)
  4. If the rule is ``skip: true`` → 404, same as 'no rule matched'.
     Used to hide admin routes from agents while still keeping them
     reachable from the platform's own front-end. (M5.8)
  5. Otherwise the rule has a ``scope``. We:
       a. Extract the ``Authorization: Bearer <jwt>`` header.
       b. Validate the JWT (signature + iss + aud + exp + nbf + iat)
          via :class:`JwtValidator`.
       c. Check the route's required scope is present in the JWT's
          ``scopes`` claim.
       d. Forward to the handler if both gates pass.
  6. Failures produce a structured JSON error response (M5.9) with
     a stable ``error_code`` so platforms can show consistent
     messaging to agents.

Locked behaviour (don't change without an explicit decision):

  - **Lazy bootstrap.** YAML + JwksClient + JwtValidator are built
    on the FIRST dispatch, not in ``__init__``. Construction is
    side-effect-free so a missing YAML at app-import time doesn't
    crash; the failure surfaces on the first request instead.
  - **Single in-flight bootstrap.** Under thundering-herd traffic
    at boot, all concurrent first requests are coalesced via
    ``asyncio.Lock`` so the YAML is parsed exactly once.
  - **Authorization header is case-insensitive on scheme.** Per
    RFC 7235 § 2.1, credential schemes are case-insensitive;
    ``Bearer`` and ``bearer`` are both accepted. The token itself
    is taken verbatim.
  - **No-rule path returns 404, not 403.** A route absent from the
    YAML is treated as non-existent from the agent's perspective;
    revealing that "the route exists but you can't use it" would
    leak topology to the agent.
  - **JWKS errors are 500, not 401.** A JWT we can't verify is
    different from a JWT we verified and rejected. The former is
    operationally significant (something's wrong on our side); the
    latter is normal failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Sequence
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from mudraid_platform_middleware._jwks_client import JwksClient
from mudraid_platform_middleware._jwt_validator import JwtValidator
from mudraid_platform_middleware._route_matcher import RouteMatcher
from mudraid_platform_middleware._v2_control_loop import (
    DecideResult,
    V2Decision,
    V2RequestFacts,
    evaluate_v2,
    is_reserved_header,
    valid_tool_name,
)
from mudraid_platform_middleware._yaml_loader import ScopesYaml, load_scopes_yaml
from mudraid_platform_middleware.exceptions import (
    MudraIDInvalidTokenError,
    MudraIDJwksError,
    MudraIDScopesYamlError,
)
from mudraid_platform_middleware.v2 import DecideContext, V2Config

_logger = logging.getLogger("mudraid_platform_middleware")

_DEFAULT_YAML_PATH = "mudraid_scopes.yaml"

# ── There is deliberately NO default JWKS URL ────────────────────────────────
#
# This used to compile in ``https://api.mudraid.io/.well-known/jwks.json``, then
# ``…mudraid.ai/…`` once the domain was corrected. Both are wrong for the same
# structural reason: a MudraID hostname baked into the package makes the build
# environment-specific, and the same artifact must run against staging and
# production with only configuration changing.
#
# It was also wrong in a quieter way. A JWKS default is the worst place to guess
# — an unconfigured platform would fetch its SIGNING KEYS from that host, and
# whoever answered would be deciding which tokens the platform accepts. Guessing
# a production origin for a staging deployment fails CLOSED at best and reaches
# the wrong environment at worst.
#
# So the URL is required. An unconfigured middleware raises at construction with
# a message naming the setting, instead of failing on its first request against
# a host nobody chose. The docs already told customers to set it; this makes the
# package agree with the docs.
_JWKS_URL_ENV = "MUDRAID_JWKS_URL"

# Reason (from JwtValidator) → stable error_code surfaced to the agent.
# Locked so a future renaming of internal reasons doesn't break a
# platform's error-handling code.
_REASON_TO_ERROR_CODE = {
    "malformed": "INVALID_TOKEN",
    "expired": "EXPIRED_TOKEN",
    "not_yet_valid": "TOKEN_NOT_YET_VALID",
    "wrong_audience": "WRONG_AUDIENCE",
    "wrong_issuer": "WRONG_ISSUER",
    "invalid_signature": "INVALID_TOKEN",
}


class MudraIDMiddleware(BaseHTTPMiddleware):
    """Starlette / FastAPI middleware enforcing MudraID-issued JWTs.

    Args:
        app: The ASGI app to wrap.
        scopes_yaml_path: Override the default location of
            ``mudraid_scopes.yaml``. Defaults to ``./mudraid_scopes.yaml``
            resolved against the process working directory. (V1 mode only.)
        jwks_url: MudraID's JWKS endpoint for the environment this platform is
            registered with. Falls back to the ``MUDRAID_JWKS_URL`` env var.
            REQUIRED in V1 mode — there is no default, because a compiled-in
            host would make this build environment-specific and would have the
            middleware fetch its signing keys from somewhere nobody chose.
            (V1 mode only.)
        mode: ``"v1"`` (default) selects the static route-scope enforcement;
            ``"v2"`` selects the V2 control loop and REQUIRES ``v2_config``.
            V1 behaviour is byte-identical whether or not V2 is available.
        v2_config: The :class:`~mudraid_platform_middleware.v2.V2Config` selecting V2
            mode. Must be supplied iff ``mode="v2"``; must be absent otherwise.
    """

    def __init__(
        self,
        app: ASGIApp,
        scopes_yaml_path: str | None = None,
        jwks_url: str | None = None,
        expected_issuer: str | Sequence[str] | None = None,
        mode: str = "v1",
        v2_config: V2Config | None = None,
    ) -> None:
        super().__init__(app)
        if mode not in ("v1", "v2"):
            raise ValueError(f"mode must be 'v1' or 'v2', not {mode!r}")
        if mode == "v2" and v2_config is None:
            raise ValueError("mode='v2' requires a v2_config")
        if mode == "v1" and v2_config is not None:
            raise ValueError("v2_config is only valid with mode='v2'")
        self._mode = mode
        self._v2_config = v2_config
        self._scopes_yaml_path = scopes_yaml_path or _DEFAULT_YAML_PATH
        self._jwks_url = (jwks_url or os.getenv(_JWKS_URL_ENV, "").strip()) or ""
        if mode == "v1" and not self._jwks_url:
            # Raised at construction, not on the first request: a platform that
            # boots and then rejects every token is far harder to diagnose than
            # one that refuses to boot and says why.
            raise ValueError(
                "no JWKS URL configured. Set the jwks_url argument or the "
                f"{_JWKS_URL_ENV} environment variable to the "
                "/.well-known/jwks.json path on the MudraID environment your "
                "platform is registered with — the portal prints it on the "
                "credential screen. There is no default: a compiled-in host "
                "would make this build environment-specific."
            )
        # Issuer the validator will accept. Defaults to None → the validator's
        # own default single issuer. A transition SET (for a verifier-before-
        # issuer rollout, charter §16) may be supplied here or via the
        # comma-separated env var MUDRAID_EXPECTED_ISSUERS (0 → default,
        # 1 → single issuer, ≥2 → accept any of them).
        self._expected_issuer = self._resolve_expected_issuer(expected_issuer)
        # Bootstrapped state — all built lazily on first dispatch.
        self._scopes: ScopesYaml | None = None
        self._matcher: RouteMatcher | None = None
        self._validator: JwtValidator | None = None
        self._bootstrap_lock = asyncio.Lock()

    @staticmethod
    def _resolve_expected_issuer(
        expected_issuer: str | Sequence[str] | None,
    ) -> str | Sequence[str] | None:
        """Resolve the accepted issuer(s): explicit arg wins, else env.

        ``MUDRAID_EXPECTED_ISSUERS`` is comma-separated:
          - unset/empty → ``None`` (validator uses its default single issuer)
          - one value   → that single issuer (str)
          - two or more → a tuple accepted as a transition set
        """
        if expected_issuer is not None:
            return expected_issuer
        raw = os.getenv("MUDRAID_EXPECTED_ISSUERS", "")
        values = tuple(v.strip() for v in raw.split(",") if v.strip())
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return values

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        # V2 mode is a wholly separate control loop; it never touches the
        # static YAML bootstrap below. V1 mode falls through unchanged.
        if self._mode == "v2":
            return await self._dispatch_v2(request, call_next)

        # 1. Ensure the four building blocks are wired. A YAML schema
        #    error here is an operator-side configuration problem;
        #    we turn it into a structured 500 instead of letting
        #    Starlette throw a generic uncaught exception. The
        #    failure is NOT cached — the next request will re-attempt
        #    the bootstrap so an operator who fixes the YAML doesn't
        #    have to restart the process.
        try:
            await self._ensure_bootstrapped()
        except MudraIDScopesYamlError as exc:
            _logger.error("MudraIDMiddleware bootstrap failed: %s", exc)
            return _error(
                500,
                "MIDDLEWARE_NOT_READY",
                "scope configuration is unavailable; check the server logs",
            )
        # An explicit guard rather than ``assert``: the matcher / validator
        # are guaranteed non-None after a successful ``_ensure_bootstrapped``,
        # but an ``assert`` is stripped under ``python -O`` whereas this
        # check survives and gives mypy the same narrowing.
        if self._matcher is None or self._validator is None:  # pragma: no cover
            return _error(500, "MIDDLEWARE_NOT_READY", "internal bootstrap invariant violation")

        # 2. Match (method, path) against the YAML's route list.
        rule = self._matcher.match(request.method, request.url.path)
        if rule is None:
            # Not covered by mudraid_scopes.yaml — treat as nonexistent
            # for agents. A platform front-end accessing the same path
            # outside this middleware (e.g. on a different port) is
            # unaffected.
            return _route_not_found()

        # 3 / 4. Public and skip handling — early exits, no token check.
        if rule.public:
            return await call_next(request)
        if rule.skip:
            return _route_not_found()

        # 5. Scope-gated route. Pull the Bearer token off the request.
        token = _extract_bearer_token(request)
        if token is None:
            return _error(
                401,
                "MISSING_TOKEN",
                "Authorization header missing or not in 'Bearer <token>' form",
            )

        # 6a. Verify the JWT.
        try:
            claims = await self._validator.validate(token)
        except MudraIDInvalidTokenError as exc:
            error_code = _REASON_TO_ERROR_CODE.get(exc.reason, "INVALID_TOKEN")
            return _error(401, error_code, str(exc))
        except MudraIDJwksError as exc:
            # We couldn't verify — distinct from "we verified and
            # rejected". The agent / operator needs to retry, not
            # fix their credentials.
            _logger.warning("JWKS unavailable while verifying request: %s", exc)
            return _error(500, "JWKS_UNAVAILABLE", "could not verify token signature")

        # 6b. Scope membership. The required scope MUST be present in
        # the token's `scopes` claim.
        token_scopes = claims.get("scopes")
        if not isinstance(token_scopes, list) or rule.scope not in token_scopes:
            return _error(
                403,
                "MISSING_SCOPE",
                f"required scope '{rule.scope}' not present in token",
            )

        # 7. Every gate passed. Forward.
        return await call_next(request)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _ensure_bootstrapped(self) -> None:
        if self._matcher is not None and self._validator is not None:
            return
        async with self._bootstrap_lock:
            if self._matcher is not None and self._validator is not None:
                return
            scopes = load_scopes_yaml(self._scopes_yaml_path)
            jwks_client = JwksClient(jwks_url=self._jwks_url)
            # Pass expected_issuer only when configured, so the default
            # construction (single literal issuer) stays byte-identical.
            if self._expected_issuer is None:
                validator = JwtValidator(
                    jwks_client=jwks_client,
                    expected_audience=scopes.platform_id,
                )
            else:
                validator = JwtValidator(
                    jwks_client=jwks_client,
                    expected_audience=scopes.platform_id,
                    expected_issuer=self._expected_issuer,
                )
            # Publish atomically — every cross-field invariant
            # (matcher built from same scopes the validator is bound
            # to) holds at the moment we expose them.
            self._scopes = scopes
            self._matcher = RouteMatcher(scopes.routes)
            self._validator = validator
            _logger.info(
                "MudraIDMiddleware bootstrapped: platform_id=%s, %d routes",
                scopes.platform_id,
                len(scopes.routes),
            )

    # ------------------------------------------------------------------
    # V2 mode
    # ------------------------------------------------------------------

    async def _dispatch_v2(self, request: Request, call_next: Any) -> Response:
        """Run the V2 enforcement control loop before the handler.

        The DECISION tree lives in :func:`evaluate_v2`; this method only
        extracts facts from the live request, drives the injected
        :class:`DecideClient`, and translates the resulting
        :class:`V2Decision` into either a forward (handler runs) or a
        structured deny. It never optimistically allows.
        """
        cfg = self._v2_config
        assert cfg is not None  # guaranteed by the constructor's mode check

        # 0. Unprotected surface: pass through untouched — nothing is stripped
        #    or decided (mirrors the contract's surface_not_protected outcome).
        protected = cfg.is_protected(request.url.path)
        if not protected:
            return await call_next(request)

        # 1. Strip reserved x-mudraid-* headers FIRST — before ANY evaluation,
        #    and even on requests that will be denied. Client-supplied trusted
        #    context is never accepted as an input fact.
        presented = _strip_reserved_headers(request)

        # 2. Fact extraction. Only a protected POST needs a body read + parse;
        #    control verbs and non-POST methods are classified before framing,
        #    exactly as the control loop's branch order requires.
        body_readable = True
        body_too_large = False
        json_shape = "not_json"
        jsonrpc: str | None = None
        rpc_method: str | None = None
        tool_name: str | None = None
        action: str | None = None
        action_mapped = False

        if protected and request.method.upper() == "POST":
            body, body_too_large, body_readable = await _read_bounded_body(
                request, cfg.max_body_bytes
            )
            if body_readable and not body_too_large:
                json_shape, payload = _classify_json(body)
                if json_shape == "object" and payload is not None:
                    raw_jsonrpc = payload.get("jsonrpc")
                    jsonrpc = raw_jsonrpc if isinstance(raw_jsonrpc, str) else None
                    raw_method = payload.get("method")
                    rpc_method = raw_method if isinstance(raw_method, str) else None
                    params = payload.get("params")
                    if isinstance(params, dict):
                        raw_name = params.get("name")
                        tool_name = raw_name if isinstance(raw_name, str) else None
                    if rpc_method == "tools/call" and valid_tool_name(tool_name):
                        # Exact, case-sensitive canonical action resolution.
                        assert tool_name is not None
                        action = cfg.decide_client.resolve_action(tool_name)
                        action_mapped = action is not None

        facts = V2RequestFacts(
            protected=protected,
            reserved_headers_presented=presented,
            bundle_active=bool(cfg.decide_client.bundle_active),
            method=request.method,
            body_readable=body_readable,
            body_too_large=body_too_large,
            json_shape=json_shape,
            jsonrpc=jsonrpc,
            rpc_method=rpc_method,
            tool_name=tool_name,
            action_mapped=action_mapped,
            action=action,
        )

        decision = await evaluate_v2(
            facts,
            self._make_decide(cfg, request),
            public_methods=cfg.public_methods,
        )

        if decision.forward:
            # Trusted context is injected ONLY after a bound allow — and only
            # when every name/value is a legal header field. An unencodable or
            # CRLF-bearing value is a deny, not a 500 and not a smuggled header.
            if not _inject_trusted_context(request, decision.trusted_context):
                _logger.error("V2 refusing to forward: trusted context is not a legal header set")
                return _error(
                    503,
                    "ENFORCE_TRUSTED_CONTEXT_UNREPRESENTABLE",
                    "the authorized decision could not be conveyed to the handler; "
                    "the request was not forwarded",
                )
            return await call_next(request)
        return _v2_error(decision)

    def _make_decide(self, cfg: V2Config, request: Request) -> Any:
        """Wrap the injected client's ``decide`` so any failure deny-closes.

        A timeout or transport error must never propagate as an exception that
        could bubble past the control loop and be mistaken for a soft failure;
        it is normalized to a deny-closed :class:`DecideResult`. No token or
        secret is logged — only the exception type name.

        The middleware applies its OWN deadline (``cfg.decide_timeout_sec``)
        rather than trusting the injected client to bound itself. The contract
        promises ``on_timeout=deny``; a client that never returns would otherwise
        hold the protected request open forever, which is neither a deny nor an
        allow but an indefinite hang — the one outcome the failure vocabulary has
        no word for. ``asyncio.wait_for`` converts it into the word it does have.
        """
        timeout = cfg.decide_timeout_sec
        # The authority establishes identity ONLY from the presented
        # protected-action token, and deny-closes when none is supplied. The raw
        # header is captured here, on the live request, because the control loop
        # deliberately knows nothing about transports — and forwarded ONLY to the
        # internal authority. It is never logged.
        context = DecideContext(
            presented_bearer=request.headers.get("authorization"),
            http_method=request.method,
            path=request.url.path,
            correlation_id=request.headers.get("x-correlation-id"),
        )

        async def _decide(action: str) -> DecideResult:
            try:
                call = cfg.decide_client.decide(action, context)
                if timeout is None:
                    return await call
                return await asyncio.wait_for(call, timeout)
            except asyncio.TimeoutError:
                # Covers both the client's own timeout and ours; the control
                # loop maps "timeout" to deadline_exceeded -> deny-closed 503.
                return DecideResult("timeout")
            except asyncio.CancelledError:
                # The caller is going away (client disconnect, server shutdown).
                # Never swallowed into a decision: re-raised so the request dies
                # rather than being recorded as an authority outcome.
                raise
            except Exception as exc:  # noqa: BLE001 — deny-closed on ANY failure
                _logger.warning("V2 /decide unavailable: %s", type(exc).__name__)
                return DecideResult("error")

        return _decide


# ---- helpers (module-level so dispatch stays readable) -------------------


def _extract_bearer_token(request: Request) -> str | None:
    """Return the token portion of the ``Authorization`` header.

    Returns ``None`` when the header is missing, doesn't start with
    ``Bearer`` (case-insensitive), or has an empty token after the
    scheme. The middleware treats all three the same — there's no
    credential to validate.
    """
    raw = request.headers.get("Authorization")
    if not raw:
        return None
    parts = raw.split(" ", 1)
    if len(parts) != 2:
        return None
    scheme, token = parts[0], parts[1].strip()
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _route_not_found() -> Response:
    """Single 404 shape for both 'not in YAML' and 'skip: true' rules.

    Locked: the agent must NOT be able to distinguish these. Revealing
    that the route exists-but-is-hidden would leak topology.
    """
    return _error(404, "ROUTE_NOT_FOUND", "no agent route at this path")


def _error(status_code: int, error_code: str, message: str) -> Response:
    """Build the structured error response shape locked in docs/openapi.yaml.

    Body: ``{"error_code": "...", "message": "..."}``. No stack traces,
    no upstream library detail — just the stable code + a human-readable
    message safe to show in an agent's logs.
    """
    return JSONResponse(
        status_code=status_code,
        content={"error_code": error_code, "message": message},
    )


# ---- V2-mode helpers -----------------------------------------------------


def _v2_error(decision: V2Decision) -> Response:
    """Render a denied :class:`V2Decision` in the locked error-body shape.

    Reuses :func:`_error`, so V2 denies carry the same
    ``{"error_code", "message"}`` shape platforms already build alerting on.
    The ``error_code`` values mirror the portable adapter-decision contract so
    an agent sees consistent codes across Kong and this middleware.
    """
    error_code = decision.error_code or "ENFORCE_DENIED"
    return _error(decision.http_status, error_code, decision.message)


def _strip_reserved_headers(request: Request) -> tuple[str, ...]:
    """Remove every reserved ``x-mudraid-*`` header from the live request.

    Mutates ``request.scope["headers"]`` in place so the downstream handler
    never observes client-forged trusted context, and returns the names that
    were presented (original case preserved) so the control loop can record
    what was stripped. Called FIRST, before any evaluation.
    """
    presented: list[str] = []
    kept: list[tuple[bytes, bytes]] = []
    for raw_name, raw_value in request.scope.get("headers", []):
        name = raw_name.decode("latin-1")
        if is_reserved_header(name):
            presented.append(name)
            continue
        kept.append((raw_name, raw_value))
    request.scope["headers"] = kept
    return tuple(presented)


# A legal HTTP field name (RFC 9110 §5.1 token) and a legal field value
# (RFC 9110 §5.5 field-content: visible ASCII, SP and HTAB — notably NO CR, LF
# or NUL). Trusted context is validated against these before it is written into
# the ASGI scope: an unvalidated value carrying CRLF would append attacker- or
# operator-chosen headers to what the handler sees, and a non-latin-1 value
# would raise UnicodeEncodeError on the ALLOW path — a 500 on a request the
# authority permitted.
_FIELD_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_FIELD_VALUE_RE = re.compile(r"^[\t\x20-\x7e\x80-\xff]*$")


def _is_legal_header(name: str, value: str) -> bool:
    """Whether ``(name, value)`` can be safely written into the ASGI scope."""
    return bool(_FIELD_NAME_RE.match(name)) and bool(_FIELD_VALUE_RE.match(value))


def _inject_trusted_context(request: Request, trusted_context: tuple[tuple[str, str], ...]) -> bool:
    """Append trusted context headers to the live request (allow path only).

    The reserved-header strip has already removed any client-supplied
    ``x-mudraid-*`` headers, so the handler sees ONLY the values the middleware
    itself injects after a bound allow.

    Returns ``False`` — injecting NOTHING — when any pair is not a legal header
    field. The caller turns that into a deny: a bound allow the middleware cannot
    convey intact is not an allow it may forward on. Validation is all-or-nothing
    so the handler never sees a partial context that would read as a complete one.
    """
    validated: list[tuple[bytes, bytes]] = []
    for name, value in trusted_context:
        if not _is_legal_header(name, value):
            return False
        validated.append((name.encode("latin-1"), value.encode("latin-1")))
    request.scope["headers"] = list(request.scope.get("headers", [])) + validated
    return True


async def _read_bounded_body(request: Request, max_bytes: int) -> tuple[bytes, bool, bool]:
    """Read the request body under a bounded-framing limit.

    Returns ``(body, too_large, readable)``. A ``Content-Length`` over the limit
    is rejected without reading; a body that exceeds the limit ON THE WIRE is
    rejected as soon as it does, without buffering the remainder; a body that
    cannot be read (client disconnect, etc.) is reported unreadable.

    THE BOUND IS ENFORCED WHILE READING, NOT AFTER. ``await request.body()``
    buffers the whole body first and only then permits a size check, so the
    ``Content-Length`` pre-check was the only real bound — and a chunked request
    declares no ``Content-Length`` at all. A client could therefore stream an
    unbounded body into memory through a limit whose entire purpose is to say it
    cannot. Streaming with a running total closes that: the read stops at the
    first chunk that crosses ``max_bytes`` and the rest is never accumulated.

    On the under-limit path the assembled body is published as
    ``request._body`` — the same attribute ``starlette.requests.Request.body()``
    sets and the same one ``BaseHTTPMiddleware._CachedRequest.wrapped_receive``
    replays downstream. Consuming the stream WITHOUT publishing it would hand the
    wrapped handler an empty body, so this is load-bearing, not an optimisation.
    On the over-limit path nothing is published, which is safe because the caller
    denies (413) and never invokes the handler.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_bytes:
                return b"", True, True
        except ValueError:
            pass

    # Already buffered by an earlier reader (another middleware, or a re-entry):
    # nothing left to stream, so bound what we have.
    cached = getattr(request, "_body", None)
    if cached is not None:
        return (b"", True, True) if len(cached) > max_bytes else (cached, False, True)

    chunks: list[bytes] = []
    total = 0
    try:
        async for chunk in request.stream():
            total += len(chunk)
            if total > max_bytes:
                # Stop reading here. The connection is abandoned mid-body, which
                # is the correct cost of refusing an oversized frame — the
                # alternative is to finish buffering something already rejected.
                return b"", True, True
            chunks.append(chunk)
    except Exception:  # noqa: BLE001 — any read failure is "unreadable", deny-closed
        return b"", False, False

    body = b"".join(chunks)
    request._body = body  # noqa: SLF001 - see docstring: this IS Request.body()'s contract
    return body, False, True


def _classify_json(body: bytes) -> tuple[str, dict[str, Any] | None]:
    """Classify a raw body as ``object`` / ``array`` / ``scalar`` / ``not_json``.

    Only an ``object`` yields a parsed payload; the other shapes are terminal
    (a JSON array is a batch, rejected wholesale — never partially evaluated).
    """
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return "not_json", None
    if isinstance(parsed, dict):
        return "object", parsed
    if isinstance(parsed, list):
        return "array", None
    return "scalar", None
