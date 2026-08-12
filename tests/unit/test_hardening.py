"""Hardening regressions for the V2 middleware and the JWKS client.

Each test here holds a defect that was present and is now closed. They are
grouped by the property they defend, and every one is written so that reverting
the corresponding fix turns it red — a test that would pass against the
pre-fix file is not evidence about the fix.

The five properties:

  1. ``V2Config.public_methods`` is HONOURED. It was a declared, typed,
     documented knob that ``evaluate_v2`` never read, so an operator narrowing
     the control/discovery allowlist got no narrowing at all.
  2. The bounded-framing limit BOUNDS THE READ. It was applied after
     ``await request.body()`` had already buffered the whole body, and the
     ``Content-Length`` pre-check does not exist for a chunked request.
  3. Trusted context is a LEGAL HEADER SET or the request does not forward. An
     unencodable value raised on the allow path; a CRLF-bearing one would have
     been written into the ASGI scope verbatim.
  4. ``/decide`` has an ADAPTER-SIDE DEADLINE. The contract promises
     ``on_timeout=deny``; a client that never returns produced neither.
  5. The JWKS reactive refresh is RATE-LIMITED BY OUTCOME — bounding an
     unauthenticated caller's ability to drive outbound fetches, without
     delaying a real rotation (locked decision D4).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI
from httpx import ASGITransport
from starlette.requests import Request

from mudraid_platform_middleware import (
    DecideContext,
    DecideResult,
    MudraIDMiddleware,
    V2Config,
)
from mudraid_platform_middleware._jwks_client import JwksClient
from mudraid_platform_middleware._route_matcher import RouteMatcher
from mudraid_platform_middleware._v2_control_loop import (
    DEFAULT_PUBLIC_METHODS,
    KONG_DEFAULT_PUBLIC_METHODS,
    V2RequestFacts,
    evaluate_v2,
)
from mudraid_platform_middleware._yaml_loader import RouteRule
from mudraid_platform_middleware.exceptions import MudraIDJwksError
from tests.unit.test_middleware_v2 import FakeDecideClient, _make_v2_app

JWKS_URL = "https://api.mudraid.test/.well-known/jwks.json"


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def _never_decides(action: str) -> DecideResult:  # pragma: no cover - guard
    raise AssertionError(f"/decide must not be called for action {action!r}")


# ---------------------------------------------------------------------------
# 1. V2Config.public_methods is honoured
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_public_methods_none_selects_the_contract_default() -> None:
    """Unset means the portable contract's allowlist — the prior behaviour."""
    for method in sorted(DEFAULT_PUBLIC_METHODS):
        decision = await evaluate_v2(
            V2RequestFacts(rpc_method=method), _never_decides, public_methods=None
        )
        assert decision.outcome == "allow", method
        assert decision.reason_code == "control_plane_passthrough"


@pytest.mark.asyncio
async def test_narrowed_public_methods_denies_the_removed_method() -> None:
    """The knob that did nothing. ``resources/list`` is in the default set, so
    a pre-fix ``evaluate_v2`` passed it through no matter what was configured."""
    decision = await evaluate_v2(
        V2RequestFacts(rpc_method="resources/list"),
        _never_decides,
        public_methods=KONG_DEFAULT_PUBLIC_METHODS,
    )
    assert decision.outcome == "deny"
    assert decision.error_code == "ENFORCE_MESSAGE_NOT_ALLOWED"
    assert decision.http_status == 403


@pytest.mark.asyncio
async def test_empty_public_methods_is_honoured_not_re_expanded() -> None:
    """An empty set is the strictest legal configuration, not "unset"."""
    decision = await evaluate_v2(
        V2RequestFacts(rpc_method="initialize"), _never_decides, public_methods=frozenset()
    )
    assert decision.outcome == "deny"
    assert decision.error_code == "ENFORCE_MESSAGE_NOT_ALLOWED"


@pytest.mark.asyncio
async def test_widened_public_methods_passes_a_non_default_method() -> None:
    decision = await evaluate_v2(
        V2RequestFacts(rpc_method="completion/complete"),
        _never_decides,
        public_methods=frozenset({"completion/complete"}),
    )
    assert decision.outcome == "allow"
    assert decision.reason_code == "control_plane_passthrough"


@pytest.mark.asyncio
async def test_notifications_still_pass_under_an_empty_allowlist() -> None:
    """``notifications/*`` is a prefix rule, deliberately separate from the
    allowlist — narrowing the allowlist must not silently also close it."""
    decision = await evaluate_v2(
        V2RequestFacts(rpc_method="notifications/cancelled"),
        _never_decides,
        public_methods=frozenset(),
    )
    assert decision.outcome == "allow"
    assert decision.reason_code == "notification_passthrough"


@pytest.mark.asyncio
async def test_config_public_methods_reaches_the_live_middleware() -> None:
    """End-to-end: the knob has to survive the trip through ``_dispatch_v2``,
    not merely be accepted by ``evaluate_v2``."""
    fake = FakeDecideClient()
    app = _make_v2_app(fake, public_methods=KONG_DEFAULT_PUBLIC_METHODS)
    async with _client(app) as http:
        denied = await http.post(
            "/mcp", json={"jsonrpc": "2.0", "method": "resources/list", "params": {}}
        )
        allowed = await http.post(
            "/mcp", json={"jsonrpc": "2.0", "method": "tools/list", "params": {}}
        )
    assert denied.status_code == 403
    assert denied.json()["error_code"] == "ENFORCE_MESSAGE_NOT_ALLOWED"
    assert allowed.status_code == 200


def test_the_two_adapters_agree_on_the_public_method_default() -> None:
    """Parity, not a recorded divergence.

    This test used to assert that the two defaults DIFFERED — it guarded the
    documentation of a gap instead of closing it, so identical facts produced
    different outcomes at a Kong gateway and at this middleware. The Python
    default is now the conservative Kong set, and this asserts the agreement.
    """
    assert DEFAULT_PUBLIC_METHODS == KONG_DEFAULT_PUBLIC_METHODS
    assert DEFAULT_PUBLIC_METHODS == frozenset({"initialize", "ping", "tools/list"})


def test_disclosure_methods_are_not_public_by_default() -> None:
    """``resources/list`` and ``prompts/list`` enumerate a surface's contents.

    They are disclosure, so they require explicit configuration rather than
    arriving permitted.
    """
    assert "resources/list" not in DEFAULT_PUBLIC_METHODS
    assert "prompts/list" not in DEFAULT_PUBLIC_METHODS


# ---------------------------------------------------------------------------
# 2. The bounded-framing limit bounds the READ
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chunked_oversized_body_is_refused_without_buffering_it_all() -> None:
    """No ``Content-Length``, so the pre-check cannot fire. Pre-fix, the whole
    body was buffered by ``request.body()`` and only then measured.

    The chunk counter is the assertion that matters: it proves the read STOPPED
    rather than that the request was eventually refused.
    """
    fake = FakeDecideClient()
    app = _make_v2_app(fake, max_body_bytes=1024)

    sent = 0

    async def chunks() -> Any:
        nonlocal sent
        for _ in range(500):  # 500 KiB if fully consumed
            sent += 1
            yield b"x" * 1024

    async with _client(app) as http:
        response = await http.post("/mcp", content=chunks())

    assert response.status_code == 413
    assert response.json()["error_code"] == "ENFORCE_BODY_TOO_LARGE"
    # Two chunks is enough to cross a 1 KiB limit; anything near 500 means the
    # read ran to completion and the "bound" bounded nothing.
    assert sent <= 4, f"read {sent} chunks past a 1 KiB limit"


@pytest.mark.asyncio
async def test_under_limit_body_still_reaches_the_handler_intact() -> None:
    """The streaming read must publish what it consumed.

    Consuming ``request.stream()`` without republishing it hands
    ``BaseHTTPMiddleware``'s wrapped receive an EMPTY body — the handler would
    run on an allow and see nothing. This is the test that catches that.
    """
    fake = FakeDecideClient()
    app = _make_v2_app(fake)
    payload = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "issue_refund"}}
    async with _client(app) as http:
        response = await http.post("/mcp", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["ran"] is True
    # The handler's own ``await request.body()`` returned the real bytes.
    assert '"issue_refund"' in body["body"]


@pytest.mark.asyncio
async def test_declared_content_length_over_the_limit_is_refused_unread() -> None:
    """The cheap pre-check is retained: an honest oversized declaration is
    refused before a single byte is read."""
    fake = FakeDecideClient()
    app = _make_v2_app(fake, max_body_bytes=16)
    async with _client(app) as http:
        response = await http.post("/mcp", content=b"y" * 4096)
    assert response.status_code == 413
    assert response.json()["error_code"] == "ENFORCE_BODY_TOO_LARGE"


# ---------------------------------------------------------------------------
# 3. Trusted context must be a legal header set
# ---------------------------------------------------------------------------


class _HostileActionClient(FakeDecideClient):
    """Maps a tool to an action name that is not a legal header value."""

    def __init__(self, action: str) -> None:
        super().__init__(action_map={"issue_refund": action})


@pytest.mark.parametrize(
    ("action", "why"),
    [
        ("refunds.issue\r\nx-forged: 1", "CRLF would append a header downstream"),
        ("refunds.issue\nx-forged: 1", "bare LF is enough on some parsers"),
        ("refunds.€issue", "non-latin-1 raised UnicodeEncodeError on the allow path"),
    ],
)
@pytest.mark.asyncio
async def test_unrepresentable_trusted_context_denies_instead_of_forwarding(
    action: str, why: str
) -> None:
    fake = _HostileActionClient(action)
    app = _make_v2_app(fake)
    payload = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "issue_refund"}}
    async with _client(app) as http:
        response = await http.post("/mcp", json=payload)

    assert response.status_code == 503, why
    assert response.json()["error_code"] == "ENFORCE_TRUSTED_CONTEXT_UNREPRESENTABLE"
    # And the handler did NOT run — a decision we could not convey is not one
    # we may act on.
    assert "ran" not in response.json()


@pytest.mark.asyncio
async def test_legal_trusted_context_still_injects() -> None:
    fake = FakeDecideClient()
    app = _make_v2_app(fake)
    payload = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "issue_refund"}}
    async with _client(app) as http:
        response = await http.post("/mcp", json=payload)
    body = response.json()
    assert body["action_key"] == "refunds.issue"
    assert body["decision_id"] == "dec-abc"


# ---------------------------------------------------------------------------
# 4. The adapter's own /decide deadline
# ---------------------------------------------------------------------------


class _HangingDecideClient(FakeDecideClient):
    """A client that never returns — the failure mode with no word for it."""

    async def decide(self, action: str, context: DecideContext) -> DecideResult:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")  # pragma: no cover


@pytest.mark.asyncio
async def test_a_hanging_decide_client_is_deny_closed_by_our_own_deadline() -> None:
    app = _make_v2_app(_HangingDecideClient(), decide_timeout_sec=0.05)
    payload = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "issue_refund"}}
    async with _client(app) as http:
        response = await asyncio.wait_for(http.post("/mcp", json=payload), timeout=5)

    assert response.status_code == 503
    assert response.json()["error_code"] == "ENFORCE_DECIDE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_the_deadline_does_not_fire_on_a_prompt_client() -> None:
    fake = FakeDecideClient()
    app = _make_v2_app(fake, decide_timeout_sec=5.0)
    payload = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "issue_refund"}}
    async with _client(app) as http:
        response = await http.post("/mcp", json=payload)
    assert response.status_code == 200
    assert fake.decide_calls == ["refunds.issue"]


def test_the_default_deadline_matches_the_kong_plugin() -> None:
    """``schema.lua`` defaults ``decide_timeout_ms`` to 2000. Two adapters
    claiming one contract should not disagree about when it has expired."""
    assert V2Config(decide_client=FakeDecideClient()).decide_timeout_sec == 2.0


# ---------------------------------------------------------------------------
# 5. JWKS reactive refresh is rate-limited by OUTCOME
# ---------------------------------------------------------------------------


def _jwks_body(*kids: str) -> dict[str, Any]:
    return {
        "keys": [
            {"kty": "RSA", "use": "sig", "alg": "RS256", "kid": k, "n": "AQAB", "e": "AQAB"}
            for k in kids
        ]
    }


@pytest.mark.asyncio
async def test_a_flood_of_unknown_kids_costs_one_fetch_not_one_each() -> None:
    """The amplifier. Pre-fix, each invented ``kid`` drove one outbound JWKS
    request; an unauthenticated caller could therefore aim this middleware's
    request rate at MudraID's own key endpoint."""
    with respx.mock() as r:
        route = r.get(JWKS_URL).mock(return_value=httpx.Response(200, json=_jwks_body("kid-1")))
        client = JwksClient(JWKS_URL, refresh_cooldown_sec=60.0)

        await client.get_key("kid-1")  # cold fetch: 1
        for i in range(25):
            with pytest.raises(MudraIDJwksError):
                await client.get_key(f"invented-{i}")

    # 1 cold fetch + at most 1 reactive fetch that returned the same key set.
    assert route.call_count <= 2, f"{route.call_count} fetches for 25 invented kids"


@pytest.mark.asyncio
async def test_a_real_rotation_is_still_picked_up_immediately() -> None:
    """Locked decision D4 survives the rate limit: the backoff is armed by an
    UNPRODUCTIVE fetch, and a rotation is by definition productive."""
    with respx.mock() as r:
        route = r.get(JWKS_URL).mock(
            side_effect=[
                httpx.Response(200, json=_jwks_body("kid-old")),
                httpx.Response(200, json=_jwks_body("kid-old", "kid-new")),
                httpx.Response(200, json=_jwks_body("kid-old", "kid-new", "kid-newer")),
            ]
        )
        client = JwksClient(JWKS_URL, refresh_cooldown_sec=3600.0)

        await client.get_key("kid-old")
        # Each rotation clears the backoff, so the NEXT unknown kid still
        # fetches at full speed even inside a one-hour cooldown.
        assert (await client.get_key("kid-new"))["kid"] == "kid-new"
        assert (await client.get_key("kid-newer"))["kid"] == "kid-newer"

    assert route.call_count == 3


@pytest.mark.asyncio
async def test_a_failing_endpoint_is_not_redialled_on_every_request() -> None:
    with respx.mock() as r:
        route = r.get(JWKS_URL).mock(side_effect=httpx.ConnectError("refused"))
        client = JwksClient(JWKS_URL, refresh_cooldown_sec=60.0)

        for _ in range(10):
            with pytest.raises(MudraIDJwksError):
                await client.get_key("kid-1")

    assert route.call_count == 1, f"{route.call_count} dials at a host known to be down"


@pytest.mark.asyncio
async def test_an_expired_cache_in_backoff_refuses_rather_than_serving() -> None:
    """The security half of the rate limit.

    Falling through to a cache past its TTL would let a failing endpoint extend
    the life of every key in it — including one rotated OUT, which is exactly
    the revocation the TTL bounds.
    """
    with respx.mock() as r:
        r.get(JWKS_URL).mock(
            side_effect=[
                httpx.Response(200, json=_jwks_body("kid-1")),
                httpx.ConnectError("refused"),
            ]
        )
        client = JwksClient(JWKS_URL, cache_ttl_sec=0.0, refresh_cooldown_sec=60.0)

        await client.get_key("kid-1")  # populated, but TTL 0 -> instantly stale
        with pytest.raises(MudraIDJwksError):
            await client.get_key("kid-1")  # the failing refresh
        # Now in backoff with an expired cache holding kid-1. It must NOT serve.
        with pytest.raises(MudraIDJwksError, match="past its TTL"):
            await client.get_key("kid-1")


@pytest.mark.asyncio
async def test_operator_refresh_ignores_the_backoff() -> None:
    with respx.mock() as r:
        route = r.get(JWKS_URL).mock(return_value=httpx.Response(200, json=_jwks_body("kid-1")))
        client = JwksClient(JWKS_URL, refresh_cooldown_sec=3600.0)

        await client.get_key("kid-1")
        with pytest.raises(MudraIDJwksError):
            await client.get_key("ghost")  # arms the backoff
        await client.refresh()  # an operator drill must still do something

    assert route.call_count == 3


@pytest.mark.asyncio
async def test_non_signing_and_non_rsa_keys_are_not_admitted() -> None:
    """A key published for encryption is not a key for checking signatures
    (RFC 7517 §4.2), and this verifier accepts RS256 and nothing else."""
    body = {
        "keys": [
            {"kty": "RSA", "use": "sig", "kid": "sig-key", "n": "AQAB", "e": "AQAB"},
            {"kty": "RSA", "use": "enc", "kid": "enc-key", "n": "AQAB", "e": "AQAB"},
            {"kty": "EC", "use": "sig", "kid": "ec-key", "crv": "P-256", "x": "a", "y": "b"},
            {"kty": "oct", "kid": "symmetric-key", "k": "AQAB"},
        ]
    }
    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json=body))
        client = JwksClient(JWKS_URL)

        assert (await client.get_key("sig-key"))["kid"] == "sig-key"
        for rejected in ("enc-key", "ec-key", "symmetric-key"):
            with pytest.raises(MudraIDJwksError, match="unknown kid"):
                await client.get_key(rejected)


@pytest.mark.asyncio
async def test_an_oversized_jwks_body_is_refused_before_parsing() -> None:
    with respx.mock() as r:
        r.get(JWKS_URL).mock(
            return_value=httpx.Response(200, content=b"{" + b"a" * 2_000_000 + b"}")
        )
        client = JwksClient(JWKS_URL)
        with pytest.raises(MudraIDJwksError, match="exceeds"):
            await client.get_key("kid-1")


# ---------------------------------------------------------------------------
# 6. Route matching is anchored at end-of-STRING
# ---------------------------------------------------------------------------


def test_a_trailing_newline_does_not_satisfy_a_route_pattern() -> None:
    """Python's ``$`` also matches before a trailing newline, and an ASGI server
    percent-decodes the path — so ``/health%0a`` arrives as ``"/health\\n"``.
    Route identity should not turn on a regex dialect's convenience."""
    matcher = RouteMatcher([RouteRule(method="GET", path="/health", public=True)])
    assert matcher.match("GET", "/health") is not None
    assert matcher.match("GET", "/health\n") is None
    assert matcher.match("GET", "/health\r\n") is None


def test_parametric_segments_are_unchanged_by_the_anchor() -> None:
    """The anchor change is about the pattern's END, not about what ``{id}``
    accepts.

    ``[^/]+`` is a negated class, so it matches a newline like any other
    non-slash byte: ``/items/42\\n`` still matches ``/items/{id}`` and always
    did. That is the safe direction — the parameter simply captured a newline
    and the rule's scope gate applies to it exactly as it would to ``42``. The
    case ``\\Z`` fixes is a LITERAL pattern, where ``$`` let a path that is not
    the route satisfy the route.
    """
    matcher = RouteMatcher([RouteRule(method="GET", path="/items/{id}", scope="items:read")])
    assert matcher.match("GET", "/items/42") is not None
    assert matcher.match("GET", "/items/42/comments") is None
    # Matches, and is still scope-gated — no authority is granted by the newline.
    assert matcher.match("GET", "/items/42\n") is not None


# ---------------------------------------------------------------------------
# 7. Protected-surface matching
# ---------------------------------------------------------------------------
#
# Two rules compose, pulling in opposite directions on purpose:
#
#   * every SPELLING of the request path is considered, so an encoded or
#     dot-segmented path cannot skip the control loop — this can only ADD
#     protection;
#   * each spelling is tested at a SEGMENT BOUNDARY, so a neighbouring route
#     that merely starts with the same characters is not swept in — this can
#     only REMOVE over-protection.
#
# A lexical `startswith` over the raw path — what both adapters used to do —
# was wrong in both directions at once.

#: The protected-path case table, plus the encoding cases. Asserted here AND,
#: verbatim, in kong/tests/lua/test_mudraid_enforce.lua, so the two adapters
#: cannot drift apart on the rule that decides whether enforcement runs at all.
PROTECTED_PATH_CASES: list[tuple[str, bool, str]] = [
    ("/mcp", True, "the exact path"),
    ("/mcp/", True, "a trailing slash is the same surface"),
    ("/mcp/tools", True, "a descendant"),
    ("/mcp//tools", True, "an empty segment still lands inside the surface"),
    ("/mcpfoo", False, "a different route that shares a character prefix"),
    ("/mcp-evil", False, "likewise"),
    ("/mcpevil/steal", False, "a longer neighbour"),
    ("/mc", False, "a shorter path"),
    ("/mcp?x=1", True, "a query string must not escape the surface"),
    ("/mcp/tools?a=b", True, "a query on a descendant"),
    ("/%6dcp/messages", True, "percent-encoded — was a total bypass"),
    ("/mcp/../mcp/tools", True, "dot-segmented — was a total bypass"),
    ("/x%252fmcp", False, "decode-once: %252f does not invent a match"),
    ("/MCP", False, "paths are case-sensitive (RFC 3986 §6.2.2.1)"),
]


def _cfg(*prefixes: str) -> V2Config:
    return V2Config(decide_client=FakeDecideClient(), protected_paths=prefixes)


@pytest.mark.parametrize(("path", "expected", "why"), PROTECTED_PATH_CASES)
def test_the_two_adapters_agree_on_the_protected_surface(
    path: str, expected: bool, why: str
) -> None:
    """This table is the contract. Kong asserts the same rows."""
    assert _cfg("/mcp").is_protected(path) is expected, why


@pytest.mark.asyncio
async def test_neighbouring_route_passes_through_untouched() -> None:
    fake = FakeDecideClient()
    app = FastAPI()
    app.add_middleware(
        MudraIDMiddleware,
        mode="v2",
        v2_config=V2Config(decide_client=fake, protected_paths=("/mcp",)),
    )

    @app.post("/mcp-metrics")
    async def metrics(request: Request) -> dict:
        # A reserved header reaching the handler proves nothing was stripped,
        # i.e. the request was genuinely treated as unprotected.
        return {"ran": True, "forged": request.headers.get("x-mudraid-decision-id")}

    async with _client(app) as http:
        response = await http.post(
            "/mcp-metrics",
            json={"scrape": True},
            headers={"x-mudraid-decision-id": "client-supplied"},
        )

    assert response.status_code == 200
    assert response.json()["ran"] is True
    assert fake.decide_calls == [], "an unprotected route must not consult the authority"


@pytest.mark.asyncio
async def test_the_protected_surface_itself_still_enforces() -> None:
    """The tightening must not have turned the surface off."""
    fake = FakeDecideClient()
    app = _make_v2_app(fake, protected_paths=("/mcp",))
    payload = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "issue_refund"}}
    async with _client(app) as http:
        response = await http.post("/mcp", json=payload)
    assert response.status_code == 200
    assert fake.decide_calls == ["refunds.issue"]


def test_root_protects_everything() -> None:
    assert _cfg("/").is_protected("/anything/at/all") is True
    assert _cfg("/").is_protected("/") is True


def test_a_configured_trailing_slash_is_the_same_surface() -> None:
    assert _cfg("/mcp/").is_protected("/mcp/tools") is True
    assert _cfg("/mcp/").is_protected("/mcpfoo") is False


def test_each_configured_surface_gets_its_own_boundary() -> None:
    cfg = _cfg("/mcp", "/rpc")
    assert cfg.is_protected("/rpc/call") is True
    assert cfg.is_protected("/rpcfoo") is False


def test_none_still_means_every_route() -> None:
    cfg = V2Config(decide_client=FakeDecideClient())
    assert cfg.protected_paths is None
    assert cfg.is_protected("/anything") is True


# ---- ambiguous configuration fails at construction ------------------------


@pytest.mark.parametrize(
    ("prefix", "fragment"),
    [
        ("", "non-empty"),
        ("mcp", "absolute"),
        ("/mcp?x=1", "query string"),
        ("/mcp#frag", "query string"),
        ("/%6dcp", "percent-escapes"),
        ("/mcp/../admin", "'.' or '..'"),
        ("/mcp/./x", "'.' or '..'"),
        ("/mcp//x", "empty segments"),
    ],
)
def test_an_ambiguous_protected_path_is_refused_at_construction(prefix: str, fragment: str) -> None:
    """Construction is startup for this middleware.

    A prefix whose meaning depends on interpretation is refused where the
    operator can still see it, rather than resolved into something they did not
    write and then enforced for the life of the process.
    """
    with pytest.raises(ValueError, match=re.escape(fragment)):
        _cfg(prefix)


@pytest.mark.parametrize("prefix", ["/mcp", "/mcp/", "/mcp///", "/", "/a/b/c"])
def test_an_unambiguous_protected_path_is_accepted_and_normalized(prefix: str) -> None:
    cfg = _cfg(prefix)
    assert cfg.protected_paths is not None
    (normalized,) = cfg.protected_paths
    assert normalized == ("/" if prefix.strip("/") == "" else prefix.rstrip("/"))


def test_a_bare_string_is_not_silently_iterated_per_character() -> None:
    """``protected_paths="/mcp"`` is iterable, so it would become one prefix per
    CHARACTER — including ``"/"``, which protects every route on the app. A
    plausible typo that silently inverts the configuration is worth catching."""
    with pytest.raises(ValueError, match="not a single string"):
        V2Config(decide_client=FakeDecideClient(), protected_paths="/mcp")  # type: ignore[arg-type]
