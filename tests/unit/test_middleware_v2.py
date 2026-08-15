"""The explicit V2 adapter mode on ``MudraIDMiddleware``.

Two layers of test:

  1. **Contract conformance.** :func:`evaluate_v2` is the middleware's native
     encoding of the portable adapter-decision contract
     (``mudraid.adapter.decision/1``). :data:`MIRRORED_CORPUS` mirrors the
     positive / boundary / hostile fixtures of
     ``shared/mudraid_contracts/.../adapter-decision-corpus.json`` — the
     semantic oracle — and asserts the middleware reaches the SAME normalized
     outcome for each. The corpus is mirrored inline (not imported) so the SDK
     keeps zero dependency on the internal ``mudraid_contracts`` package.

  2. **End-to-end.** The real middleware in ``mode="v2"`` is mounted on a
     FastAPI app and driven through a real ASGI transport with an injected fake
     :class:`DecideClient` (no network): a bound allow forwards and injects
     trusted context; every honest-failure mode (no-bundle, unmapped,
     decide-timeout, decide-error, batch, oversized) deny-closes; and
     client-forged ``x-mudraid-*`` headers are stripped on every path.

Plus a regression test that ``mode="v1"`` (the default) is byte-for-byte the
static route-scope behaviour it always was.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from fastapi import FastAPI
from httpx import ASGITransport
from mudraid_platform_middleware import DecideContext, DecideResult, MudraIDMiddleware, V2Config
from mudraid_platform_middleware._v2_control_loop import V2RequestFacts, evaluate_v2
from starlette.requests import Request

from tests.conftest import baseline_claims, sign_jwt

# ---------------------------------------------------------------------------
# 1. Contract conformance — mirrored corpus driven through evaluate_v2
# ---------------------------------------------------------------------------

_TOOL_AT_MAX = "a" * 512
_TOOL_OVER_MAX = "a" * 513

#: Positive / boundary / hostile fixtures mirrored from the validator's
#: adapter-decision corpus. Each is ``(id, facts, expect)`` where ``expect``
#: mirrors the corpus's ``adapter_code`` under this middleware's ``error_code``.
MIRRORED_CORPUS: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    (
        "positive_allow_mapped_tool_decide_allow",
        {
            "tool_name": "issue_refund",
            "action_mapped": True,
            "decide": "allow",
        },
        {
            "outcome": "allow",
            "reason_code": "authorized",
            "reason_tier": "authorization",
            "http_status": 200,
            "error_code": None,
            "stripped": [],
        },
    ),
    (
        "positive_allow_strips_reserved_header",
        {
            "tool_name": "issue_refund",
            "action_mapped": True,
            "decide": "allow",
            "reserved_headers_presented": ["x-mudraid-decision-id", "authorization"],
        },
        {
            "outcome": "allow",
            "reason_code": "authorized",
            "reason_tier": "authorization",
            "http_status": 200,
            "error_code": None,
            "stripped": ["x-mudraid-decision-id"],
        },
    ),
    (
        "positive_control_verb_get_passes",
        {"method": "GET"},
        {
            "outcome": "allow",
            "reason_code": "control_plane_passthrough",
            "reason_tier": "transport",
            "http_status": 200,
            "error_code": None,
            "stripped": [],
        },
    ),
    (
        "positive_public_method_initialize_passes",
        {"rpc_method": "initialize"},
        {
            "outcome": "allow",
            "reason_code": "control_plane_passthrough",
            "reason_tier": "transport",
            "http_status": 200,
            "error_code": None,
            "stripped": [],
        },
    ),
    (
        "positive_client_notification_passes",
        {"rpc_method": "notifications/progress"},
        {
            "outcome": "allow",
            "reason_code": "notification_passthrough",
            "reason_tier": "transport",
            "http_status": 200,
            "error_code": None,
            "stripped": [],
        },
    ),
    (
        "positive_unprotected_surface_passthrough",
        {"protected": False, "reserved_headers_presented": ["x-mudraid-decision-id"]},
        {
            "outcome": "allow",
            "reason_code": "surface_not_protected",
            "reason_tier": "transport",
            "http_status": 200,
            "error_code": None,
            "stripped": [],
        },
    ),
    (
        "boundary_tool_name_at_max_len",
        {"tool_name": _TOOL_AT_MAX, "action_mapped": True, "decide": "allow"},
        {
            "outcome": "allow",
            "reason_code": "authorized",
            "reason_tier": "authorization",
            "http_status": 200,
            "error_code": None,
            "stripped": [],
        },
    ),
    (
        "boundary_tool_name_over_max_len",
        {"tool_name": _TOOL_OVER_MAX, "action_mapped": True, "decide": "allow"},
        {
            "outcome": "deny",
            "reason_code": "malformed_request",
            "reason_tier": "transport",
            "http_status": 400,
            "error_code": "ENFORCE_MALFORMED_REQUEST",
            "stripped": [],
        },
    ),
    (
        "boundary_empty_tool_name",
        {"tool_name": "", "action_mapped": True, "decide": "allow"},
        {
            "outcome": "deny",
            "reason_code": "malformed_request",
            "reason_tier": "transport",
            "http_status": 400,
            "error_code": "ENFORCE_MALFORMED_REQUEST",
            "stripped": [],
        },
    ),
    (
        "boundary_delete_control_verb_passes",
        {"method": "DELETE"},
        {
            "outcome": "allow",
            "reason_code": "control_plane_passthrough",
            "reason_tier": "transport",
            "http_status": 200,
            "error_code": None,
            "stripped": [],
        },
    ),
    (
        "boundary_public_method_tools_list_passes",
        {"rpc_method": "tools/list"},
        {
            "outcome": "allow",
            "reason_code": "control_plane_passthrough",
            "reason_tier": "transport",
            "http_status": 200,
            "error_code": None,
            "stripped": [],
        },
    ),
    (
        "boundary_decide_deny_explicit_reason",
        {
            "tool_name": "issue_refund",
            "action_mapped": True,
            "decide": "deny",
            "decide_reason": "amount_limit_exceeded",
        },
        {
            "outcome": "deny",
            "reason_code": "amount_limit_exceeded",
            "reason_tier": "authorization",
            "http_status": 403,
            "error_code": "ENFORCE_DECISION_DENY",
            "stripped": [],
        },
    ),
    (
        "hostile_reserved_header_injection",
        {
            "tool_name": "issue_refund",
            "action_mapped": True,
            "decide": "allow",
            "reserved_headers_presented": [
                "x-mudraid-decision-id",
                "x-mudraid-action-key",
                "X-MudraID-Bundle-Version",
                "cookie",
            ],
        },
        {
            "outcome": "allow",
            "reason_code": "authorized",
            "reason_tier": "authorization",
            "http_status": 200,
            "error_code": None,
            "stripped": [
                "x-mudraid-decision-id",
                "x-mudraid-action-key",
                "X-MudraID-Bundle-Version",
            ],
        },
    ),
    (
        "hostile_no_bundle_fail_closed",
        {
            "bundle_active": False,
            "tool_name": "issue_refund",
            "action_mapped": True,
            "decide": "allow",
        },
        {
            "outcome": "not_safely_decided",
            "reason_code": "adapter_config_stale",
            "reason_tier": "authorization",
            "http_status": 503,
            "error_code": "ENFORCE_NO_VALID_BUNDLE",
            "stripped": [],
        },
    ),
    (
        "hostile_batch_jsonrpc_rejected",
        {"json_shape": "array"},
        {
            "outcome": "deny",
            "reason_code": "batch_unsupported",
            "reason_tier": "transport",
            "http_status": 400,
            "error_code": "ENFORCE_BATCH_UNSUPPORTED",
            "stripped": [],
        },
    ),
    (
        "hostile_scalar_body_malformed",
        {"json_shape": "scalar"},
        {
            "outcome": "deny",
            "reason_code": "malformed_request",
            "reason_tier": "transport",
            "http_status": 400,
            "error_code": "ENFORCE_MALFORMED_REQUEST",
            "stripped": [],
        },
    ),
    (
        "hostile_wrong_jsonrpc_version",
        {"jsonrpc": "1.0", "tool_name": "issue_refund", "action_mapped": True},
        {
            "outcome": "deny",
            "reason_code": "malformed_request",
            "reason_tier": "transport",
            "http_status": 400,
            "error_code": "ENFORCE_MALFORMED_REQUEST",
            "stripped": [],
        },
    ),
    (
        "hostile_non_post_method_denied",
        {"method": "PUT"},
        {
            "outcome": "deny",
            "reason_code": "method_not_allowed",
            "reason_tier": "transport",
            "http_status": 405,
            "error_code": "ENFORCE_METHOD_NOT_ALLOWED",
            "stripped": [],
        },
    ),
    (
        "hostile_body_too_large_denied",
        {"body_too_large": True},
        {
            "outcome": "deny",
            "reason_code": "body_too_large",
            "reason_tier": "transport",
            "http_status": 413,
            "error_code": "ENFORCE_BODY_TOO_LARGE",
            "stripped": [],
        },
    ),
    (
        "hostile_body_unreadable_denied",
        {"body_readable": False},
        {
            "outcome": "deny",
            "reason_code": "body_unreadable",
            "reason_tier": "transport",
            "http_status": 400,
            "error_code": "ENFORCE_BODY_UNREADABLE",
            "stripped": [],
        },
    ),
    (
        "hostile_message_not_allowed",
        {"rpc_method": "resources/subscribe"},
        {
            "outcome": "deny",
            "reason_code": "message_not_allowed",
            "reason_tier": "transport",
            "http_status": 403,
            "error_code": "ENFORCE_MESSAGE_NOT_ALLOWED",
            "stripped": [],
        },
    ),
    (
        "hostile_unmapped_action_denied",
        {"tool_name": "unknown_tool", "action_mapped": False},
        {
            "outcome": "deny",
            "reason_code": "action_unmapped",
            "reason_tier": "authorization",
            "http_status": 403,
            "error_code": "ENFORCE_ACTION_UNMAPPED",
            "stripped": [],
        },
    ),
    (
        "hostile_decide_timeout_deny_closed",
        {"tool_name": "issue_refund", "action_mapped": True, "decide": "timeout"},
        {
            "outcome": "not_safely_decided",
            "reason_code": "deadline_exceeded",
            "reason_tier": "authorization",
            "http_status": 503,
            "error_code": "ENFORCE_DECIDE_UNAVAILABLE",
            "stripped": [],
        },
    ),
    (
        "hostile_decide_error_deny_closed",
        {"tool_name": "issue_refund", "action_mapped": True, "decide": "error"},
        {
            "outcome": "not_safely_decided",
            "reason_code": "authority_source_unavailable",
            "reason_tier": "authorization",
            "http_status": 503,
            "error_code": "ENFORCE_DECIDE_UNAVAILABLE",
            "stripped": [],
        },
    ),
    (
        "hostile_decide_credential_unconfigured_deny_closed",
        {"tool_name": "issue_refund", "action_mapped": True, "decide": "credential_unconfigured"},
        {
            "outcome": "not_safely_decided",
            "reason_code": "adapter_config_stale",
            "reason_tier": "authorization",
            "http_status": 503,
            "error_code": "ENFORCE_DECIDE_UNAVAILABLE",
            "stripped": [],
        },
    ),
    (
        "hostile_decide_unreachable_deny_closed",
        {"tool_name": "issue_refund", "action_mapped": True, "decide": "unreachable"},
        {
            "outcome": "not_safely_decided",
            "reason_code": "authority_source_unavailable",
            "reason_tier": "authorization",
            "http_status": 503,
            "error_code": "ENFORCE_DECIDE_UNAVAILABLE",
            "stripped": [],
        },
    ),
    (
        "hostile_decide_unconfigured_deny_closed",
        {"tool_name": "issue_refund", "action_mapped": True, "decide": "unconfigured"},
        {
            "outcome": "not_safely_decided",
            "reason_code": "adapter_config_stale",
            "reason_tier": "authorization",
            "http_status": 503,
            "error_code": "ENFORCE_DECIDE_UNAVAILABLE",
            "stripped": [],
        },
    ),
    (
        "hostile_decide_deny_reason_not_deny_falls_back",
        {
            "tool_name": "issue_refund",
            "action_mapped": True,
            "decide": "deny",
            "decide_reason": "authorized",
        },
        {
            "outcome": "deny",
            "reason_code": "policy_rule_denied",
            "reason_tier": "authorization",
            "http_status": 403,
            "error_code": "ENFORCE_DECISION_DENY",
            "stripped": [],
        },
    ),
    (
        "hostile_reserved_header_injection_stripped_on_deny",
        {
            "tool_name": "issue_refund",
            "action_mapped": False,
            "reserved_headers_presented": ["x-mudraid-decision-id", "authorization"],
        },
        {
            "outcome": "deny",
            "reason_code": "action_unmapped",
            "reason_tier": "authorization",
            "http_status": 403,
            "error_code": "ENFORCE_ACTION_UNMAPPED",
            "stripped": ["x-mudraid-decision-id"],
        },
    ),
]


def _facts_from(fixture: dict[str, Any]) -> V2RequestFacts:
    known = set(V2RequestFacts.__dataclass_fields__)
    kwargs = {k: v for k, v in fixture.items() if k in known and k != "reserved_headers_presented"}
    kwargs["reserved_headers_presented"] = tuple(
        fixture.get("reserved_headers_presented", ()) or ()
    )
    return V2RequestFacts(**kwargs)


def _decide_from(fixture: dict[str, Any]) -> Any:
    async def _d(action: str) -> DecideResult:
        return DecideResult(fixture.get("decide", "error"), reason=fixture.get("decide_reason"))

    return _d


@pytest.mark.parametrize(
    "case_id,facts,expect", MIRRORED_CORPUS, ids=[c[0] for c in MIRRORED_CORPUS]
)
@pytest.mark.asyncio
async def test_evaluate_v2_matches_contract_corpus(
    case_id: str, facts: dict[str, Any], expect: dict[str, Any]
) -> None:
    """The native V2 control loop reaches the SAME outcome as the contract."""
    decision = await evaluate_v2(_facts_from(facts), _decide_from(facts))
    assert decision.outcome == expect["outcome"], case_id
    assert decision.reason_code == expect["reason_code"], case_id
    assert decision.reason_tier == expect["reason_tier"], case_id
    assert decision.http_status == expect["http_status"], case_id
    assert decision.error_code == expect["error_code"], case_id
    assert list(decision.stripped_reserved_headers) == expect["stripped"], case_id


# ---------------------------------------------------------------------------
# 2. End-to-end — real middleware in mode="v2" with an injected fake client
# ---------------------------------------------------------------------------


class FakeDecideClient:
    """Hermetic :class:`DecideClient` — no network.

    Configurable bundle state, action map, and ``/decide`` behaviour (a fixed
    result, or a raised exception to exercise the deny-closed wrapper).
    """

    def __init__(
        self,
        *,
        bundle_active: bool = True,
        action_map: dict[str, str] | None = None,
        decide_result: DecideResult | None = None,
        decide_exc: BaseException | None = None,
    ) -> None:
        self._bundle_active = bundle_active
        self._action_map = (
            action_map if action_map is not None else {"issue_refund": "refunds.issue"}
        )
        self._decide_result = decide_result or DecideResult("allow", decision_id="dec-abc")
        self._decide_exc = decide_exc
        self.decide_calls: list[str] = []
        #: Context of each call, so a test can assert the caller's credential
        #: actually reached the seam rather than trusting that it did.
        self.decide_contexts: list[DecideContext] = []

    @property
    def bundle_active(self) -> bool:
        return self._bundle_active

    def resolve_action(self, tool_name: str) -> str | None:
        return self._action_map.get(tool_name)

    async def decide(self, action: str, context: DecideContext) -> DecideResult:
        self.decide_calls.append(action)
        self.decide_contexts.append(context)
        if self._decide_exc is not None:
            raise self._decide_exc
        return self._decide_result


def _make_v2_app(client: FakeDecideClient, **cfg_kwargs: Any) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        MudraIDMiddleware,
        mode="v2",
        v2_config=V2Config(decide_client=client, **cfg_kwargs),
    )

    @app.post("/mcp")
    async def mcp(request: Request) -> dict:
        body = await request.body()
        return {
            "ran": True,
            "body": body.decode("utf-8"),
            "action_key": request.headers.get("x-mudraid-action-key"),
            "decision_id": request.headers.get("x-mudraid-decision-id"),
        }

    @app.get("/mcp")
    async def mcp_get() -> dict:
        return {"ran": True, "verb": "get"}

    # The rest of the Streamable-HTTP verb surface, each mounted explicitly so a
    # pass-through reaches a REAL handler and the test cannot mistake a 405 from
    # the router for a middleware allow. HEAD is registered via api_route because
    # FastAPI does not derive it from the GET above.
    @app.api_route("/mcp", methods=["HEAD"])
    async def mcp_head() -> dict:
        return {"ran": True, "verb": "head"}

    @app.options("/mcp")
    async def mcp_options() -> dict:
        return {"ran": True, "verb": "options"}

    @app.delete("/mcp")
    async def mcp_delete(request: Request) -> dict:
        return {
            "ran": True,
            "verb": "delete",
            "decision_id": request.headers.get("x-mudraid-decision-id"),
        }

    @app.put("/mcp")
    async def mcp_put() -> dict:  # pragma: no cover - must never be reached
        return {"ran": True, "verb": "put"}

    @app.post("/open/echo")
    async def open_echo(request: Request) -> dict:
        return {"ran": True, "decision_id": request.headers.get("x-mudraid-decision-id")}

    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


_TOOLS_CALL = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "issue_refund"}}


@pytest.mark.asyncio
async def test_v2_mapped_allow_forwards_to_handler_and_injects_context() -> None:
    client = FakeDecideClient()
    app = _make_v2_app(client)

    async with _client(app) as c:
        resp = await c.post("/mcp", json=_TOOLS_CALL)

    assert resp.status_code == 200
    body = resp.json()
    assert body["ran"] is True
    # Trusted context injected only after the bound allow.
    assert body["action_key"] == "refunds.issue"
    assert body["decision_id"] == "dec-abc"
    # The body reached the handler intact.
    assert '"tools/call"' in body["body"]
    assert client.decide_calls == ["refunds.issue"]


@pytest.mark.asyncio
async def test_v2_allow_strips_client_forged_trusted_context() -> None:
    """A client-forged x-mudraid-* header is stripped; the handler sees ONLY
    the middleware-injected decision id, never the spoofed one."""
    client = FakeDecideClient()
    app = _make_v2_app(client)

    async with _client(app) as c:
        resp = await c.post(
            "/mcp",
            json=_TOOLS_CALL,
            headers={"x-mudraid-decision-id": "SPOOFED", "x-mudraid-action-key": "SPOOFED"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["decision_id"] == "dec-abc"
    assert body["action_key"] == "refunds.issue"


@pytest.mark.asyncio
async def test_v2_decide_deny_is_403_and_handler_not_run() -> None:
    client = FakeDecideClient(decide_result=DecideResult("deny", reason="amount_limit_exceeded"))
    app = _make_v2_app(client)

    async with _client(app) as c:
        resp = await c.post("/mcp", json=_TOOLS_CALL)

    assert resp.status_code == 403
    assert resp.json()["error_code"] == "ENFORCE_DECISION_DENY"


@pytest.mark.asyncio
async def test_v2_decide_timeout_status_deny_closed_503() -> None:
    client = FakeDecideClient(decide_result=DecideResult("timeout"))
    app = _make_v2_app(client)

    async with _client(app) as c:
        resp = await c.post("/mcp", json=_TOOLS_CALL)

    assert resp.status_code == 503
    assert resp.json()["error_code"] == "ENFORCE_DECIDE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_v2_decide_raised_exception_deny_closed_503() -> None:
    """A raised transport error from the decide client must deny-close, never
    bubble out or be mistaken for an allow."""
    client = FakeDecideClient(decide_exc=RuntimeError("connection reset"))
    app = _make_v2_app(client)

    async with _client(app) as c:
        resp = await c.post("/mcp", json=_TOOLS_CALL)

    assert resp.status_code == 503
    assert resp.json()["error_code"] == "ENFORCE_DECIDE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_v2_no_bundle_fail_closed_503() -> None:
    client = FakeDecideClient(bundle_active=False)
    app = _make_v2_app(client)

    async with _client(app) as c:
        resp = await c.post("/mcp", json=_TOOLS_CALL)

    assert resp.status_code == 503
    assert resp.json()["error_code"] == "ENFORCE_NO_VALID_BUNDLE"
    assert client.decide_calls == []  # never reached /decide


@pytest.mark.asyncio
async def test_v2_unmapped_action_denied_403() -> None:
    client = FakeDecideClient(action_map={})  # nothing maps
    app = _make_v2_app(client)

    async with _client(app) as c:
        resp = await c.post("/mcp", json=_TOOLS_CALL)

    assert resp.status_code == 403
    assert resp.json()["error_code"] == "ENFORCE_ACTION_UNMAPPED"
    assert client.decide_calls == []


@pytest.mark.asyncio
async def test_v2_batch_array_rejected_400() -> None:
    client = FakeDecideClient()
    app = _make_v2_app(client)

    async with _client(app) as c:
        resp = await c.post("/mcp", json=[_TOOLS_CALL, _TOOLS_CALL])

    assert resp.status_code == 400
    assert resp.json()["error_code"] == "ENFORCE_BATCH_UNSUPPORTED"


@pytest.mark.asyncio
async def test_v2_control_verb_get_passes_without_decide() -> None:
    client = FakeDecideClient()
    app = _make_v2_app(client)

    async with _client(app) as c:
        resp = await c.get("/mcp")

    assert resp.status_code == 200
    assert resp.json() == {"ran": True, "verb": "get"}
    assert client.decide_calls == []


@pytest.mark.asyncio
async def test_v2_body_too_large_denied_413() -> None:
    client = FakeDecideClient()
    app = _make_v2_app(client, max_body_bytes=16)

    async with _client(app) as c:
        resp = await c.post("/mcp", json=_TOOLS_CALL)  # well over 16 bytes

    assert resp.status_code == 413
    assert resp.json()["error_code"] == "ENFORCE_BODY_TOO_LARGE"


@pytest.mark.asyncio
async def test_v2_unprotected_path_passes_through_untouched() -> None:
    """A path outside ``protected_paths`` is not enforced: nothing is decided
    and client headers are left intact (they are simply not trusted context on
    an unprotected surface)."""
    client = FakeDecideClient()
    app = _make_v2_app(client, protected_paths=("/mcp",))

    async with _client(app) as c:
        resp = await c.post(
            "/open/echo", json={"x": 1}, headers={"x-mudraid-decision-id": "client-value"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["ran"] is True
    assert body["decision_id"] == "client-value"  # not stripped on an unprotected surface
    assert client.decide_calls == []


# ---------------------------------------------------------------------------
# 2b. V2 is MCP-ONLY, and this section pins exactly what that buys and costs
#
# LAUNCH SCOPING (owner direction 2026-08-13). MudraID's two enforcement paths
# are now separated cleanly: ordinary REST APIs use the V1 route/scope
# middleware (which enforces method + route, GET and DELETE included); MCP
# servers use this V2 loop or the Kong `mudraid-enforce` plugin.
#
# The behaviour below is DELIBERATE and is NOT changing: on a V2 surface,
# GET/HEAD/OPTIONS pass as Streamable-HTTP transport and DELETE passes as MCP
# session control, with no `/decide` call at all. That is right for MCP — those
# verbs open, poll and close an SSE session and cannot invoke a tool — and it
# would be a serious hole on a REST API, where a GET reads and a DELETE
# destroys. So it is not fixed here by protocol sniffing or a customer toggle;
# it is fixed by making V2 unattachable to a non-MCP surface (see
# platform-integration-service: registration requires `surface_type ==
# "mcp_server"`, and an ABSENT classification is rejected).
#
# These tests exist so that the transport passes are pinned as INTENDED
# behaviour with their justification attached, and so that the one invariant
# holding the whole arrangement up — every `tools/call` reaches live
# authorization — is asserted directly rather than inferred.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verb", ["GET", "HEAD", "OPTIONS"])
@pytest.mark.asyncio
async def test_v2_streamable_http_transport_verbs_pass_without_decide(verb: str) -> None:
    """GET/HEAD/OPTIONS are Streamable-HTTP transport: they open or probe the
    session and carry no JSON-RPC request, so they cannot invoke a tool. They
    pass, and they must not consume a decision."""
    client = FakeDecideClient()
    app = _make_v2_app(client)

    async with _client(app) as c:
        resp = await c.request(verb, "/mcp")

    assert resp.status_code == 200, verb
    assert client.decide_calls == [], verb


@pytest.mark.asyncio
async def test_v2_delete_passes_as_mcp_session_control_not_as_a_decided_action() -> None:
    """DELETE terminates an MCP session (MCP-Session-Id teardown), it does not
    delete a resource — hence no `/decide`. The assertion is deliberately
    two-sided: it passes AND it reaches the handler with no trusted context,
    because trusted context is minted only by an authorized decision and a
    transport pass is not one."""
    client = FakeDecideClient()
    app = _make_v2_app(client)

    async with _client(app) as c:
        resp = await c.delete("/mcp", headers={"x-mudraid-decision-id": "SPOOFED"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["verb"] == "delete"
    assert client.decide_calls == []
    # The client-supplied context header was stripped and NOT replaced: a
    # transport pass never fabricates an authorization it did not obtain.
    assert body["decision_id"] is None


@pytest.mark.asyncio
async def test_v2_non_streamable_http_verb_is_denied() -> None:
    """The transport passes are an enumerated set, not "anything that is not a
    POST". A PUT is neither transport nor a JSON-RPC message, so it denies even
    though a handler exists for it."""
    client = FakeDecideClient()
    app = _make_v2_app(client)

    async with _client(app) as c:
        resp = await c.put("/mcp")

    assert resp.status_code == 405
    assert resp.json()["error_code"] == "ENFORCE_METHOD_NOT_ALLOWED"
    assert client.decide_calls == []


@pytest.mark.parametrize(
    "tool,expect_status,expect_code",
    [
        ("issue_refund", 200, None),
        ("issue_refund", 403, "ENFORCE_DECISION_DENY"),
    ],
    ids=["allow", "deny"],
)
@pytest.mark.asyncio
async def test_v2_tools_call_always_reaches_live_authorization(
    tool: str, expect_status: int, expect_code: str | None
) -> None:
    """THE LOAD-BEARING INVARIANT. Whatever the transport verbs are allowed to
    do, a `tools/call` — the only JSON-RPC message that can act — must reach a
    live `/decide` on every single request. Both outcomes are exercised so the
    test cannot be satisfied by a client that never answers."""
    result = (
        DecideResult("allow", decision_id="dec-abc")
        if expect_code is None
        else DecideResult("deny", reason="policy_rule_denied")
    )
    client = FakeDecideClient(decide_result=result)
    app = _make_v2_app(client)

    async with _client(app) as c:
        for _ in range(3):
            resp = await c.post(
                "/mcp", json={"jsonrpc": "2.0", "method": "tools/call", "params": {"name": tool}}
            )
            assert resp.status_code == expect_status
            if expect_code is not None:
                assert resp.json()["error_code"] == expect_code

    # Three calls, three decisions — nothing is cached, memoized or short-circuited.
    assert client.decide_calls == ["refunds.issue"] * 3


@pytest.mark.asyncio
async def test_v2_unknown_jsonrpc_method_is_denied_not_ignored() -> None:
    """A JSON-RPC method outside the control/discovery allowlist denies rather
    than slipping through because action extraction found nothing to decide."""
    client = FakeDecideClient()
    app = _make_v2_app(client)

    async with _client(app) as c:
        resp = await c.post("/mcp", json={"jsonrpc": "2.0", "method": "resources/read"})

    assert resp.status_code == 403
    assert resp.json()["error_code"] == "ENFORCE_MESSAGE_NOT_ALLOWED"
    assert client.decide_calls == []


@pytest.mark.parametrize(
    "status", ["unreachable", "error", "unconfigured", "credential_unconfigured"]
)
@pytest.mark.asyncio
async def test_v2_decide_outage_and_credential_failures_deny_closed(status: str) -> None:
    """Every way the authority call can fail to answer — the authority is
    unreachable, it errors, the adapter is unconfigured, or its credential is
    missing/invalid — is deny-closed 503. An adapter that cannot authenticate
    itself must not degrade into an allow."""
    client = FakeDecideClient(decide_result=DecideResult(status))
    app = _make_v2_app(client)

    async with _client(app) as c:
        resp = await c.post("/mcp", json=_TOOLS_CALL)

    assert resp.status_code == 503, status
    assert resp.json()["error_code"] == "ENFORCE_DECIDE_UNAVAILABLE", status


@pytest.mark.asyncio
async def test_v2_reserved_headers_stripped_before_forwarding_on_every_path() -> None:
    """Reserved `x-mudraid-*` headers are stripped BEFORE evaluation, on the
    pass-through path as well as the decided one, so client-asserted trusted
    context is never an input fact and never reaches the handler."""
    client = FakeDecideClient()
    app = _make_v2_app(client)
    forged = {
        "x-mudraid-decision-id": "SPOOFED",
        "x-mudraid-action-key": "SPOOFED",
        "X-MudraID-Agent-Id": "SPOOFED",  # mixed case: the match is case-insensitive
    }

    async with _client(app) as c:
        # Transport pass — nothing decided, so nothing is injected either.
        passed = await c.delete("/mcp", headers=forged)
        # Authorized call — the handler sees the MIDDLEWARE's values, not the
        # client's.
        decided = await c.post("/mcp", json=_TOOLS_CALL, headers=forged)

    assert passed.json()["decision_id"] is None
    assert decided.json()["decision_id"] == "dec-abc"
    assert decided.json()["action_key"] == "refunds.issue"


@pytest.mark.asyncio
async def test_v2_missing_bundle_denies_before_any_credential_is_used() -> None:
    """No verified signed bundle → the surface cannot be safely decided at all.
    Asserted on a `tools/call` AND shown not to consult the authority: an
    adapter with a stale or absent bundle fails closed rather than asking about
    an action map it does not have."""
    client = FakeDecideClient(bundle_active=False)
    app = _make_v2_app(client)

    async with _client(app) as c:
        resp = await c.post("/mcp", json=_TOOLS_CALL)

    assert resp.status_code == 503
    assert resp.json()["error_code"] == "ENFORCE_NO_VALID_BUNDLE"
    assert client.decide_calls == []


# ---------------------------------------------------------------------------
# 3. V1 regression — the default mode is unchanged
# ---------------------------------------------------------------------------

JWKS_URL = "https://api.mudraid.test/.well-known/jwks.json"
PLATFORM_ID = "plt-test"


def _v1_yaml(routes_yaml: str) -> str:
    return f"platform_id: {PLATFORM_ID}\nversion: 1\nroutes:\n{routes_yaml}"


def _make_v1_app(yaml_path: Path) -> FastAPI:
    app = FastAPI()
    # NOTE: no `mode=` argument — the default must remain V1.
    app.add_middleware(MudraIDMiddleware, scopes_yaml_path=str(yaml_path), jwks_url=JWKS_URL)

    @app.get("/items")
    async def list_items() -> dict:
        return {"items": ["a", "b"]}

    @app.delete("/items/{item_id}")
    async def delete_item(item_id: str) -> dict:
        return {"deleted": item_id}

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


def test_v2_config_rejected_in_v1_mode() -> None:
    """Passing a v2_config without mode='v2' is a construction error — the two
    modes never silently blend."""
    app = FastAPI()
    with pytest.raises(ValueError, match="only valid with mode='v2'"):
        MudraIDMiddleware(app, v2_config=V2Config(decide_client=FakeDecideClient()))


def test_v2_mode_requires_config() -> None:
    app = FastAPI()
    with pytest.raises(ValueError, match="requires a v2_config"):
        MudraIDMiddleware(app, mode="v2")


def test_unknown_mode_rejected() -> None:
    app = FastAPI()
    with pytest.raises(ValueError, match="mode must be"):
        MudraIDMiddleware(app, mode="v3")


@pytest.mark.asyncio
async def test_v1_default_mode_public_route_unchanged(tmp_path: Path) -> None:
    """The default (no ``mode=``) middleware still enforces the static
    route-scope contract: a public route forwards with no token, unchanged."""
    yaml_path = tmp_path / "mudraid_scopes.yaml"
    yaml_path.write_text(
        _v1_yaml("  - method: GET\n    path: /health\n    public: true\n"), encoding="utf-8"
    )
    app = _make_v1_app(yaml_path)

    async with _client(app) as c:
        resp = await c.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_v1_default_mode_scoped_route_still_requires_token(tmp_path: Path) -> None:
    """A scoped V1 route with no token is still a 401 MISSING_TOKEN — proving a
    successful path is never inferred and V2 has not leaked into V1."""
    yaml_path = tmp_path / "mudraid_scopes.yaml"
    yaml_path.write_text(
        _v1_yaml("  - method: GET\n    path: /items\n    scope: items:read\n"), encoding="utf-8"
    )
    app = _make_v1_app(yaml_path)

    async with _client(app) as c:
        resp = await c.get("/items")

    assert resp.status_code == 401
    assert resp.json()["error_code"] == "MISSING_TOKEN"


@pytest.mark.asyncio
async def test_v1_default_mode_valid_token_reaches_handler(
    tmp_path: Path,
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """The V1 happy path still works end-to-end under the default mode."""
    yaml_path = tmp_path / "mudraid_scopes.yaml"
    yaml_path.write_text(
        _v1_yaml("  - method: GET\n    path: /items\n    scope: items:read\n"), encoding="utf-8"
    )
    app = _make_v1_app(yaml_path)
    token = sign_jwt(rsa_private_key, baseline_claims(audience=PLATFORM_ID, scopes=["items:read"]))

    router = respx.mock(assert_all_called=False)
    router.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))
    with router:
        async with _client(app) as c:
            resp = await c.get("/items", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200
    assert resp.json() == {"items": ["a", "b"]}


# ---------------------------------------------------------------------------
# 3b. V1 MUST NOT BE WEAKENED — the GET/DELETE guarantee REST customers get
#
# The launch scoping (owner direction 2026-08-13) points every ordinary REST
# API at this V1 route/scope middleware, precisely BECAUSE V2 passes
# GET/HEAD/OPTIONS/DELETE as MCP transport. That makes V1's method-level
# enforcement of read and destructive verbs the load-bearing half of the
# decision, and a regression here would be the worst possible outcome of the
# change: customers moved off V2 onto a path that turned out not to protect the
# very verbs they were moved for.
#
# So it is asserted directly, in both directions — refused without a satisfying
# token, served with one — rather than assumed from the V2 tests above.
# ---------------------------------------------------------------------------

_V1_GET_AND_DELETE = (
    "  - method: GET\n    path: /items\n    scope: items:read\n"
    "  - method: DELETE\n    path: /items/{item_id}\n    scope: items:delete\n"
)


@pytest.mark.parametrize(
    "verb,path,scope",
    [("GET", "/items", "items:read"), ("DELETE", "/items/abc", "items:delete")],
)
@pytest.mark.asyncio
async def test_v1_still_protects_get_and_delete_routes(
    tmp_path: Path, verb: str, path: str, scope: str
) -> None:
    """No token → 401 on BOTH a read route and a destructive one. This is the
    behaviour V2 deliberately does not have and the reason REST surfaces are
    directed here."""
    yaml_path = tmp_path / "mudraid_scopes.yaml"
    yaml_path.write_text(_v1_yaml(_V1_GET_AND_DELETE), encoding="utf-8")
    app = _make_v1_app(yaml_path)

    async with _client(app) as c:
        resp = await c.request(verb, path)

    assert resp.status_code == 401, verb
    assert resp.json()["error_code"] == "MISSING_TOKEN", verb


@pytest.mark.parametrize(
    "verb,path,granted,expect",
    [
        # The right scope for the right verb → served.
        ("GET", "/items", ["items:read"], 200),
        ("DELETE", "/items/abc", ["items:delete"], 200),
        # A token good for the READ route does NOT unlock the DELETE route.
        # Method-level enforcement is the whole point; a per-path-only check
        # would pass this and would be exactly the regression to catch.
        ("DELETE", "/items/abc", ["items:read"], 403),
    ],
    ids=["get-with-scope", "delete-with-scope", "delete-with-only-read-scope"],
)
@pytest.mark.asyncio
async def test_v1_get_and_delete_enforce_their_own_scopes(
    tmp_path: Path,
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
    verb: str,
    path: str,
    granted: list[str],
    expect: int,
) -> None:
    yaml_path = tmp_path / "mudraid_scopes.yaml"
    yaml_path.write_text(_v1_yaml(_V1_GET_AND_DELETE), encoding="utf-8")
    app = _make_v1_app(yaml_path)
    token = sign_jwt(rsa_private_key, baseline_claims(audience=PLATFORM_ID, scopes=granted))

    router = respx.mock(assert_all_called=False)
    router.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))
    with router:
        async with _client(app) as c:
            resp = await c.request(verb, path, headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == expect, (verb, granted)
