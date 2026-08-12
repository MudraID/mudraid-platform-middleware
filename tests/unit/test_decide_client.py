"""The production DecideClient: bundle verification, and the live decision.

These tests exist to catch the failures that would be invisible in staging and
expensive in production:

  - a verifier that accepts a bundle the signer did not sign;
  - a client that cannot forward the caller's credential, and therefore denies
    every real request while passing every fake one;
  - any path where "we could not decide" quietly becomes "allow".
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import httpx
import pytest

from mudraid_platform_middleware._bundle import BundleRefused, verify_bundle
from mudraid_platform_middleware._canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    loads_strict,
    sha256_hex,
)
from mudraid_platform_middleware._contract import (
    MAX_RESPONSE_BYTES,
    REQUEST_CONTRACT,
    RESPONSE_CONTRACT,
)
from mudraid_platform_middleware.decide_client import HttpDecideClient, InsecureDecideDestination
from mudraid_platform_middleware.v2 import DecideContext

SECRET = "bundle-signing-secret"
DECIDE_URL = "https://enforcement.internal/api/v2/enforcement/decide"
CHANNEL_URL = "https://platform.internal"

#: The whole customer configuration: one base URL, one adapter token.
BASE_URL = "https://api.staging.mudraid.ai"


# ---------------------------------------------------------------------------
# Fixtures — a bundle signed exactly the way the control plane signs one
# ---------------------------------------------------------------------------


def _content(**overrides: Any) -> dict[str, Any]:
    content = {
        "surface": {
            "platform_id": "plat_1",
            "environment": "staging",
            "canonical_resource_uri": "https://mcp.example.com/mcp",
        },
        "evaluation": {
            "mode": "live",
            "on_timeout": "deny",
            "on_error": "deny",
            "on_unmapped_action": "deny",
            "on_stale_bundle": "deny",
            "forward": "once",
            "decide_required": True,
            "retry_forwarded_request": False,
        },
        "matcher": {
            "kind": "mcp_tool_exact",
            "actions": [
                {
                    "tool_name": "issue_refund",
                    "action_key": "refunds.issue",
                    "action_version": 3,
                    "mapping_id": "map_1",
                    "mapping_revision": 2,
                    "risk_class": "high",
                    "required_scopes": ["refunds:write"],
                }
            ],
        },
        "trusted_context": {"strip_request_header_prefixes": ["x-mudraid-"]},
    }
    content.update(overrides)
    return content


def _signed(version: int = 7, secret: str = SECRET, **content_overrides: Any) -> dict[str, Any]:
    """A served bundle, signed the way ``bundle_compiler.py`` signs one."""
    payload = {
        "schema_version": "1.0",
        "bundle_version": version,
        "content": _content(**content_overrides),
    }
    canon = canonical_json_bytes(payload)
    return {
        "bundle_version": version,
        "schema_version": "1.0",
        "payload": payload,
        "payload_digest": hashlib.sha256(canon).hexdigest(),
        "signing_key_id": "local-hmac-v1",
        "signature": hmac.new(secret.encode(), canon, hashlib.sha256).hexdigest(),
    }


# ---------------------------------------------------------------------------
# 1. Canonical bytes ARE the signer's bytes
# ---------------------------------------------------------------------------


def test_canonical_bytes_match_the_signers_definition() -> None:
    """The signer's definition, restated. If these diverge, nothing verifies."""
    value = {"b": 1, "a": {"z": [1, 2], "y": "ü"}}
    assert canonical_json_bytes(value) == json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def test_non_ascii_is_escaped_and_keys_are_sorted() -> None:
    assert canonical_json_bytes({"b": "ü", "a": 1}) == b'{"a":1,"b":"\\u00fc"}'


def test_nan_and_infinity_are_refused_rather_than_carried_into_a_digest() -> None:
    """Python accepts these by default; no signer ever emitted one."""
    for token in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(CanonicalizationError):
            loads_strict('{"x": %s}' % token)


def test_non_integer_numbers_have_no_canonical_form() -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json_bytes({"x": 1.5})


# ---------------------------------------------------------------------------
# 2. Bundle verification refuses what it must
# ---------------------------------------------------------------------------


def test_a_correctly_signed_bundle_verifies() -> None:
    bundle = verify_bundle(_signed(), secret=SECRET)
    assert bundle.bundle_version == 7
    assert bundle.resolve("issue_refund") is not None
    assert bundle.surface["environment"] == "staging"


def test_a_tampered_payload_is_a_digest_mismatch_not_a_signature_failure() -> None:
    """Tampering is reported as its own fact, regardless of key material."""
    served = _signed()
    served["payload"]["content"]["matcher"]["actions"][0]["tool_name"] = "drain_account"
    with pytest.raises(BundleRefused) as exc:
        verify_bundle(served, secret=SECRET)
    assert exc.value.code == "BUNDLE_DIGEST_MISMATCH"


def test_a_bundle_signed_with_the_wrong_secret_is_refused() -> None:
    with pytest.raises(BundleRefused) as exc:
        verify_bundle(_signed(secret="attacker"), secret=SECRET)
    assert exc.value.code == "BUNDLE_SIGNATURE_INVALID"


def test_no_secret_refuses_rather_than_trusting_unsigned() -> None:
    for secret in (None, ""):
        with pytest.raises(BundleRefused) as exc:
            verify_bundle(_signed(), secret=secret)
        assert exc.value.code == "BUNDLE_SIGNING_SECRET_UNCONFIGURED"


def test_an_unknown_schema_version_is_refused_not_best_effort_parsed() -> None:
    served = _signed()
    served["schema_version"] = "2.0"
    served["payload"]["schema_version"] = "2.0"
    with pytest.raises(BundleRefused) as exc:
        verify_bundle(served, secret=SECRET)
    assert exc.value.code == "BUNDLE_SCHEMA_UNSUPPORTED"


@pytest.mark.parametrize("field", ["platform_id", "environment", "canonical_resource_uri"])
def test_an_unbound_surface_is_refused_at_activation(field: str) -> None:
    """A null resource uri deny-closes every request; refuse it up front."""
    surface = {
        "platform_id": "plat_1",
        "environment": "staging",
        "canonical_resource_uri": "https://mcp.example.com/mcp",
    }
    surface[field] = None
    with pytest.raises(BundleRefused) as exc:
        verify_bundle(_signed(surface=surface), secret=SECRET)
    assert exc.value.code == "BUNDLE_SURFACE_UNBOUND"


def test_a_weaker_evaluation_contract_is_refused_never_downgraded() -> None:
    evaluation = dict(_content()["evaluation"])
    evaluation["on_timeout"] = "allow"
    with pytest.raises(BundleRefused) as exc:
        verify_bundle(_signed(evaluation=evaluation), secret=SECRET)
    assert exc.value.code == "BUNDLE_EVALUATION_UNSUPPORTED"


def test_an_unknown_matcher_kind_never_degrades_to_fuzzy_matching() -> None:
    with pytest.raises(BundleRefused) as exc:
        verify_bundle(_signed(matcher={"kind": "mcp_tool_prefix", "actions": []}), secret=SECRET)
    assert exc.value.code == "BUNDLE_MATCHER_UNSUPPORTED"


def test_a_duplicate_tool_name_is_ambiguous_and_refused_before_use() -> None:
    """Ambiguity must never be resolved by iteration order at request time."""
    matcher = {
        "kind": "mcp_tool_exact",
        "actions": [
            {"tool_name": "t", "action_key": "a.one"},
            {"tool_name": "t", "action_key": "a.two"},
        ],
    }
    with pytest.raises(BundleRefused) as exc:
        verify_bundle(_signed(matcher=matcher), secret=SECRET)
    assert exc.value.code == "BUNDLE_MATCHER_AMBIGUOUS"


def test_a_lower_version_is_a_rollback_and_is_refused() -> None:
    active = verify_bundle(_signed(version=7), secret=SECRET)
    with pytest.raises(BundleRefused) as exc:
        verify_bundle(_signed(version=6), secret=SECRET, active=active)
    assert exc.value.code == "BUNDLE_VERSION_REGRESSION"


def test_the_same_version_with_different_bytes_is_a_conflict() -> None:
    active = verify_bundle(_signed(version=7), secret=SECRET)
    other = _signed(
        version=7,
        matcher={"kind": "mcp_tool_exact", "actions": [{"tool_name": "x", "action_key": "a.x"}]},
    )
    with pytest.raises(BundleRefused) as exc:
        verify_bundle(other, secret=SECRET, active=active)
    assert exc.value.code == "BUNDLE_VERSION_CONFLICT"


def test_the_same_version_with_the_same_digest_is_a_no_op() -> None:
    active = verify_bundle(_signed(version=7), secret=SECRET)
    again = verify_bundle(_signed(version=7), secret=SECRET, active=active)
    assert again.no_change is True


def test_a_boolean_bundle_version_is_not_an_integer() -> None:
    """`bool` subclasses `int`; True must not read as version 1."""
    served = _signed()
    served["bundle_version"] = True
    with pytest.raises(BundleRefused) as exc:
        verify_bundle(served, secret=SECRET)
    assert exc.value.code == "BUNDLE_RESPONSE_INVALID"


def test_resolution_is_exact_and_case_sensitive() -> None:
    bundle = verify_bundle(_signed(), secret=SECRET)
    assert bundle.resolve("issue_refund") is not None
    for miss in ("Issue_Refund", "issue_refun", "issue_refundX", "", "x" * 600):
        assert bundle.resolve(miss) is None


# ---------------------------------------------------------------------------
# 3. The client — the credential, and every deny-closed path
# ---------------------------------------------------------------------------


def _client(handler: Any, **kwargs: Any) -> HttpDecideClient:
    """One base URL and one adapter token — the customer configuration.

    ``adapter_token`` is now the credential the decision route authenticates,
    so a test that used to clear ``service_secret`` clears the token instead.
    """
    transport = httpx.MockTransport(handler)
    return HttpDecideClient(
        base_url=BASE_URL,
        adapter_token=kwargs.pop("service_secret", "adapter-token"),
        bundle_signing_secret=SECRET,
        client=httpx.AsyncClient(transport=transport),
        **kwargs,
    )


def _serving(bundle: dict[str, Any] | None, decide: dict[str, Any] | None = None) -> Any:
    """A transport serving a bundle and a contract-valid decision.

    The decision echoes the request's ``decision_id`` and carries the response
    contract version, because anything less is no longer a readable answer.
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/keys"):
            # The client fetches verification keys before each bundle. These
            # fixtures are HMAC-signed, so an empty set is correct here — the
            # asymmetric path has its own suite.
            captured["keys_fetched"] = True
            return httpx.Response(200, json={"keys": []})
        if request.url.path.endswith("/bundle"):
            captured["bundle_auth"] = request.headers.get("authorization")
            if bundle is None:
                return httpx.Response(404)
            return httpx.Response(200, json=bundle)
        body = json.loads(request.content)
        captured["decide_body"] = body
        captured["decide_headers"] = dict(request.headers)
        if decide is not None:
            return httpx.Response(200, json=decide)
        return httpx.Response(
            200,
            json={
                "schema_version": RESPONSE_CONTRACT,
                "decision_id": body["decision_id"],
                "decision": "allow",
                # Required now: a decision with no timestamp is unbounded in
                # age, and a captured allow would stay usable forever.
                "decided_at": _now_iso(),
            },
        )

    handler.captured = captured  # type: ignore[attr-defined]
    return handler


def _now_iso() -> str:
    """A decision timestamp that is fresh whenever the suite runs.

    Hard-coding one would make these tests pass today and fail silently in a
    month, which is the worst way to learn that freshness is enforced.
    """
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _bound(decision: str = "allow", **overrides: Any) -> Any:
    """A handler answering with a valid response, with fields overridden."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bundle"):
            return httpx.Response(200, json=_signed())
        body = json.loads(request.content)
        payload: dict[str, Any] = {
            "schema_version": RESPONSE_CONTRACT,
            "decision_id": body["decision_id"],
            "decision": decision,
            "decided_at": _now_iso(),
        }
        payload.update(overrides)
        payload = {k: v for k, v in payload.items() if v is not _OMIT}
        return httpx.Response(200, json=payload)

    return handler


_OMIT = object()


def _ctx(**kwargs: Any) -> DecideContext:
    """A fully specified context; every field is explicit by design."""
    return DecideContext(
        presented_bearer=kwargs.pop("presented_bearer", "Bearer caller-jwt"),
        http_method=kwargs.pop("http_method", "POST"),
        path=kwargs.pop("path", "/mcp"),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_the_callers_credential_reaches_the_authority() -> None:
    """THE reason the seam carries context.

    The authority establishes identity only from this token and denies when it
    is absent, so a client that dropped it would deny every real request while
    passing every test that did not check this.
    """
    handler = _serving(_signed())
    client = _client(handler)
    await client.refresh_once()
    result = await client.decide(
        "refunds.issue",
        _ctx(correlation_id="corr-1"),
    )
    assert result.status == "allow"
    body = handler.captured["decide_body"]
    assert body["presented_authorization"] == "Bearer caller-jwt"
    # The ADAPTER's bearer, not the old global service secret. That secret named
    # no tenant, so it could not say WHICH adapter was calling — which is what
    # lets enforcement bind the decision to a surface at all.
    assert handler.captured["decide_headers"]["authorization"] == "Bearer adapter-token"
    assert handler.captured["decide_headers"]["x-correlation-id"] == "corr-1"


@pytest.mark.asyncio
async def test_the_surface_on_the_envelope_comes_from_the_signed_bundle() -> None:
    """Never from the request — the surface is what binds a grant."""
    handler = _serving(_signed())
    client = _client(handler)
    await client.refresh_once()
    await client.decide("refunds.issue", _ctx())
    surface = handler.captured["decide_body"]["surface"]
    assert surface == {
        "platform_id": "plat_1",
        "environment": "staging",
        "canonical_resource_uri": "https://mcp.example.com/mcp",
    }


@pytest.mark.asyncio
async def test_action_metadata_from_the_bundle_is_forwarded() -> None:
    handler = _serving(_signed())
    client = _client(handler)
    await client.refresh_once()
    await client.decide("refunds.issue", _ctx())
    action = handler.captured["decide_body"]["action"]
    assert action["action_key"] == "refunds.issue"
    assert action["tool_name"] == "issue_refund"
    assert action["required_scopes"] == ["refunds:write"]


@pytest.mark.asyncio
async def test_the_bundle_fetch_presents_the_adapter_token() -> None:
    handler = _serving(_signed())
    client = _client(handler)
    await client.refresh_once()
    assert handler.captured["bundle_auth"] == "Bearer adapter-token"


@pytest.mark.asyncio
async def test_no_bundle_means_not_active_and_nothing_resolves() -> None:
    client = _client(_serving(None))
    assert await client.refresh_once() is False
    assert client.bundle_active is False
    assert client.resolve_action("issue_refund") is None


@pytest.mark.asyncio
async def test_a_refused_bundle_leaves_the_previous_one_active() -> None:
    """A control plane serving garbage must not disarm working enforcement."""
    good = _signed(version=7)
    state = {"serve": good}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=state["serve"])

    client = _client(handler)
    await client.refresh_once()
    assert client.bundle_active is True

    tampered = _signed(version=8)
    tampered["signature"] = "0" * 64
    state["serve"] = tampered
    assert await client.refresh_once() is False
    assert client.bundle_active is True
    assert client.bundle_version == 7


@pytest.mark.asyncio
async def test_an_unmapped_tool_resolves_to_nothing() -> None:
    client = _client(_serving(_signed()))
    await client.refresh_once()
    assert client.resolve_action("issue_refund") == "refunds.issue"
    assert client.resolve_action("drain_account") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "expected"),
    [("allow", "allow"), ("deny", "deny"), ("maybe", "error"), ("ALLOW", "error")],
)
async def test_decision_vocabulary_is_never_optimistically_read(
    decision: str, expected: str
) -> None:
    client = _client(_bound(decision))
    await client.refresh_once()
    result = await client.decide("refunds.issue", _ctx())
    assert result.status == expected


@pytest.mark.asyncio
async def test_a_bare_allow_is_not_a_readable_answer() -> None:
    """The response every naive implementation accepts, and this one must not.

    Without a contract version and a bound decision id, "allow" is five
    characters anything on the path could have produced.
    """
    client = _client(_serving(_signed(), decide={"decision": "allow"}))
    await client.refresh_once()
    assert (await client.decide("refunds.issue", _ctx())).status == "error"


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [_OMIT, None, "", "mudraid.enforce.decide-response/2", 1])
async def test_an_absent_or_unsupported_response_contract_denies(version: Any) -> None:
    client = _client(_bound(schema_version=version))
    await client.refresh_once()
    assert (await client.decide("refunds.issue", _ctx())).status == "error"


@pytest.mark.asyncio
@pytest.mark.parametrize("decision_id", [_OMIT, None, "", 12345, "x" * 400])
async def test_a_missing_or_unusable_decision_id_denies(decision_id: Any) -> None:
    client = _client(_bound(decision_id=decision_id))
    await client.refresh_once()
    assert (await client.decide("refunds.issue", _ctx())).status == "error"


@pytest.mark.asyncio
async def test_a_decision_id_for_a_different_request_is_refused() -> None:
    """Binding is what stops one request's genuine allow answering another."""
    client = _client(_bound(decision_id="some-other-decision"))
    await client.refresh_once()
    assert (await client.decide("refunds.issue", _ctx())).status == "error"


@pytest.mark.asyncio
async def test_the_returned_decision_id_is_the_one_we_sent() -> None:
    handler = _serving(_signed())
    client = _client(handler)
    await client.refresh_once()
    result = await client.decide("refunds.issue", _ctx())
    assert result.decision_id == handler.captured["decide_body"]["decision_id"]


@pytest.mark.asyncio
async def test_an_oversized_response_is_refused_before_it_is_read() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bundle"):
            return httpx.Response(200, json=_signed())
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "schema_version": RESPONSE_CONTRACT,
                "decision_id": body["decision_id"],
                "decision": "allow",
                "padding": "x" * (MAX_RESPONSE_BYTES + 1),
            },
        )

    client = _client(handler)
    await client.refresh_once()
    assert (await client.decide("refunds.issue", _ctx())).status == "error"


@pytest.mark.asyncio
async def test_the_request_carries_the_versioned_contract() -> None:
    handler = _serving(_signed())
    client = _client(handler)
    await client.refresh_once()
    await client.decide("refunds.issue", _ctx())
    assert handler.captured["decide_body"]["schema_version"] == REQUEST_CONTRACT


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404, 500, 503])
async def test_a_non_200_from_the_authority_is_not_a_deny_but_an_error(status: int) -> None:
    """We could not consult the authority; that is not the authority denying."""
    bundle = _signed()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bundle"):
            return httpx.Response(200, json=bundle)
        return httpx.Response(status, json={"detail": "no"})

    client = _client(handler)
    await client.refresh_once()
    result = await client.decide("refunds.issue", _ctx())
    assert result.status == "error"


@pytest.mark.asyncio
async def test_a_timeout_is_deny_closed_as_a_timeout() -> None:
    bundle = _signed()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bundle"):
            return httpx.Response(200, json=bundle)
        raise httpx.ReadTimeout("too slow", request=request)

    client = _client(handler)
    await client.refresh_once()
    result = await client.decide("refunds.issue", _ctx())
    assert result.status == "timeout"


@pytest.mark.asyncio
async def test_a_transport_failure_is_deny_closed_as_unreachable() -> None:
    bundle = _signed()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bundle"):
            return httpx.Response(200, json=bundle)
        raise httpx.ConnectError("refused", request=request)

    client = _client(handler)
    await client.refresh_once()
    result = await client.decide("refunds.issue", _ctx())
    assert result.status == "unreachable"


@pytest.mark.asyncio
async def test_malformed_json_from_the_authority_is_an_error() -> None:
    bundle = _signed()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bundle"):
            return httpx.Response(200, json=bundle)
        return httpx.Response(200, content=b"{not json")

    client = _client(handler)
    await client.refresh_once()
    result = await client.decide("refunds.issue", _ctx())
    assert result.status == "error"


@pytest.mark.asyncio
async def test_without_a_service_secret_the_call_is_not_made_anonymously() -> None:
    """A private authenticated route is never called without a credential."""
    attempted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bundle"):
            return httpx.Response(200, json=_signed())
        if request.url.path.endswith("/keys"):
            # Public verification keys, deliberately unauthenticated — fetching
            # them without a credential is correct and is not the thing this
            # test guards.
            return httpx.Response(200, json={"keys": []})
        attempted.append(str(request.url))
        return httpx.Response(200, json={"decision": "allow"})

    client = _client(handler, service_secret="")
    await client.refresh_once()
    result = await client.decide("refunds.issue", _ctx())
    assert result.status == "credential_unconfigured"
    assert attempted == [], "the decision route was called without a credential"


@pytest.mark.asyncio
async def test_deciding_with_no_active_bundle_is_deny_closed() -> None:
    client = _client(_serving(None))
    result = await client.decide("refunds.issue", _ctx())
    assert result.status == "unconfigured"


@pytest.mark.asyncio
async def test_one_attempt_per_decision_never_a_retry() -> None:
    """A retried decision can double-meter a root authority decision."""
    calls: list[str] = []
    bundle = _signed()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bundle"):
            return httpx.Response(200, json=bundle)
        if request.url.path.endswith("/keys"):
            return httpx.Response(200, json={"keys": []})
        calls.append("decide")
        return httpx.Response(500)

    client = _client(handler)
    await client.refresh_once()
    await client.decide("refunds.issue", _ctx())
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_an_oversized_bundle_is_refused_before_parsing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"[" + b"0," * 5_000_000 + b"0]")

    client = _client(handler)
    assert await client.refresh_once() is False
    assert client.bundle_active is False


@pytest.mark.asyncio
async def test_the_client_satisfies_the_protocol_it_claims_to() -> None:
    from mudraid_platform_middleware.v2 import DecideClient

    assert isinstance(_client(_serving(_signed())), DecideClient)


def test_content_digest_is_over_the_content_not_the_payload() -> None:
    bundle = verify_bundle(_signed(), secret=SECRET)
    assert bundle.content_digest == sha256_hex(_content())


# ---------------------------------------------------------------------------
# 4. The destination we are willing to send secrets to
# ---------------------------------------------------------------------------


def _construct(**overrides: Any) -> HttpDecideClient:
    """Construct the client the way a CUSTOMER would.

    Deliberately minimal: base_url and adapter_token, plus a transport. If this
    helper needed more to work, the one-base-URL claim would be false — so the
    helper is itself part of the assertion.

    ``bundle_signing_secret`` is supplied because most tests here exercise the
    HMAC-era bundle fixtures; a real customer adapter passes none and verifies
    the asymmetric signature instead (see
    ``test_bundle_asymmetric_signature.py``).
    """
    kwargs: dict[str, Any] = {
        "base_url": BASE_URL,
        "adapter_token": "adapter-token",
        "bundle_signing_secret": SECRET,
        "client": httpx.AsyncClient(transport=httpx.MockTransport(_serving(_signed()))),
    }
    kwargs.update(overrides)
    return HttpDecideClient(**kwargs)


@pytest.mark.parametrize(
    "url",
    [
        "http://enforcement.example.com/decide",  # cleartext to a remote host
        "https://user:pw@enforcement.example.com/d",  # credentials in the URL
        "https://enforcement.example.com/d#frag",  # fragment
        "ftp://enforcement.example.com/d",  # not http(s)
        "https:///decide",  # no host
        "not-a-url",
    ],
)
def test_an_unsafe_decide_destination_is_refused_at_construction(url: str) -> None:
    """This request carries the caller's credential AND our service secret."""
    with pytest.raises(InsecureDecideDestination):
        _construct(decide_url=url)


def test_the_channel_destination_is_held_to_the_same_rule() -> None:
    with pytest.raises(InsecureDecideDestination):
        _construct(channel_url="http://platform.example.com")


def test_cleartext_loopback_requires_the_explicit_development_override() -> None:
    with pytest.raises(InsecureDecideDestination):
        _construct(decide_url="http://127.0.0.1:8080/decide")
    client = _construct(
        decide_url="http://127.0.0.1:8080/decide",
        channel_url="http://127.0.0.1:8081",
        allow_insecure_loopback=True,
    )
    assert client.bundle_active is False


def test_the_development_override_does_not_extend_beyond_loopback() -> None:
    """A host that merely looks loopback-ish is not loopback."""
    for host in ("127.0.0.1.evil.example", "notlocalhost", "10.0.0.1"):
        with pytest.raises(InsecureDecideDestination):
            _construct(decide_url=f"http://{host}/decide", allow_insecure_loopback=True)


def test_an_origin_outside_the_allowed_set_is_refused() -> None:
    """One base URL means one origin to allow — and one to refuse.

    Simpler than it was: the allowlist used to have to cover the channel and
    decide origins separately, and covering one but not the other was a
    configuration that half-worked.
    """
    with pytest.raises(InsecureDecideDestination):
        _construct(allowed_origins={"https://enforcement.mudraid.ai"})
    # The configured origin is accepted.
    _construct(allowed_origins={BASE_URL})


@pytest.mark.asyncio
async def test_secrets_never_reach_logs_or_exception_text(caplog: Any) -> None:
    """Every credential this client holds, checked against everything it emits."""
    secrets = ("adapter-token", SECRET, "svc-secret", "Bearer caller-jwt")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bundle"):
            return httpx.Response(500)
        return httpx.Response(503, content=b"upstream boom")

    with caplog.at_level("DEBUG", logger="mudraid_platform_middleware.decide"):
        client = _client(handler)
        await client.refresh_once()
        await client.decide("refunds.issue", _ctx(presented_bearer="Bearer caller-jwt"))

    emitted = "\n".join(record.getMessage() for record in caplog.records)
    assert emitted  # the failures did produce log lines
    for secret in secrets:
        assert secret not in emitted


def test_an_unsafe_url_error_does_not_echo_the_secret() -> None:
    try:
        _construct(decide_url="https://user:hunter2@enforcement.example.com/d")
    except InsecureDecideDestination as exc:
        assert "hunter2" not in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected refusal")


# ---------------------------------------------------------------------------
# 5. Request context is explicit, validated, and cannot hide a wiring error
# ---------------------------------------------------------------------------


def test_the_context_has_no_defaults_that_could_hide_a_wiring_error() -> None:
    with pytest.raises(TypeError):
        DecideContext()  # type: ignore[call-arg]


def test_an_absent_credential_is_represented_explicitly() -> None:
    assert _ctx(presented_bearer=None).presented_bearer is None
    assert _ctx(presented_bearer="   ").presented_bearer is None


@pytest.mark.parametrize("method", ["PO ST", "GET\r\nX: y", "", "GET Z"])
def test_an_invalid_http_method_is_rejected(method: str) -> None:
    with pytest.raises(ValueError):
        _ctx(http_method=method)


@pytest.mark.parametrize("path", ["mcp", "", "http://x/mcp", "/mcp\nX: y"])
def test_an_invalid_path_is_rejected(path: str) -> None:
    with pytest.raises(ValueError):
        _ctx(path=path)


def test_the_method_is_normalized_once() -> None:
    assert _ctx(http_method="post").http_method == "POST"


@pytest.mark.parametrize("bad", ["with space", "a" * 500, "", "tab\there", None, 42])
def test_an_unusable_correlation_id_is_replaced_not_propagated(bad: Any) -> None:
    ctx = _ctx(correlation_id=bad)
    assert ctx.correlation_id and " " not in ctx.correlation_id
    assert ctx.correlation_id != bad


def test_a_usable_correlation_id_is_kept() -> None:
    assert _ctx(correlation_id="corr-1").correlation_id == "corr-1"


@pytest.mark.asyncio
async def test_no_adapter_supplied_identity_reaches_the_authority() -> None:
    """Identity comes from the verified credential, never from the adapter."""
    handler = _serving(_signed())
    client = _client(handler)
    await client.refresh_once()
    await client.decide("refunds.issue", _ctx())
    body = json.dumps(handler.captured["decide_body"])
    for forbidden in ("agent_id", "user_id", "organization_id"):
        assert forbidden not in body


@pytest.mark.asyncio
async def test_the_request_body_is_never_forwarded() -> None:
    handler = _serving(_signed())
    client = _client(handler)
    await client.refresh_once()
    await client.decide("refunds.issue", _ctx())
    body = handler.captured["decide_body"]
    assert "body" not in body and "params" not in body
    assert set(body["request"]) == {"transport", "http_method", "path"}


# ---------------------------------------------------------------------------
# One base URL: the same artifact in every environment.
# ---------------------------------------------------------------------------
def test_every_path_is_derived_from_the_one_base_url() -> None:
    """Selecting an environment is a config change, never a different build."""
    client = _construct(base_url="https://api.mudraid.ai")
    assert client._decide_url == "https://api.mudraid.ai/api/v1/adapter/enforcement/decide"
    assert client._keys_url == "https://api.mudraid.ai/api/v1/adapter/enforcement/keys"
    assert client._channel_url == "https://api.mudraid.ai"

    staging = _construct(base_url="https://api.staging.mudraid.ai")
    assert staging._decide_url.startswith("https://api.staging.mudraid.ai/")
    # Same class, same code path — only the configured origin differs.
    assert type(staging) is type(client)


def test_the_package_carries_no_mudraid_hostname() -> None:
    """A hard-coded host would make one build environment-specific.

    Checked against the SHIPPED source rather than asserted in prose: this is
    the property that lets the identical artifact be promoted from staging to
    production, so it is worth a test that fails if someone inlines a URL.
    """
    import pathlib

    import mudraid_platform_middleware

    package = pathlib.Path(mudraid_platform_middleware.__file__).parent
    offenders: list[str] = []
    for module in package.rglob("*.py"):
        for number, line in enumerate(module.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if "mudraid.ai" in code or "mudraid.io" in code:
                offenders.append(f"{module.name}:{number}: {line.strip()}")
    assert offenders == [], "a MudraID hostname is compiled into the package:\n" + "\n".join(
        offenders
    )


def test_an_override_is_validated_as_strictly_as_a_derived_path() -> None:
    """An advanced override changes WHERE a request goes, never how it is checked.

    An override that skipped validation would be a way to reintroduce every
    destination rule this client exists to enforce.
    """
    with pytest.raises(InsecureDecideDestination):
        _construct(decide_url="http://enforcement.example.com/decide")
    with pytest.raises(InsecureDecideDestination):
        _construct(keys_url="https://user:pw@enforcement.example.com/keys")


@pytest.mark.asyncio
async def test_verification_keys_are_fetched_before_the_bundle() -> None:
    """Order matters: a signed bundle with an empty key set is REFUSED.

    Fetching in the other order would make every cold start a refusal, even
    when the keys were reachable the whole time.
    """
    order: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/keys"):
            order.append("keys")
            return httpx.Response(200, json={"keys": []})
        order.append("bundle")
        return httpx.Response(200, json=_signed())

    client = _client(handler)
    await client.refresh_once()
    assert order[:2] == ["keys", "bundle"]
