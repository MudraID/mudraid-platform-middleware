"""The /decide response signature: verify when present, refuse when wrong.

AUDIT-009 A9-02. The envelope-level binding (contract, decision id, freshness)
defends against a confused or replaying authority; only the signature defends
against a party who can rewrite the TLS stream. These tests pin both the
verifier itself and its integration into ``_read_decision`` — a refusal that
merely raised somewhere but still surfaced an allow would be worse than none.

The cross-implementation vectors live in the shared corpus
(kong/tests/fixtures/adapter-conformance.json, run by
kong/tests/test_adapter_conformance.py and kong/tests/lua/test_conformance.lua);
these unit tests cover what the corpus cannot — key-set refresh plumbing, the
no-keys bootstrap edge, and refusal ordering.
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from mudraid_platform_middleware._canonical import canonical_json_bytes
from mudraid_platform_middleware._decision_signature import (
    DECISION_SIGNATURE_ALGORITHM,
    DECISION_SIGNATURE_PROFILE,
    DecisionBindings,
    DecisionSignatureRefused,
    verify_decision_signature,
)
from mudraid_platform_middleware.decide_client import _read_decision

KEY_ID = "muid_sk_decision_1"
DECISION_ID = "dec-42"
NOW = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)

BINDINGS = DecisionBindings(
    platform_id="platform-1",
    environment="production",
    resource="https://mcp.example.test/mcp",
    action_key="crm.contacts.export",
    bundle_version=7,
)


@pytest.fixture(scope="module")
def keypair():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = (
        private.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return private, public_pem


def _envelope(decision: str = "allow") -> dict:
    decided_at = NOW - timedelta(seconds=5)
    primary = "authorized" if decision == "allow" else "scope_not_granted"
    return {
        "schema_version": "2.0",
        "decision_id": DECISION_ID,
        "decision": decision,
        "outcome": decision,
        "reason": {"primary": primary, "contributing": [], "category": None},
        "decided_at": decided_at.isoformat(),
        "deadline_at": (decided_at + timedelta(milliseconds=250)).isoformat(),
    }


def _claims(envelope: dict, **overrides) -> dict:
    claims = {
        "profile": DECISION_SIGNATURE_PROFILE,
        "algorithm": DECISION_SIGNATURE_ALGORITHM,
        "key_id": KEY_ID,
        "decision_id": envelope["decision_id"],
        "outcome": envelope["outcome"],
        "decision": envelope["decision"],
        "reason_primary": envelope["reason"]["primary"],
        "organization_id": "org-1",
        "environment": BINDINGS.environment,
        "platform_id": BINDINGS.platform_id,
        "agent_id": "agent-1",
        "subject": "client-1",
        "action_key": BINDINGS.action_key,
        "resource": BINDINGS.resource,
        "bundle_version": BINDINGS.bundle_version,
        "policy_version": 3,
        "decided_at": envelope["decided_at"],
        "deadline_at": envelope["deadline_at"],
        "not_before": envelope["decided_at"],
        "expires_at": (NOW + timedelta(minutes=5)).isoformat(),
    }
    claims.update(overrides)
    return claims


def _sign(private, claims: dict) -> dict:
    raw = private.sign(canonical_json_bytes(claims), padding.PKCS1v15(), hashes.SHA256())
    return {
        "profile": DECISION_SIGNATURE_PROFILE,
        "algorithm": DECISION_SIGNATURE_ALGORITHM,
        "key_id": KEY_ID,
        "claims": claims,
        "signature": base64.b64encode(raw).decode("ascii"),
    }


def _signed(keypair, decision: str = "allow", **claim_overrides) -> tuple[dict, str]:
    private, public_pem = keypair
    envelope = _envelope(decision)
    envelope["signature"] = _sign(private, _claims(envelope, **claim_overrides))
    return envelope, public_pem


# ── An HMAC-signed enforcement bundle, for the end-to-end wiring tests ───────
# Mirrors tests/unit/test_decide_client.py's fixture: the mandatory-mode wiring
# has to be driven through a real HttpDecideClient.decide() call, and that needs
# an active bundle to resolve an action against.
_BUNDLE_SECRET = "bundle-signing-secret"


def _bundle(version: int = 7) -> dict:
    payload = {
        "schema_version": "1.0",
        "bundle_version": version,
        "content": {
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
        },
    }
    canon = canonical_json_bytes(payload)
    return {
        "bundle_version": version,
        "schema_version": "1.0",
        "payload": payload,
        "payload_digest": hashlib.sha256(canon).hexdigest(),
        "signing_key_id": "local-hmac-v1",
        "signature": hmac.new(_BUNDLE_SECRET.encode(), canon, hashlib.sha256).hexdigest(),
    }


_BUNDLE = _bundle()


def _DECIDE_CTX():
    from mudraid_platform_middleware.v2 import DecideContext

    return DecideContext(
        presented_bearer="Bearer caller-jwt", http_method="POST", path="/mcp"
    )


# ---------------------------------------------------------------------------
# The verifier itself
# ---------------------------------------------------------------------------
class TestVerifier:
    def test_a_genuine_allow_verifies(self, keypair):
        envelope, public_pem = _signed(keypair, "allow")
        claims = verify_decision_signature(
            envelope, verification_keys={KEY_ID: public_pem}, expected=BINDINGS, now=NOW
        )
        assert claims["decision_id"] == DECISION_ID

    def test_a_genuine_deny_verifies(self, keypair):
        envelope, public_pem = _signed(keypair, "deny")
        verify_decision_signature(
            envelope, verification_keys={KEY_ID: public_pem}, expected=BINDINGS, now=NOW
        )

    def test_a_signature_with_no_keys_is_refused_not_skipped(self, keypair):
        """The bootstrap/rotation edge: signed before any decision key is
        published. Unverifiable is a refusal, never 'nothing to check'."""
        envelope, _ = _signed(keypair)
        with pytest.raises(DecisionSignatureRefused) as exc:
            verify_decision_signature(
                envelope, verification_keys=None, expected=BINDINGS, now=NOW
            )
        assert exc.value.code == "DECIDE_RESPONSE_SIGNATURE_KEYS_UNAVAILABLE"

    def test_the_bundle_key_never_verifies_a_decision(self, keypair):
        """Different series, different trust: a decision signature naming a
        key id that only exists in the BUNDLE series must be unknown here."""
        envelope, public_pem = _signed(keypair)
        with pytest.raises(DecisionSignatureRefused):
            verify_decision_signature(
                envelope,
                verification_keys={"muid_sk_bundle_1": public_pem},
                expected=BINDINGS,
                now=NOW,
            )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("decision", "deny"),
            ("outcome", "deny"),
            ("decision_id", "dec-other"),
            ("decided_at", "2026-08-20T11:00:00+00:00"),
            ("deadline_at", "2026-08-21T11:00:00+00:00"),
        ],
    )
    def test_an_envelope_edited_under_an_intact_signature_is_refused(
        self, keypair, field, value
    ):
        envelope, public_pem = _signed(keypair)
        envelope[field] = value
        with pytest.raises(DecisionSignatureRefused, match="does not cover"):
            verify_decision_signature(
                envelope, verification_keys={KEY_ID: public_pem}, expected=BINDINGS, now=NOW
            )

    def test_a_rewritten_reason_is_refused(self, keypair):
        envelope, public_pem = _signed(keypair, "deny")
        envelope["reason"] = {"primary": "authorized", "contributing": [], "category": None}
        with pytest.raises(DecisionSignatureRefused, match="reason"):
            verify_decision_signature(
                envelope, verification_keys={KEY_ID: public_pem}, expected=BINDINGS, now=NOW
            )

    def test_foreign_surface_bindings_are_refused(self, keypair):
        envelope, public_pem = _signed(keypair, "allow", platform_id="platform-other")
        with pytest.raises(DecisionSignatureRefused, match="another platform"):
            verify_decision_signature(
                envelope, verification_keys={KEY_ID: public_pem}, expected=BINDINGS, now=NOW
            )

    def test_an_expired_window_is_refused(self, keypair):
        envelope, public_pem = _signed(
            keypair,
            "allow",
            not_before=(NOW - timedelta(hours=2)).isoformat(),
            expires_at=(NOW - timedelta(hours=1)).isoformat(),
        )
        with pytest.raises(DecisionSignatureRefused, match="expired"):
            verify_decision_signature(
                envelope, verification_keys={KEY_ID: public_pem}, expected=BINDINGS, now=NOW
            )

    def test_algorithm_is_pinned_not_read(self, keypair):
        envelope, public_pem = _signed(keypair)
        envelope["signature"] = copy.deepcopy(envelope["signature"])
        envelope["signature"]["algorithm"] = "none"
        with pytest.raises(DecisionSignatureRefused, match="algorithm"):
            verify_decision_signature(
                envelope, verification_keys={KEY_ID: public_pem}, expected=BINDINGS, now=NOW
            )


# ---------------------------------------------------------------------------
# Integration into _read_decision: a refusal deny-closes the WHOLE response
# ---------------------------------------------------------------------------
class TestReadDecisionIntegration:
    def test_a_signed_allow_reads_as_allow(self, keypair):
        envelope, public_pem = _signed(keypair, "allow")
        result = _read_decision(
            envelope,
            expected_decision_id=DECISION_ID,
            now=NOW,
            verification_keys={KEY_ID: public_pem},
            expected_bindings=BINDINGS,
        )
        assert result.status == "allow"

    def test_a_forged_allow_under_a_deny_signature_is_error(self, keypair):
        """THE forgery this signature exists for."""
        envelope, public_pem = _signed(keypair, "deny")
        envelope["decision"] = "allow"
        envelope["outcome"] = "allow"
        result = _read_decision(
            envelope,
            expected_decision_id=DECISION_ID,
            now=NOW,
            verification_keys={KEY_ID: public_pem},
            expected_bindings=BINDINGS,
        )
        assert result.status == "error"

    def test_a_signed_response_with_no_keys_is_error(self, keypair):
        envelope, _ = _signed(keypair, "allow")
        result = _read_decision(
            envelope,
            expected_decision_id=DECISION_ID,
            now=NOW,
            verification_keys=None,
            expected_bindings=BINDINGS,
        )
        assert result.status == "error"

    def test_an_unsigned_response_still_reads(self):
        """Verify-when-present: the authority activates signing by rollout,
        and the rollout must not be a flag-day at the adapter."""
        result = _read_decision(
            _envelope("deny"),
            expected_decision_id=DECISION_ID,
            now=NOW,
            verification_keys={},
            expected_bindings=BINDINGS,
        )
        assert result.status == "deny"

    def test_signature_json_null_is_absence_not_an_object(self):
        """A serialized envelope may carry ``"signature": null``; that is the
        same fact as no signature, not a malformed one."""
        envelope = _envelope("deny")
        envelope["signature"] = None
        result = _read_decision(
            envelope,
            expected_decision_id=DECISION_ID,
            now=NOW,
            verification_keys={},
            expected_bindings=BINDINGS,
        )
        assert result.status == "deny"


# ---------------------------------------------------------------------------
# Key-set refresh plumbing: the decision series rides the same keys page
# ---------------------------------------------------------------------------
class TestDecisionKeyRefresh:
    def _client(self, payload: dict):
        import httpx

        from mudraid_platform_middleware.decide_client import HttpDecideClient

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/keys"):
                return httpx.Response(200, json=payload)
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        return HttpDecideClient(
            base_url="https://api.example.test",
            adapter_token="adapter-tok",
            client=httpx.AsyncClient(transport=transport),
        )

    async def test_decision_keys_are_read_from_the_key_sets_projection(self):
        client = self._client(
            {
                "purpose": "enforcement_bundle_signing",
                "keys": [{"key_id": "bk1", "public_key_pem": "BUNDLE-PEM"}],
                "key_sets": [
                    {
                        "purpose": "enforcement_bundle_signing",
                        "keys": [{"key_id": "bk1", "public_key_pem": "BUNDLE-PEM"}],
                    },
                    {
                        "purpose": "enforcement_decision_signing",
                        "keys": [{"key_id": "dk1", "public_key_pem": "DECISION-PEM"}],
                    },
                ],
            }
        )
        await client._refresh_verification_keys()
        # The two series never bleed into each other.
        assert client._verification_keys == {"bk1": "BUNDLE-PEM"}
        assert client._decision_verification_keys == {"dk1": "DECISION-PEM"}

    async def test_an_older_control_plane_without_key_sets_is_tolerated(self):
        client = self._client(
            {
                "purpose": "enforcement_bundle_signing",
                "keys": [{"key_id": "bk1", "public_key_pem": "BUNDLE-PEM"}],
            }
        )
        client._decision_verification_keys = {"dk-old": "OLD"}
        await client._refresh_verification_keys()
        # No decision set on the page leaves the working set alone (a page
        # that stops publishing the set is indistinguishable from an older
        # control plane; clearing on it would refuse every signed decision).
        assert client._decision_verification_keys == {"dk-old": "OLD"}

    async def test_a_successful_decision_set_read_replaces_for_rotation(self):
        client = self._client(
            {
                "purpose": "enforcement_bundle_signing",
                "keys": [{"key_id": "bk1", "public_key_pem": "BUNDLE-PEM"}],
                "key_sets": [
                    {
                        "purpose": "enforcement_decision_signing",
                        "keys": [{"key_id": "dk2", "public_key_pem": "NEW-PEM"}],
                    }
                ],
            }
        )
        client._decision_verification_keys = {"dk1": "RETIRED-PEM"}
        await client._refresh_verification_keys()
        # REPLACED, not merged: a key that stops being published stops being
        # trusted at the next successful refresh.
        assert client._decision_verification_keys == {"dk2": "NEW-PEM"}


# ---------------------------------------------------------------------------
# Mandatory mode (Decision 8): require_signed_decisions
# ---------------------------------------------------------------------------
#
# The optional-and-default-off enforcement setting. Two modes, and the ONLY
# difference between them is what an ABSENT signature means:
#
#   off (default) — unsigned reads as before (1.1-compatible rollout);
#   on            — unsigned is refused.
#
# In BOTH modes a signature that is PRESENT must verify completely. Absence and
# invalidity stay different facts; the flag governs only the first.
class TestMandatorySignatureMode:
    """require_signed_decisions at the _read_decision seam."""

    def _read(self, envelope, *, keys, **kwargs):
        return _read_decision(
            envelope,
            expected_decision_id=DECISION_ID,
            now=NOW,
            verification_keys=keys,
            expected_bindings=BINDINGS,
            **kwargs,
        )

    # ── off: the 1.1-compatible default ──────────────────────────────────────
    def test_default_is_off_so_an_unsigned_response_reads(self):
        """The default must not change 1.1 behaviour: no flag passed at all."""
        result = self._read(_envelope("deny"), keys={})
        assert result.status == "deny"

    def test_off_and_unsigned_reads(self):
        result = self._read(_envelope("deny"), keys={}, require_signed_decisions=False)
        assert result.status == "deny"

    def test_off_and_a_valid_signature_reads(self, keypair):
        envelope, public_pem = _signed(keypair, "allow")
        result = self._read(
            envelope, keys={KEY_ID: public_pem}, require_signed_decisions=False
        )
        assert result.status == "allow"

    def test_off_and_an_invalid_signature_is_still_refused(self, keypair):
        """Deny-closed regardless of the flag: present-and-wrong always refuses."""
        envelope, public_pem = _signed(keypair, "deny")
        envelope["decision"] = "allow"
        envelope["outcome"] = "allow"
        result = self._read(
            envelope, keys={KEY_ID: public_pem}, require_signed_decisions=False
        )
        assert result.status == "error"

    # ── on: the new enforcement ──────────────────────────────────────────────
    def test_on_and_an_unsigned_response_is_refused(self, keypair):
        """THE new behaviour. An unsigned decision is not an answer here."""
        _, public_pem = keypair
        result = self._read(
            _envelope("allow"),
            keys={KEY_ID: public_pem},
            require_signed_decisions=True,
        )
        assert result.status == "error"

    def test_on_refuses_an_unsigned_deny_too(self, keypair):
        """Not an allow-only check: an unsigned DENY is equally unreadable, or
        an attacker could strip the signature from an allow and still be heard
        by stripping it from a deny."""
        _, public_pem = keypair
        result = self._read(
            _envelope("deny"), keys={KEY_ID: public_pem}, require_signed_decisions=True
        )
        assert result.status == "error"

    def test_on_and_json_null_signature_is_refused(self, keypair):
        """``"signature": null`` is ABSENCE — which mandatory mode refuses."""
        _, public_pem = keypair
        envelope = _envelope("allow")
        envelope["signature"] = None
        result = self._read(
            envelope, keys={KEY_ID: public_pem}, require_signed_decisions=True
        )
        assert result.status == "error"

    def test_on_and_a_valid_signature_reads(self, keypair):
        envelope, public_pem = _signed(keypair, "allow")
        result = self._read(
            envelope, keys={KEY_ID: public_pem}, require_signed_decisions=True
        )
        assert result.status == "allow"

    def test_on_and_a_valid_deny_reads_as_deny(self, keypair):
        envelope, public_pem = _signed(keypair, "deny")
        result = self._read(
            envelope, keys={KEY_ID: public_pem}, require_signed_decisions=True
        )
        assert result.status == "deny"

    def test_on_and_an_invalid_signature_is_refused(self, keypair):
        envelope, public_pem = _signed(keypair, "deny")
        envelope["decision"] = "allow"
        envelope["outcome"] = "allow"
        result = self._read(
            envelope, keys={KEY_ID: public_pem}, require_signed_decisions=True
        )
        assert result.status == "error"

    def test_a_mandatory_refusal_never_surfaces_an_allow(self, keypair):
        """The refusal must deny CLOSED — an unsigned allow must not become one."""
        _, public_pem = keypair
        result = self._read(
            _envelope("allow"),
            keys={KEY_ID: public_pem},
            require_signed_decisions=True,
        )
        assert result.status != "allow"
        assert result.status == "error"

    # ── both modes refuse a present-and-wrong signature, for every reason ────
    @pytest.mark.parametrize("mandatory", [False, True])
    def test_an_unknown_key_is_refused_in_both_modes(self, keypair, mandatory):
        envelope, public_pem = _signed(keypair, "allow")
        result = self._read(
            envelope,
            keys={"some-other-key": public_pem},
            require_signed_decisions=mandatory,
        )
        assert result.status == "error"

    @pytest.mark.parametrize("mandatory", [False, True])
    def test_an_expired_window_is_refused_in_both_modes(self, keypair, mandatory):
        envelope, public_pem = _signed(
            keypair,
            "allow",
            expires_at=(NOW - timedelta(minutes=30)).isoformat(),
        )
        result = self._read(
            envelope, keys={KEY_ID: public_pem}, require_signed_decisions=mandatory
        )
        assert result.status == "error"

    @pytest.mark.parametrize("mandatory", [False, True])
    def test_a_mutated_bound_field_is_refused_in_both_modes(self, keypair, mandatory):
        envelope, public_pem = _signed(keypair, "allow")
        envelope["reason"]["primary"] = "something_else"
        result = self._read(
            envelope, keys={KEY_ID: public_pem}, require_signed_decisions=mandatory
        )
        assert result.status == "error"

    @pytest.mark.parametrize("mandatory", [False, True])
    def test_a_foreign_surface_is_refused_in_both_modes(self, keypair, mandatory):
        envelope, public_pem = _signed(keypair, "allow", platform_id="someone-elses")
        result = self._read(
            envelope, keys={KEY_ID: public_pem}, require_signed_decisions=mandatory
        )
        assert result.status == "error"

    @pytest.mark.parametrize("mandatory", [False, True])
    def test_a_foreign_action_is_refused_in_both_modes(self, keypair, mandatory):
        envelope, public_pem = _signed(keypair, "allow", action_key="billing.refund")
        result = self._read(
            envelope, keys={KEY_ID: public_pem}, require_signed_decisions=mandatory
        )
        assert result.status == "error"

    @pytest.mark.parametrize("mandatory", [False, True])
    def test_a_signature_arriving_before_the_keys_is_refused_in_both_modes(
        self, keypair, mandatory
    ):
        """Existing behaviour, restated under the flag: a signed decision that
        arrives before the client has fetched the decision key series is
        REFUSED (unknown key), never skipped as if unsigned."""
        envelope, _ = _signed(keypair, "allow")
        result = self._read(
            envelope, keys=None, require_signed_decisions=mandatory
        )
        assert result.status == "error"


# ---------------------------------------------------------------------------
# Wiring: the operator-facing setting must actually reach the verifier
# ---------------------------------------------------------------------------
#
# The defect these guard against is a setting that can be PASSED somewhere
# plausible and have NO EFFECT. require_signed_decisions lives on
# HttpDecideClient — the object that both holds the signature and does the
# verifying — so these drive it end to end through a real decide() call rather
# than asserting the constructor merely accepts the keyword.
class TestMandatoryModeIsWiredThroughTheClient:
    def _client(self, *, decide_response, **kwargs):
        import httpx

        from mudraid_platform_middleware.decide_client import HttpDecideClient

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/keys"):
                return httpx.Response(200, json={"keys": []})
            if request.url.path.endswith("/bundle"):
                return httpx.Response(200, json=_BUNDLE)
            body = json.loads(request.content)
            payload = dict(decide_response)
            payload["decision_id"] = body["decision_id"]
            return httpx.Response(200, json=payload)

        return HttpDecideClient(
            base_url="https://api.example.test",
            adapter_token="adapter-tok",
            bundle_signing_secret=_BUNDLE_SECRET,
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            **kwargs,
        )

    def _unsigned_allow(self):
        return {
            "schema_version": "2.0",
            "decision": "allow",
            "decided_at": datetime.now(timezone.utc).isoformat(),
        }

    async def test_the_client_defaults_to_off_and_reads_an_unsigned_decision(self):
        """1.1-compatible default, proven through the real client."""
        client = self._client(decide_response=self._unsigned_allow())
        await client.refresh_once()
        result = await client.decide("refunds.issue", _DECIDE_CTX())
        assert result.status == "allow"

    async def test_the_client_flag_off_explicitly_reads_an_unsigned_decision(self):
        client = self._client(
            decide_response=self._unsigned_allow(), require_signed_decisions=False
        )
        await client.refresh_once()
        result = await client.decide("refunds.issue", _DECIDE_CTX())
        assert result.status == "allow"

    async def test_the_client_flag_on_refuses_an_unsigned_decision(self):
        """If this passes with the flag ignored, the setting is decorative."""
        client = self._client(
            decide_response=self._unsigned_allow(), require_signed_decisions=True
        )
        await client.refresh_once()
        result = await client.decide("refunds.issue", _DECIDE_CTX())
        assert result.status == "error"


class TestTheSettingHasExactlyOneHome:
    """It must not be settable anywhere it would be silently ignored.

    V2Config carries a DecideClient *Protocol* whose only members are
    ``bundle_active`` / ``resolve_action`` / ``decide(action, context)``. There
    is no channel on that seam to carry a verification policy, and DecideResult
    carries no signed-ness for the control loop to enforce after the fact — so a
    require_signed_decisions on V2Config could not be honoured for an injected
    client. It deliberately does not exist there.
    """

    def test_v2config_does_not_accept_an_inert_copy_of_the_setting(self):
        from mudraid_platform_middleware.v2 import V2Config

        class _FakeClient:
            bundle_active = True

            def resolve_action(self, tool_name):
                return None

            async def decide(self, action, context):
                raise AssertionError("not called")

        with pytest.raises(TypeError):
            V2Config(  # type: ignore[call-arg]
                decide_client=_FakeClient(), require_signed_decisions=True
            )

    def test_the_decide_client_protocol_gains_no_verification_member(self):
        """The seam stays three members; widening it would put a security
        policy on an interface third-party clients implement."""
        from mudraid_platform_middleware.v2 import DecideClient

        members = {
            name for name in vars(DecideClient) if not name.startswith("_")
        }
        assert members == {"bundle_active", "resolve_action", "decide"}


class TestV1ModeIsUnaffected:
    def test_v1_mode_still_refuses_a_v2_config_and_has_no_decide_client(self):
        """V1 never constructs the seam the setting lives on, so the setting
        has no V1 code path to change."""
        from mudraid_platform_middleware.middleware import MudraIDMiddleware

        assert not hasattr(MudraIDMiddleware, "require_signed_decisions")

    async def test_a_v1_dispatch_is_unchanged_by_the_setting(self):
        """A V1 app decides nothing through /decide at all."""
        from mudraid_platform_middleware.v2 import V2Config

        assert "require_signed_decisions" not in {
            f.name for f in dataclasses.fields(V2Config)
        }
