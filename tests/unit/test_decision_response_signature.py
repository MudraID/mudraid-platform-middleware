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
