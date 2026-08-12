"""Absence and invalidity are different facts, and the window depends on it.

A bundle published before asymmetric signing existed carries no signature and
must still be accepted on its HMAC — that is what makes the migration window a
window. A bundle that CARRIES a signature which does not verify must never be
treated as legacy: if invalidity fell back to the HMAC, anyone holding the
shared secret could corrupt one field and downgrade every bundle to the weaker
check.

These tests pin both halves, and the bindings that stop a genuine signature
being replayed onto the wrong surface.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from mudraid_platform_middleware._bundle import (
    _REQUIRED_EVALUATION,
    BUNDLE_SIGNATURE_ALGORITHM,
    BUNDLE_SIGNATURE_PROFILE,
    BundleRefused,
    verify_bundle,
)
from mudraid_platform_middleware._canonical import canonical_json_bytes

SECRET = "test-bundle-signing-secret"
PLATFORM = "platform-1"
ENVIRONMENT = "staging"
KEY_ID = "muid_sk_bundle_1"


@pytest.fixture(scope="module")
def keypair():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = (
        private.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return private, private_pem, public_pem


def _payload(version: int = 1) -> dict:
    return {
        "schema_version": "1.0",
        "bundle_version": version,
        "issued_at": "2026-08-11T12:00:00+00:00",
        "not_before": "2026-08-11T12:00:00+00:00",
        "previous_bundle_version": None,
        "content": {
            "surface": {
                "platform_id": PLATFORM,
                "canonical_resource_uri": "https://mcp.example.com/mcp",
                "environment": ENVIRONMENT,
            },
            # Built from the module's own _REQUIRED_EVALUATION rather than typed out
            # here: a hand-copied contract is a second copy that can go stale, and
            # this fixture exists to test the SIGNATURE, not to re-assert the
            # evaluation rules the module already enforces.
            "evaluation": {
                **_REQUIRED_EVALUATION,
                "decide_required": True,
                "retry_forwarded_request": False,
            },
            "matcher": {
                "kind": "mcp_tool_exact",
                "actions": [{"tool_name": "t", "action_key": "a"}],
            },
        },
    }


def _legacy_bundle(version: int = 1) -> dict:
    """An HMAC-only bundle: exactly what existed before asymmetric signing."""
    payload = _payload(version)
    canon = canonical_json_bytes(payload)
    return {
        "bundle_version": version,
        "schema_version": "1.0",
        "payload": payload,
        "payload_digest": hashlib.sha256(canon).hexdigest(),
        "signing_key_id": "local-hmac-v1",
        "signature": hmac.new(SECRET.encode(), canon, "sha256").hexdigest(),
    }


def _signed_bundle(keypair, version: int = 1, **claim_overrides) -> tuple[dict, dict]:
    private, _, public_pem = keypair
    bundle = _legacy_bundle(version)
    claims = {
        "profile": BUNDLE_SIGNATURE_PROFILE,
        "algorithm": BUNDLE_SIGNATURE_ALGORITHM,
        "key_id": KEY_ID,
        "payload_digest": bundle["payload_digest"],
        "bundle_version": version,
        "platform_id": PLATFORM,
        "environment": ENVIRONMENT,
        "canonical_resource_uri": "https://mcp.example.com/mcp",
        "issued_at": "2026-08-11T12:00:00+00:00",
        "not_before": "2026-08-11T12:00:00+00:00",
        "expires_at": "2026-09-10T12:00:00+00:00",
    }
    claims.update(claim_overrides)
    raw = private.sign(canonical_json_bytes(claims), padding.PKCS1v15(), hashes.SHA256())
    bundle.update(
        {
            "signature_profile": BUNDLE_SIGNATURE_PROFILE,
            "signature_algorithm": BUNDLE_SIGNATURE_ALGORITHM,
            "signature_key_id": KEY_ID,
            "signature_claims": claims,
            "signature_value": base64.b64encode(raw).decode(),
        }
    )
    return bundle, {KEY_ID: public_pem}


def _verify(bundle, keys=None, **kwargs):
    return verify_bundle(
        bundle,
        secret=SECRET,
        verification_keys=keys,
        expected_platform_id=kwargs.pop("platform_id", PLATFORM),
        expected_environment=kwargs.pop("environment", ENVIRONMENT),
        **kwargs,
    )


class TestTheWindow:
    def test_a_legacy_bundle_with_no_signature_still_verifies_on_its_hmac(self):
        """The window exists for exactly this bundle.

        Refusing it would break every surface whose bundle predates asymmetric
        signing — an enforcement outage caused by a rollout.
        """
        verified = _verify(_legacy_bundle(), keys=None)
        assert verified.bundle_version == 1

    def test_a_signed_bundle_verifies_when_the_key_is_published(self, keypair):
        bundle, keys = _signed_bundle(keypair)
        assert _verify(bundle, keys).bundle_version == 1

    def test_an_invalid_signature_never_falls_back_to_the_hmac(self, keypair):
        """THE test in this file.

        The HMAC here is perfectly valid — only the asymmetric signature is
        corrupt. If invalidity fell back, an attacker holding the shared secret
        could downgrade every bundle by flipping one byte.
        """
        bundle, keys = _signed_bundle(keypair)
        tampered = copy.deepcopy(bundle)
        tampered["signature_claims"]["environment"] = "production"

        with pytest.raises(BundleRefused) as exc:
            _verify(tampered, keys, environment="production")
        assert exc.value.code == "BUNDLE_SIGNATURE_INVALID"

    def test_a_signed_bundle_with_no_keys_available_is_refused_not_downgraded(self, keypair):
        """Carrying a signature nobody can check is not the same as carrying none.

        Falling back here would let an attacker force the weaker check simply by
        making the key endpoint unreachable.
        """
        bundle, _ = _signed_bundle(keypair)
        with pytest.raises(BundleRefused) as exc:
            _verify(bundle, keys=None)
        assert exc.value.code == "BUNDLE_VERIFICATION_KEYS_UNAVAILABLE"


class TestBindings:
    def test_a_bundle_bound_to_another_platform_is_refused(self, keypair):
        bundle, keys = _signed_bundle(keypair)
        with pytest.raises(BundleRefused, match="another platform"):
            _verify(bundle, keys, platform_id="platform-2")

    def test_a_staging_bundle_replayed_into_production_is_refused(self, keypair):
        """The signature is genuine; the placement is not."""
        bundle, keys = _signed_bundle(keypair)
        with pytest.raises(BundleRefused, match="another environment"):
            _verify(bundle, keys, environment="production")

    def test_a_signature_covering_a_different_version_is_refused(self, keypair):
        bundle, keys = _signed_bundle(keypair, version=2)
        # The claims say version 2; present it as the bundle for version 2 but
        # with the digest of a different payload.
        bundle["payload_digest"] = "b" * 64
        with pytest.raises(BundleRefused):
            _verify(bundle, keys)


class TestSubstitution:
    def test_an_unknown_key_id_is_refused(self, keypair):
        bundle, keys = _signed_bundle(keypair)
        with pytest.raises(BundleRefused, match="unknown key"):
            _verify(bundle, {"muid_sk_someone_else": next(iter(keys.values()))})

    def test_claims_naming_a_different_key_are_refused(self, keypair):
        bundle, keys = _signed_bundle(keypair)
        bundle["signature_claims"]["key_id"] = "muid_sk_elsewhere"
        with pytest.raises(BundleRefused, match="different key"):
            _verify(bundle, keys)

    @pytest.mark.parametrize("claimed", ["none", "HS256", "RS512", ""])
    def test_the_algorithm_is_compared_not_trusted(self, keypair, claimed):
        bundle, keys = _signed_bundle(keypair)
        bundle["signature_algorithm"] = claimed
        with pytest.raises(BundleRefused, match="unsupported signature algorithm"):
            _verify(bundle, keys)

    def test_an_unsupported_profile_is_refused(self, keypair):
        bundle, keys = _signed_bundle(keypair)
        bundle["signature_profile"] = "mudraid.bundle.signature/2"
        with pytest.raises(BundleRefused, match="unsupported signature profile"):
            _verify(bundle, keys)

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda b: b.update({"signature_value": "!!!not base64!!!"}),
            lambda b: b.update({"signature_claims": "not-an-object"}),
            lambda b: b.update({"signature_key_id": ""}),
            lambda b: b.update({"signature_value": base64.b64encode(b"short").decode()}),
        ],
    )
    def test_a_malformed_signature_is_refused_and_never_crashes(self, keypair, mutate):
        bundle, keys = _signed_bundle(keypair)
        mutate(bundle)
        with pytest.raises(BundleRefused):
            _verify(bundle, keys)


def test_the_three_implementations_pin_the_same_constants():
    """Python, Lua and the control plane must agree or a valid bundle is refused.

    The Lua side asserts these same two strings in its own suite; this is the
    Python half of that agreement. A drift would present as every signed bundle
    failing on one adapter only.
    """
    assert BUNDLE_SIGNATURE_PROFILE == "mudraid.bundle.signature/1"
    assert BUNDLE_SIGNATURE_ALGORITHM == "RS256"

    lua = (
        __import__("pathlib")
        .Path(__file__)
        .parents[3]
        .joinpath("kong/plugins/mudraid-enforce/bundle.lua")
    )
    if lua.is_file():
        source = lua.read_text(encoding="utf-8")
        assert f'_M.SIGNATURE_PROFILE = "{BUNDLE_SIGNATURE_PROFILE}"' in source
        assert f'_M.SIGNATURE_ALGORITHM = "{BUNDLE_SIGNATURE_ALGORITHM}"' in source
    else:  # pragma: no cover - the SDK is also published standalone
        json.dumps({})  # no-op; the constants above are still asserted


class TestCustomerAdaptersVerifyAsymmetricOnly:
    """A customer adapter never held the HMAC secret and never will.

    Handing it over would let the customer mint bundles for their own surface,
    which is the whole reason for moving to asymmetric signatures. So the
    shipped SDK configures no secret and verifies RS256 alone.
    """

    def _verify_without_secret(self, bundle, keys):
        return verify_bundle(
            bundle,
            secret=None,
            verification_keys=keys,
            expected_platform_id=PLATFORM,
            expected_environment=ENVIRONMENT,
        )

    def test_a_signed_bundle_verifies_with_no_hmac_secret_at_all(self, keypair):
        bundle, keys = _signed_bundle(keypair)
        assert self._verify_without_secret(bundle, keys).bundle_version == 1

    def test_a_corrupt_signature_still_denies_without_a_secret(self, keypair):
        """No secret is not a reason to be lenient about the one signature there is."""
        bundle, keys = _signed_bundle(keypair)
        bundle["signature_claims"]["platform_id"] = "platform-2"
        with pytest.raises(BundleRefused):
            self._verify_without_secret(bundle, keys)

    def test_neither_signature_is_refused_rather_than_accepted(self):
        """THE case that must never pass.

        No secret configured and no asymmetric signature means nothing verified
        this bundle at all. Accepting it would make every other check in this
        module decorative — the bundle is exactly the artifact an attacker would
        want to edit.
        """
        with pytest.raises(BundleRefused) as exc:
            self._verify_without_secret(_legacy_bundle(), keys=None)
        assert exc.value.code == "BUNDLE_SIGNING_SECRET_UNCONFIGURED"


class TestTheServerStopsEmittingHmac:
    """The end state of the migration, which nothing here reached.

    Every bundle in this module is built from ``_legacy_bundle``, so every one
    of them carries an HMAC ``signature`` — even the cases that verify "without
    a secret". That made the whole suite pass while two unconditional field
    checks stood in front of the signature logic:

        signature      must be hmac-sha256 hex
        signing_key_id must be a non-empty string

    So the moment the control plane stops emitting them — the point of retiring
    HMAC — every bundle is refused as BUNDLE_RESPONSE_INVALID before any
    signature is examined, and the shipped SDK that "verifies RS256 alone"
    could never accept one. The retirement had removed the requirement for the
    SECRET while leaving the requirement for the FIELD.

    These bundles carry no ``signature`` and no ``signing_key_id`` at all.
    """

    @staticmethod
    def _rs256_only(keypair, **claim_overrides) -> tuple[dict, dict]:
        bundle, keys = _signed_bundle(keypair, **claim_overrides)
        del bundle["signature"]
        del bundle["signing_key_id"]
        return bundle, keys

    def test_an_rs256_only_bundle_verifies_with_no_secret(self, keypair):
        bundle, keys = self._rs256_only(keypair)
        verified = verify_bundle(
            bundle,
            secret=None,
            verification_keys=keys,
            expected_platform_id=PLATFORM,
            expected_environment=ENVIRONMENT,
        )
        assert verified.bundle_version == 1

    def test_a_leftover_secret_does_not_refuse_an_rs256_only_bundle(self, keypair):
        """A deployment that still has the secret in its config keeps working.

        Before this, the configured secret was compared against a missing
        signature and every RS256 bundle was refused as an HMAC failure — a deny
        attributed to the wrong signature entirely, on a deployment whose
        configuration nobody had touched.
        """
        bundle, keys = self._rs256_only(keypair)
        assert _verify(bundle, keys).bundle_version == 1

    def test_an_rs256_only_bundle_whose_signature_is_wrong_still_denies(self, keypair):
        """Absence became allowed; invalidity must not have come with it."""
        bundle, keys = self._rs256_only(keypair)
        bundle["signature_claims"]["platform_id"] = "platform-2"
        with pytest.raises(BundleRefused) as exc:
            _verify(bundle, keys)
        assert exc.value.code == "BUNDLE_SIGNATURE_INVALID"

    def test_no_signature_of_any_kind_is_refused_even_with_a_secret(self):
        """A configured secret with nothing to check is not a check.

        This is the case the relaxation above could have opened, so it is
        asserted rather than assumed: dropping the HMAC field must not become a
        way to arrive at "nothing verified this" and be accepted.
        """
        bundle = _legacy_bundle()
        del bundle["signature"]
        del bundle["signing_key_id"]
        with pytest.raises(BundleRefused) as exc:
            _verify(bundle, keys=None)
        assert exc.value.code == "BUNDLE_SIGNING_SECRET_UNCONFIGURED"
        assert "nothing to check" in exc.value.detail

    def test_a_present_but_malformed_hmac_signature_is_still_refused(self, keypair):
        bundle, keys = _signed_bundle(keypair)
        bundle["signature"] = "not-hex"
        with pytest.raises(BundleRefused) as exc:
            _verify(bundle, keys)
        assert exc.value.code == "BUNDLE_RESPONSE_INVALID"

    def test_a_signature_with_no_key_id_is_refused(self, keypair):
        """Half-present is not absent. A signature nothing names the key for
        cannot be checked, and must not read as a bundle that carries none."""
        bundle, keys = _signed_bundle(keypair)
        del bundle["signing_key_id"]
        with pytest.raises(BundleRefused) as exc:
            _verify(bundle, keys)
        assert exc.value.code == "BUNDLE_RESPONSE_INVALID"
        assert "signing_key_id" in exc.value.detail
