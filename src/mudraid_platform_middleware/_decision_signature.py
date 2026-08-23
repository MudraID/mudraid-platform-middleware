"""Verify the asymmetric ``/decide`` response signature (AUDIT-009 A9-02).

The decision response used to be authenticated by the transport plus the
decision-id / freshness binding in ``decide_client._read_decision`` — real
controls, and not a signature: a party able to subvert the TLS path could forge
an ALLOW. The authority now signs every decision response the same way it signs
enforcement bundles: RS256 over the canonical JSON of a CLAIMS object that
BINDS the decision — protocol profile + version, decision id, outcome and
stable reason code, tenant / environment / platform surface / agent, action and
canonical resource, bundle and policy versions, decided-at and a bounded
validity window.

VERIFY WHEN PRESENT, REFUSE WHEN PRESENT-AND-WRONG. A response carrying no
``signature`` field is read exactly as before — the authority rolls out signing
by activation, and an adapter that refused every unsigned response would turn
that activation into a flag-day. A response CARRYING a signature must verify
completely or the whole response is refused: absence and invalidity are
different facts, and collapsing them would let an attacker downgrade by
corrupting one field. (Stripping the signature entirely remains possible for a
party who can already rewrite the TLS stream — closing that requires the
authority-side activation this verifier is the client half of; nothing in this
package may describe unsigned responses as authenticated beyond the transport.)

This mirrors ``_bundle._verify_asymmetric`` deliberately — same pinned-constant
posture, same claims-carried-verbatim rule, same canonical bytes — and conforms
to the shared corpus in ``kong/tests/fixtures/adapter-conformance.json``, which
both this verifier and the Kong plugin's ``decide.lua`` are tested against.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from mudraid_platform_middleware._canonical import CanonicalizationError, canonical_json_bytes

__all__ = [
    "DECISION_SIGNATURE_ALGORITHM",
    "DECISION_SIGNATURE_PROFILE",
    "DecisionBindings",
    "DecisionSignatureRefused",
    "verify_decision_signature",
]

#: Pinned, not negotiated. Mirrors ``decision_signature.py`` on the authority
#: and ``decide.lua`` in Kong — three implementations, one set of constants
#: they are each checked against.
DECISION_SIGNATURE_PROFILE = "mudraid.decision.signature/1"
DECISION_SIGNATURE_ALGORITHM = "RS256"

#: Refuse absurd clock skew rather than silently accepting it. Matches the
#: authority's bundle/decision verifier tolerance.
_MAX_CLOCK_SKEW = timedelta(minutes=5)


class DecisionSignatureRefused(Exception):
    """A present decision signature could not be trusted. Deny-closed.

    ``code`` is a stable machine reason for logs; the caller answers every
    refusal identically (the response is not acted on), so nothing on the wire
    distinguishes "wrong key" from "wrong surface" for an attacker probing.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class DecisionBindings:
    """What THIS adapter knows the decision must be bound to.

    Each ``None`` skips its comparison — the adapter holds no expectation to
    enforce — while a present value that the signed claims contradict is a
    refusal. The values come from the VERIFIED bundle surface and the mapped
    action, never from the response being checked.
    """

    platform_id: str | None = None
    environment: str | None = None
    resource: str | None = None
    action_key: str | None = None
    bundle_version: int | None = None


def _refuse(detail: str) -> DecisionSignatureRefused:
    return DecisionSignatureRefused("DECIDE_RESPONSE_SIGNATURE_INVALID", detail)


def verify_decision_signature(
    decoded: Mapping[str, Any],
    *,
    verification_keys: Mapping[str, str] | None,
    expected: DecisionBindings | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify ``decoded['signature']``, or raise. Returns the verified claims.

    The caller has already established the envelope-level binding (supported
    contract, decision id equal to the request's, freshness); this function
    establishes that MudraID SIGNED that envelope, for THIS surface and action.
    Order mirrors the authority's reference verifier: structural checks, the
    asymmetric operation, and only then are the claims compared against the
    envelope in hand and the adapter's own expectations.
    """
    signature = decoded.get("signature")
    if not isinstance(signature, dict):
        raise _refuse("signature is not an object")
    if not verification_keys:
        # A signed response with no keys to check it against is unverifiable,
        # which is a refusal — never "nothing to check".
        raise DecisionSignatureRefused(
            "DECIDE_RESPONSE_SIGNATURE_KEYS_UNAVAILABLE",
            "the response carries a signature but no decision verification "
            "keys are available",
        )

    if signature.get("profile") != DECISION_SIGNATURE_PROFILE:
        raise _refuse("unsupported signature profile")
    # Compared against the pinned constant, never read out of the signature and
    # used — the ``alg: none`` lesson.
    if signature.get("algorithm") != DECISION_SIGNATURE_ALGORITHM:
        raise _refuse("unsupported signature algorithm")

    key_id = signature.get("key_id")
    if not isinstance(key_id, str) or not key_id:
        raise _refuse("signature names no key")
    public_pem = verification_keys.get(key_id)
    if not public_pem:
        # Unknown or retired key — the rotation refusal. A key no longer
        # published is a key whose signatures are no longer trusted.
        raise _refuse("signature names an unknown key")

    claims = signature.get("claims")
    if not isinstance(claims, dict):
        raise _refuse("signature carries no claims")
    if claims.get("key_id") != key_id:
        raise _refuse("claims name a different key than the signature")
    if claims.get("profile") != DECISION_SIGNATURE_PROFILE:
        raise _refuse("claims name a different signature profile")
    if claims.get("algorithm") != DECISION_SIGNATURE_ALGORITHM:
        raise _refuse("claims name a different signature algorithm")

    encoded = signature.get("signature")
    if not isinstance(encoded, str) or not encoded:
        raise _refuse("signature is absent")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise _refuse("signature is not valid base64") from exc

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
    except ImportError as exc:  # pragma: no cover - cryptography is a hard dep
        # A missing verifier must never read as a valid signature.
        raise _refuse("asymmetric verification is unavailable") from exc

    try:
        key = serialization.load_pem_public_key(public_pem.encode("utf-8"))
    except (ValueError, TypeError) as exc:
        raise _refuse("verification key could not be loaded") from exc
    if not isinstance(key, rsa.RSAPublicKey):
        raise _refuse("verification key is not RSA")

    try:
        message = canonical_json_bytes(claims)
    except CanonicalizationError as exc:
        raise _refuse("claims could not be canonicalized") from exc
    try:
        key.verify(raw, message, padding.PKCS1v15(), hashes.SHA256())
    except InvalidSignature as exc:
        raise _refuse("signature does not verify") from exc

    # ── Only now are the claims trustworthy enough to be compared ────────────
    #
    # Claims ↔ envelope: the unsigned envelope copy of every signed field must
    # agree byte-for-byte, so an edit to the envelope alone is a refusal.
    for field in ("decision_id", "outcome", "decision", "decided_at", "deadline_at"):
        if claims.get(field) != decoded.get(field):
            raise _refuse(f"signature does not cover this envelope ({field})")
    reason = decoded.get("reason")
    envelope_reason_primary = reason.get("primary") if isinstance(reason, dict) else None
    if claims.get("reason_primary") != envelope_reason_primary:
        raise _refuse("signature does not cover this envelope (reason)")

    # Claims ↔ this adapter's own surface and request.
    if expected is not None:
        if expected.platform_id is not None and claims.get("platform_id") != expected.platform_id:
            raise _refuse("decision is bound to another platform")
        if expected.environment is not None and claims.get("environment") != expected.environment:
            raise _refuse("decision is bound to another environment")
        if expected.resource is not None and claims.get("resource") != expected.resource:
            raise _refuse("decision is bound to another resource")
        if expected.action_key is not None and claims.get("action_key") != expected.action_key:
            raise _refuse("decision is bound to another action")
        if (
            expected.bundle_version is not None
            and claims.get("bundle_version") != expected.bundle_version
        ):
            raise _refuse("decision is bound to another bundle version")

    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    not_before = _instant(claims.get("not_before"))
    expires_at = _instant(claims.get("expires_at"))
    if not_before is None or expires_at is None:
        raise _refuse("signature carries no usable validity window")
    if moment + _MAX_CLOCK_SKEW < not_before:
        raise _refuse("decision signature is not yet valid")
    if moment - _MAX_CLOCK_SKEW > expires_at:
        raise _refuse("decision signature has expired")

    return dict(claims)


def _instant(value: Any) -> datetime | None:
    """Parse an ISO-8601 instant, or ``None``. A naive timestamp is refused."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)
