"""Signed-bundle verification — nothing served is trusted before this passes.

The adapter channel serves a bundle describing which surface is protected and
which tool names map to which canonical action. That description decides whether
a request is enforced at all, so it is verified BEFORE any of it is trusted:

  1. shape and type checks on the served envelope;
  2. canonical re-serialization of the payload (the same bytes the control plane
     hashed and signed);
  3. ``payload_digest == SHA-256(canonical(payload))``;
  4. ``signature == HMAC-SHA256(signing secret, canonical(payload))``, compared
     in constant time;
  5. contract checks — schema version, surface binding, matcher kind (exact
     only, never fuzzy), the deny-closed evaluation contract;
  6. monotonic version rules against the currently active bundle.

ANY failure refuses the bundle. The caller keeps the last valid bundle active,
and if none exists the protected surface fails CLOSED. An unsigned or tampered
bundle is never activated — there is no "best effort" acceptance path, because a
bundle is exactly the artifact an attacker would want to edit.

This mirrors ``kong/plugins/mudraid-enforce/bundle.lua`` check for check. The two
adapters must refuse the same bundles for the same reasons, or "portable
enforcement" is a claim with a gap in it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Any

from mudraid_platform_middleware._canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    sha256_hex,
)

__all__ = ["BundleRefused", "VerifiedBundle", "verify_bundle"]

#: Bundle schema versions this adapter implements. An unknown version is
#: refused, never "best effort" parsed. Tracks ``BUNDLE_SCHEMA_VERSION`` in
#: ``services/platform-integration-service/app/application/bundle_compiler.py``.
SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0"})

#: The only evaluation contract this adapter implements: live, decide-required,
#: deny-closed, forward-once. A bundle declaring anything else (a future
#: snapshot mode, say) is refused until this adapter ships verified support for
#: it — never silently downgraded to something weaker.
_REQUIRED_EVALUATION: dict[str, str] = {
    "mode": "live",
    "on_timeout": "deny",
    "on_error": "deny",
    "on_unmapped_action": "deny",
    "on_stale_bundle": "deny",
    "forward": "once",
}

#: Surface fields the adapter forwards verbatim on ``/decide``. A grant is bound
#: to the exact environment and canonical resource, so a bundle omitting them
#: describes a surface on which every protected request deny-closes. That is
#: refused at ACTIVATION rather than discovered per request.
_REQUIRED_SURFACE_FIELDS = ("platform_id", "environment", "canonical_resource_uri")

#: Bounded action-name length, matching the matcher's ``MAX_TOOL_NAME_LEN``.
MAX_TOOL_NAME_LEN = 512

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


class BundleRefused(Exception):
    """A served bundle failed verification and must not be activated.

    ``code`` is the stable typed reason (``BUNDLE_SIGNATURE_INVALID`` and
    friends), matching the Lua plugin's vocabulary so operators reading either
    adapter's logs see the same word for the same defect.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class VerifiedBundle:
    """A bundle that passed every check, and only then."""

    bundle_version: int
    payload_digest: str
    content_digest: str
    signing_key_id: str
    surface: dict[str, Any]
    actions: dict[str, dict[str, Any]]
    strip_prefixes: tuple[str, ...]
    no_change: bool = False

    def resolve(self, tool_name: str) -> dict[str, Any] | None:
        """The EXACT canonical action for ``tool_name``, or ``None``.

        Exact and case-sensitive — never fuzzy, prefix or regex. An unmapped
        name resolves to ``None``, which the control loop turns into a deny.
        """
        if not isinstance(tool_name, str) or not tool_name:
            return None
        if len(tool_name.encode("utf-8")) > MAX_TOOL_NAME_LEN:
            return None
        return self.actions.get(tool_name)


def _is_bound_string(value: Any) -> bool:
    """A usable, non-empty, non-whitespace string.

    JSON ``null`` decodes to ``None``, and the control plane still emits it for a
    surface row whose ``canonical_resource_uri`` was never backfilled — so the
    check is an explicit type test, not a truthiness test.
    """
    return isinstance(value, str) and value.strip() != ""


def _build_action_index(actions: Any) -> dict[str, dict[str, Any]]:
    """Index the action corpus by exact tool name, refusing ambiguity.

    A duplicate ``tool_name`` is rejected at BUILD time, so runtime can never
    resolve an overlap by iteration order.
    """
    if not isinstance(actions, list) or not actions:
        raise BundleRefused("BUNDLE_CONTENT_INVALID", "matcher.actions is empty")
    index: dict[str, dict[str, Any]] = {}
    for action in actions:
        if not isinstance(action, dict):
            raise BundleRefused("BUNDLE_ACTION_TOOL_NAME_INVALID", "action is not an object")
        name = action.get("tool_name")
        if not isinstance(name, str) or name == "" or len(name.encode("utf-8")) > MAX_TOOL_NAME_LEN:
            raise BundleRefused("BUNDLE_ACTION_TOOL_NAME_INVALID", "tool_name is not usable")
        if name in index:
            raise BundleRefused("BUNDLE_MATCHER_AMBIGUOUS", f"duplicate tool_name {name!r}")
        index[name] = action
    return index


def _check_surface(surface: Any) -> None:
    if not isinstance(surface, dict):
        raise BundleRefused("BUNDLE_SURFACE_UNBOUND", "surface missing")
    for key in _REQUIRED_SURFACE_FIELDS:
        if not _is_bound_string(surface.get(key)):
            raise BundleRefused("BUNDLE_SURFACE_UNBOUND", f"surface.{key} is not a bound value")


def _check_evaluation(evaluation: Any) -> None:
    if not isinstance(evaluation, dict):
        raise BundleRefused("BUNDLE_EVALUATION_UNSUPPORTED", "evaluation missing")
    for key, expected in _REQUIRED_EVALUATION.items():
        if evaluation.get(key) != expected:
            raise BundleRefused(
                "BUNDLE_EVALUATION_UNSUPPORTED", f"evaluation.{key} must be {expected!r}"
            )
    # `is not True` rather than a truthiness test: 1 is not the contract.
    if evaluation.get("decide_required") is not True:
        raise BundleRefused(
            "BUNDLE_EVALUATION_UNSUPPORTED", "evaluation.decide_required must be true"
        )
    if evaluation.get("retry_forwarded_request") is not False:
        raise BundleRefused(
            "BUNDLE_EVALUATION_UNSUPPORTED", "evaluation.retry_forwarded_request must be false"
        )


#: Pinned, not negotiated. Mirrors ``bundle_signature.py`` on the control plane
#: and ``bundle.lua`` in Kong — three implementations, one set of constants they
#: are each checked against.
BUNDLE_SIGNATURE_PROFILE = "mudraid.bundle.signature/1"
BUNDLE_SIGNATURE_ALGORITHM = "RS256"


def _verify_asymmetric(
    fetched: dict,
    *,
    verification_keys: dict[str, str] | None,
    expected_platform_id: str | None,
    expected_environment: str | None,
) -> None:
    """Verify the RS256 bundle signature, or raise ``BundleRefused``.

    The claims are verified as the signer serialized them and only THEN compared
    against the bundle in hand. A signature that verifies proves MudraID
    produced those claims; it says nothing about whether they describe this
    bundle, which is what the digest and binding comparisons establish.
    """
    if not verification_keys:
        raise BundleRefused(
            "BUNDLE_VERIFICATION_KEYS_UNAVAILABLE",
            "the bundle carries a signature but no verification keys are available",
        )
    if fetched.get("signature_profile") != BUNDLE_SIGNATURE_PROFILE:
        raise BundleRefused("BUNDLE_SIGNATURE_INVALID", "unsupported signature profile")
    # Compared against the pinned constant, never read out of the signature and
    # used — the ``alg: none`` lesson.
    if fetched.get("signature_algorithm") != BUNDLE_SIGNATURE_ALGORITHM:
        raise BundleRefused("BUNDLE_SIGNATURE_INVALID", "unsupported signature algorithm")

    key_id = fetched.get("signature_key_id")
    if not isinstance(key_id, str) or not key_id:
        raise BundleRefused("BUNDLE_SIGNATURE_INVALID", "signature names no key")
    public_pem = verification_keys.get(key_id)
    if not public_pem:
        # Unknown or retired. A key that is no longer published is a key whose
        # signatures are no longer trusted.
        raise BundleRefused("BUNDLE_SIGNATURE_INVALID", "signature names an unknown key")

    claims = fetched.get("signature_claims")
    if not isinstance(claims, dict):
        raise BundleRefused("BUNDLE_SIGNATURE_INVALID", "signature carries no claims")
    if claims.get("key_id") != key_id:
        raise BundleRefused(
            "BUNDLE_SIGNATURE_INVALID", "claims name a different key than the signature"
        )

    encoded = fetched.get("signature_value")
    if not isinstance(encoded, str) or not encoded:
        raise BundleRefused("BUNDLE_SIGNATURE_INVALID", "signature is absent")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise BundleRefused("BUNDLE_SIGNATURE_INVALID", "signature is not valid base64") from exc

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
    except ImportError as exc:  # pragma: no cover - cryptography is a hard dep
        # A missing verifier must never read as a valid signature.
        raise BundleRefused(
            "BUNDLE_SIGNATURE_INVALID", "asymmetric verification is unavailable"
        ) from exc

    try:
        key = serialization.load_pem_public_key(public_pem.encode("utf-8"))
    except (ValueError, TypeError) as exc:
        raise BundleRefused(
            "BUNDLE_SIGNATURE_INVALID", "verification key could not be loaded"
        ) from exc
    if not isinstance(key, rsa.RSAPublicKey):
        raise BundleRefused("BUNDLE_SIGNATURE_INVALID", "verification key is not RSA")

    try:
        key.verify(raw, canonical_json_bytes(claims), padding.PKCS1v15(), hashes.SHA256())
    except InvalidSignature as exc:
        raise BundleRefused("BUNDLE_SIGNATURE_INVALID", "signature does not verify") from exc

    # Only now are the claims trustworthy enough to compare.
    if claims.get("payload_digest") != fetched.get("payload_digest"):
        raise BundleRefused("BUNDLE_SIGNATURE_INVALID", "signature does not cover this payload")
    if claims.get("bundle_version") != fetched.get("bundle_version"):
        raise BundleRefused(
            "BUNDLE_SIGNATURE_INVALID", "signature covers a different bundle version"
        )
    if expected_platform_id and claims.get("platform_id") != expected_platform_id:
        raise BundleRefused("BUNDLE_SIGNATURE_INVALID", "bundle is bound to another platform")
    if expected_environment and claims.get("environment") != expected_environment:
        raise BundleRefused("BUNDLE_SIGNATURE_INVALID", "bundle is bound to another environment")


def verify_bundle(
    fetched: Any,
    *,
    secret: str | None,
    active: VerifiedBundle | None = None,
    verification_keys: dict[str, str] | None = None,
    expected_platform_id: str | None = None,
    expected_environment: str | None = None,
) -> VerifiedBundle:
    """Verify one fetched bundle, or refuse it.

    Args:
        fetched: the decoded ``GET /api/v1/internal/enforcement/bundle`` response.
        secret: the adapter's bundle signing secret. Absent or empty means
            nothing can be verified, which is refused — never "trust unsigned".
        active: the currently active bundle, for the monotonic version rules.

    Raises:
        BundleRefused: on any failure, with a typed ``code``.
    """
    if not isinstance(fetched, dict):
        raise BundleRefused("BUNDLE_RESPONSE_INVALID", "response is not an object")

    version = fetched.get("bundle_version")
    # `bool` is a subclass of `int`, so an explicit exclusion is required or
    # `True` would be accepted as version 1.
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise BundleRefused("BUNDLE_RESPONSE_INVALID", "bundle_version is not a positive integer")
    if fetched.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise BundleRefused(
            "BUNDLE_SCHEMA_UNSUPPORTED",
            f"schema_version {fetched.get('schema_version')!r} is not supported",
        )
    payload = fetched.get("payload")
    if not isinstance(payload, dict):
        raise BundleRefused("BUNDLE_RESPONSE_INVALID", "payload missing")
    payload_digest = fetched.get("payload_digest")
    if not isinstance(payload_digest, str) or not _HEX64.match(payload_digest):
        raise BundleRefused("BUNDLE_RESPONSE_INVALID", "payload_digest is not sha256 hex")
    # The HMAC fields are required only WHEN PRESENT, which is not a weakening.
    #
    # These two were unconditional, and that made the HMAC retirement incomplete
    # in a way no signature test would have found: the block below stopped
    # requiring the SECRET, while this stayed requiring the FIELD. The moment
    # the control plane stops emitting `signature`, every bundle — including one
    # carrying a perfectly good RS256 signature — is refused here as
    # BUNDLE_RESPONSE_INVALID, before any signature logic runs. A malformed
    # envelope and an asymmetrically-signed bundle would report as the same
    # thing, and a shipped SDK that "configures no secret and verifies the RS256
    # signature only" could never actually accept one.
    #
    # Present-but-malformed is still a refusal, and that is the half that must
    # not move: absence and invalidity are different facts (see below).
    signature = fetched.get("signature")
    has_hmac_signature = signature is not None
    signing_key_id = fetched.get("signing_key_id")
    if has_hmac_signature:
        if not isinstance(signature, str) or not _HEX64.match(signature):
            raise BundleRefused("BUNDLE_RESPONSE_INVALID", "signature is not hmac-sha256 hex")
        if not isinstance(signing_key_id, str) or signing_key_id == "":
            raise BundleRefused(
                "BUNDLE_RESPONSE_INVALID",
                "signature is present but signing_key_id is missing; nothing names "
                "the key it was produced with",
            )
    else:
        # With no HMAC signature there is no HMAC key, so the key that
        # authenticated this bundle is the asymmetric one. Reporting the absent
        # HMAC field would leave every downstream log and acknowledgement
        # unable to say which key vouched for the bundle it applied.
        signing_key_id = fetched.get("signature_key_id")

    # The SIGNED payload is authoritative; the unsigned wrapper must agree with
    # it, or the served envelope is describing a different bundle than the one
    # the signature covers.
    if payload.get("schema_version") != fetched.get("schema_version"):
        raise BundleRefused(
            "BUNDLE_ENVELOPE_MISMATCH", "payload.schema_version disagrees with response"
        )
    if payload.get("bundle_version") != version:
        raise BundleRefused(
            "BUNDLE_ENVELOPE_MISMATCH", "payload.bundle_version disagrees with response"
        )

    try:
        canon = canonical_json_bytes(payload)
    except CanonicalizationError as exc:
        raise BundleRefused("BUNDLE_CANONICALIZATION_FAILED", str(exc)) from exc

    # Digest BEFORE signature, deliberately: a digest mismatch is tampering or
    # corruption regardless of key material, and deserves its own typed fact
    # rather than being reported as a signature failure.
    if hashlib.sha256(canon).hexdigest() != payload_digest:
        raise BundleRefused(
            "BUNDLE_DIGEST_MISMATCH", "payload_digest does not match canonical payload"
        )

    # ── Which signature must this verifier check? ────────────────────────────
    #
    # A CUSTOMER adapter never held the HMAC secret and never will — handing it
    # over would let the customer mint bundles for their own surface, which is
    # the whole reason for moving to asymmetric signatures. So the shipped SDK
    # configures no secret and verifies the RS256 signature only.
    #
    # MudraID's own deployed gateway is the other case: it holds the secret and
    # its bundles may predate asymmetric signing, so it verifies the HMAC and
    # treats the asymmetric signature as an additional check when present.
    #
    # What must NEVER happen is BOTH being absent. That is a bundle nothing
    # verified, and accepting it would make every check above decorative.
    has_secret = isinstance(secret, str) and secret != ""
    has_asymmetric = fetched.get("signature_value") is not None

    if not has_asymmetric and not (has_secret and has_hmac_signature):
        # Nothing here can be authenticated. The detail distinguishes the ways
        # to arrive, because "configure a secret" is useless advice to someone
        # who already has one and received a bundle carrying no signature.
        if has_secret:
            detail = (
                "the bundle carries neither an asymmetric signature nor an HMAC "
                "signature, so the configured signing secret had nothing to check; "
                "unsigned trust is refused"
            )
        elif has_hmac_signature:
            detail = (
                "the bundle carries no asymmetric signature and no signing secret is "
                "configured to check its HMAC; unsigned trust is refused"
            )
        else:
            detail = (
                "the bundle carries no signature of any kind and no signing secret is "
                "configured; unsigned trust is refused"
            )
        raise BundleRefused("BUNDLE_SIGNING_SECRET_UNCONFIGURED", detail)

    # BOTH halves are required to CHECK an HMAC — a secret to check with, and a
    # signature to check. A configured secret alone is not a check.
    #
    # The `has_hmac_signature` half is what lets a deployment that still has the
    # secret configured keep working once the control plane stops emitting HMAC.
    # Without it, compare_digest against a None signature raises or fails, and
    # every RS256-signed bundle is refused as an HMAC failure — a deny
    # attributed to the wrong signature entirely.
    #
    # Not a downgrade path: stripping the HMAC does not help an attacker,
    # because whatever remains must still verify on its own, and if nothing
    # does, the refusal above already fired.
    if has_secret and has_hmac_signature:
        expected = hmac.new(secret.encode("utf-8"), canon, "sha256").hexdigest()
        # compare_digest, not `!=`: this is an authentication tag, and a plain
        # comparison short-circuits on the first differing byte.
        if not hmac.compare_digest(expected, signature):
            raise BundleRefused("BUNDLE_SIGNATURE_INVALID", "HMAC signature verification failed")

    # ── The asymmetric signature ─────────────────────────────────────────────
    #
    # ABSENCE AND INVALIDITY ARE DIFFERENT FACTS. A bundle published before
    # asymmetric signing existed carries none, and must still be accepted on its
    # HMAC — that is what makes the migration window a window. A bundle that
    # CARRIES one which does not verify is never treated as legacy: if
    # invalidity fell back to the HMAC, anyone holding the shared secret could
    # corrupt a single field and downgrade every bundle to the weaker check.
    #
    # Present-but-invalid always refuses, and the reason is typed separately so
    # an operator can tell a downgrade attempt from an unsigned legacy bundle.
    if fetched.get("signature_value") is not None:
        _verify_asymmetric(
            fetched,
            verification_keys=verification_keys,
            expected_platform_id=expected_platform_id,
            expected_environment=expected_environment,
        )

    content = payload.get("content")
    if not isinstance(content, dict):
        raise BundleRefused("BUNDLE_CONTENT_INVALID", "payload.content missing")
    _check_surface(content.get("surface"))
    _check_evaluation(content.get("evaluation"))

    matcher = content.get("matcher")
    if not isinstance(matcher, dict) or matcher.get("kind") != "mcp_tool_exact":
        # An unknown matcher kind must never degrade to fuzzy or partial
        # matching — that would silently widen what counts as a mapped action.
        kind = matcher.get("kind") if isinstance(matcher, dict) else None
        raise BundleRefused("BUNDLE_MATCHER_UNSUPPORTED", f"matcher.kind {kind!r} is not supported")
    index = _build_action_index(matcher.get("actions"))

    # Monotonic version rules: a LOWER version is refused (a control-plane
    # rollback republishes a HIGHER number, never a lower one, so a lower number
    # is a rollback attack or a stale mirror); the same version with different
    # bytes is a conflict; the same version with the same digest is a no-op.
    no_change = False
    if active is not None:
        if version < active.bundle_version:
            raise BundleRefused(
                "BUNDLE_VERSION_REGRESSION",
                f"served version {version} < active version {active.bundle_version}",
            )
        if version == active.bundle_version:
            if payload_digest != active.payload_digest:
                raise BundleRefused(
                    "BUNDLE_VERSION_CONFLICT", "same bundle_version with a different payload_digest"
                )
            no_change = True

    try:
        content_digest = sha256_hex(content)
    except CanonicalizationError as exc:  # pragma: no cover - payload already canonicalized
        raise BundleRefused("BUNDLE_CANONICALIZATION_FAILED", str(exc)) from exc

    trusted = content.get("trusted_context")
    prefixes: tuple[str, ...] = ()
    if isinstance(trusted, dict):
        raw = trusted.get("strip_request_header_prefixes")
        if isinstance(raw, list):
            prefixes = tuple(p for p in raw if isinstance(p, str) and p)

    return VerifiedBundle(
        bundle_version=version,
        payload_digest=payload_digest,
        content_digest=content_digest,
        signing_key_id=signing_key_id,
        surface=dict(content["surface"]),
        actions=index,
        strip_prefixes=prefixes,
        no_change=no_change,
    )
