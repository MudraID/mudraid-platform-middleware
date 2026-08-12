"""M5.5 — JwtValidator: signature + claim verification.

Each test signs a real JWT with the session-scoped RSA key from
conftest, mocks the JWKS endpoint with respx to return the matching
public JWK, and asserts the validator either decodes the claims or
raises ``MudraIDInvalidTokenError`` with the right ``reason``.

The conftest helpers (``sign_jwt``, ``baseline_claims``) keep each
test compact — most cases only override the one or two claims they
care about exercising.
"""

from __future__ import annotations

from typing import Any

import httpx
import jwt as pyjwt
import pytest
import respx
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

from mudraid_platform_middleware import MudraIDInvalidTokenError, MudraIDJwksError
from mudraid_platform_middleware._jwks_client import JwksClient
from mudraid_platform_middleware._jwt_validator import JwtValidator
from tests.conftest import baseline_claims, sign_jwt

JWKS_URL = "https://api.mudraid.test/.well-known/jwks.json"
AUDIENCE = "plt-test"


def _validator(
    audience: str = AUDIENCE,
    leeway: float = 30.0,
    issuer=None,
) -> JwtValidator:
    kwargs = {
        "jwks_client": JwksClient(JWKS_URL, cache_ttl_sec=3600),
        "expected_audience": audience,
        "leeway_sec": leeway,
    }
    if issuer is not None:
        kwargs["expected_issuer"] = issuer
    return JwtValidator(**kwargs)


# ---- construction --------------------------------------------------------


def test_constructor_rejects_empty_audience() -> None:
    """The middleware must always pass the platform_id as audience; an
    empty value would mean 'accept any audience', which is a security
    hole. Refuse to construct."""
    with pytest.raises(ValueError, match="expected_audience"):
        JwtValidator(
            jwks_client=JwksClient(JWKS_URL),
            expected_audience="",
        )


# ---- happy path -----------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_token_decodes_and_returns_claims(
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    token = sign_jwt(rsa_private_key, baseline_claims())

    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))

        claims = await _validator().validate(token)

    assert claims["aud"] == AUDIENCE
    assert claims["iss"] == "mudraid-identity"
    assert claims["scopes"] == ["items:read"]
    assert claims["sub"] == "agent-test-1"


@pytest.mark.asyncio
async def test_returned_claims_include_jti_and_kid_traceability(
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """The middleware will surface jti in audit logs; lock that it
    flows through the validator intact."""
    token = sign_jwt(rsa_private_key, baseline_claims())

    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))

        claims = await _validator().validate(token)

    assert claims["jti"] == "jti-test-1"


# ---- expiry --------------------------------------------------------------


@pytest.mark.asyncio
async def test_expired_token_is_rejected_with_reason_expired(
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    claims = baseline_claims(expires_in=-600)  # expired 10 minutes ago
    token = sign_jwt(rsa_private_key, claims)

    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))

        with pytest.raises(MudraIDInvalidTokenError) as exc:
            await _validator().validate(token)

    assert exc.value.reason == "expired"


@pytest.mark.asyncio
async def test_recently_expired_token_within_leeway_is_accepted(
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """Clock skew between MudraID and the platform host can make a
    just-issued token look briefly expired. The 30 s default leeway
    absorbs that. Without this leeway, a clock 5 s slow would
    reject every fresh token."""
    claims = baseline_claims(expires_in=-10)  # 10 s in the past
    token = sign_jwt(rsa_private_key, claims)

    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))

        await _validator(leeway=30.0).validate(token)  # no exception


# ---- not-before ----------------------------------------------------------


@pytest.mark.asyncio
async def test_token_with_future_nbf_is_rejected_with_reason_not_yet_valid(
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    claims = baseline_claims(not_before_offset=600)  # nbf 10 min in the future
    token = sign_jwt(rsa_private_key, claims)

    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))

        with pytest.raises(MudraIDInvalidTokenError) as exc:
            await _validator().validate(token)

    assert exc.value.reason == "not_yet_valid"


# ---- audience ------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrong_audience_is_rejected_with_reason_wrong_audience(
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """A JWT minted for platform A must NOT validate at platform B
    even if both trust MudraID. Locks the per-platform binding."""
    claims = baseline_claims(audience="plt-OTHER-PLATFORM")
    token = sign_jwt(rsa_private_key, claims)

    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))

        with pytest.raises(MudraIDInvalidTokenError) as exc:
            await _validator(audience="plt-MY-PLATFORM").validate(token)

    assert exc.value.reason == "wrong_audience"


# ---- issuer --------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrong_issuer_is_rejected_with_reason_wrong_issuer(
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """A token whose ``iss`` isn't MudraID is rejected even if it
    happens to be signed with the right key. Belt-and-braces: a
    future MudraID rebrand can update ``expected_issuer``; this
    test catches drift."""
    claims = baseline_claims(issuer="not-mudraid")
    token = sign_jwt(rsa_private_key, claims)

    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))

        with pytest.raises(MudraIDInvalidTokenError) as exc:
            await _validator().validate(token)

    assert exc.value.reason == "wrong_issuer"


# ---- issuer transition set (Phase 1a / charter §16) ----------------------


def test_constructor_rejects_empty_issuer_string() -> None:
    with pytest.raises(ValueError, match="expected_issuer"):
        JwtValidator(
            jwks_client=JwksClient(JWKS_URL),
            expected_audience=AUDIENCE,
            expected_issuer="",
        )


def test_constructor_rejects_empty_issuer_set() -> None:
    with pytest.raises(ValueError, match="expected_issuer"):
        JwtValidator(
            jwks_client=JwksClient(JWKS_URL),
            expected_audience=AUDIENCE,
            expected_issuer=[],
        )


@pytest.mark.asyncio
async def test_issuer_in_transition_set_is_accepted(
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """During a verifier-before-issuer rollout the middleware accepts both
    the legacy and the new issuer. A token carrying either must validate."""
    issuer_set = ("mudraid-identity", "https://identity.mudraid.example")
    for iss in issuer_set:
        token = sign_jwt(rsa_private_key, baseline_claims(issuer=iss))
        with respx.mock() as r:
            r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))
            claims = await _validator(issuer=issuer_set).validate(token)
        assert claims["iss"] == iss


@pytest.mark.asyncio
async def test_issuer_not_in_transition_set_is_rejected(
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    issuer_set = ("mudraid-identity", "https://identity.mudraid.example")
    token = sign_jwt(rsa_private_key, baseline_claims(issuer="https://evil.example"))

    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))
        with pytest.raises(MudraIDInvalidTokenError) as exc:
            await _validator(issuer=issuer_set).validate(token)

    assert exc.value.reason == "wrong_issuer"


@pytest.mark.asyncio
async def test_missing_issuer_with_transition_set_is_malformed(
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """`iss` presence is still required even when a set is configured; a
    token with no `iss` fails as malformed, not wrong_issuer."""
    claims = baseline_claims()
    del claims["iss"]
    token = sign_jwt(rsa_private_key, claims)

    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))
        with pytest.raises(MudraIDInvalidTokenError) as exc:
            await _validator(issuer=("a", "b")).validate(token)

    assert exc.value.reason == "malformed"


# ---- signature -----------------------------------------------------------


@pytest.mark.asyncio
async def test_tampered_signature_is_rejected(
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    token = sign_jwt(rsa_private_key, baseline_claims())
    # Flip a character in the MIDDLE of the signature segment. Flipping the
    # LAST base64url char is unreliable: the final char carries padding bits,
    # so some flips decode to the same signature bytes and the token still
    # verifies (this caused intermittent "DID NOT RAISE" failures). A middle
    # char's six bits are all significant, so the tamper always changes the
    # signature.
    header, body, sig = token.split(".")
    mid = len(sig) // 2
    bad_token = f"{header}.{body}.{sig[:mid] + ('A' if sig[mid] != 'A' else 'B') + sig[mid + 1:]}"

    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))

        with pytest.raises(MudraIDInvalidTokenError) as exc:
            await _validator().validate(bad_token)

    # PyJWT may report this as either invalid_signature OR malformed
    # depending on whether the segment still base64-decodes cleanly.
    # Either is acceptable — what matters is we don't accept it.
    assert exc.value.reason in {"invalid_signature", "malformed"}


@pytest.mark.asyncio
async def test_token_signed_with_a_different_key_is_rejected(
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """Signed with the right algorithm but the wrong key. PyJWT
    catches the signature mismatch."""
    from cryptography.hazmat.primitives.asymmetric import rsa as _rsa

    other_key = _rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = sign_jwt(other_key, baseline_claims())

    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))

        with pytest.raises(MudraIDInvalidTokenError) as exc:
            await _validator().validate(token)

    assert exc.value.reason == "invalid_signature"


# ---- malformed token -----------------------------------------------------


@pytest.mark.asyncio
async def test_unparseable_token_is_rejected_with_reason_malformed() -> None:
    with pytest.raises(MudraIDInvalidTokenError) as exc:
        await _validator().validate("not.a.jwt")
    assert exc.value.reason == "malformed"


@pytest.mark.asyncio
async def test_token_without_kid_header_is_rejected_with_reason_malformed(
    rsa_private_key: RSAPrivateKey,
) -> None:
    """No kid → we can't look up a key → fail loud. Falling back to
    'just try the first key' would be a security hole during
    rotation when both old and new keys are valid."""
    # PyJWT lets us omit headers; this builds a JWT with no kid.
    token = pyjwt.encode(
        baseline_claims(),
        key=rsa_private_key,
        algorithm="RS256",
        # No headers={"kid": ...} — default header is empty.
    )

    with pytest.raises(MudraIDInvalidTokenError) as exc:
        await _validator().validate(token)
    assert exc.value.reason == "malformed"


# ---- missing required claims ---------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["exp", "iat", "nbf", "aud", "iss"])
async def test_missing_required_claim_is_rejected_with_reason_malformed(
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
    missing: str,
) -> None:
    """Every claim in the locked JWT shape (M2.11) is required.
    Dropping any of them must fail loudly — defense in depth against
    a future MudraID change that accidentally stops emitting one."""
    claims = baseline_claims()
    del claims[missing]
    token = sign_jwt(rsa_private_key, claims)

    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))

        with pytest.raises(MudraIDInvalidTokenError) as exc:
            await _validator().validate(token)

    assert exc.value.reason == "malformed"


# ---- JWKS errors propagate ----------------------------------------------


@pytest.mark.asyncio
async def test_unknown_kid_propagates_as_jwks_error(
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """When the JWT references a kid not in the JWKS (even after
    refresh), JwksClient raises MudraIDJwksError. The validator
    must NOT catch and re-raise as MudraIDInvalidTokenError — the
    middleware handles each error type differently (an unknown
    kid is a 500-ish "we couldn't even check", not a 401-ish
    "your token was bad")."""
    token = sign_jwt(rsa_private_key, baseline_claims(), kid="rogue-kid")

    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))

        with pytest.raises(MudraIDJwksError):
            await _validator().validate(token)


@pytest.mark.asyncio
async def test_jwks_fetch_failure_propagates(
    rsa_private_key: RSAPrivateKey,
) -> None:
    token = sign_jwt(rsa_private_key, baseline_claims())

    with respx.mock() as r:
        r.get(JWKS_URL).mock(side_effect=httpx.ConnectError("boom"))

        with pytest.raises(MudraIDJwksError):
            await _validator().validate(token)


# ---- algorithm pinning ---------------------------------------------------


@pytest.mark.asyncio
async def test_validator_only_accepts_rs256(
    rsa_public_jwk: dict[str, Any],
) -> None:
    """The classic JWT alg-confusion family of attacks: an attacker
    presents an HS256 token where the validator expected RS256. The
    validator's ``algorithms=["RS256"]`` pin must reject this even
    if the underlying signature would otherwise verify.

    We sign with a plain string secret — the point is that the
    incoming algorithm is HS256, full stop. The validator should
    refuse to even try verifying a non-RS256 token.

    (PyJWT itself blocks the more aggressive variant of this attack
    — passing the RSA public key as the HMAC secret — by refusing
    to encode it, which is good defense-in-depth at their layer.
    That doesn't change our requirement to pin algorithms at decode.)
    """
    hs_token = pyjwt.encode(
        baseline_claims(),
        key="totally-not-the-real-key",
        algorithm="HS256",
        headers={"kid": "test-kid-1"},
    )

    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))

        with pytest.raises(MudraIDInvalidTokenError) as exc:
            await _validator().validate(hs_token)

    # PyJWT reports this as InvalidAlgorithmError, which our mapper
    # routes through the catch-all → reason="malformed". Either
    # malformed or invalid_signature is acceptable; the critical
    # thing is the token is NOT accepted.
    assert exc.value.reason in {"malformed", "invalid_signature"}
