"""Shared fixtures + helpers for middleware tests.

The signing helpers are session-scoped because RSA key generation is
~1–2 s and we don't want to pay that cost per test.
"""

from __future__ import annotations

import json
import time
from typing import Any

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from jwt.algorithms import RSAAlgorithm


@pytest.fixture(scope="session")
def rsa_private_key() -> RSAPrivateKey:
    """One RSA key pair for the entire test session."""
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="session")
def rsa_public_jwk(rsa_private_key: RSAPrivateKey) -> dict[str, Any]:
    """JWK form of the test session's public key, ready to drop into
    a mocked JWKS response. The ``kid`` field is set to a stable
    sentinel so tests can build matching JWT headers."""
    pem_jwk = json.loads(RSAAlgorithm.to_jwk(rsa_private_key.public_key()))
    pem_jwk["kid"] = "test-kid-1"
    pem_jwk["use"] = "sig"
    pem_jwk["alg"] = "RS256"
    return pem_jwk


def sign_jwt(
    private_key: RSAPrivateKey,
    claims: dict[str, Any],
    *,
    kid: str = "test-kid-1",
) -> str:
    """Helper to build a signed JWT from a claims dict.

    Tests call this with whatever claims they're exercising. Anything
    not in the dict is omitted from the JWT — so a test that drops
    ``exp`` can verify the validator rejects an expiry-less token.
    """
    return pyjwt.encode(
        claims,
        key=private_key,
        algorithm="RS256",
        headers={"kid": kid},
    )


def baseline_claims(
    *,
    audience: str = "plt-test",
    issuer: str = "mudraid-identity",
    expires_in: int = 900,
    not_before_offset: int = 0,
    scopes: list[str] | None = None,
) -> dict[str, Any]:
    """A claim set that should pass every validator gate."""
    now = int(time.time())
    return {
        "iss": issuer,
        "sub": "agent-test-1",
        "aud": audience,
        "scopes": scopes if scopes is not None else ["items:read"],
        "iat": now,
        "nbf": now + not_before_offset,
        "exp": now + expires_in,
        "jti": "jti-test-1",
    }
