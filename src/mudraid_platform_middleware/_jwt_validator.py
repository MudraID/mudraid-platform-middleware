"""JWT validation against MudraID's JWKS.

Pulls the right public key out of the JWKS by ``kid``, then uses
PyJWT to verify the signature and the locked claim set:

  - ``iss``  — must equal ``expected_issuer`` (default
                ``"mudraid-identity"``), or, when ``expected_issuer`` is a
                sequence, be one of the configured transition set
  - ``aud``  — must equal the platform's ``platform_id`` (passed
                in by the middleware at construction)
  - ``exp``  — must be in the future (with leeway)
  - ``nbf``  — must not be in the future (with leeway)
  - ``iat``  — must be present (locked claim per backend M2.11)

On success the decoded claim dict is returned to the middleware,
which then checks ``scopes`` membership against the route's
required scope. The scope check intentionally lives in the
middleware, not here — this module is the JWT machinery, the
middleware is the policy.

PyJWT's exception types are mapped to a single
:class:`MudraIDInvalidTokenError` with a precise ``reason`` field
so the structured 401 response (M5.9) can surface a stable error
code without leaking the upstream message.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jwt

from mudraid_platform_middleware._jwks_client import JwksClient
from mudraid_platform_middleware.exceptions import MudraIDInvalidTokenError

_DEFAULT_ISSUER = "mudraid-identity"
_DEFAULT_LEEWAY_SEC = 30.0  # Locked at 30 s per Phase 3 plan risk A3.


class JwtValidator:
    """Verify a JWT against MudraID's JWKS + the expected claims."""

    def __init__(
        self,
        jwks_client: JwksClient,
        expected_audience: str,
        expected_issuer: str | Sequence[str] = _DEFAULT_ISSUER,
        leeway_sec: float = _DEFAULT_LEEWAY_SEC,
    ) -> None:
        if not expected_audience:
            raise ValueError("expected_audience is required")
        self._jwks = jwks_client
        self._audience = expected_audience
        self._leeway = leeway_sec
        # `expected_issuer` accepts a single issuer (str) OR a transition SET
        # (sequence of str) so a rolling issuer migration can accept both the
        # old and new issuer during the window (standards charter §16,
        # verifier-before-issuer). PyJWT's `issuer=` only does an exact compare
        # — a set is therefore enforced here by membership, not handed to PyJWT.
        # NOTE: `str` is itself a `Sequence`, so the str branch MUST come first.
        if isinstance(expected_issuer, str):
            if not expected_issuer:
                raise ValueError("expected_issuer is required")
            self._issuer_for_pyjwt: str | None = expected_issuer
            self._allowed_issuers: frozenset[str] | None = None
        else:
            allowed = frozenset(expected_issuer)
            if not allowed:
                raise ValueError("expected_issuer set must not be empty")
            self._issuer_for_pyjwt = None
            self._allowed_issuers = allowed

    async def validate(self, token: str) -> dict[str, Any]:
        """Return the decoded claims on success.

        Raises:
            MudraIDInvalidTokenError: with a ``reason`` in
                {malformed, expired, not_yet_valid, wrong_audience,
                 wrong_issuer, invalid_signature}.
            MudraIDJwksError: bubbled up from the JWKS client when
                the key cannot be fetched or doesn't exist for the
                JWT's ``kid``.
        """
        # 1. Parse the JWT header (NOT the signature yet) to find kid.
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise MudraIDInvalidTokenError("malformed", "could not parse JWT header") from exc

        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise MudraIDInvalidTokenError("malformed", "JWT header missing 'kid'")

        # 2. Resolve the public key. JWKS errors propagate.
        #
        # The JwksClient owns the JWK->key parse and memoises it per key
        # (keyed by kid, reset on every JWKS refresh), so the expensive
        # ``RSAAlgorithm.from_jwk`` runs once per key per refresh rather than
        # on every request. The trust decision (which kids exist, freshness,
        # reactive refresh on unknown kid) still lives entirely in get_key,
        # which get_public_key wraps — so caching the parse never widens the
        # set of accepted keys.
        public_key = await self._jwks.get_public_key(kid)
        from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

        if not isinstance(public_key, RSAPublicKey):
            raise MudraIDInvalidTokenError(
                "malformed",
                f"JWK for kid={kid} produced a {type(public_key).__name__}, "
                "expected RSAPublicKey",
            )

        # 3. Verify signature + claim set in one shot.
        try:
            claims = jwt.decode(
                token,
                key=public_key,
                algorithms=["RS256"],
                audience=self._audience,
                # str path: PyJWT enforces the exact issuer. set path:
                # `None` skips PyJWT's check and we enforce membership below
                # (PyJWT's `issuer=` can't take a sequence).
                issuer=self._issuer_for_pyjwt,
                leeway=self._leeway,
                # ``require`` adds belt-and-braces presence checks for
                # claims PyJWT only validates conditionally. ``exp``
                # and ``aud`` are required by default once we pass
                # ``audience=``; ``iat`` and ``nbf`` are not, but the
                # locked JWT shape (M2.11) emits both — refusing a
                # JWT that lacks them is the right defensive posture.
                options={"require": ["exp", "iat", "nbf", "aud", "iss"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise MudraIDInvalidTokenError("expired", str(exc)) from exc
        except jwt.ImmatureSignatureError as exc:
            raise MudraIDInvalidTokenError("not_yet_valid", str(exc)) from exc
        except jwt.InvalidAudienceError as exc:
            raise MudraIDInvalidTokenError("wrong_audience", str(exc)) from exc
        except jwt.InvalidIssuerError as exc:
            raise MudraIDInvalidTokenError("wrong_issuer", str(exc)) from exc
        except jwt.InvalidSignatureError as exc:
            raise MudraIDInvalidTokenError("invalid_signature", str(exc)) from exc
        except jwt.MissingRequiredClaimError as exc:
            raise MudraIDInvalidTokenError("malformed", str(exc)) from exc
        except jwt.PyJWTError as exc:
            # Catch-all for DecodeError, InvalidTokenError, etc.
            raise MudraIDInvalidTokenError("malformed", str(exc)) from exc

        # Transition-set issuer enforcement (set path only). `iss` is
        # guaranteed present by the `require` option above, so a missing
        # issuer has already failed as `malformed`.
        if self._allowed_issuers is not None and claims.get("iss") not in self._allowed_issuers:
            raise MudraIDInvalidTokenError(
                "wrong_issuer", "token issuer is not in the accepted issuer set"
            )

        return claims
