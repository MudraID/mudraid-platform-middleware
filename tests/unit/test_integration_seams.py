"""M5.10 — Cross-module wiring tests.

The per-module suites (yaml_loader, route_matcher, jwks_client,
jwt_validator, middleware_dispatch) each lock their own behaviour
cleanly. What was missing was a small set of tests that cross the
seams between them — places where two modules cooperate and a
future refactor could silently break the contract without any
single module's tests failing.

Scope of this file:

  - Exception hierarchy: subclass relationships + ``except`` semantics
  - JWT-rotation end-to-end: unknown kid → JWKS refresh → 200
  - Query-string handling (Starlette strips, we trust it)
  - Method × scope dimensions: same path, different methods, different
    scopes, different tokens
  - Bootstrap failure recovery: bad YAML now, fix YAML, next request works
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

from mudraid_platform_middleware import (
    MudraIDInvalidTokenError,
    MudraIDJwksError,
    MudraIDMiddleware,
    MudraIDMiddlewareError,
    MudraIDScopesYamlError,
)
from tests.conftest import baseline_claims, sign_jwt

JWKS_URL = "https://api.mudraid.test/.well-known/jwks.json"
PLATFORM_ID = "plt-test"


def _yaml(routes_yaml: str) -> str:
    return f"platform_id: {PLATFORM_ID}\n" f"version: 1\n" f"routes:\n{routes_yaml}"


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "mudraid_scopes.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def _make_app(yaml_path: Path) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        MudraIDMiddleware,
        scopes_yaml_path=str(yaml_path),
        jwks_url=JWKS_URL,
    )

    @app.get("/items")
    async def list_items() -> dict:
        return {"ok": "GET"}

    @app.post("/items")
    async def create_item() -> dict:
        return {"ok": "POST"}

    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    )


# ---- exception hierarchy --------------------------------------------------


@pytest.mark.parametrize(
    "exception_class",
    [
        MudraIDScopesYamlError,
        MudraIDJwksError,
        MudraIDInvalidTokenError,
    ],
)
def test_every_specific_exception_subclasses_mudraid_middleware_error(
    exception_class: type[MudraIDMiddlewareError],
) -> None:
    """Locks the inheritance chain — one ``except MudraIDMiddlewareError``
    catches every middleware-raised failure. Public-API contract; a
    future contributor who adds a new exception that forgets the
    inheritance will be caught here."""
    assert issubclass(exception_class, MudraIDMiddlewareError)
    assert issubclass(exception_class, Exception)


def test_single_except_clause_catches_every_middleware_error() -> None:
    """The platform-operator-facing version of the same contract:
    ``except MudraIDMiddlewareError`` catches every flavour."""
    errors = [
        MudraIDScopesYamlError("x"),
        MudraIDJwksError("x"),
        MudraIDInvalidTokenError("malformed", "x"),
    ]
    for exc in errors:
        try:
            raise exc
        except MudraIDMiddlewareError as caught:
            assert caught is exc
        else:  # pragma: no cover
            pytest.fail(f"{type(exc).__name__} was not caught by MudraIDMiddlewareError")


def test_specific_except_clauses_do_not_catch_sibling_errors() -> None:
    """``except MudraIDJwksError`` must NOT catch a
    MudraIDInvalidTokenError — locks the precise-handling contract."""
    try:
        raise MudraIDInvalidTokenError("malformed", "x")
    except MudraIDJwksError:  # pragma: no cover
        pytest.fail("MudraIDJwksError must not catch MudraIDInvalidTokenError")
    except MudraIDInvalidTokenError:
        pass


def test_invalid_token_error_exposes_reason_attribute() -> None:
    """The structured error response (M5.9) maps ``reason`` to
    ``error_code``. Locks the public attribute name so a future
    refactor of the validator can't break the response shape
    without breaking this test first."""
    exc = MudraIDInvalidTokenError("expired", "token gone bad")
    assert exc.reason == "expired"


# ---- the headline cross-module case: JWT rotation end-to-end ------------


@pytest.mark.asyncio
async def test_jwt_rotation_via_kid_refresh_end_to_end(
    tmp_path: Path,
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """The single most important integration scenario the middleware
    has to handle: MudraID rotated its signing key while the
    middleware was running, and the next agent presents a token
    signed under the new key.

    Steps:
      1. Agent presents JWT signed under kid-1; JWKS has kid-1 → 200.
      2. Agent presents JWT signed under kid-2; JWKS still has only
         kid-1 → JwksClient refreshes once, this time the JWKS
         response includes kid-2 → middleware validates → 200.

    Without the reactive-refresh contract (locked decision D4), the
    second request would fail with INVALID_TOKEN even though the
    JWT is perfectly valid — and every agent in production would
    start hitting that on the next MudraID rotation. This test
    locks the recovery."""
    yaml_path = _write_yaml(
        tmp_path,
        _yaml("  - method: GET\n    path: /items\n    scope: items:read\n"),
    )
    app = _make_app(yaml_path)

    # A second key pair for the "rotated" signing identity.
    from cryptography.hazmat.primitives.asymmetric import rsa as _rsa

    rotated_key = _rsa.generate_private_key(public_exponent=65537, key_size=2048)
    import json as _json

    from jwt.algorithms import RSAAlgorithm

    rotated_jwk = _json.loads(RSAAlgorithm.to_jwk(rotated_key.public_key()))
    rotated_jwk["kid"] = "kid-rotated"
    rotated_jwk["use"] = "sig"
    rotated_jwk["alg"] = "RS256"

    token_old = sign_jwt(
        rsa_private_key,
        baseline_claims(audience=PLATFORM_ID, scopes=["items:read"]),
        kid="test-kid-1",  # the conftest session key's kid
    )
    token_new = sign_jwt(
        rotated_key,
        baseline_claims(audience=PLATFORM_ID, scopes=["items:read"]),
        kid="kid-rotated",
    )

    # JWKS responses: first call has only the original kid; second
    # call (the refresh triggered by the unknown kid) has both.
    jwks_responses = [
        httpx.Response(200, json={"keys": [rsa_public_jwk]}),
        httpx.Response(200, json={"keys": [rsa_public_jwk, rotated_jwk]}),
    ]

    with respx.mock() as r:
        route = r.get(JWKS_URL).mock(side_effect=jwks_responses)

        async with _client(app) as c:
            r1 = await c.get("/items", headers={"Authorization": f"Bearer {token_old}"})
            r2 = await c.get("/items", headers={"Authorization": f"Bearer {token_new}"})

    assert r1.status_code == 200, r1.json()
    assert r2.status_code == 200, r2.json()
    assert route.call_count == 2, "exactly one refresh on the unknown kid"


# ---- query string handling -----------------------------------------------


@pytest.mark.asyncio
async def test_query_string_does_not_affect_route_match(
    tmp_path: Path,
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """Starlette's ``request.url.path`` excludes the query string.
    A real-world request ``GET /items?limit=10`` must match the
    rule for ``/items``. Without this trust, every paginated agent
    call would fail."""
    yaml_path = _write_yaml(
        tmp_path,
        _yaml("  - method: GET\n    path: /items\n    scope: items:read\n"),
    )
    app = _make_app(yaml_path)
    token = sign_jwt(rsa_private_key, baseline_claims(audience=PLATFORM_ID, scopes=["items:read"]))

    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))

        async with _client(app) as c:
            resp = await c.get(
                "/items?limit=10&cursor=abc",
                headers={"Authorization": f"Bearer {token}"},
            )

    assert resp.status_code == 200


# ---- method × scope independence -----------------------------------------


@pytest.mark.asyncio
async def test_same_path_two_methods_two_scopes_independent_enforcement(
    tmp_path: Path,
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """GET /items needs items:read; POST /items needs items:write.
    A token with items:read can GET but not POST. Locks the
    method-as-part-of-the-match-key contract end-to-end."""
    yaml_path = _write_yaml(
        tmp_path,
        _yaml(
            "  - method: GET\n    path: /items\n    scope: items:read\n"
            "  - method: POST\n    path: /items\n    scope: items:write\n"
        ),
    )
    app = _make_app(yaml_path)
    read_only_token = sign_jwt(
        rsa_private_key,
        baseline_claims(audience=PLATFORM_ID, scopes=["items:read"]),
    )

    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))

        async with _client(app) as c:
            get_resp = await c.get("/items", headers={"Authorization": f"Bearer {read_only_token}"})
            post_resp = await c.post(
                "/items", headers={"Authorization": f"Bearer {read_only_token}"}
            )

    assert get_resp.status_code == 200
    assert post_resp.status_code == 403
    assert post_resp.json()["error_code"] == "MISSING_SCOPE"
    assert "items:write" in post_resp.json()["message"]


# ---- scope-string semantics ----------------------------------------------


@pytest.mark.asyncio
async def test_scope_match_is_exact_string_no_wildcards(
    tmp_path: Path,
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """``items:read`` does NOT match a token scope of ``items:*`` or
    ``items``. v1 locks exact-string matching — wildcards are a
    v1.x feature once we have telemetry on real demand. Without
    this lock, a future contributor adding wildcard support could
    silently broaden every existing scope."""
    yaml_path = _write_yaml(
        tmp_path,
        _yaml("  - method: GET\n    path: /items\n    scope: items:read\n"),
    )
    app = _make_app(yaml_path)
    token = sign_jwt(
        rsa_private_key,
        baseline_claims(audience=PLATFORM_ID, scopes=["items:*"]),  # wildcard-ish
    )

    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))

        async with _client(app) as c:
            resp = await c.get("/items", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 403
    assert resp.json()["error_code"] == "MISSING_SCOPE"


# ---- bootstrap failure recovery -----------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_failure_is_recoverable_on_subsequent_request(
    tmp_path: Path,
) -> None:
    """If the YAML is bad on the first dispatch, the middleware fails
    that request — but does NOT permanently cache the failure. The
    operator can fix the YAML and the next request succeeds without
    a restart.

    This is the operator-friendly half of the lazy-bootstrap design:
    a transient or correctable misconfiguration doesn't brick the
    process forever."""
    yaml_path = _write_yaml(tmp_path, "this is not valid YAML schema")
    app = _make_app(yaml_path)

    async with _client(app) as c:
        bad = await c.get("/items")
        # The middleware turns the bootstrap failure into a structured
        # 500 instead of letting it propagate as an uncaught exception.
        # MIDDLEWARE_NOT_READY is the locked error_code for this path.
        assert bad.status_code == 500
        assert bad.json()["error_code"] == "MIDDLEWARE_NOT_READY"

        # Operator fixes the YAML.
        _write_yaml(
            tmp_path,
            _yaml("  - method: GET\n    path: /items\n    public: true\n"),
        )

        good = await c.get("/items")

    # Subsequent request must re-attempt the bootstrap and succeed —
    # locks the "transient failures are recoverable without a restart"
    # operator-friendly half of the lazy-bootstrap design.
    assert good.status_code == 200
    assert good.json() == {"ok": "GET"}


# ---- repeated tokens / no per-token leakage -----------------------------


@pytest.mark.asyncio
async def test_repeated_requests_with_different_tokens_each_succeed(
    tmp_path: Path,
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """The middleware must NOT cache validation results per-token —
    every request re-validates. Without this, a token revocation
    that hits MudraID's enforcement layer wouldn't take effect at
    the middleware until JWKS rotation.

    We can't observe 'didn't cache' directly, but we can confirm
    multiple distinct tokens each independently validate."""
    yaml_path = _write_yaml(
        tmp_path,
        _yaml("  - method: GET\n    path: /items\n    scope: items:read\n"),
    )
    app = _make_app(yaml_path)

    # Three different tokens (different jti / sub combinations).
    def _token(sub: str, jti: str) -> str:
        claims = baseline_claims(audience=PLATFORM_ID, scopes=["items:read"])
        claims["sub"] = sub
        claims["jti"] = jti
        return sign_jwt(rsa_private_key, claims)

    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))

        async with _client(app) as c:
            for sub, jti in [("agent-A", "j1"), ("agent-B", "j2"), ("agent-A", "j3")]:
                resp = await c.get(
                    "/items",
                    headers={"Authorization": f"Bearer {_token(sub, jti)}"},
                )
                assert resp.status_code == 200
