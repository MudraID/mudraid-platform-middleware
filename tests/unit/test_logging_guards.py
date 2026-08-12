"""M7.5 — Anti-leak guard: JWTs and Authorization headers must never
land in middleware logs, even at DEBUG verbosity.

Runs every dispatch branch (public route, happy scoped path, every
locked error_code surface) under the ``mudraid_platform_middleware`` logger at
DEBUG, captures every record emitted along the way, and asserts the
captured stream contains:

  - none of the JWT body the client sent
  - no ``Bearer ...`` Authorization header value
  - none of the test agent's distinctive subject claim

The sentinel values are deliberately chosen so a substring match
won't false-positive on common dispatch text (e.g. "Bearer" alone
appears in error messages — only the full attacker-readable token is
considered a leak).

If a future contributor adds a ``logger.debug("got header %s", auth)``
or ``logger.info("decoded claims=%s", claims)`` along the dispatch
path, one of these tests fires before the leak ships.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI
from httpx import ASGITransport

from mudraid_platform_middleware import MudraIDMiddleware
from tests.conftest import baseline_claims, sign_jwt

JWKS_URL = "https://api.mudraid.test/.well-known/jwks.json"
PLATFORM_ID = "11111111-2222-3333-4444-555555555555"

# Distinctive markers that should never reach a log stream. Long
# enough that any substring match means a real leak — not an accidental
# collision with framework log text.
_AGENT_SUB = "agent-LEAK-CANARY-zzzzzzzzzzzzzzzz"
_JWT_LEAK_JTI = "jti-LEAK-CANARY-zzzzzzzzzzzzzzzzz"


def _yaml_body() -> str:
    return f"""platform_id: {PLATFORM_ID}
version: 1
routes:
  - method: GET
    path: /health
    public: true
  - method: GET
    path: /api/v1/items
    scope: items:read
  - method: POST
    path: /api/v1/items
    scope: items:write
"""


def _build_app(yaml_path: Path) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        MudraIDMiddleware,
        scopes_yaml_path=str(yaml_path),
        jwks_url=JWKS_URL,
    )

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True}

    @app.get("/api/v1/items")
    async def list_items() -> dict:
        return {"items": []}

    @app.post("/api/v1/items", status_code=201)
    async def create_items() -> dict:
        return {"created": True}

    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    )


def _mock_jwks(rsa_public_jwk: dict[str, Any]) -> respx.MockRouter:
    router = respx.mock(assert_all_called=False)
    router.get(JWKS_URL).mock(
        return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}),
    )
    return router


def _build_token(private_key, **claim_overrides: Any) -> str:
    claims = baseline_claims(audience=PLATFORM_ID, scopes=["items:read"])
    claims["sub"] = _AGENT_SUB
    claims["jti"] = _JWT_LEAK_JTI
    claims.update(claim_overrides)
    return sign_jwt(private_key, claims)


def _captured(caplog: pytest.LogCaptureFixture) -> str:
    """Concatenate every captured log line into one string for
    substring assertions. Includes both the formatted message and
    its args via the standard ``%s`` substitution path."""
    parts: list[str] = []
    for rec in caplog.records:
        # ``getMessage`` formats the record's msg with its args — same
        # path a real handler would take, so anything a leak would
        # actually emit is what we scan.
        parts.append(rec.getMessage())
    return "\n".join(parts)


def _assert_no_leak(captured: str, *, jwt: str | None) -> None:
    assert (
        _AGENT_SUB not in captured
    ), f"agent sub claim leaked into middleware logs: {_AGENT_SUB!r}"
    assert (
        _JWT_LEAK_JTI not in captured
    ), f"jti claim leaked into middleware logs: {_JWT_LEAK_JTI!r}"
    if jwt is not None:
        assert jwt not in captured, "raw JWT leaked into middleware logs"
        # The Authorization header value itself is also forbidden.
        assert (
            f"Bearer {jwt}" not in captured
        ), "full Bearer Authorization header leaked into middleware logs"


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_scoped_request_does_not_log_token_or_claims(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    rsa_private_key,
    rsa_public_jwk: dict[str, Any],
) -> None:
    yaml_path = tmp_path / "scopes.yaml"
    yaml_path.write_text(_yaml_body(), encoding="utf-8")
    app = _build_app(yaml_path)
    token = _build_token(rsa_private_key)

    caplog.set_level(logging.DEBUG, logger="mudraid_platform_middleware")
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/api/v1/items", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    _assert_no_leak(_captured(caplog), jwt=token)


@pytest.mark.asyncio
async def test_invalid_token_path_does_not_log_offending_token(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """A malformed-token rejection is the easiest place to accidentally
    log the offending value while trying to "explain" the failure.
    Locks the contract that the rejection path never echoes the
    attacker-controlled token back into a log."""
    yaml_path = tmp_path / "scopes.yaml"
    yaml_path.write_text(_yaml_body(), encoding="utf-8")
    app = _build_app(yaml_path)
    bogus = "Bearer.LEAK.CANARY.aaaaaaaaaaaaaaaaaaaaa"

    caplog.set_level(logging.DEBUG, logger="mudraid_platform_middleware")
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/api/v1/items", headers={"Authorization": f"Bearer {bogus}"})
    assert resp.status_code == 401

    captured = _captured(caplog)
    assert bogus not in captured, "rejection path logged the bogus token verbatim"
    assert f"Bearer {bogus}" not in captured


@pytest.mark.asyncio
async def test_expired_token_path_does_not_log_token_value(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    rsa_private_key,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """Exercises the JWT-decoded-but-expired path so the validator
    knows the JWT is well-formed and could (incorrectly) include the
    raw value in its rejection log. Lock: it does not."""
    yaml_path = tmp_path / "scopes.yaml"
    yaml_path.write_text(_yaml_body(), encoding="utf-8")
    app = _build_app(yaml_path)
    now = int(time.time())
    claims = baseline_claims(audience=PLATFORM_ID, scopes=["items:read"])
    claims["sub"] = _AGENT_SUB
    claims["jti"] = _JWT_LEAK_JTI
    claims["iat"] = now - 3600
    claims["nbf"] = now - 3600
    claims["exp"] = now - 600
    token = sign_jwt(rsa_private_key, claims)

    caplog.set_level(logging.DEBUG, logger="mudraid_platform_middleware")
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/api/v1/items", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    _assert_no_leak(_captured(caplog), jwt=token)


@pytest.mark.asyncio
async def test_missing_scope_path_does_not_log_token_or_claims(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    rsa_private_key,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """The scope-check rejection knows the JWT was valid — easy place
    to log the (now trusted-by-signature) JWT in a "diagnostic"
    message. Lock: it must not, because operators routinely ship
    these logs to third-party aggregators."""
    yaml_path = tmp_path / "scopes.yaml"
    yaml_path.write_text(_yaml_body(), encoding="utf-8")
    app = _build_app(yaml_path)
    token = _build_token(rsa_private_key, scopes=["items:read"])

    caplog.set_level(logging.DEBUG, logger="mudraid_platform_middleware")
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.post(
                "/api/v1/items",
                headers={"Authorization": f"Bearer {token}"},
                json={"name": "x"},
            )
    assert resp.status_code == 403
    _assert_no_leak(_captured(caplog), jwt=token)


@pytest.mark.asyncio
async def test_public_route_does_not_log_authorization_when_present(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    rsa_private_key,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """Even on a public route the middleware sees the Authorization
    header (the agent may carry it for any reason — outbound proxies,
    multi-platform sessions, etc.). The middleware must not log it."""
    yaml_path = tmp_path / "scopes.yaml"
    yaml_path.write_text(_yaml_body(), encoding="utf-8")
    app = _build_app(yaml_path)
    token = _build_token(rsa_private_key)

    caplog.set_level(logging.DEBUG, logger="mudraid_platform_middleware")
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/health", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    _assert_no_leak(_captured(caplog), jwt=token)
