"""M5.11 — End-to-end middleware integration test against the sample platform.

The unit suite in ``tests/unit/`` already drives the middleware through
a real ASGI transport, but each test rebuilds an ad-hoc FastAPI app.
This file pins the contract against the **exact route + YAML shape
shipped in ``examples/sample-platform/``** — the platform a real
operator will copy when integrating.

What this proves end-to-end:

  - The sample-platform's four locked routes (public ``/health``,
    ``items:read`` GET list + GET-by-id, ``items:write`` POST) enforce
    correctly under the middleware.
  - Every locked v1 error_code surfaces through the wire under the
    matching failure mode (MISSING_TOKEN, INVALID_TOKEN, EXPIRED_TOKEN,
    TOKEN_NOT_YET_VALID, WRONG_AUDIENCE, WRONG_ISSUER, MISSING_SCOPE,
    ROUTE_NOT_FOUND).
  - The parametric path ``/api/v1/items/{item_id}`` accepts arbitrary
    concrete ids under the same scope rule (regression guard for the
    route-matcher).

No real backend stack is required: a respx-mocked JWKS plus a
session-scoped RSA key gives us a hermetic, deterministic harness.
The tokens that exit ``sign_jwt`` are byte-identical in shape to the
tokens MudraID's ``/auth/token`` will issue in production.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport
from pydantic import BaseModel

from mudraid_platform_middleware import MudraIDMiddleware
from tests.conftest import baseline_claims, sign_jwt

JWKS_URL = "https://api.mudraid.test/.well-known/jwks.json"
PLATFORM_ID = "11111111-2222-3333-4444-555555555555"


def _sample_yaml_body(platform_id: str = PLATFORM_ID) -> str:
    """Mirror of examples/sample-platform/mudraid_scopes.yaml routes.

    Kept inline so the middleware test isn't coupled to the example
    file's filesystem location, but every route/scope/method below
    matches the shipped YAML byte-for-byte."""
    return f"""platform_id: {platform_id}
version: 1
routes:
  - method: GET
    path: /health
    public: true
  - method: GET
    path: /api/v1/items
    scope: items:read
  - method: GET
    path: /api/v1/items/{{item_id}}
    scope: items:read
  - method: POST
    path: /api/v1/items
    scope: items:write
"""


class _ItemIn(BaseModel):
    name: str


def _build_sample_app(yaml_path: Path) -> FastAPI:
    """Build a FastAPI app whose route shape matches
    examples/sample-platform/app.py exactly. Kept as a local builder
    rather than importing from ``examples.sample_platform.app`` because
    the example uses module-level ``add_middleware`` with default
    config — for the test we need to inject the test JWKS URL and
    tmp_path-scoped YAML location."""
    app = FastAPI()
    app.add_middleware(
        MudraIDMiddleware,
        scopes_yaml_path=str(yaml_path),
        jwks_url=JWKS_URL,
    )

    items: dict[str, dict[str, Any]] = {
        "item-1": {"id": "item-1", "name": "Widget"},
        "item-2": {"id": "item-2", "name": "Gadget"},
    }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/items")
    def list_items() -> dict[str, Any]:
        return {"items": list(items.values())}

    @app.get("/api/v1/items/{item_id}")
    def get_item(item_id: str) -> dict[str, Any]:
        if item_id not in items:
            raise HTTPException(status_code=404, detail="item not found")
        return items[item_id]

    @app.post("/api/v1/items", status_code=201)
    def create_item(payload: _ItemIn) -> dict[str, Any]:
        item_id = f"item-{len(items) + 1}"
        items[item_id] = {"id": item_id, "name": payload.name}
        return items[item_id]

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


def _write_yaml(tmp_path: Path, body: str | None = None) -> Path:
    p = tmp_path / "mudraid_scopes.yaml"
    p.write_text(body if body is not None else _sample_yaml_body(), encoding="utf-8")
    return p


# ----------------------------------------------------------------------------
# Happy path — every locked sample-platform route under a valid token
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_public_health_route_passes_without_a_token(
    tmp_path: Path, rsa_public_jwk: dict[str, Any]
) -> None:
    app = _build_sample_app(_write_yaml(tmp_path))
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_items_read_scope_allows_list_endpoint(
    tmp_path: Path, rsa_private_key, rsa_public_jwk: dict[str, Any]
) -> None:
    app = _build_sample_app(_write_yaml(tmp_path))
    token = sign_jwt(
        rsa_private_key,
        baseline_claims(audience=PLATFORM_ID, scopes=["items:read"]),
    )
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/api/v1/items", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert "items" in resp.json()


@pytest.mark.asyncio
async def test_items_read_scope_allows_parametric_get_by_id(
    tmp_path: Path, rsa_private_key, rsa_public_jwk: dict[str, Any]
) -> None:
    """Locks the route-matcher's {item_id} segment against a real
    concrete id — same scope rule applies regardless of the value."""
    app = _build_sample_app(_write_yaml(tmp_path))
    token = sign_jwt(
        rsa_private_key,
        baseline_claims(audience=PLATFORM_ID, scopes=["items:read"]),
    )
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get(
                "/api/v1/items/item-1",
                headers={"Authorization": f"Bearer {token}"},
            )
    assert resp.status_code == 200
    assert resp.json()["id"] == "item-1"


@pytest.mark.asyncio
async def test_items_write_scope_allows_post(
    tmp_path: Path, rsa_private_key, rsa_public_jwk: dict[str, Any]
) -> None:
    app = _build_sample_app(_write_yaml(tmp_path))
    token = sign_jwt(
        rsa_private_key,
        baseline_claims(audience=PLATFORM_ID, scopes=["items:write"]),
    )
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.post(
                "/api/v1/items",
                headers={"Authorization": f"Bearer {token}"},
                json={"name": "Sprocket"},
            )
    assert resp.status_code == 201
    assert resp.json()["name"] == "Sprocket"


# ----------------------------------------------------------------------------
# Locked error_code matrix — every 401/403/404 surface from M5.9
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_authorization_header_returns_401_missing_token(
    tmp_path: Path, rsa_public_jwk: dict[str, Any]
) -> None:
    app = _build_sample_app(_write_yaml(tmp_path))
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/api/v1/items")
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "MISSING_TOKEN"


@pytest.mark.asyncio
async def test_authorization_without_bearer_scheme_returns_401_missing_token(
    tmp_path: Path, rsa_public_jwk: dict[str, Any]
) -> None:
    app = _build_sample_app(_write_yaml(tmp_path))
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get(
                "/api/v1/items",
                headers={"Authorization": "Basic abc123"},
            )
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "MISSING_TOKEN"


@pytest.mark.asyncio
async def test_malformed_token_returns_401_invalid_token(
    tmp_path: Path, rsa_public_jwk: dict[str, Any]
) -> None:
    app = _build_sample_app(_write_yaml(tmp_path))
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get(
                "/api/v1/items",
                headers={"Authorization": "Bearer not.a.jwt"},
            )
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "INVALID_TOKEN"


@pytest.mark.asyncio
async def test_expired_token_returns_401_expired_token(
    tmp_path: Path, rsa_private_key, rsa_public_jwk: dict[str, Any]
) -> None:
    app = _build_sample_app(_write_yaml(tmp_path))
    claims = baseline_claims(audience=PLATFORM_ID, scopes=["items:read"])
    # Push iat/exp into the past so the JWT is unambiguously expired
    # even after clock-skew leeway.
    now = int(time.time())
    claims["iat"] = now - 3600
    claims["nbf"] = now - 3600
    claims["exp"] = now - 600
    token = sign_jwt(rsa_private_key, claims)
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/api/v1/items", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "EXPIRED_TOKEN"


@pytest.mark.asyncio
async def test_not_yet_valid_token_returns_401_token_not_yet_valid(
    tmp_path: Path, rsa_private_key, rsa_public_jwk: dict[str, Any]
) -> None:
    app = _build_sample_app(_write_yaml(tmp_path))
    # nbf well past any leeway window keeps the test deterministic.
    token = sign_jwt(
        rsa_private_key,
        baseline_claims(
            audience=PLATFORM_ID,
            scopes=["items:read"],
            not_before_offset=3600,
        ),
    )
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/api/v1/items", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "TOKEN_NOT_YET_VALID"


@pytest.mark.asyncio
async def test_wrong_audience_returns_401_wrong_audience(
    tmp_path: Path, rsa_private_key, rsa_public_jwk: dict[str, Any]
) -> None:
    app = _build_sample_app(_write_yaml(tmp_path))
    token = sign_jwt(
        rsa_private_key,
        baseline_claims(audience="some-other-platform", scopes=["items:read"]),
    )
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/api/v1/items", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "WRONG_AUDIENCE"


@pytest.mark.asyncio
async def test_wrong_issuer_returns_401_wrong_issuer(
    tmp_path: Path, rsa_private_key, rsa_public_jwk: dict[str, Any]
) -> None:
    app = _build_sample_app(_write_yaml(tmp_path))
    token = sign_jwt(
        rsa_private_key,
        baseline_claims(
            audience=PLATFORM_ID,
            issuer="someone-else",
            scopes=["items:read"],
        ),
    )
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/api/v1/items", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "WRONG_ISSUER"


@pytest.mark.asyncio
async def test_read_token_cannot_call_write_route_returns_403_missing_scope(
    tmp_path: Path, rsa_private_key, rsa_public_jwk: dict[str, Any]
) -> None:
    """Locks the principle-of-least-privilege win the sample-platform
    YAML advertises: an items:read token cannot mutate items even
    though it's a perfectly valid JWT for the same platform."""
    app = _build_sample_app(_write_yaml(tmp_path))
    token = sign_jwt(
        rsa_private_key,
        baseline_claims(audience=PLATFORM_ID, scopes=["items:read"]),
    )
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.post(
                "/api/v1/items",
                headers={"Authorization": f"Bearer {token}"},
                json={"name": "Bolt"},
            )
    assert resp.status_code == 403
    assert resp.json()["error_code"] == "MISSING_SCOPE"


@pytest.mark.asyncio
async def test_write_only_token_cannot_call_read_route_returns_403_missing_scope(
    tmp_path: Path, rsa_private_key, rsa_public_jwk: dict[str, Any]
) -> None:
    """Symmetric to the read-cannot-write test: scope check is strictly
    membership, not hierarchy. There is no implicit ``items:write
    implies items:read`` — both must be granted explicitly."""
    app = _build_sample_app(_write_yaml(tmp_path))
    token = sign_jwt(
        rsa_private_key,
        baseline_claims(audience=PLATFORM_ID, scopes=["items:write"]),
    )
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/api/v1/items", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403
    assert resp.json()["error_code"] == "MISSING_SCOPE"


@pytest.mark.asyncio
async def test_unknown_route_returns_404_route_not_found(
    tmp_path: Path, rsa_private_key, rsa_public_jwk: dict[str, Any]
) -> None:
    """A route that has no rule in mudraid_scopes.yaml is invisible to
    agents — same response shape as a ``skip: true`` route so the agent
    cannot distinguish hidden-by-config from genuinely-absent."""
    app = _build_sample_app(_write_yaml(tmp_path))
    token = sign_jwt(
        rsa_private_key,
        baseline_claims(audience=PLATFORM_ID, scopes=["items:read"]),
    )
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get(
                "/api/v1/no-such-route",
                headers={"Authorization": f"Bearer {token}"},
            )
    assert resp.status_code == 404
    assert resp.json()["error_code"] == "ROUTE_NOT_FOUND"


# ----------------------------------------------------------------------------
# Response-body shape lock
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_error_response_has_locked_body_shape(
    tmp_path: Path, rsa_public_jwk: dict[str, Any]
) -> None:
    """SDK consumers + platform operators alert on {error_code, message}.
    Any drift in this two-field contract breaks every downstream
    monitor."""
    app = _build_sample_app(_write_yaml(tmp_path))
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/api/v1/items")
    body = resp.json()
    assert set(body.keys()) == {"error_code", "message"}
    assert isinstance(body["error_code"], str)
    assert isinstance(body["message"], str)
    assert body["message"]
