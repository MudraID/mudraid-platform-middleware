"""M5.6 / M5.7 / M5.8 / M5.9 — dispatch + public + skip + error responses.

These tests mount the real ``MudraIDMiddleware`` on a real FastAPI app
and fire requests through ``httpx.AsyncClient(transport=ASGITransport(...))``.
Each test writes its own ``mudraid_scopes.yaml`` to ``tmp_path`` and
mocks the JWKS endpoint with ``respx`` so no live MudraID is needed.

Reading the responses through a real ASGI transport (rather than
poking ``middleware.dispatch`` directly with synthetic Request
objects) catches a class of bugs unit tests miss — header
case-folding, body-encoding, and Starlette's own response handling
all participate in the test.
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

from mudraid_platform_middleware import MudraIDMiddleware
from tests.conftest import baseline_claims, sign_jwt

JWKS_URL = "https://api.mudraid.test/.well-known/jwks.json"
PLATFORM_ID = "plt-test"


def _yaml(platform_id: str = PLATFORM_ID, routes_yaml: str = "") -> str:
    """Build a `mudraid_scopes.yaml` body for a test.

    Empty ``routes_yaml`` produces ``routes: []`` rather than the
    bare ``routes:`` (which YAML parses to ``None`` and the loader
    correctly rejects)."""
    routes_section = f"routes:\n{routes_yaml}" if routes_yaml else "routes: []\n"
    return f"platform_id: {platform_id}\nversion: 1\n{routes_section}"


def _make_app(yaml_path: Path) -> FastAPI:
    """Build a FastAPI app with the middleware applied and a small
    set of test routes covering every dispatch branch."""
    app = FastAPI()
    app.add_middleware(
        MudraIDMiddleware,
        scopes_yaml_path=str(yaml_path),
        jwks_url=JWKS_URL,
    )

    @app.get("/items")
    async def list_items() -> dict:
        return {"items": ["a", "b"]}

    @app.post("/items")
    async def create_item() -> dict:
        return {"created": True}

    @app.get("/items/{id}")
    async def get_item(id: str) -> dict:
        return {"id": id}

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/internal/admin")
    async def admin() -> dict:
        return {"admin": True}

    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    """Async test client that drives the ASGI app directly — no real
    network sockets are involved, so it's hermetic and fast."""
    return httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    )


def _mock_jwks(rsa_public_jwk: dict[str, Any]) -> respx.MockRouter:
    """Return a respx mock that returns the canonical JWKS body.

    ``assert_all_called=False`` is deliberate: many tests register
    this mock but exercise paths (skip routes, no-rule routes,
    missing Authorization headers) that never reach JWKS lookup.
    Failing those tests for an "unused mock" would be punishing
    the test for being correct."""
    router = respx.mock(assert_all_called=False)
    router.get(JWKS_URL).mock(
        return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}),
    )
    return router


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "mudraid_scopes.yaml"
    p.write_text(body, encoding="utf-8")
    return p


# ---- public routes (M5.7) ------------------------------------------------


@pytest.mark.asyncio
async def test_public_route_does_not_require_token(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        _yaml(routes_yaml="  - method: GET\n    path: /health\n    public: true\n"),
    )
    app = _make_app(yaml_path)

    async with _client(app) as c:
        resp = await c.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---- skip routes (M5.8) --------------------------------------------------


@pytest.mark.asyncio
async def test_skip_route_returns_404_even_with_valid_token(
    tmp_path: Path,
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """``skip: true`` hides the route from agents entirely. Even a
    fully-valid JWT must produce a 404 — the agent must NOT be able
    to discover that the admin route exists."""
    yaml_path = _write_yaml(
        tmp_path,
        _yaml(routes_yaml="  - method: POST\n    path: /internal/admin\n    skip: true\n"),
    )
    app = _make_app(yaml_path)
    token = sign_jwt(rsa_private_key, baseline_claims(audience=PLATFORM_ID, scopes=["admin:do"]))

    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.post("/internal/admin", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 404
    assert resp.json()["error_code"] == "ROUTE_NOT_FOUND"


# ---- no-rule routes ------------------------------------------------------


@pytest.mark.asyncio
async def test_route_not_in_yaml_returns_404(tmp_path: Path) -> None:
    """A route the YAML doesn't mention is treated as nonexistent
    from the agent's perspective — same shape as `skip: true` so the
    agent can't distinguish 'hidden by config' from 'no such rule'."""
    yaml_path = _write_yaml(tmp_path, _yaml())  # no routes
    app = _make_app(yaml_path)

    async with _client(app) as c:
        resp = await c.get("/items")

    assert resp.status_code == 404
    assert resp.json()["error_code"] == "ROUTE_NOT_FOUND"


# ---- missing / malformed Authorization header (M5.9 — MISSING_TOKEN) ----


@pytest.mark.asyncio
async def test_missing_authorization_header_returns_401(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        _yaml(routes_yaml="  - method: GET\n    path: /items\n    scope: items:read\n"),
    )
    app = _make_app(yaml_path)

    async with _client(app) as c:
        resp = await c.get("/items")

    assert resp.status_code == 401
    assert resp.json()["error_code"] == "MISSING_TOKEN"


@pytest.mark.asyncio
async def test_non_bearer_authorization_returns_401(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        _yaml(routes_yaml="  - method: GET\n    path: /items\n    scope: items:read\n"),
    )
    app = _make_app(yaml_path)

    async with _client(app) as c:
        resp = await c.get("/items", headers={"Authorization": "Basic abc123"})

    assert resp.status_code == 401
    assert resp.json()["error_code"] == "MISSING_TOKEN"


@pytest.mark.asyncio
async def test_empty_bearer_token_returns_401(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        _yaml(routes_yaml="  - method: GET\n    path: /items\n    scope: items:read\n"),
    )
    app = _make_app(yaml_path)

    async with _client(app) as c:
        resp = await c.get("/items", headers={"Authorization": "Bearer "})

    assert resp.status_code == 401
    assert resp.json()["error_code"] == "MISSING_TOKEN"


@pytest.mark.asyncio
async def test_bearer_scheme_is_case_insensitive(
    tmp_path: Path,
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """RFC 7235 § 2.1: credential schemes are case-insensitive.
    ``bearer`` and ``Bearer`` must both work."""
    yaml_path = _write_yaml(
        tmp_path,
        _yaml(routes_yaml="  - method: GET\n    path: /items\n    scope: items:read\n"),
    )
    app = _make_app(yaml_path)
    token = sign_jwt(rsa_private_key, baseline_claims(audience=PLATFORM_ID, scopes=["items:read"]))

    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/items", headers={"Authorization": f"bearer {token}"})

    assert resp.status_code == 200


# ---- happy path: valid token with required scope -------------------------


@pytest.mark.asyncio
async def test_valid_token_with_required_scope_reaches_handler(
    tmp_path: Path,
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        _yaml(routes_yaml="  - method: GET\n    path: /items\n    scope: items:read\n"),
    )
    app = _make_app(yaml_path)
    token = sign_jwt(rsa_private_key, baseline_claims(audience=PLATFORM_ID, scopes=["items:read"]))

    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/items", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200
    assert resp.json() == {"items": ["a", "b"]}


@pytest.mark.asyncio
async def test_parametric_path_routes_to_correct_rule(
    tmp_path: Path,
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """``/items/{id}`` rule applies to ``/items/42``. Locks the
    integration between the matcher (M5.3) and the dispatcher."""
    yaml_path = _write_yaml(
        tmp_path,
        _yaml(routes_yaml="  - method: GET\n    path: /items/{id}\n    scope: items:read\n"),
    )
    app = _make_app(yaml_path)
    token = sign_jwt(rsa_private_key, baseline_claims(audience=PLATFORM_ID, scopes=["items:read"]))

    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/items/42", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200
    assert resp.json() == {"id": "42"}


# ---- scope failures (M5.9 — MISSING_SCOPE) ------------------------------


@pytest.mark.asyncio
async def test_valid_token_without_required_scope_returns_403(
    tmp_path: Path,
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """JWT verified fine, but the route requires `items:write` and
    the token only carries `items:read`. 403, not 401 — auth was
    valid; authorisation failed."""
    yaml_path = _write_yaml(
        tmp_path,
        _yaml(routes_yaml="  - method: POST\n    path: /items\n    scope: items:write\n"),
    )
    app = _make_app(yaml_path)
    token = sign_jwt(rsa_private_key, baseline_claims(audience=PLATFORM_ID, scopes=["items:read"]))

    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.post("/items", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 403
    body = resp.json()
    assert body["error_code"] == "MISSING_SCOPE"
    assert "items:write" in body["message"]


@pytest.mark.asyncio
async def test_token_with_no_scopes_claim_at_all_is_rejected(
    tmp_path: Path,
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """A JWT minted without a ``scopes`` claim (shouldn't happen with
    the M2.8 backend fix, but defensive) must still be denied if the
    route requires a scope. Locks the defence — never default to
    'allow' on missing scope data."""
    yaml_path = _write_yaml(
        tmp_path,
        _yaml(routes_yaml="  - method: GET\n    path: /items\n    scope: items:read\n"),
    )
    app = _make_app(yaml_path)
    claims = baseline_claims(audience=PLATFORM_ID)
    del claims["scopes"]
    token = sign_jwt(rsa_private_key, claims)

    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/items", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 403
    assert resp.json()["error_code"] == "MISSING_SCOPE"


# ---- token-shape failures (M5.9) -----------------------------------------


@pytest.mark.asyncio
async def test_expired_token_returns_401_with_expired_token_code(
    tmp_path: Path,
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        _yaml(routes_yaml="  - method: GET\n    path: /items\n    scope: items:read\n"),
    )
    app = _make_app(yaml_path)
    token = sign_jwt(
        rsa_private_key,
        baseline_claims(audience=PLATFORM_ID, scopes=["items:read"], expires_in=-600),
    )

    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/items", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 401
    assert resp.json()["error_code"] == "EXPIRED_TOKEN"


@pytest.mark.asyncio
async def test_wrong_audience_returns_401_with_wrong_audience_code(
    tmp_path: Path,
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """A JWT minted for a different platform must be rejected. Locks
    the cross-platform-token-rejection contract end-to-end."""
    yaml_path = _write_yaml(
        tmp_path,
        _yaml(routes_yaml="  - method: GET\n    path: /items\n    scope: items:read\n"),
    )
    app = _make_app(yaml_path)
    token = sign_jwt(
        rsa_private_key,
        baseline_claims(audience="plt-DIFFERENT-PLATFORM", scopes=["items:read"]),
    )

    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/items", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 401
    assert resp.json()["error_code"] == "WRONG_AUDIENCE"


@pytest.mark.asyncio
async def test_unparseable_token_returns_401_with_invalid_token_code(
    tmp_path: Path,
    rsa_public_jwk: dict[str, Any],
) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        _yaml(routes_yaml="  - method: GET\n    path: /items\n    scope: items:read\n"),
    )
    app = _make_app(yaml_path)

    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/items", headers={"Authorization": "Bearer not.a.jwt"})

    assert resp.status_code == 401
    assert resp.json()["error_code"] == "INVALID_TOKEN"


# ---- JWKS unavailable (M5.9 — JWKS_UNAVAILABLE) -------------------------


@pytest.mark.asyncio
async def test_jwks_network_failure_returns_500(
    tmp_path: Path,
    rsa_private_key: RSAPrivateKey,
) -> None:
    """JWKS unreachable = we can't verify any signature. 500, not
    401 — this is an operator-side problem; the agent's credentials
    might be fine. The distinction matters for retry semantics:
    agents typically retry 5xx but not 401."""
    yaml_path = _write_yaml(
        tmp_path,
        _yaml(routes_yaml="  - method: GET\n    path: /items\n    scope: items:read\n"),
    )
    app = _make_app(yaml_path)
    token = sign_jwt(rsa_private_key, baseline_claims(audience=PLATFORM_ID, scopes=["items:read"]))

    with respx.mock() as r:
        r.get(JWKS_URL).mock(side_effect=httpx.ConnectError("network down"))

        async with _client(app) as c:
            resp = await c.get("/items", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 500
    assert resp.json()["error_code"] == "JWKS_UNAVAILABLE"


# ---- response shape contract --------------------------------------------


@pytest.mark.asyncio
async def test_error_response_has_locked_body_shape(tmp_path: Path) -> None:
    """Locked shape: ``{"error_code": "...", "message": "..."}``.
    No extra fields, no stack traces, no upstream error library
    detail. Platforms build alerting on the ``error_code`` field —
    if its name or position changes silently, their dashboards
    break."""
    yaml_path = _write_yaml(
        tmp_path,
        _yaml(routes_yaml="  - method: GET\n    path: /items\n    scope: items:read\n"),
    )
    app = _make_app(yaml_path)

    async with _client(app) as c:
        resp = await c.get("/items")

    body = resp.json()
    assert set(body.keys()) == {"error_code", "message"}
    assert isinstance(body["error_code"], str) and body["error_code"]
    assert isinstance(body["message"], str) and body["message"]


# ---- bootstrap behaviour -------------------------------------------------


@pytest.mark.asyncio
async def test_lazy_bootstrap_does_not_fetch_jwks_until_a_scoped_route_is_hit(
    tmp_path: Path,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """A request that only hits public routes should never touch
    the JWKS endpoint. Without this, every cold-boot health check
    would needlessly fetch JWKS."""
    yaml_path = _write_yaml(
        tmp_path,
        _yaml(
            routes_yaml=(
                "  - method: GET\n    path: /health\n    public: true\n"
                "  - method: GET\n    path: /items\n    scope: items:read\n"
            )
        ),
    )
    app = _make_app(yaml_path)

    with respx.mock(assert_all_called=False) as r:
        jwks_route = r.get(JWKS_URL).mock(
            return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}),
        )

        async with _client(app) as c:
            await c.get("/health")
            await c.get("/health")
            await c.get("/health")

        assert jwks_route.call_count == 0, "public-only traffic must not trigger JWKS fetches"


@pytest.mark.asyncio
async def test_repeated_requests_bootstrap_yaml_only_once(
    tmp_path: Path,
    rsa_private_key: RSAPrivateKey,
    rsa_public_jwk: dict[str, Any],
) -> None:
    """The YAML is read on the first dispatch and never again.
    Locks the load-once contract (locked decision D4 mirror —
    'redeploy to pick up changes')."""
    yaml_path = _write_yaml(
        tmp_path,
        _yaml(routes_yaml="  - method: GET\n    path: /items\n    scope: items:read\n"),
    )
    app = _make_app(yaml_path)
    token = sign_jwt(rsa_private_key, baseline_claims(audience=PLATFORM_ID, scopes=["items:read"]))

    # Mutate the YAML between requests — if the middleware re-read
    # it, the second request would see the new (broken) shape and
    # 404 the route.
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            r1 = await c.get("/items", headers={"Authorization": f"Bearer {token}"})

            yaml_path.write_text("platform_id: x\nversion: 1\nroutes: []\n", encoding="utf-8")

            r2 = await c.get("/items", headers={"Authorization": f"Bearer {token}"})

    # Both requests succeed against the original YAML — confirming
    # the middleware didn't reload mid-process.
    assert r1.status_code == 200
    assert r2.status_code == 200
