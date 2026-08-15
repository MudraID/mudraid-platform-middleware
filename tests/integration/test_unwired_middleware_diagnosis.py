"""Diagnostic: what a plain REST surface returns, wired and unwired.

WHY THIS FILE EXISTS. A tester protected a plain REST API
(``POST /api/v1/tasks`` -> ``tasks:write``) with this package, granted their
agent only ``tasks:read``, and then replayed the request carrying ONLY the raw
agent credential headers — no ``Authorization: Bearer`` at all — and got a
``201`` with the resource created. The portal showed the surface Verified and
Active with both scopes and both route mappings configured, so the natural
reading was that the middleware had authorized a request it should have denied.

It had not. This file pins the four outcomes the middleware CAN produce for
that exact request, so the one outcome it cannot produce is written down as a
test rather than as an argument:

  - wired + no Bearer, raw credential headers present -> 401 MISSING_TOKEN.
    The raw ``X-MudraID-API-Key-ID`` / ``X-MudraID-Secret`` pair is not an
    authentication method here. No branch reads those names; a long-lived agent
    credential is never a substitute for a minted token.
  - wired + the route absent from the YAML (a prefix or trailing-slash typo)
    -> 404 ROUTE_NOT_FOUND. An unmatched route is DENIED, not allowed: a
    mis-typed path removes the route from the agent's world, it does not remove
    the guard.
  - wired + an unloadable ``mudraid_scopes.yaml`` -> 500 MIDDLEWARE_NOT_READY.
  - NOT wired -> 201, the handler runs. This is the only configuration of the
    four that reproduces what staging returned.

The positive control sits alongside them: the same wired app, a properly
scoped token, still creates the task. A middleware that denied everything would
satisfy the three negatives and be useless.

These tests characterize shipped behaviour; they are not the red control for a
fix. Nothing in the enforcement path is changed by the commit that adds them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI
from httpx import ASGITransport
from pydantic import BaseModel

from mudraid_platform_middleware import MudraIDMiddleware
from tests.conftest import baseline_claims, sign_jwt

JWKS_URL = "https://api.mudraid.test/.well-known/jwks.json"
PLATFORM_ID = "11111111-2222-3333-4444-555555555555"

# The two headers the tester sent. Values here are obviously synthetic — the
# point of the test is the header NAMES, which no code path in this package
# reads.
RAW_CREDENTIAL_HEADERS = {
    "X-MudraID-API-Key-ID": "muid_kid_example_not_a_real_key",
    "X-MudraID-Secret": "not-a-real-secret",
}

# The portal-shaped YAML for the tester's surface: two scopes, two mappings.
TASKS_YAML = f"""platform_id: {PLATFORM_ID}
version: 1
routes:
  - method: GET
    path: /api/v1/tasks
    scope: tasks:read
  - method: POST
    path: /api/v1/tasks
    scope: tasks:write
"""


class _TaskIn(BaseModel):
    title: str


def _routes(app: FastAPI) -> FastAPI:
    """The tester's app shape: one GET list, one POST create returning 201."""
    tasks: list[dict[str, Any]] = []

    @app.get("/api/v1/tasks")
    def list_tasks() -> dict[str, Any]:
        return {"tasks": tasks}

    @app.post("/api/v1/tasks", status_code=201)
    def create_task(payload: _TaskIn) -> dict[str, Any]:
        task = {"id": len(tasks) + 1, "title": payload.title}
        tasks.append(task)
        return task

    return app


def _wired_app(yaml_path: Path) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        MudraIDMiddleware,
        scopes_yaml_path=str(yaml_path),
        jwks_url=JWKS_URL,
    )
    return _routes(app)


def _unwired_app() -> FastAPI:
    """The same app with the package installed but never added to the stack."""
    return _routes(FastAPI())


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


def _mock_jwks(rsa_public_jwk: dict[str, Any]) -> respx.MockRouter:
    router = respx.mock(assert_all_called=False)
    router.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"keys": [rsa_public_jwk]}))
    return router


def _write_yaml(tmp_path: Path, body: str = TASKS_YAML) -> Path:
    p = tmp_path / "mudraid_scopes.yaml"
    p.write_text(body, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# (b) refuted — raw agent credentials are not an authentication method
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_credential_headers_do_not_authenticate_a_scope_gated_route(
    tmp_path: Path, rsa_public_jwk: dict[str, Any]
) -> None:
    """The tester's exact request against a WIRED middleware: 401, not 201."""
    app = _wired_app(_write_yaml(tmp_path))
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.post(
                "/api/v1/tasks",
                headers=RAW_CREDENTIAL_HEADERS,
                json={"title": "Testing Declared operating limits"},
            )
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "MISSING_TOKEN"


@pytest.mark.asyncio
async def test_raw_credential_headers_do_not_substitute_for_a_missing_scope(
    tmp_path: Path, rsa_private_key, rsa_public_jwk: dict[str, Any]
) -> None:
    """A read-only agent presenting its raw credential alongside a valid
    read-scoped token still cannot write: the credential adds nothing."""
    app = _wired_app(_write_yaml(tmp_path))
    token = sign_jwt(
        rsa_private_key,
        baseline_claims(audience=PLATFORM_ID, scopes=["tasks:read"]),
    )
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.post(
                "/api/v1/tasks",
                headers={**RAW_CREDENTIAL_HEADERS, "Authorization": f"Bearer {token}"},
                json={"title": "Testing Declared operating limits"},
            )
    assert resp.status_code == 403
    assert resp.json()["error_code"] == "MISSING_SCOPE"


# ---------------------------------------------------------------------------
# (c) refuted as a fail-open — an unmatched route denies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/tasks/",  # trailing slash: a different route, deliberately
        "/tasks",  # the mapping's prefix mis-typed
        "/api/v2/tasks",  # wrong version segment
        "/API/v1/tasks",  # case differs; URL paths are case-sensitive
    ],
)
@pytest.mark.asyncio
async def test_a_path_no_rule_matches_is_denied_not_allowed(
    path: str, tmp_path: Path, rsa_public_jwk: dict[str, Any]
) -> None:
    """A configured surface fails CLOSED on a path its YAML does not cover.

    This is the fail-open question, answered: an operator who mis-types a path
    loses the ROUTE, not the GUARD. 404 rather than 403 is deliberate — it does
    not tell the agent that a route it may not use exists.
    """
    app = _wired_app(_write_yaml(tmp_path))
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.post(path, headers=RAW_CREDENTIAL_HEADERS, json={"title": "x"})
    assert resp.status_code == 404
    assert resp.json()["error_code"] == "ROUTE_NOT_FOUND"


@pytest.mark.asyncio
async def test_a_surface_whose_yaml_will_not_load_denies_every_request(
    tmp_path: Path, rsa_public_jwk: dict[str, Any]
) -> None:
    """The other configuration failure: 500, and the handler never runs."""
    app = _wired_app(_write_yaml(tmp_path, body="platform_id: x\nversion: 1\n"))
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.post("/api/v1/tasks", json={"title": "x"})
    assert resp.status_code == 500
    assert resp.json()["error_code"] == "MIDDLEWARE_NOT_READY"


# ---------------------------------------------------------------------------
# (a) confirmed — only an unwired app produces the observed 201
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unwired_app_creates_the_task_with_no_token_at_all() -> None:
    """The staging observation, reproduced: no middleware in the ASGI stack.

    Installing the package and writing ``mudraid_scopes.yaml`` does not put
    anything in the request path. ``add_middleware`` is what does.
    """
    app = _unwired_app()
    async with _client(app) as c:
        resp = await c.post(
            "/api/v1/tasks",
            headers=RAW_CREDENTIAL_HEADERS,
            json={"title": "Testing Declared operating limits"},
        )
    assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Positive control — a correctly wired middleware still allows a scoped write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_correctly_scoped_write_still_reaches_the_handler(
    tmp_path: Path, rsa_private_key, rsa_public_jwk: dict[str, Any]
) -> None:
    app = _wired_app(_write_yaml(tmp_path))
    token = sign_jwt(
        rsa_private_key,
        baseline_claims(audience=PLATFORM_ID, scopes=["tasks:read", "tasks:write"]),
    )
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.post(
                "/api/v1/tasks",
                headers={"Authorization": f"Bearer {token}"},
                json={"title": "Testing Declared operating limits"},
            )
    assert resp.status_code == 201
    assert resp.json()["title"] == "Testing Declared operating limits"


@pytest.mark.asyncio
async def test_the_read_scoped_agent_can_still_read(
    tmp_path: Path, rsa_private_key, rsa_public_jwk: dict[str, Any]
) -> None:
    app = _wired_app(_write_yaml(tmp_path))
    token = sign_jwt(
        rsa_private_key,
        baseline_claims(audience=PLATFORM_ID, scopes=["tasks:read"]),
    )
    with _mock_jwks(rsa_public_jwk):
        async with _client(app) as c:
            resp = await c.get("/api/v1/tasks", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
