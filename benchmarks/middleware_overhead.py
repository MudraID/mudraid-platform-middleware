"""M6.5 — Middleware per-request overhead baseline.

Measures the cost the MudraID middleware adds on top of a vanilla
FastAPI handler. The harness mirrors the M5.11 integration test setup
exactly — same ASGI transport, same RSA-signed JWTs, same JWKS mock —
so the numbers are reproducible from a fresh clone with no live backend.

Two scenarios are measured:

  * **public route** — middleware matches the path, sees ``public:
    true``, returns control to the handler. No JWT, no JWKS, no scope
    check. Lower bound on middleware overhead.

  * **scope-gated route** — middleware matches the path, extracts the
    Bearer token, verifies the RS256 signature against the cached
    JWKS, checks every claim (iss/aud/exp/nbf/iat) and the required
    scope. This is the steady-state production path; the JWKS cache
    is warmed by one preceding request so cold-cache fetch time does
    not contaminate the measurement.

The baseline pair (no-middleware FastAPI app, same routes) is also
measured so the *delta* — the actual overhead the middleware adds —
is visible, not just the raw timings.

Run::

    python -m benchmarks.middleware_overhead              # default: 2000 iterations
    python -m benchmarks.middleware_overhead --iters 5000

The script is intentionally synchronous + single-process. The point is
to characterise a single request's overhead so operators can capacity-
plan, not to measure server throughput under concurrency.

Numbers are written to stdout in a copy-pasteable table; record the
result against the current commit hash so regressions are caught at
review time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import jwt as pyjwt
import respx
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from httpx import ASGITransport
from jwt.algorithms import RSAAlgorithm

from mudraid_platform_middleware import MudraIDMiddleware

JWKS_URL = "https://api.mudraid.test/.well-known/jwks.json"
PLATFORM_ID = "11111111-2222-3333-4444-555555555555"


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
"""


def _build_routes(app: FastAPI) -> None:
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/items")
    def list_items() -> dict[str, Any]:
        return {"items": [{"id": "item-1", "name": "Widget"}]}


def _make_middleware_app(yaml_path: Path) -> FastAPI:
    """The sample-platform shape with the real middleware applied."""
    app = FastAPI()
    app.add_middleware(
        MudraIDMiddleware,
        scopes_yaml_path=str(yaml_path),
        jwks_url=JWKS_URL,
    )
    _build_routes(app)
    return app


def _make_bare_app() -> FastAPI:
    """Identical routes, no middleware — gives the dispatch-only floor
    so the middleware *delta* is what's reported, not absolute timings
    (which depend mostly on FastAPI/Starlette/asyncio scheduling)."""
    app = FastAPI()
    _build_routes(app)
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    )


def _build_keys() -> tuple[rsa.RSAPrivateKey, dict[str, Any]]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    public_jwk["kid"] = "bench-kid"
    public_jwk["use"] = "sig"
    public_jwk["alg"] = "RS256"
    return private_key, public_jwk


def _sign_token(private_key: rsa.RSAPrivateKey) -> str:
    now = int(time.time())
    claims = {
        "iss": "mudraid-identity",
        "sub": "agent-bench",
        "aud": PLATFORM_ID,
        "scopes": ["items:read"],
        "iat": now,
        "nbf": now,
        "exp": now + 900,
        "jti": "jti-bench",
    }
    return pyjwt.encode(
        claims,
        key=private_key,
        algorithm="RS256",
        headers={"kid": "bench-kid"},
    )


async def _time_loop(
    client: httpx.AsyncClient,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    iters: int,
) -> list[float]:
    """Send ``iters`` sequential requests and return per-request wall
    times in microseconds.

    Sequential, not concurrent: we're measuring a single request's
    overhead, not throughput. Concurrency would inject scheduler
    contention into every sample."""
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        resp = await client.get(path, headers=headers or {})
        elapsed = (time.perf_counter() - t0) * 1_000_000
        if resp.status_code >= 400:
            raise RuntimeError(
                f"unexpected non-2xx during benchmark: {resp.status_code} {resp.text}"
            )
        samples.append(elapsed)
    return samples


def _summarise(samples: list[float]) -> dict[str, float]:
    samples_sorted = sorted(samples)
    return {
        "p50": statistics.median(samples_sorted),
        "p95": samples_sorted[int(0.95 * (len(samples_sorted) - 1))],
        "p99": samples_sorted[int(0.99 * (len(samples_sorted) - 1))],
        "min": samples_sorted[0],
        "max": samples_sorted[-1],
        "mean": statistics.fmean(samples_sorted),
    }


def _format_row(label: str, stats: dict[str, float]) -> str:
    return (
        f"  {label:<32} "
        f"p50={stats['p50']:7.1f}us  "
        f"p95={stats['p95']:7.1f}us  "
        f"p99={stats['p99']:7.1f}us  "
        f"min={stats['min']:7.1f}us  "
        f"max={stats['max']:7.1f}us"
    )


async def _run(iters: int, warmup: int) -> None:
    private_key, public_jwk = _build_keys()
    token = _sign_token(private_key)

    yaml_path = Path("_bench_scopes.yaml")
    yaml_path.write_text(_yaml_body(), encoding="utf-8")

    try:
        # --- baseline (no middleware) ------------------------------------
        bare_app = _make_bare_app()
        async with _client(bare_app) as bare_client:
            await _time_loop(bare_client, "/health", iters=warmup)
            bare_public = await _time_loop(bare_client, "/health", iters=iters)
            bare_scoped = await _time_loop(bare_client, "/api/v1/items", iters=iters)

        # --- middleware path with respx-mocked JWKS ----------------------
        with respx.mock(assert_all_called=False) as router:
            router.get(JWKS_URL).mock(
                return_value=httpx.Response(200, json={"keys": [public_jwk]}),
            )
            mw_app = _make_middleware_app(yaml_path)
            async with _client(mw_app) as mw_client:
                # warm up — first scoped request triggers YAML load,
                # JWKS fetch, validator init. We don't want those one-
                # off costs polluting the steady-state sample.
                await _time_loop(
                    mw_client,
                    "/api/v1/items",
                    headers={"Authorization": f"Bearer {token}"},
                    iters=warmup,
                )
                await _time_loop(mw_client, "/health", iters=warmup)

                mw_public = await _time_loop(mw_client, "/health", iters=iters)
                mw_scoped = await _time_loop(
                    mw_client,
                    "/api/v1/items",
                    headers={"Authorization": f"Bearer {token}"},
                    iters=iters,
                )
    finally:
        yaml_path.unlink(missing_ok=True)

    bare_public_s = _summarise(bare_public)
    bare_scoped_s = _summarise(bare_scoped)
    mw_public_s = _summarise(mw_public)
    mw_scoped_s = _summarise(mw_scoped)

    print()
    print("=" * 96)
    print(f"  MudraID middleware overhead — {iters} iters per scenario " f"(warmup={warmup})")
    print(f"  Python {sys.version.split()[0]} on " f"{platform.system()} {platform.release()}")
    print("=" * 96)
    print()
    print("  Per-request wall time:")
    print(_format_row("baseline /health (no MW)", bare_public_s))
    print(_format_row("baseline /api/v1/items (no MW)", bare_scoped_s))
    print(_format_row("middleware /health (public)", mw_public_s))
    print(_format_row("middleware /api/v1/items (scoped)", mw_scoped_s))
    print()
    print("  Middleware delta (mw - bare, p50):")
    print(f"    public route:      " f"{mw_public_s['p50'] - bare_public_s['p50']:+7.1f}us")
    print(f"    scope-gated route: " f"{mw_scoped_s['p50'] - bare_scoped_s['p50']:+7.1f}us")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure MudraIDMiddleware per-request overhead.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=2000,
        help="samples per scenario (default: 2000)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=200,
        help="warmup iterations before measurement (default: 200)",
    )
    args = parser.parse_args()
    asyncio.run(_run(iters=args.iters, warmup=args.warmup))


if __name__ == "__main__":
    main()
