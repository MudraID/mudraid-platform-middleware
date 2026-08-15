# mudraid-platform-middleware

FastAPI / Starlette middleware for [MudraID](https://mudraid.ai) — a trust
layer for AI agents.

Drop it into your FastAPI app, point it at your `mudraid_scopes.yaml`,
and every scope-gated route is enforced before reaching your handler.
**No decorators, no per-route auth code, zero changes to existing route files.**

```python
# MUDRAID_JWKS_URL must be set in the environment — there is no default,
# and the middleware refuses to construct without it.
from fastapi import FastAPI
from mudraid_platform_middleware import MudraIDMiddleware

app = FastAPI()
app.add_middleware(MudraIDMiddleware)

@app.get("/api/v1/items")
def list_items():
    return {"items": [...]}   # gets to here only after JWT + scope check pass
```

That's the entire integration. Two lines, one environment variable, and a
YAML file in your project root — then [verify it is actually in the
stack](#4-verify-the-integration-before-trusting-it), because adding the
package is not the same act as adding the middleware.

---

## What the middleware actually does

For every incoming request:

1. **Match** `(method, path)` against rules from `mudraid_scopes.yaml`.
2. **Bypass** routes flagged `public: true` (health checks, etc.).
3. **404** routes flagged `skip: true` or absent from the YAML —
   they're invisible to agents.
4. **Extract** the `Authorization: Bearer <jwt>` header.
5. **Verify** signature, `iss`, `aud == platform_id`, `exp`, `nbf`, `iat`
   using MudraID's JWKS (cached 1 h; reactive-refresh on unknown `kid`
   so MudraID key rotation is invisible).
6. **Check** the route's required scope is in the JWT's `scopes` claim.
7. **Forward** to the route handler if every gate passes; otherwise
   return a structured `{"error_code", "message"}` 401 / 403 / 404 / 500
   without the handler ever being called.

All of this happens before your route code runs — your handler only
sees authenticated, authorised, scope-checked requests, *once the
middleware is actually in the ASGI stack*. Confirm that first:
see [Verify the integration before trusting
it](#4-verify-the-integration-before-trusting-it).

---

## Installation

```bash
pip install mudraid-platform-middleware==1.1.0
```

Requires Python 3.10+. Brings in `pyjwt[crypto]`, `cryptography`,
`httpx`, `pyyaml`, and `starlette` as runtime deps.

---

## Quickstart

### 1. Register your platform on MudraID

Sign in to the MudraID portal (or call the API), create a new
platform, verify your domain ownership (DNS TXT record), and define
the scopes your routes will require.

### 2. Download `mudraid_scopes.yaml`

The portal generates a file mapping every one of your routes to a
scope, a `public: true` bypass, or a `skip: true` 404. Drop it into
your project root next to `main.py`.

```yaml
platform_id: 4f8e9c1d-2b4f-5e6a-7b8c-9d0e1f2a3b4c
version: 1
routes:
  - method: GET
    path: /health
    public: true
  - method: GET
    path: /api/v1/items
    scope: items:read
  - method: GET
    path: /api/v1/items/{item_id}
    scope: items:read
  - method: POST
    path: /api/v1/items
    scope: items:write
```

### 3. Set the JWKS URL and add two lines to your FastAPI app

`MUDRAID_JWKS_URL` is required and has no default. Take it from the credential
screen in the portal — it is the `/.well-known/jwks.json` path on the MudraID
environment your platform is registered with:

```bash
export MUDRAID_JWKS_URL="<the JWKS URL from your credential screen>"
```

```python
from fastapi import FastAPI
from mudraid_platform_middleware import MudraIDMiddleware

app = FastAPI()
app.add_middleware(MudraIDMiddleware)
```

Pass it in code instead if you would rather not use the environment:
`app.add_middleware(MudraIDMiddleware, jwks_url="https://…/.well-known/jwks.json")`.
With neither set, the middleware raises `ValueError` when Starlette builds the
ASGI stack — it will not boot and quietly accept nothing.

That's the entire change.

Your existing route handlers stay untouched — no decorators, no auth
parameters, no per-route checks. The middleware enforces the rules
in `mudraid_scopes.yaml` before requests reach them.

### 4. Verify the integration before trusting it

`curl` a scope-gated route with no `Authorization` header. A wired middleware
answers `401 MISSING_TOKEN` — or `404 ROUTE_NOT_FOUND` if the path does not
match your YAML.

```bash
curl -i http://localhost:8000/api/v1/items
# → 401 {"error_code":"MISSING_TOKEN", ...}     the middleware is in the stack
# → 404 {"error_code":"ROUTE_NOT_FOUND", ...}   in the stack; your YAML does not
#                                               cover this path
```

**If your own handler runs instead — a `200`, a `201`, your JSON — the
middleware is not in the ASGI stack.** Installing the package and downloading
`mudraid_scopes.yaml` does not put it there; `add_middleware` does. There is no
other state that produces your handler's response: wired with no token is `401`,
wired with the wrong scope is `403`, wired with a path your YAML does not cover
is `404`, and wired with unreadable YAML is `500`. Reaching the handler means
nothing enforced.

**The portal cannot detect this for you.** **Verified** means your domain was
proven by a DNS TXT record, and **Active** means the surface may use the API.
Both are set the moment DNS verification succeeds. Neither means MudraID has
observed a single enforced request.

---

## Configuration reference

| Setting | Env var | Constructor kwarg | Default |
|---|---|---|---|
| YAML path | — | `scopes_yaml_path=` | `./mudraid_scopes.yaml` (cwd) |
| MudraID JWKS URL | `MUDRAID_JWKS_URL` | `jwks_url=` | **none — required; construction raises without it** |

**There is no default JWKS URL, deliberately.** A MudraID hostname compiled
into the package would make the build environment-specific, and — worse — an
unconfigured platform would fetch its *signing keys* from a host nobody chose,
so whoever answered would be deciding which tokens the platform accepts. In V1
mode the middleware therefore raises `ValueError` when it is constructed with
neither `jwks_url=` nor `MUDRAID_JWKS_URL` set, naming the setting. Starlette
constructs middleware when it builds the ASGI stack, so an unconfigured app
fails there rather than booting and rejecting every token afterwards.

Point it at the `/.well-known/jwks.json` path on the MudraID environment your
platform is registered with — the portal prints it on the credential screen.

Everything else (cache TTLs, clock-skew leeway, request-body limits)
is internal and stable in v1.

### V2 mode (`mode="v2"`) — live tool-action enforcement for MCP servers

> **V2 is for MCP servers only. Read this before configuring it.**
>
> MudraID V2 provides live action authorization for MCP Streamable HTTP tool
> calls. MCP transport and session requests remain subject to the MCP server's
> normal HTTP/OAuth authentication. For ordinary REST APIs, use MudraID's
> route/scope middleware, which enforces the configured HTTP method and
> route—including GET and DELETE.
>
> This is not a preference. The V2 control loop treats `GET`/`HEAD`/`OPTIONS` as
> Streamable-HTTP transport and `DELETE` as MCP session control, and makes no
> `/decide` call for any of them. That is correct for MCP Streamable HTTP and
> wrong for a REST API, where a `GET` reads and a `DELETE` destroys. Point V2 at
> your MCP endpoint; point V1 at everything else.

V2 adds signed-bundle action enforcement on top of V1's token and scope
checking, for MCP surfaces: every `tools/call` on a protected surface resolves
to a canonical action and gets a live authority decision, and anything that
cannot be decided is denied rather than allowed. MCP control and discovery
messages — `initialize`, `ping`, `tools/list` and the rest of `public_methods` —
pass a protected surface without a decision; they do not receive a `/decide`
verdict, and nothing here authenticates the MCP transport or session on your
server's behalf.

`HttpDecideClient` is the supported implementation — it fetches and verifies the
signed bundle, keeps it refreshed, resolves tool names to canonical actions, and
makes the authenticated `/decide` call.

Two settings configure it, and selecting an environment means changing them —
never changing the package, the build or the code path:

| Setting | What it is |
|---|---|
| `MUDRAID_BASE_URL` | The MudraID origin for your environment. The client derives every path it needs from this. |
| `MUDRAID_ADAPTER_TOKEN` | Your adapter's bearer, shown once at registration. It must differ between environments — reusing a staging credential in production breaks environment isolation. |

Both are printed by the MudraID portal when you register an adapter. You do not
configure a signing secret or a service secret: bundle signatures are verified
with public keys the client fetches for itself.

```python
from mudraid_platform_middleware import HttpDecideClient, MudraIDMiddleware, V2Config

decide_client = HttpDecideClient(
    base_url=os.environ["MUDRAID_BASE_URL"],
    adapter_token=os.environ["MUDRAID_ADAPTER_TOKEN"],
)

@app.on_event("startup")
async def _start_enforcement() -> None:
    # Fetches and verifies the first bundle, then refreshes in the background.
    # Until one verifies, protected surfaces fail CLOSED — never open.
    await decide_client.start()

@app.on_event("shutdown")
async def _stop_enforcement() -> None:
    await decide_client.aclose()

app.add_middleware(
    MudraIDMiddleware,
    mode="v2",
    v2_config=V2Config(decide_client=decide_client, protected_paths=("/mcp",)),
)
```

> **Staging-qualified, not generally production-ready.** The decision exchange
> is versioned in both directions and every response is bound to the request that
> asked for it, but the response is authenticated by the transport alone. A
> signed decision response is required before this is described as a finished
> public package. Deploy it behind a version you control, and measure it
> end-to-end against your own authority first.

| Setting | `V2Config` field | Default |
|---|---|---|
| Protected surfaces | `protected_paths=` | `None` (every route runs the V2 loop) |
| Bounded body limit | `max_body_bytes=` | `1048576` (1 MiB) — Kong defaults to 128 KiB |
| Control/discovery allowlist | `public_methods=` | `DEFAULT_PUBLIC_METHODS` |
| `/decide` deadline | `decide_timeout_sec=` | `2.0` |

**Set `protected_paths` to your MCP Streamable HTTP endpoints — for example
`("/mcp",)`.** The `None` default runs the V2 loop on every route the app
serves, which is only correct when every route on the app *is* an MCP
Streamable HTTP endpoint. On an app that also serves ordinary REST routes,
leaving it `None` applies MCP framing rules to routes that are not MCP: name the
MCP endpoints explicitly and enforce the REST routes with V1.

**`protected_paths` matches whole path segments, not characters.** A configured
`/mcp` protects `/mcp`, `/mcp/` and `/mcp/tools` — and does *not* protect
`/mcpfoo` or `/mcp-metrics`, which are different routes that merely start with
the same letters:

| request | `/mcp` configured |
|---|---|
| `/mcp` · `/mcp/` · `/mcp/tools` · `/mcp//tools` | protected |
| `/mcp?x=1` · `/%6dcp/messages` · `/mcp/../mcp/tools` | protected — every spelling is tested |
| `/mcpfoo` · `/mcp-evil` · `/mcpevil/steal` · `/MCP` | **not** protected |

The Kong plugin applies the identical rule, and a contract test runs this table
through both implementations so they cannot drift apart.

Entries are validated at construction. A prefix whose meaning depends on
interpretation — `/mcp/../admin`, `/%6dcp`, `/mcp?x=1` — raises `ValueError`
there rather than resolving to something you did not write. A trailing slash is
accepted and normalized away.

**`public_methods` differs from the Kong plugin's default, deliberately.** This
middleware defaults to the portable contract's allowlist — `initialize`, `ping`,
`tools/list`, `resources/list`, `prompts/list` — while `mudraid-enforce`'s
schema defaults to the narrower `initialize`, `ping`, `tools/list`. A Kong
surface left on its default therefore **denies** `resources/list` and
`prompts/list` (403 `ENFORCE_MESSAGE_NOT_ALLOWED`) where this middleware on its
default passes them.

If you need the two adapters to agree, set it explicitly on both rather than
relying on either default:

```python
from mudraid_platform_middleware.v2 import KONG_DEFAULT_PUBLIC_METHODS

V2Config(decide_client=client, public_methods=KONG_DEFAULT_PUBLIC_METHODS)
```

An empty set is honoured as an empty set — the strictest configuration, where
every message other than `tools/call` and `notifications/*` denies.

**Two defaults differ between the adapters**, and both are configurable on both
sides. If you rely on a workload behaving identically on Kong and here, set them
explicitly rather than trusting either default:

| Setting | Kong plugin | This middleware |
|---|---|---|
| `public_methods` | `initialize`, `ping`, `tools/list` | those plus `resources/list`, `prompts/list` |
| body limit | 128 KiB | 1 MiB |

A contract test pins this inventory, so a change to either default fails until
this table is updated in the same commit.

**`decide_timeout_sec` is the middleware's own deadline**, applied around your
`DecideClient` whether or not it sets one of its own. It expires deny-closed
(503 `ENFORCE_DECIDE_UNAVAILABLE`), matching the plugin's `decide_timeout_ms`
default of 2000. Set it to `None` only if your client is known to bound itself.

### If you registered an adapter declaring the `http` protocol mode

`mcp_streamable_http` is now the only protocol mode a V2 enforcement adapter may
declare. A bare `http` mode is no longer accepted, and an existing record that
carries it is **not** silently reinterpreted as MCP — a record that says `http`
described a surface V2 cannot correctly enforce, and quietly reading it as
something else would leave a REST API's `GET` and `DELETE` unauthorized while
the record claimed otherwise. The record fails validation on its next write,
visibly.

Two ways forward, both operator decisions rather than automatic ones:

- **The surface is an ordinary REST API.** Move it to the V1 route/scope
  middleware, which enforces the configured HTTP method and route including
  `GET` and `DELETE`, and retire the V2 adapter record.
- **The surface really is MCP Streamable HTTP.** Reclassify it: re-register the
  adapter with `mcp_streamable_http` and set `protected_paths` to the MCP
  endpoints, e.g. `/mcp`.

```python
# Local development against the docker-compose stack:
app.add_middleware(
    MudraIDMiddleware,
    scopes_yaml_path="./mudraid_scopes.yaml",
    jwks_url="http://localhost:8011/.well-known/jwks.json",
)
```

---

## Error responses

All errors come back as a single JSON shape so platforms can build
alerting on a stable contract:

```json
{
  "error_code": "MISSING_SCOPE",
  "message": "required scope 'items:write' not present in token"
}
```

| Status | `error_code` | Cause |
|---|---|---|
| 401 | `MISSING_TOKEN` | `Authorization` header missing or not `Bearer` |
| 401 | `INVALID_TOKEN` | Malformed JWT or invalid signature |
| 401 | `EXPIRED_TOKEN` | `exp` claim is in the past |
| 401 | `TOKEN_NOT_YET_VALID` | `nbf` claim is in the future |
| 401 | `WRONG_AUDIENCE` | `aud` doesn't match this platform's `platform_id` |
| 401 | `WRONG_ISSUER` | `iss` is not `mudraid-identity` |
| 403 | `MISSING_SCOPE` | JWT valid but required scope absent |
| 404 | `ROUTE_NOT_FOUND` | No YAML rule OR `skip: true` (intentionally indistinguishable) |
| 500 | `JWKS_UNAVAILABLE` | Couldn't fetch JWKS to verify (operator-side, not credential-side) |
| 500 | `MIDDLEWARE_NOT_READY` | YAML couldn't be loaded or parsed |

Behind that contract is a unit suite covering every failure shape, the JWKS
rotation and rate-limit paths, concurrency safety, and an anti-leak guarantee
that no JWT or secret ever appears in middleware logs. Run it with
`pytest` from `sdks/mudraid-middleware-python` for the current count — a number
written here goes stale the first time anyone adds a test, and a stale number
reads as a measurement.

---

## Diagnostics

Enable the middleware's structured logs to watch enforcement happen:

```python
import logging
logging.getLogger("mudraid_platform_middleware").setLevel(logging.INFO)
```

Bootstrap, JWKS fetches, 401 retries, and 500-level events surface at
INFO/WARNING.

---

## Performance

Per-request overhead the middleware adds on top of a vanilla FastAPI
handler, measured against the sample-platform route shape with a
warm JWKS cache. Numbers are µs of wall time per request, sequential
(single-process — capacity-planning numbers, not throughput).

| Route type                  | Middleware p50 | Middleware p99 | Delta vs. bare app (p50) |
|-----------------------------|---------------:|---------------:|-------------------------:|
| `public: true`              |         527 µs |         811 µs |                  +130 µs |
| Scope-gated (RS256 + scope) |         722 µs |        1403 µs |                  +310 µs |

> Measured on Python 3.14, Windows 11, 2000 iterations per scenario,
> 200-iteration warm-up. Reproduce with
> `python -m benchmarks.middleware_overhead --iters 2000` from the
> package root.

The scope-gated delta is dominated by RS256 signature verification.
JWKS fetch happens once at first request and is not part of the
steady-state cost.

---

## Operator playbook

### Updating scopes / routes

The middleware loads `mudraid_scopes.yaml` once at first request and
holds the result in memory. To pick up YAML changes: **redeploy**
(or restart your process). Hot-reload is intentionally not supported
in v1 — the single-source-of-truth contract between portal and
running middleware is easier to reason about with a restart in the
middle.

### MudraID key rotation

When MudraID rotates its signing key, the middleware:

1. Receives a JWT signed under the new `kid`
2. Looks up the `kid` in its cached JWKS — not found
3. Re-fetches JWKS (single in-flight refresh under contention)
4. Verifies against the new key — success

**Zero operator intervention required.** Locked by integration tests
and the unit-level "unknown-kid reactive refresh" contract.

### JWKS endpoint unreachable

The middleware returns `500 JWKS_UNAVAILABLE` for affected requests.
The failure is logged at WARNING. The cache is **not** invalidated on
failure — a transient network blip doesn't take down every cached
key.

The endpoint is **not re-dialled once per request** while it is down. A fetch
that teaches the client nothing — one that failed, or one that returned a key
set identical to the cache it replaced — arms a short backoff (30 s by default,
`refresh_cooldown_sec=`). A fetch that *changes* the key set clears it, so a real
rotation is picked up at full speed and the unknown-kid refresh contract is
unaffected.

What the backoff bounds is a caller inventing `kid` values: without it, every
unverified bearer token could force one outbound request to MudraID's key
endpoint. Outcomes are unchanged — an unknown kid was a refusal before and still
is.

One consequence worth knowing: if the cache is **past its TTL** and the client is
in backoff, requests are refused rather than served from the expired cache.
Serving it would extend the lifetime of every key in it, including one rotated
*out*.

### Misconfigured `mudraid_scopes.yaml`

The middleware lazily loads the YAML on first request. A schema
error returns `500 MIDDLEWARE_NOT_READY` and is logged at ERROR. The
failure is **not** cached — once you fix the YAML, the next request
succeeds without a restart.

---

## Try it locally

A runnable sample platform ships in [`examples/sample-platform/`](../../examples/sample-platform/).
It's a minimal FastAPI app with the middleware applied, callable
from the matching sample agent in
[`examples/sample-agent/`](../../examples/sample-agent/).

```bash
# From the repo root:
docker compose up -d                                       # backend
docker compose --profile samples up -d sample-platform     # this middleware in action

curl http://localhost:8020/health
# → {"status":"ok"}             (public route, no JWT required)

curl http://localhost:8020/api/v1/items
# → 401 {"error_code":"MISSING_TOKEN", "message":"..."}
```

The sample-platform README walks through the full setup including
the manual `platform_id` swap before first run.

---

## How it fits into the wider system

```
        agent making a request               your FastAPI app
        ─────────────────────────             ────────────────────────
        agent.get(url) ──── HTTP ────► MudraIDMiddleware
                                           │
                                           ▼
                                       (verify JWT + check scope)
                                           │
                                           ▼
                                       your route handler
                                           │
                                           ▼
                                       response

        ↕
        ┌──────────────────────────┐
        │     MudraID backend       │
        │  /auth/token + /jwks      │
        └──────────────────────────┘
```

The matching client-side package is
[`mudraid-sdk`](../mudraid-sdk-python/) for Python agents.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
