"""M5.4 — JwksClient: fetch, cache, reactive refresh, concurrent safety.

Uses ``respx`` to mock httpx so no live MudraID is required. The
tests cover each locked behaviour from the module docstring with
one dedicated case, plus a thread-of-coroutines test that locks
the single-in-flight-refresh contract.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from mudraid_platform_middleware import MudraIDJwksError
from mudraid_platform_middleware._jwks_client import JwksClient

JWKS_URL = "https://api.mudraid.test/.well-known/jwks.json"


def _jwk(kid: str) -> dict:
    """Build a minimally-shaped JWK that's good enough for tests.

    The middleware never decodes the key bytes itself — that's M5.5's
    job via PyJWT. For this module we only care that ``kid`` lookup
    works, so the rest of the JWK is intentionally opaque."""
    return {
        "kid": kid,
        "kty": "RSA",
        "use": "sig",
        "alg": "RS256",
        "n": "stub-modulus",
        "e": "AQAB",
    }


def _jwks_body(*kids: str) -> dict:
    return {"keys": [_jwk(k) for k in kids]}


def _client(**overrides) -> JwksClient:
    kwargs = {"jwks_url": JWKS_URL, "cache_ttl_sec": 3600.0, "timeout_sec": 1.0}
    kwargs.update(overrides)
    return JwksClient(**kwargs)


# ---- construction --------------------------------------------------------


def test_constructor_rejects_empty_url() -> None:
    with pytest.raises(ValueError, match="jwks_url"):
        JwksClient(jwks_url="")


def test_construction_does_not_fetch() -> None:
    """Lazy bootstrap — building a client must not touch the network.
    Without this lock, ``MudraIDMiddleware(app)`` would block on
    JWKS fetch during app startup, even before the first request."""
    # ``as r`` was unused: the assertion is respx's own — assert_all_called
    # flags an unexpected call on exit, so the context manager IS the check.
    with respx.mock(assert_all_called=True):
        _client()


# ---- happy path: cold + warm cache ---------------------------------------


@pytest.mark.asyncio
async def test_first_get_key_fetches_jwks() -> None:
    with respx.mock() as r:
        route = r.get(JWKS_URL).mock(return_value=httpx.Response(200, json=_jwks_body("kid-1")))

        result = await _client().get_key("kid-1")

    assert result["kid"] == "kid-1"
    assert route.called
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_second_get_key_hits_cache_no_http() -> None:
    with respx.mock() as r:
        route = r.get(JWKS_URL).mock(return_value=httpx.Response(200, json=_jwks_body("kid-1")))

        client = _client()
        await client.get_key("kid-1")
        await client.get_key("kid-1")

    assert route.call_count == 1, "cache hit must skip the network"


@pytest.mark.asyncio
async def test_multiple_kids_returned_independently() -> None:
    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json=_jwks_body("kid-a", "kid-b")))

        client = _client()
        a = await client.get_key("kid-a")
        b = await client.get_key("kid-b")

    assert a["kid"] == "kid-a"
    assert b["kid"] == "kid-b"


# ---- reactive refresh on unknown kid -------------------------------------


@pytest.mark.asyncio
async def test_unknown_kid_triggers_one_refresh_and_then_returns() -> None:
    """Locked decision D4: an unknown kid means MudraID rotated; refresh
    once and try again. If the new JWKS contains the kid, return it."""
    responses = [
        httpx.Response(200, json=_jwks_body("kid-old")),
        httpx.Response(200, json=_jwks_body("kid-old", "kid-new")),
    ]
    with respx.mock() as r:
        route = r.get(JWKS_URL).mock(side_effect=responses)

        client = _client()
        await client.get_key("kid-old")  # cold fetch
        new = await client.get_key("kid-new")  # triggers refresh

    assert new["kid"] == "kid-new"
    assert route.call_count == 2, "exactly one refresh on unknown kid"


@pytest.mark.asyncio
async def test_unknown_kid_after_refresh_raises_jwks_error() -> None:
    """If even the freshly-fetched JWKS lacks the requested kid, we
    raise loudly. Better an explicit 401 to the agent than silently
    passing through an unverifiable token."""
    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json=_jwks_body("only-kid")))

        client = _client()

        with pytest.raises(MudraIDJwksError, match="unknown kid"):
            await client.get_key("ghost-kid")


@pytest.mark.asyncio
async def test_unknown_kid_does_not_loop_after_failed_refresh() -> None:
    """A third HTTP call must not happen — locks the single-retry
    contract so a misbehaving JWKS endpoint can't burn our request
    budget."""
    with respx.mock() as r:
        route = r.get(JWKS_URL).mock(return_value=httpx.Response(200, json=_jwks_body("kid-1")))

        client = _client()
        with pytest.raises(MudraIDJwksError):
            await client.get_key("absent-kid")

    # 1 fetch attempt total — the unknown kid did not cause a second
    # refresh because there was no cache to invalidate (the first
    # fetch IS the refresh).
    assert route.call_count == 1


# ---- TTL expiry ----------------------------------------------------------


@pytest.mark.asyncio
async def test_expired_cache_triggers_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """After the TTL expires, the next ``get_key`` triggers a fresh
    fetch even when the kid is already in the (stale) cache. Locks
    the periodic-refresh half of the rotation handling."""
    fake_now = [1_000_000.0]

    def time_stub() -> float:
        return fake_now[0]

    import mudraid_platform_middleware._jwks_client as jwks_mod

    monkeypatch.setattr(jwks_mod.time, "time", time_stub)

    with respx.mock() as r:
        route = r.get(JWKS_URL).mock(return_value=httpx.Response(200, json=_jwks_body("kid-1")))

        client = _client(cache_ttl_sec=60.0)
        await client.get_key("kid-1")

        # Advance past TTL.
        fake_now[0] += 120.0

        await client.get_key("kid-1")

    assert route.call_count == 2, "TTL expiry must trigger a refresh"


# ---- explicit refresh ----------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_refresh_bypasses_cache() -> None:
    """The operator hook — useful for rotation drills and tests."""
    with respx.mock() as r:
        route = r.get(JWKS_URL).mock(return_value=httpx.Response(200, json=_jwks_body("kid-1")))

        client = _client()
        await client.get_key("kid-1")
        await client.refresh()
        await client.refresh()

    assert route.call_count == 3, "every refresh() call hits the network"


# ---- error mapping -------------------------------------------------------


@pytest.mark.asyncio
async def test_network_error_raises_jwks_error() -> None:
    with respx.mock() as r:
        r.get(JWKS_URL).mock(side_effect=httpx.ConnectError("boom"))

        with pytest.raises(MudraIDJwksError, match="could not fetch"):
            await _client().get_key("kid-1")


@pytest.mark.asyncio
async def test_non_200_raises_jwks_error() -> None:
    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(500, text="server error"))

        with pytest.raises(MudraIDJwksError, match="status 500"):
            await _client().get_key("kid-1")


@pytest.mark.asyncio
async def test_non_json_body_raises_jwks_error() -> None:
    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, text="<html/>"))

        with pytest.raises(MudraIDJwksError, match="non-JSON"):
            await _client().get_key("kid-1")


@pytest.mark.asyncio
async def test_response_without_keys_array_raises_jwks_error() -> None:
    with respx.mock() as r:
        r.get(JWKS_URL).mock(return_value=httpx.Response(200, json={"foo": "bar"}))

        with pytest.raises(MudraIDJwksError, match="keys"):
            await _client().get_key("kid-1")


@pytest.mark.asyncio
async def test_keys_with_malformed_entries_are_silently_dropped() -> None:
    """Tolerate malformed entries (drop them) but ensure the
    well-formed ones remain usable. Without this, a single bad row
    in the JWKS response would brick the entire middleware."""
    with respx.mock() as r:
        r.get(JWKS_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "keys": [
                        "this is not a dict",
                        {"kty": "RSA"},  # missing kid
                        _jwk("kid-good"),
                    ]
                },
            )
        )

        client = _client()
        good = await client.get_key("kid-good")
        assert good["kid"] == "kid-good"

        with pytest.raises(MudraIDJwksError):
            await client.get_key("missing-kid")


@pytest.mark.asyncio
async def test_failed_fetch_does_not_publish_partial_cache() -> None:
    """If the second (refresh) fetch fails, the cache from the first
    fetch must remain intact — otherwise a transient network blip
    would invalidate every cached key."""
    with respx.mock() as r:
        # First call: succeeds with kid-1 only.
        # Second call (the unknown-kid refresh): fails.
        r.get(JWKS_URL).mock(
            side_effect=[
                httpx.Response(200, json=_jwks_body("kid-1")),
                httpx.ConnectError("boom"),
            ]
        )

        client = _client()
        await client.get_key("kid-1")

        with pytest.raises(MudraIDJwksError):
            await client.get_key("unknown-kid")

        # Original kid still works — the cache wasn't wiped by the
        # failed refresh.
        assert await client.get_key("kid-1") == _jwk("kid-1")


# ---- concurrency ---------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_first_get_key_makes_only_one_http_call() -> None:
    """20 coroutines all hitting a cold cache simultaneously must
    produce exactly ONE outbound HTTP call. Otherwise a thundering-
    herd of agent requests at boot would stampede MudraID's JWKS
    endpoint."""
    with respx.mock() as r:

        async def respond(request):
            # Tiny await so the lock-coalescing path actually runs —
            # without it the first coroutine completes before the
            # second even enters get_key, and the test passes trivially.
            await asyncio.sleep(0.005)
            return httpx.Response(200, json=_jwks_body("kid-1"))

        route = r.get(JWKS_URL).mock(side_effect=respond)

        client = _client()

        async def worker() -> dict:
            return await client.get_key("kid-1")

        results = await asyncio.gather(*[worker() for _ in range(20)])

    assert all(r["kid"] == "kid-1" for r in results)
    assert route.call_count == 1, "double-checked lock must coalesce concurrent first calls"
