"""Fetch + cache MudraID's JWKS for JWT verification.

The middleware verifies every incoming JWT against MudraID's
published key set rather than calling MudraID on each request.
This module is responsible for keeping that key set fresh:

  - **Lazy bootstrap.** The first ``get_key`` call fetches JWKS.
    Construction is side-effect-free so a misconfigured startup
    doesn't crash on import.
  - **1-hour TTL** (configurable). After the TTL expires the next
    ``get_key`` triggers a refresh.
  - **Reactive refresh on unknown kid.** Locked decision D4 from
    the Phase 3 plan: a JWT carrying a ``kid`` we don't have
    triggers ONE refresh attempt, then a re-lookup. If the kid is
    still missing after refresh, ``MudraIDJwksError`` is raised —
    we never silently pass an unverifiable token through.
  - **Single in-flight refresh.** Concurrent ``get_key`` calls
    that all miss the cache coalesce into one HTTP request via
    ``asyncio.Lock``; the late-arrivers re-read the cache after
    the holder's fetch publishes.

The module is async because the middleware host (Starlette's
``BaseHTTPMiddleware``) is async — a blocking sync HTTP call here
would stall the event loop on every cold-cache request.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
from jwt.algorithms import RSAAlgorithm

from mudraid_platform_middleware.exceptions import MudraIDJwksError

_DEFAULT_TTL_SEC = 3600.0
_DEFAULT_TIMEOUT_SEC = 5.0
#: Minimum interval between two fetches triggered by anything OTHER than the
#: cache's TTL expiring — i.e. by the reactive unknown-kid path. See
#: :meth:`JwksClient.get_key` for why this exists.
_DEFAULT_REFRESH_COOLDOWN_SEC = 30.0
#: Hard ceiling on a JWKS response body. The key set is a handful of small
#: public keys; anything at this scale is a misconfigured or hostile endpoint,
#: and parsing it would be the expensive half.
_MAX_JWKS_BYTES = 1_048_576


class JwksClient:
    """Async JWKS fetcher with a TTL cache and rate-limited reactive-refresh."""

    def __init__(
        self,
        jwks_url: str,
        cache_ttl_sec: float = _DEFAULT_TTL_SEC,
        timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
        refresh_cooldown_sec: float = _DEFAULT_REFRESH_COOLDOWN_SEC,
    ) -> None:
        if not jwks_url:
            raise ValueError("jwks_url is required")
        self._url = jwks_url
        self._ttl = cache_ttl_sec
        self._timeout = timeout_sec
        self._cooldown = refresh_cooldown_sec
        # Wall-clock instant before which no further fetch may be dialled. Set
        # ONLY by a fetch that taught us nothing — one that failed, or one that
        # returned a key set identical to the one already cached. A fetch that
        # changed the key set clears it, so a real rotation is never delayed.
        # See ``get_key`` for why the distinction is the whole mechanism.
        self._backoff_until: float = 0.0
        # kid -> JWK dict. Empty means "not yet bootstrapped" — but
        # so does "bootstrapped against an empty key set", so don't
        # treat emptiness as a stale signal; use ``_fetched_at``.
        self._cache: dict[str, dict[str, Any]] = {}
        # kid -> parsed public-key object. A memoised view of ``_cache``,
        # populated lazily on first lookup and reset atomically alongside
        # ``_cache`` on every ``_fetch`` (see ``_fetch``). Tying it to the
        # same lifecycle is the security-critical bit: when a key is rotated
        # or revoked out of JWKS, its parsed form is dropped in the same swap,
        # so a removed key can never keep verifying tokens past the JWKS TTL.
        self._parsed: dict[str, Any] = {}
        self._fetched_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get_key(self, kid: str) -> dict[str, Any]:
        """Return the JWK whose ``kid`` matches.

        Fetches JWKS on first call. Triggers a refresh if the cache is stale, or
        if ``kid`` isn't in it AND the reactive-refresh cooldown has elapsed.
        Raises :class:`MudraIDJwksError` when the kid is unknown — the middleware
        translates this to a 500 with a "token signature unverifiable" reason.

        WHY THE RATE LIMIT. Locked decision D4 is "an unknown kid triggers ONE
        refresh, then a re-lookup". Per TOKEN that is right and it is preserved
        exactly. Implemented per REQUEST with no floor between fetches, though,
        it lets every caller who can present a bearer token — which is every
        caller, since the token has not been verified yet — force one outbound
        HTTPS request to the identity service by inventing a ``kid``. A loop of
        garbage tokens becomes a request-rate-multiplied load amplifier pointed
        at MudraID's own JWKS endpoint, from every platform running this
        middleware at once. Because the fetch runs under a single
        ``asyncio.Lock`` with a 5 s timeout, a slow JWKS host turns that into a
        queue every legitimate cold-cache request also waits in.

        THE LIMIT IS ON UNPRODUCTIVE FETCHES, NOT ON REFRESH. A blanket cooldown
        would have bought the same protection by breaking D4 — a key rotated
        moments after a fetch would go unseen for the cooldown. So the backoff is
        armed only by a fetch that TAUGHT US NOTHING: one that failed, or one
        whose key set was byte-identical to the cache it replaced. A fetch that
        actually changed the key set clears it. The consequences fall out:

          * first unknown kid after a rotation -> fetch, key set changes, no
            backoff. D4 holds, and holds at full speed.
          * attacker's stream of invented kids -> the first costs one fetch that
            returns the same keys, arming the backoff; every kid after it is
            refused with no upstream call at all.
          * JWKS endpoint down -> one attempt, then quiet until the cooldown.

        No outcome changes, only a rate. An unknown kid was a refusal before and
        is a refusal now.

        A STALE CACHE IN BACKOFF REFUSES; IT IS NOT SERVED. Falling through to a
        cache whose TTL has expired would let a failing endpoint silently extend
        the lifetime of keys — including one rotated OUT of the key set, which is
        precisely the revocation this TTL exists to bound.
        """
        # Fast path: cache fresh AND kid known — no lock needed.
        if self._is_fresh() and kid in self._cache:
            return self._cache[kid]

        async with self._lock:
            # Re-check under the lock. Another coroutine may have
            # refreshed while we were waiting; if so, prefer its
            # work rather than re-fetching.
            if not self._is_fresh():
                if not self._may_attempt():
                    raise MudraIDJwksError(
                        f"JWKS cache from {self._url} is past its TTL and the last "
                        "fetch attempt neither succeeded nor changed the key set; "
                        "refusing to verify against an expired key set"
                    )
                await self._fetch()
            elif kid not in self._cache and self._may_attempt():
                # Fresh cache, unknown kid: the reactive-refresh path (D4).
                await self._fetch()

            jwk = self._cache.get(kid)
            if jwk is None:
                raise MudraIDJwksError(
                    f"JWT signed with unknown kid {kid!r}; " "key not present in MudraID JWKS"
                )
            return jwk

    async def get_public_key(self, kid: str) -> Any:
        """Return the *parsed* public key for ``kid``.

        Wraps :meth:`get_key` (which owns all freshness / reactive-refresh /
        unknown-kid trust logic) and memoises the expensive
        ``RSAAlgorithm.from_jwk`` parse so it runs once per key per JWKS
        refresh instead of once per request. The parse cache shares
        ``get_key``'s lifecycle exactly: ``_fetch`` resets ``_parsed`` in the
        same atomic swap as ``_cache``, so a rotated/revoked key drops its
        parsed form too.

        A malformed JWK raises here on lookup (``from_jwk`` propagates),
        matching the previous behaviour where the validator parsed inline.
        """
        jwk = await self.get_key(kid)
        # No ``await`` between this read and the write below, so the
        # ``_parsed`` dict we populate is guaranteed to be the one published
        # by the fetch that produced ``jwk`` — no cross-refresh mixing.
        cached = self._parsed.get(kid)
        if cached is not None:
            return cached
        public_key = RSAAlgorithm.from_jwk(json.dumps(jwk))
        self._parsed[kid] = public_key
        return public_key

    async def refresh(self) -> None:
        """Force a JWKS fetch regardless of cache freshness.

        Provided as an operator hook — the middleware doesn't call
        this on the hot path. Useful for ops-driven rotation drills
        and for tests.

        Ignores the unproductive-fetch backoff: an operator asking for a refresh
        is not the traffic the backoff exists to bound, and a rotation drill that
        silently did nothing would be worse than useless.
        """
        async with self._lock:
            self._backoff_until = 0.0
            await self._fetch()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _is_fresh(self) -> bool:
        return self._fetched_at > 0.0 and (time.time() - self._fetched_at) < self._ttl

    def _may_attempt(self) -> bool:
        """Whether another fetch may be dialled now.

        True unless an unproductive fetch armed the backoff and it has not yet
        elapsed. A cold start is never delayed (``_backoff_until`` is 0). Called
        only under ``_lock``.
        """
        return time.time() >= self._backoff_until

    async def _fetch(self) -> None:
        """Replace ``self._cache`` with a freshly-fetched JWKS.

        Called only from inside the lock. The cache is replaced
        atomically: either every kid in the new response is present
        afterwards, or — if the fetch raises — the cache is left
        untouched and the exception propagates.
        """
        # Every exit that is not a successful, key-set-CHANGING fetch arms the
        # backoff. Arming it up front and clearing it only on the productive
        # path means a return added here later cannot forget to.
        self._backoff_until = time.time() + self._cooldown
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(self._url)
        except httpx.HTTPError as exc:
            raise MudraIDJwksError(f"could not fetch JWKS from {self._url}: {exc}") from exc

        # A JWKS is a handful of small public keys. Refusing an oversized body
        # before ``response.json()` keeps a misconfigured or hostile endpoint
        # from making the verifier pay to parse it.
        if len(response.content) > _MAX_JWKS_BYTES:
            raise MudraIDJwksError(
                f"JWKS response from {self._url} exceeds {_MAX_JWKS_BYTES} bytes"
            )

        if response.status_code != 200:
            raise MudraIDJwksError(
                f"JWKS endpoint returned status {response.status_code} " f"from {self._url}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise MudraIDJwksError(
                f"JWKS endpoint returned non-JSON body from {self._url}"
            ) from exc

        keys = payload.get("keys") if isinstance(payload, dict) else None
        if not isinstance(keys, list):
            raise MudraIDJwksError(f"JWKS response is missing a 'keys' array (from {self._url})")

        new_cache: dict[str, dict[str, Any]] = {}
        for entry in keys:
            if not isinstance(entry, dict):
                # Tolerate malformed entries — drop them silently.
                # Verification will fail loudly on lookup if the
                # JWT references one of them.
                continue
            kid = entry.get("kid")
            if not (isinstance(kid, str) and kid):
                continue
            # Only keys this verifier could legitimately use are admitted.
            # ``kty`` must be RSA (the validator accepts RS256 and nothing
            # else), and a key published for ENCRYPTION is not a key for
            # checking signatures — RFC 7517 §4.2 says so, and admitting one
            # would let a key set widen the verifying set by accident. Dropping
            # them here means a JWT naming one is an unknown kid, which is
            # already a refusal.
            if entry.get("kty") != "RSA":
                continue
            use = entry.get("use")
            if use is not None and use != "sig":
                continue
            new_cache[kid] = entry

        # Did this fetch teach us anything? Compared BEFORE the swap, on the kid
        # set alone: that is the only thing an unknown-kid lookup can be waiting
        # for, so it is the right definition of "productive" for the backoff this
        # answer arms or clears.
        changed = new_cache.keys() != self._cache.keys()

        # Atomic swap. Failures above prevent us from reaching here
        # so we never publish a partial cache. ``_parsed`` is reset to empty
        # in the SAME swap so no parsed key outlives its JWK entry — keys
        # rotated/revoked out of this fetch can never be served from the
        # parse cache afterwards.
        self._cache = new_cache
        self._parsed = {}
        self._fetched_at = time.time()
        if changed:
            # A rotation landed. Whatever a caller is looking for, it may be in
            # here now, and the next unknown kid deserves a fetch at full speed.
            self._backoff_until = 0.0
