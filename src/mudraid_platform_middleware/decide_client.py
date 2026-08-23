"""The supported :class:`~mudraid_platform_middleware.v2.DecideClient`.

This is the production implementation of the V2 seam: it holds the adapter's
verified signed bundle, resolves tool names to canonical actions from it, and
makes the authenticated live ``/decide`` call for every protected invocation.

It is the Python counterpart of the Kong plugin's ``channel.lua`` +
``bundle.lua`` + ``decide.lua``, and it is deliberately built to reach the same
outcomes for the same facts — a platform enforcing with this middleware and one
enforcing at a Kong gateway must not disagree about whether a call is allowed.

DENY-CLOSED IN EVERY DIRECTION, which is the whole design:

  - no verified bundle yet → :attr:`bundle_active` is ``False`` → the control
    loop fails closed with 503, never "allow until we know better";
  - a served bundle that fails verification is REFUSED and the last good bundle
    stays active; if there is no last good bundle the surface keeps failing
    closed;
  - ``/decide`` unconfigured, credential-less, unreachable, slow, non-200, or
    answering in a vocabulary this client does not recognise → a failure
    :class:`DecideResult`, which the control loop deny-closes;
  - one attempt per decision, never a retry: a retried decision can double-meter
    a root authority decision, so the caller denies instead.

NOTHING SECRET IS EVER LOGGED. The adapter token, the bundle signing secret, the
service secret and the caller's presented credential are all held or forwarded
but never written to a log line — failures log status codes and exception type
names only.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import uuid
from datetime import datetime, timedelta, timezone
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

import httpx

from mudraid_platform_middleware._bundle import BundleRefused, VerifiedBundle, verify_bundle
from mudraid_platform_middleware._canonical import loads_strict
from mudraid_platform_middleware._contract import (
    DECISION_ALLOW,
    DECISION_DENY,
    MAX_DECISION_ID_LEN,
    MAX_REASON_LEN,
    MAX_RESPONSE_BYTES,
    REQUEST_CONTRACT,
    SUPPORTED_RESPONSE_CONTRACTS,
)
from mudraid_platform_middleware._decision_signature import (
    DecisionBindings,
    DecisionSignatureRefused,
    verify_decision_signature,
)
from mudraid_platform_middleware._v2_control_loop import DecideResult
from mudraid_platform_middleware.v2 import DecideContext

__all__ = ["HttpDecideClient", "InsecureDecideDestination"]


class InsecureDecideDestination(ValueError):
    """The configured ``/decide`` destination is not one we may send secrets to.

    Raised at CONSTRUCTION, not at request time. The client forwards the caller's
    credential and this workload's service secret on every decision, so an
    unverifiable or unexpected destination is a misconfiguration to fail on at
    startup rather than a per-request risk to discover in production.
    """


_logger = logging.getLogger("mudraid_platform_middleware.decide")

#: Schemes we will send credentials over. ``http`` is permitted only to a
#: loopback host, and only when the caller opts in explicitly.
_SECURE_SCHEME = "https"

#: The adapter channel path serving the signed bundle.
# ── The canonical adapter-facing paths, derived from ONE base URL ───────────
#
# A customer configures MUDRAID_BASE_URL and MUDRAID_ADAPTER_TOKEN and nothing
# else. Every address below is derived from the first of those, so the SAME
# artifact runs against staging and production with no rebuild and no
# per-environment code — which is the point of the one-base-URL model.
#
# These are relative paths, never hostnames. No customer artifact may carry a
# hard-coded staging, production or internal hostname, and a relative path
# cannot become one by accident.
_BUNDLE_PATH = "/api/v1/adapter/enforcement/bundle"
_DECIDE_PATH = "/api/v1/adapter/enforcement/decide"
_KEYS_PATH = "/api/v1/adapter/enforcement/keys"
_HEARTBEAT_PATH = "/api/v1/adapter/enforcement/heartbeat"
_ACK_PATH = "/api/v1/adapter/enforcement/acknowledgements"

#: The legacy internal bundle path, kept only so a MudraID-operated adapter
#: mid-migration can be pointed at it explicitly. Never derived from base_url,
#: because a customer must never reach for it.
_LEGACY_BUNDLE_PATH = "/api/v1/internal/enforcement/bundle"

#: The versioned request contract this client stamps on every decision. Both
#: adapters send this exact string and the authority validates it — see
#: ``_contract.py`` for why request and response versions are named separately.
ENVELOPE_SCHEMA = REQUEST_CONTRACT

#: Adapter identity carried on the envelope, distinguishing this adapter from
#: the Kong plugin in the authority's evidence.
_ADAPTER_TYPE = "python_platform_middleware"

#: Refuse an absurdly large bundle body before parsing it. The real artifact is
#: kilobytes; anything at this scale is a fault or an attack, and parsing it
#: first would be the expensive way to find out.
_MAX_BUNDLE_BYTES = 8 * 1024 * 1024


def _validate_destination(
    url: str,
    *,
    allowed_origins: frozenset[str] | None,
    allow_insecure_loopback: bool,
    label: str,
) -> str:
    """Return ``url`` if we may send credentials to it, else refuse.

    The checks are ordered so the message names the first real problem rather
    than a downstream symptom of it.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise InsecureDecideDestination(f"{label} must be an absolute http(s) URL; got {url!r}")
    if parts.username or parts.password:
        # Credentials in a URL end up in logs, process lists and error strings.
        raise InsecureDecideDestination(f"{label} must not embed credentials in the URL")
    if parts.fragment:
        # A fragment is never sent to a server, so its presence means the
        # configured value is not the URL the operator thinks it is.
        raise InsecureDecideDestination(f"{label} must not contain a fragment")
    host = parts.hostname
    if not host:
        raise InsecureDecideDestination(f"{label} has no host")
    try:
        port = parts.port
    except ValueError as exc:
        raise InsecureDecideDestination(f"{label} has an invalid port") from exc
    if port is not None and not (0 < port < 65536):
        raise InsecureDecideDestination(f"{label} has an out-of-range port")

    if parts.scheme != _SECURE_SCHEME:
        if not allow_insecure_loopback:
            raise InsecureDecideDestination(
                f"{label} is cleartext http. This request carries the caller's "
                "credential and this workload's service secret, so https is "
                "required. Set allow_insecure_loopback=True for local "
                "development against a loopback address only."
            )
        if not _is_loopback(host):
            raise InsecureDecideDestination(
                f"{label} is cleartext http to a non-loopback host; the "
                "development override covers loopback only"
            )

    if allowed_origins is not None:
        origin = f"{parts.scheme}://{parts.netloc}".lower()
        if origin not in allowed_origins:
            raise InsecureDecideDestination(
                f"{label} origin {origin!r} is not in the configured allowed origins"
            )
    return url


def _is_loopback(host: str) -> bool:
    """Whether ``host`` names this machine, checked structurally.

    ``ipaddress`` rather than a string prefix, so ``127.0.0.1`` and ``127.9.9.9``
    are both recognised and ``127.0.0.1.evil.example`` is not mistaken for either.
    """
    host = host.strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class HttpDecideClient:
    """Bundle lifecycle + authenticated ``/decide``, over HTTP.

    The whole customer configuration is two values::

        client = HttpDecideClient(
            base_url=os.environ["MUDRAID_BASE_URL"],
            adapter_token=os.environ["MUDRAID_ADAPTER_TOKEN"],
        )

    Selecting an environment is a change to ``MUDRAID_BASE_URL`` and a different
    environment-issued token. It is never a different build, a different package
    or a different code path — the same artifact runs against staging and
    production, which is what makes promoting one meaningful.

    Args:
        base_url: The MudraID origin for this environment. Every path the client
            uses — bundle, decision, verification keys — is derived from it, so
            no customer artifact carries a MudraID hostname.
        adapter_token: This adapter instance's bearer, issued by the environment
            it belongs to. It names ONE adapter, which is what lets enforcement
            derive the tenant and surface; this client never names a platform id.
        channel_url, decide_url, keys_url: Advanced overrides for MudraID-operated
            deployments and local testing. Each replaces one derived path and is
            validated exactly as strictly — supplying one changes where a request
            goes, never how carefully it is checked. Not the customer setup.
        bundle_signing_secret: The legacy HMAC secret. Absent for every customer
            adapter, and deliberately so: anyone who can verify with it can also
            SIGN with it, so a customer holding it could mint bundles for their
            own surface. The SDK verifies the RS256 bundle signature instead.
            MudraID's own gateway supplies it during the migration window.
        poll_interval_seconds: Cadence of the background bundle refresh.
        channel_timeout_seconds: Deadline for one channel request.
        decide_timeout_seconds: Deadline for one ``/decide`` call. The middleware
            applies its own deadline as well; this one bounds the transport.
        require_signed_decisions: Whether an UNSIGNED ``/decide`` response is
            refused. Optional and ``False`` by default, so 1.1-compatible
            behaviour is unchanged.

            ``False`` — an unsigned response is read as before; a signature that
            IS present must still verify completely or the whole response is
            refused (verify-when-present).

            ``True`` — an unsigned response is refused as well. This is the mode
            that closes signature-stripping by a party who can terminate TLS,
            and the only mode in which a decision response is meaningfully
            authenticated by its signature.

            In BOTH modes an invalid, unknown-key, expired or binding-mismatched
            signature is refused, and every refusal deny-closes as ``"error"``.
            Turn this on only for a surface the authority already signs for:
            MudraID does not sign decision responses today, so enabling it now
            denies every decision. It has no effect on V1 mode, which makes no
            ``/decide`` call.
        client: An injected :class:`httpx.AsyncClient`, for tests or for a
            platform that needs its own transport, proxies or TLS context. When
            supplied it is NOT closed by :meth:`aclose` — whoever owns it closes
            it.
    """

    def __init__(
        self,
        *,
        base_url: str,
        adapter_token: str,
        # ── Advanced overrides. NOT the normal customer setup. ───────────────
        #
        # Present for MudraID-operated deployments and local testing: an
        # in-VPC gateway that reaches enforcement over a private address, or a
        # test pointing at a loopback stub. Each one REPLACES a path that would
        # otherwise be derived from base_url, and each is validated exactly as
        # strictly. Supplying one does not relax any check — it only changes
        # where the request goes.
        channel_url: str | None = None,
        decide_url: str | None = None,
        keys_url: str | None = None,
        # The HMAC bundle secret. Absent for every customer adapter: anyone who
        # can verify with it can also SIGN with it, so a customer holding it
        # could mint bundles for their own surface. MudraID's own gateway
        # supplies it during the migration window; the SDK verifies the RS256
        # signature instead.
        bundle_signing_secret: str | None = None,
        poll_interval_seconds: float = 30.0,
        channel_timeout_seconds: float = 5.0,
        decide_timeout_seconds: float = 2.0,
        allowed_origins: frozenset[str] | set[str] | None = None,
        allow_insecure_loopback: bool = False,
        # A9-02 / Decision 8. Optional, and OFF by default so 1.1-compatible
        # behaviour is unchanged for every existing deployment.
        #
        #   False — an UNSIGNED decision response is read as before; a PRESENT
        #           signature must still verify completely or the response is
        #           refused (verify-when-present);
        #   True  — an UNSIGNED decision response is REFUSED, closing the
        #           signature-stripping gap against a TLS-terminating party.
        #
        # Only turn this on for a surface the authority actually signs for;
        # until then it would deny-close every decision. It lives HERE, on the
        # client that holds the signature and does the verifying, rather than on
        # V2Config: the DecideClient seam is a Protocol whose decide() carries no
        # channel for a verification policy, so a copy of this setting on the
        # config object could not be honoured for an injected client.
        require_signed_decisions: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        origins = (
            frozenset(o.rstrip("/").lower() for o in allowed_origins)
            if allowed_origins is not None
            else None
        )
        # The one address a customer configures. Validated first, and every
        # derived path inherits that validation because it inherits the origin.
        base_url = _validate_destination(
            base_url,
            allowed_origins=origins,
            allow_insecure_loopback=allow_insecure_loopback,
            label="base_url",
        ).rstrip("/")
        self._base_url = base_url

        def _resolve(override: str | None, path: str, label: str) -> str:
            """An explicit override, validated the same way, or the derived path."""
            if override is None:
                return f"{base_url}{path}"
            return _validate_destination(
                override,
                allowed_origins=origins,
                allow_insecure_loopback=allow_insecure_loopback,
                label=label,
            )

        self._channel_url = _resolve(channel_url, "", "channel_url").rstrip("/")
        self._decide_url = _resolve(decide_url, _DECIDE_PATH, "decide_url")
        self._keys_url = _resolve(keys_url, _KEYS_PATH, "keys_url")

        self._adapter_token = adapter_token
        # None for a customer adapter — see the constructor note. verify_bundle
        # accepts that and verifies the asymmetric signature alone.
        self._bundle_signing_secret = bundle_signing_secret
        # Fetched from the keys endpoint, refreshed alongside the bundle. Empty
        # until the first fetch succeeds, which means a signed bundle arriving
        # first is REFUSED rather than trusted — the correct order.
        self._verification_keys: dict[str, str] = {}
        # A9-02: the DECISION-response verification keys — a DIFFERENT series
        # from the bundle keys, served on the same page. Kept separate so a
        # decision can never verify against the bundle key or vice versa.
        # Empty until published, which refuses any SIGNED decision (unknown
        # key) while still reading unsigned ones — verify-when-present.
        self._decision_verification_keys: dict[str, str] = {}
        # Whether an UNSIGNED decision response is refused. See the constructor
        # note; default False keeps 1.1-compatible verify-when-present.
        self._require_signed_decisions = require_signed_decisions
        self._poll_interval = poll_interval_seconds
        self._channel_timeout = channel_timeout_seconds
        self._decide_timeout = decide_timeout_seconds

        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()

        self._bundle: VerifiedBundle | None = None
        self._by_action_key: dict[str, dict[str, Any]] = {}
        self._refresh_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    # ---- lifecycle -------------------------------------------------------

    async def start(self, *, wait_for_first: bool = True) -> None:
        """Fetch the first bundle and begin refreshing in the background.

        ``wait_for_first`` performs one synchronous fetch before returning, so a
        platform that starts serving immediately does not spend its first
        requests failing closed. It is not an error for that first fetch to
        fail — the surface simply stays failed-closed until a bundle verifies,
        which is the correct posture, so startup is not blocked on the control
        plane being reachable.
        """
        if wait_for_first:
            await self.refresh_once()
        if self._refresh_task is None:
            self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def aclose(self) -> None:
        """Stop refreshing and release the transport if this client owns it."""
        task = self._refresh_task
        self._refresh_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> HttpDecideClient:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def _refresh_verification_keys(self) -> None:
        """Refresh the bundle verification keys. Never clears a working set.

        A fetch failure LEAVES the previous keys in place. That is deliberate
        and is the one degradation worth allowing here: the keys are public,
        already-fetched material, and dropping them because the endpoint blipped
        would refuse every signed bundle — turning a transient control-plane
        outage into an enforcement outage.

        Rotation still takes effect, because a successful fetch REPLACES the set
        rather than merging into it: a key that stops being published stops
        being trusted at the next successful refresh.
        """
        try:
            response = await self._client.get(
                self._keys_url,
                headers={"Accept": "application/json"},
                timeout=self._channel_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("verification keys unreachable: %s", type(exc).__name__)
            return
        if response.status_code != 200:
            _logger.warning("verification keys refused: HTTP %s", response.status_code)
            return
        try:
            payload = response.json()
        except ValueError:
            _logger.warning("verification keys response was not JSON")
            return

        entries = payload.get("keys") if isinstance(payload, dict) else None
        if not isinstance(entries, list) or not entries:
            # An empty set is not a successful refresh. Replacing a working set
            # with nothing would refuse every signed bundle.
            _logger.warning("verification keys response carried no keys")
            return

        fresh = _key_map(entries)
        if fresh:
            self._verification_keys = fresh

        # A9-02: the decision-response key series, from the additive
        # ``key_sets`` projection. Same replace-never-clear discipline, per
        # series: a page that stops carrying the decision set (older control
        # plane, or a series not yet rotated) leaves the working set alone,
        # while a successful decision-set read REPLACES it so rotation takes
        # effect at the next refresh.
        key_sets = payload.get("key_sets") if isinstance(payload, dict) else None
        if isinstance(key_sets, list):
            for key_set in key_sets:
                if not isinstance(key_set, dict):
                    continue
                if key_set.get("purpose") != "enforcement_decision_signing":
                    continue
                fresh_decision = _key_map(key_set.get("keys"))
                if fresh_decision:
                    self._decision_verification_keys = fresh_decision

    async def _refresh_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._poll_interval)
                await self.refresh_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — a poll failure is never fatal
                _logger.warning("bundle refresh tick failed: %s", type(exc).__name__)

    # ---- bundle ----------------------------------------------------------

    async def refresh_once(self) -> bool:
        """Fetch and verify one bundle. Returns whether an activation happened.

        A failure here is deliberately quiet and non-fatal: the previously
        verified bundle stays active, because a control plane that is briefly
        unreachable must not disarm enforcement that is already correct.
        """
        # Keys FIRST. A signed bundle arriving with an empty key set is refused
        # rather than trusted, so fetching in the other order would turn every
        # cold start into a refusal even when the keys were reachable all along.
        await self._refresh_verification_keys()
        try:
            fetched = await self._fetch_bundle()
        except Exception as exc:  # noqa: BLE001
            _logger.warning("bundle fetch failed: %s", type(exc).__name__)
            return False
        if fetched is None:
            return False
        try:
            verified = verify_bundle(
                fetched,
                secret=self._bundle_signing_secret,
                verification_keys=self._verification_keys or None,
                active=self._bundle,
            )
        except BundleRefused as exc:
            # Typed and loud: a bundle that fails verification is the single
            # most security-relevant event this client can observe.
            _logger.error("bundle REFUSED (%s): %s", exc.code, exc.detail)
            return False
        if verified.no_change:
            return False
        async with self._lock:
            self._bundle = verified
            self._by_action_key = {
                key: action
                for action in verified.actions.values()
                if isinstance(key := action.get("action_key"), str) and key
            }
        _logger.info(
            "bundle version %d activated (%d action(s))",
            verified.bundle_version,
            len(verified.actions),
        )
        return True

    async def _fetch_bundle(self) -> dict[str, Any] | None:
        response = await self._client.get(
            f"{self._channel_url}{_BUNDLE_PATH}",
            headers={
                "Authorization": f"Bearer {self._adapter_token}",
                "Accept": "application/json",
            },
            timeout=self._channel_timeout,
        )
        if response.status_code == 404:
            # Nothing published for this surface yet. Not an error; the surface
            # stays failed-closed until something is.
            return None
        if response.status_code != 200:
            _logger.warning("bundle fetch status %d", response.status_code)
            return None
        raw = response.content
        if len(raw) > _MAX_BUNDLE_BYTES:
            _logger.error("bundle response exceeds the bounded size; refusing to parse")
            return None
        decoded = loads_strict(raw)
        if not isinstance(decoded, dict):
            _logger.warning("bundle response is not a JSON object")
            return None
        return decoded

    # ---- DecideClient ----------------------------------------------------

    @property
    def bundle_active(self) -> bool:
        """Whether a verified signed bundle is currently active."""
        return self._bundle is not None

    @property
    def bundle_version(self) -> int | None:
        """Active bundle version, or ``None`` when no bundle is active."""
        return self._bundle.bundle_version if self._bundle else None

    def resolve_action(self, tool_name: str) -> str | None:
        """The exact canonical action key for ``tool_name``, or ``None``.

        ``None`` covers every not-exactly-mapped case — no bundle, unknown name,
        over-long name, or a mapping carrying no usable ``action_key`` — and the
        control loop turns all of them into the same deny. An action the client
        cannot name is one it cannot ask the authority about.
        """
        bundle = self._bundle
        if bundle is None:
            return None
        action = bundle.resolve(tool_name)
        if action is None:
            return None
        key = action.get("action_key")
        if not isinstance(key, str) or not key:
            _logger.warning("mapped tool resolves to an action with no usable action_key")
            return None
        return key

    async def decide(self, action: str, context: DecideContext) -> DecideResult:
        """One live authority decision for a mapped canonical action.

        Single attempt by design — a retry here could double-meter a root
        decision, so the caller denies instead of retrying.
        """
        bundle = self._bundle
        if bundle is None:
            # The control loop should have failed closed before reaching this,
            # but the client does not rely on the caller to be correct.
            return DecideResult("unconfigured", reason="adapter_config_stale")
        if not self._decide_url:
            return DecideResult("unconfigured", reason="adapter_config_stale")
        if not self._adapter_token:
            # The decision route authenticates the ADAPTER, not a shared
            # workload. With no credential to present the call is never made
            # anonymously.
            return DecideResult("credential_unconfigured", reason="adapter_config_stale")

        decision_id = uuid.uuid4().hex
        envelope = self._envelope(action, context, bundle, decision_id)
        try:
            response = await self._client.post(
                self._decide_url,
                json=envelope,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-Correlation-ID": context.correlation_id,
                    # The adapter's own bearer. It names one adapter, which is
                    # what lets enforcement bind the decision to a surface; the
                    # old global service secret named no tenant at all and must
                    # never reach a customer.
                    "Authorization": f"Bearer {self._adapter_token}",
                },
                timeout=self._decide_timeout,
            )
        except httpx.TimeoutException:
            return DecideResult("timeout")
        except Exception as exc:  # noqa: BLE001 — any transport failure deny-closes
            _logger.warning("/decide unreachable: %s", type(exc).__name__)
            return DecideResult("unreachable")

        if response.status_code != 200:
            # Includes 401/403 from a rejected service secret. Never treated as
            # a deny decision — it is an authority we could not consult.
            _logger.warning("/decide status %d", response.status_code)
            return DecideResult("error")

        raw = response.content
        if len(raw) > MAX_RESPONSE_BYTES:
            # Refused before parsing: an oversized decision is not a decision.
            _logger.warning("/decide response exceeds the bounded size")
            return DecideResult("error")
        try:
            decoded = loads_strict(raw)
        except Exception:  # noqa: BLE001
            return DecideResult("error")
        surface = bundle.surface
        return _read_decision(
            decoded,
            expected_decision_id=decision_id,
            # A9-02: verify the response signature WHEN PRESENT, against the
            # decision key series and the bindings this adapter can vouch for —
            # the SIGNED bundle's surface and the mapped action, never anything
            # taken from the response being checked.
            verification_keys=self._decision_verification_keys or None,
            expected_bindings=DecisionBindings(
                platform_id=surface.get("platform_id"),
                environment=surface.get("environment"),
                resource=surface.get("canonical_resource_uri"),
                action_key=action,
                bundle_version=bundle.bundle_version,
            ),
            # Decision 8: when on, an unsigned response is refused rather than
            # read. Off by default — see the constructor note.
            require_signed_decisions=self._require_signed_decisions,
        )

    def _envelope(
        self,
        action: str,
        context: DecideContext,
        bundle: VerifiedBundle,
        decision_id: str,
    ) -> dict[str, Any]:
        """Build the ``/decide`` request envelope.

        Mirrors the Kong plugin's envelope field for field, so the authority
        receives one adapter vocabulary. The surface block is copied verbatim
        from the SIGNED bundle — never from request-supplied data — because it is
        what binds a grant to an exact environment and canonical resource.
        """
        mapped = self._by_action_key.get(action, {})
        surface = bundle.surface
        return {
            "schema_version": ENVELOPE_SCHEMA,
            "decision_id": decision_id,
            "correlation_id": context.correlation_id,
            "adapter": {"type": _ADAPTER_TYPE, "version": _adapter_version()},
            "bundle": {
                "version": bundle.bundle_version,
                "payload_digest": bundle.payload_digest,
            },
            "surface": {
                "platform_id": surface.get("platform_id"),
                "environment": surface.get("environment"),
                "canonical_resource_uri": surface.get("canonical_resource_uri"),
            },
            "action": {
                "action_key": action,
                "action_version": mapped.get("action_version"),
                "tool_name": mapped.get("tool_name"),
                "mapping_id": mapped.get("mapping_id"),
                "mapping_revision": mapped.get("mapping_revision"),
                "risk_class": mapped.get("risk_class"),
                "required_scopes": mapped.get("required_scopes"),
            },
            "request": {
                "transport": "mcp_streamable_http",
                "http_method": context.http_method,
                "path": context.path,
            },
            # The wire field keeps the authority's name; the seam uses
            # `presented_bearer`. Canonicalized once, in DecideContext.
            "presented_authorization": context.presented_bearer,
        }


def _key_map(entries: Any) -> dict[str, str]:
    """``key_id -> public_key_pem`` from one published key list."""
    keys: dict[str, str] = {}
    if not isinstance(entries, list):
        return keys
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key_id = entry.get("key_id")
        pem = entry.get("public_key_pem")
        if isinstance(key_id, str) and key_id and isinstance(pem, str) and pem:
            keys[key_id] = pem
    return keys


def _bounded_str(value: Any, limit: int) -> str | None:
    """A non-empty string within ``limit``, or ``None``."""
    if not isinstance(value, str):
        return None
    if not value or len(value) > limit:
        return None
    return value


def _read_decision(
    decoded: Any,
    *,
    expected_decision_id: str,
    now: datetime | None = None,
    verification_keys: dict[str, str] | None = None,
    expected_bindings: DecisionBindings | None = None,
    require_signed_decisions: bool = False,
) -> DecideResult:
    """Turn an authority response into a result, or refuse to read it.

    A bare ``{"decision": "allow"}`` is NOT a sufficient answer, and this is the
    function that says so. Without a supported response contract and a decision
    id bound to the request we sent, an "allow" is a string of five characters
    that anything on the path could have produced — including a replay of a
    different request's genuine allow.

    WHAT AUTHENTICATES THIS RESPONSE, STATED PLAINLY (A9-02)
    --------------------------------------------------------
    Two layers, and they are honestly different:

    1. Validated HTTPS to a verified destination, plus the strict binding
       below (contract, decision id, freshness). Real controls; they defend
       against a confused or replaying authority and against cross-request
       mixups, and they do NOT defend against a party who can terminate TLS.
    2. The authority's RS256 response signature, verified WHEN PRESENT against
       the published decision key series and this adapter's own surface/action
       bindings (``_decision_signature``). A response carrying a signature that
       does not verify completely — wrong key, wrong surface, mutated field,
       expired window — is refused whole. What a response carrying NO signature
       means is the operator's choice, via ``require_signed_decisions``:

         * off (default) — an unsigned response is read as before, because the
           authority activates signing by rollout and refusing every unsigned
           response would make that activation a flag-day. In this mode
           signature-stripping by a TLS-terminating party is NOT closed;
         * on — an unsigned response is refused. This is the mode in which
           layer 2 actually authenticates the decision, and it is only
           deployable once the authority signs for the surface in question.

       Nothing in this package may describe decision responses as
       cryptographically authenticated while the flag is off.

    Every failure below returns a result the control loop deny-closes.
    """
    if not isinstance(decoded, dict):
        _logger.warning("/decide response is not an object")
        return DecideResult("error")

    version = decoded.get("schema_version")
    if version not in SUPPORTED_RESPONSE_CONTRACTS:
        # Absent and unsupported are the same refusal: we cannot know what the
        # fields mean, so we cannot act on them.
        _logger.warning("/decide response contract is absent or unsupported")
        return DecideResult("error")

    decision_id = _bounded_str(decoded.get("decision_id"), MAX_DECISION_ID_LEN)
    if decision_id is None:
        _logger.warning("/decide response carries no usable decision_id")
        return DecideResult("error")
    if decision_id != expected_decision_id:
        # The answer is for some other question. Binding is what makes a
        # response non-replayable across requests.
        _logger.error("/decide decision_id does not match the request")
        return DecideResult("error")

    # ── Freshness ────────────────────────────────────────────────────────────
    #
    # A decision is an answer about a moment. Without a bound on age, a captured
    # allow stays usable indefinitely — and because the decision id is echoed
    # from the request, an attacker who can replay a whole exchange could pair a
    # stale allow with a fresh request id. ``decided_at`` and ``deadline_at``
    # are what close that, and an absent or unparseable one is a refusal rather
    # than an unbounded default.
    moment = now or datetime.now(timezone.utc)
    decided_at = _instant(decoded.get("decided_at"))
    if decided_at is None:
        _logger.warning("/decide response carries no usable decided_at")
        return DecideResult("error")
    if decided_at - _CLOCK_SKEW > moment:
        _logger.warning("/decide response is dated in the future")
        return DecideResult("error")
    if moment - decided_at > _MAX_DECISION_AGE:
        _logger.warning("/decide response is too old to act on")
        return DecideResult("error")

    # ``deadline_at`` is the authority's own bound on this decision. Optional,
    # because not every deployment sets one; enforced when present.
    deadline_at = _instant(decoded.get("deadline_at"))
    if deadline_at is not None and moment - _CLOCK_SKEW > deadline_at:
        _logger.warning("/decide response is past its deadline")
        return DecideResult("error")

    # ── Signature (A9-02): verify when present; require it when configured ───
    #
    # Placed AFTER the envelope-level binding so the signature is checked on a
    # response already known to answer THIS request, and BEFORE the outcome is
    # read so no allow/deny is ever surfaced from a response whose signature
    # failed.
    #
    # A PRESENT signature must verify completely, in BOTH modes — unknown key,
    # mutated bound field, foreign surface/action, expired window all refuse
    # the whole response. ``require_signed_decisions`` governs only what an
    # ABSENT signature means, because absence and invalidity are different
    # facts and collapsing them would let an attacker downgrade by corrupting
    # one field rather than stripping the whole signature:
    #
    #   off (default) — absent is read as before, keeping the authority's
    #                   signing rollout from becoming a flag-day (1.1-compatible);
    #   on            — absent is refused, which is what closes the stripping
    #                   gap for a party who can terminate TLS.
    #
    # Both refusals deny CLOSED: "error", never an allow.
    if decoded.get("signature") is not None:
        try:
            verify_decision_signature(
                decoded,
                verification_keys=verification_keys,
                expected=expected_bindings,
                now=moment,
            )
        except DecisionSignatureRefused as exc:
            _logger.error("/decide response signature REFUSED (%s)", exc.code)
            return DecideResult("error")
    elif require_signed_decisions:
        # Mandatory mode. Note this also covers ``"signature": null``, which is
        # absence, not a malformed signature object.
        _logger.error("/decide response carries no signature and require_signed_decisions is on")
        return DecideResult("error")

    # ── Outcome, and the governed reason code ────────────────────────────────
    decision = decoded.get("decision")
    reason = _governed_reason(decoded)
    if decision == DECISION_ALLOW:
        return DecideResult("allow", reason=reason, decision_id=decision_id)
    if decision == DECISION_DENY:
        return DecideResult("deny", reason=reason, decision_id=decision_id)
    _logger.warning("/decide answered an unrecognised decision")
    return DecideResult("error")


#: How stale a decision may be before it stops being an answer about now.
_MAX_DECISION_AGE = timedelta(seconds=60)

#: Real clocks disagree; a decision is not the place to discover it.
_CLOCK_SKEW = timedelta(seconds=30)


def _instant(value: Any) -> datetime | None:
    """Parse an ISO-8601 instant, or ``None``. A naive timestamp is refused.

    Refused rather than assumed UTC: the ambiguity is a whole timezone wide, and
    guessing errs in the direction that makes a decision look FRESHER than it is
    for any eastward offset.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _governed_reason(decoded: dict) -> str | None:
    """The governed reason code, from wherever this authority puts it.

    THIS WAS SILENTLY DROPPING EVERY REASON. The authority emits ``reason`` as
    an OBJECT — ``{"primary": ..., "contributing": [...], "category": ...}`` —
    and the previous implementation passed that dict straight to a string
    bounder, which answered ``None`` for every real response. The deny reason an
    operator needs in order to explain a refusal was therefore never surfaced,
    and nothing failed: a dropped reason looks exactly like a decision that had
    no reason to give.
    """
    reason = decoded.get("reason")
    if isinstance(reason, dict):
        return _bounded_str(reason.get("primary"), MAX_REASON_LEN)
    # A flat string, or the older ``reason_code`` spelling.
    return _bounded_str(decoded.get("reason_code") or reason, MAX_REASON_LEN)


def _adapter_version() -> str:
    """This package's version, for the authority's adapter evidence."""
    from mudraid_platform_middleware import __version__

    return __version__
