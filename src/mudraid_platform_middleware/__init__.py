"""mudraid-middleware — FastAPI/Starlette middleware for MudraID.

Drop-in scope enforcement for platforms registered with the MudraID trust
layer. Two mutually exclusive modes:

  - ``mode="v1"`` (default) verifies MudraID-issued JWTs against the route
    scopes declared in ``mudraid_scopes.yaml``;
  - ``mode="v2"`` runs the portable enforcement control loop — reserved header
    strip, bounded JSON-RPC framing, exact canonical action resolution and a
    live, deny-closed ``/decide`` call.

See :class:`MudraIDMiddleware` for the per-request flow and
:class:`~mudraid_platform_middleware.v2.V2Config` for V2 configuration.
"""

from mudraid_platform_middleware.decide_client import HttpDecideClient
from mudraid_platform_middleware.exceptions import (
    MudraIDInvalidTokenError,
    MudraIDJwksError,
    MudraIDMiddlewareError,
    MudraIDScopesYamlError,
)
from mudraid_platform_middleware.middleware import MudraIDMiddleware
from mudraid_platform_middleware.v2 import (
    DecideClient,
    DecideContext,
    DecideResult,
    V2Config,
)

__all__ = [
    "MudraIDMiddleware",
    "MudraIDMiddlewareError",
    "MudraIDScopesYamlError",
    "MudraIDJwksError",
    "MudraIDInvalidTokenError",
    # V2 mode
    "V2Config",
    "DecideClient",
    "DecideContext",
    "HttpDecideClient",
    "DecideResult",
]

# Single source of truth is pyproject.toml (issue #116): read the installed
# package metadata, falling back to the packaged literal for a source checkout
# where the distribution isn't installed. Keep the fallback equal to pyproject's
# version so the two never diverge again.
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("mudraid-platform-middleware")
except PackageNotFoundError:  # source checkout, not pip-installed
    # MUST match pyproject's `version`. `test_version_single_source` pins the
    # two together, and it caught this line being left behind when 1.2.0 was
    # bumped to 1.2.1 for KAN-163 — which is precisely the drift issue #116
    # created the guard for.
    __version__ = "1.2.1"
