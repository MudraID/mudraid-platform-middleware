"""Middleware error hierarchy.

The middleware's normal failure mode is *not* a Python exception —
it returns a structured HTTP response (401, 403, 404) to the agent.
These exceptions are reserved for startup-time and operator-facing
failures that should bring down the process rather than be quietly
returned as errors:

  - Malformed ``mudraid_scopes.yaml``: dev mistake; raise at startup
    so it surfaces in deploy logs, not during the first request.
  - Unreachable / malformed JWKS at startup: same.

Subclasses are added in M5.2+ as the modules that raise them are
introduced.
"""

from __future__ import annotations


class MudraIDMiddlewareError(Exception):
    """Base class for every error raised by the MudraID middleware.

    Catching this base will catch every middleware-originated failure
    (YAML schema errors, JWKS bootstrap errors, etc.) without also
    catching the routine exceptions thrown by Starlette / FastAPI
    that aren't ours.
    """


class MudraIDScopesYamlError(MudraIDMiddlewareError):
    """``mudraid_scopes.yaml`` is missing, unreadable, or malformed.

    Raised at process startup (by the YAML loader, not by request
    handling) so the failure surfaces in deploy logs the first time
    the integrator runs the service, not on the first agent request.

    The message includes the offending line number when YAML syntax
    is the cause, and the route index when a schema rule is violated.
    """


class MudraIDJwksError(MudraIDMiddlewareError):
    """Could not fetch or use MudraID's JWKS.

    Covers four distinct failure shapes:

      - HTTP fetch failed (network error, timeout, non-2xx response)
      - Response body was not valid JSON
      - Response JSON did not contain a usable ``keys`` array
      - A JWT presented a ``kid`` that is not in the JWKS even after
        a refresh

    The middleware's per-request path catches this and returns a
    structured 401 / 500 to the agent depending on cause; the
    exception itself is never surfaced directly to the agent.
    """


class MudraIDInvalidTokenError(MudraIDMiddlewareError):
    """The presented JWT failed one of the validation gates.

    Carries a machine-readable ``reason`` so the middleware can
    surface a precise ``error_code`` in the structured response to
    the agent without leaking implementation details from the
    upstream PyJWT exception.

    Reasons (the only values this class will produce in v1):

      - ``"malformed"``         — header / body unparseable, or a
                                  required claim was missing
      - ``"expired"``           — ``exp`` is in the past (with leeway)
      - ``"not_yet_valid"``     — ``nbf`` is in the future (with leeway)
      - ``"wrong_audience"``    — ``aud`` doesn't bind to this platform
      - ``"wrong_issuer"``      — ``iss`` is not the expected MudraID
      - ``"invalid_signature"`` — signature did not verify against
                                  the matching JWKS key

    The original ``PyJWTError`` (if any) is chained via ``__cause__``
    for diagnostics without ever appearing in the message we surface
    to the agent.
    """

    def __init__(self, reason: str, message: str | None = None) -> None:
        super().__init__(message or reason)
        self.reason = reason
