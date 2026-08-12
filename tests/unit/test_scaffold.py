"""M5.1 smoke tests for the middleware scaffold.

These assert only that the package is importable and the public
names exist. Behavioural tests are added in M5.2+ as each module
gains real logic.
"""

from __future__ import annotations


def test_public_api_is_importable() -> None:
    """A platform operator's
    ``from mudraid_platform_middleware import MudraIDMiddleware``
    must work from a fresh install."""
    from mudraid_platform_middleware import MudraIDMiddleware, MudraIDMiddlewareError

    assert MudraIDMiddleware is not None
    assert issubclass(MudraIDMiddlewareError, Exception)


def test_version_is_exposed() -> None:
    """Tooling and operator scripts need ``mudraid_platform_middleware.__version__``."""
    import mudraid_platform_middleware

    assert isinstance(mudraid_platform_middleware.__version__, str)
    assert mudraid_platform_middleware.__version__.count(".") == 2  # semver-shaped


def test_middleware_subclasses_starlette_basehttpmiddleware() -> None:
    """Locks the inheritance contract — ``app.add_middleware(MudraIDMiddleware)``
    requires it to be a Starlette ``BaseHTTPMiddleware``. If a future
    refactor swaps the base class to a custom ASGI wrapper, FastAPI's
    ``add_middleware`` machinery may break in subtle ways."""
    from starlette.middleware.base import BaseHTTPMiddleware

    from mudraid_platform_middleware import MudraIDMiddleware

    assert issubclass(MudraIDMiddleware, BaseHTTPMiddleware)


def test_middleware_constructor_accepts_optional_yaml_path() -> None:
    """Construction must succeed with no extra args, and accept the
    documented ``scopes_yaml_path=`` override — otherwise downstream
    tasks can't write tests with custom fixture YAML paths."""

    from mudraid_platform_middleware import MudraIDMiddleware

    # Dummy ASGI app — the middleware constructor only stores it.
    async def app(scope, receive, send):  # pragma: no cover — never called here
        ...

    # A JWKS URL is now REQUIRED in V1 mode — there is no compiled-in default,
    # because a MudraID hostname in the package would make the build
    # environment-specific. Supplying it here is the whole change to this test.
    jwks = "https://api.example.test/.well-known/jwks.json"
    MudraIDMiddleware(app, jwks_url=jwks)
    MudraIDMiddleware(app, jwks_url=jwks, scopes_yaml_path="/tmp/custom_scopes.yaml")


# test_dispatch_raises_until_m5_6_implements_it — removed in M5.6
# along with the NotImplementedError stub it was guarding. Real
# behavioural coverage of dispatch lives in test_middleware_dispatch.py.
