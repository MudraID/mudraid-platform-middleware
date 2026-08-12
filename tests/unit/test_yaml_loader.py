"""M5.2 — YAML loader: parse, validate, fail loud.

Every test writes a single ``mudraid_scopes.yaml`` into a tmp_path
and asserts what the loader does with it. Schema rules are checked
one at a time so failure messages point at the exact rule that broke.

The schema-rule cases are deliberately tedious — they lock the
contract that the portal-side YAML exporter and the middleware
loader agree on. If either side drifts, one of these tests will
flag it before a customer hits it in production.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mudraid_platform_middleware import MudraIDScopesYamlError
from mudraid_platform_middleware._yaml_loader import RouteRule, ScopesYaml, load_scopes_yaml


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "mudraid_scopes.yaml"
    path.write_text(content, encoding="utf-8")
    return path


VALID_YAML = """\
platform_id: plt_abc
version: 1
routes:
  - method: GET
    path: /api/v1/items
    scope: items:read
  - method: POST
    path: /api/v1/items
    scope: items:write
  - method: GET
    path: /health
    public: true
  - method: POST
    path: /internal/admin
    skip: true
"""


# ---- happy path -----------------------------------------------------------


def test_valid_yaml_parses_into_frozen_dataclass(tmp_path: Path) -> None:
    path = _write(tmp_path, VALID_YAML)

    parsed = load_scopes_yaml(path)

    assert isinstance(parsed, ScopesYaml)
    assert parsed.platform_id == "plt_abc"
    assert parsed.version == 1
    assert len(parsed.routes) == 4

    # Routes preserve YAML order.
    assert parsed.routes[0] == RouteRule(method="GET", path="/api/v1/items", scope="items:read")
    assert parsed.routes[1] == RouteRule(method="POST", path="/api/v1/items", scope="items:write")
    assert parsed.routes[2] == RouteRule(method="GET", path="/health", public=True)
    assert parsed.routes[3] == RouteRule(method="POST", path="/internal/admin", skip=True)


def test_parsed_structure_is_immutable(tmp_path: Path) -> None:
    """Locked behaviour: the YAML is the source of truth; the
    parsed structure must not be mutable at runtime so the
    middleware can't drift from what's on disk."""
    parsed = load_scopes_yaml(_write(tmp_path, VALID_YAML))

    with pytest.raises(Exception):  # FrozenInstanceError
        parsed.platform_id = "evil"  # type: ignore[misc]
    with pytest.raises(Exception):
        parsed.routes[0].scope = "evil"  # type: ignore[misc]


def test_empty_routes_list_is_valid(tmp_path: Path) -> None:
    """A platform that has registered with MudraID but hasn't mapped
    any routes yet is a valid state — every request will 404 to an
    agent (no rule matches), which is the developer's choice."""
    path = _write(tmp_path, "platform_id: plt_a\nversion: 1\nroutes: []\n")

    parsed = load_scopes_yaml(path)
    assert parsed.routes == ()


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
def test_every_locked_http_method_is_accepted(tmp_path: Path, method: str) -> None:
    yaml = f"""\
platform_id: plt_a
version: 1
routes:
  - method: {method}
    path: /x
    public: true
"""
    parsed = load_scopes_yaml(_write(tmp_path, yaml))
    assert parsed.routes[0].method == method


def test_root_path_slash_is_accepted(tmp_path: Path) -> None:
    yaml = """\
platform_id: plt_a
version: 1
routes:
  - method: GET
    path: /
    public: true
"""
    parsed = load_scopes_yaml(_write(tmp_path, yaml))
    assert parsed.routes[0].path == "/"


def test_scope_whitespace_is_stripped(tmp_path: Path) -> None:
    """Defensive: a trailing space in the YAML shouldn't make the
    middleware refuse to match `items:read`."""
    yaml = """\
platform_id: plt_a
version: 1
routes:
  - method: GET
    path: /x
    scope: "  items:read  "
"""
    parsed = load_scopes_yaml(_write(tmp_path, yaml))
    assert parsed.routes[0].scope == "items:read"


# ---- file-level errors ---------------------------------------------------


def test_missing_file_raises_with_actionable_message(tmp_path: Path) -> None:
    with pytest.raises(MudraIDScopesYamlError, match="not found"):
        load_scopes_yaml(tmp_path / "absent.yaml")


def test_malformed_yaml_reports_line_number(tmp_path: Path) -> None:
    """A developer with a broken file shouldn't have to bisect — the
    PyYAML line number is bubbled up into the message.

    The payload below is genuinely unparseable: an unclosed
    single-quoted scalar that runs to EOF. PyYAML surfaces the
    starting line of the broken token, which the loader bubbles
    up into the error message."""
    path = _write(
        tmp_path,
        "platform_id: plt_a\nversion: 1\nroutes:\n  - method: GET\n    path: 'unclosed\n",
    )

    with pytest.raises(MudraIDScopesYamlError, match=r"line \d+"):
        load_scopes_yaml(path)


def test_top_level_not_a_mapping_is_rejected(tmp_path: Path) -> None:
    """A bare list at the top level (a common mistake — copying just
    the routes array) is caught with a clear message."""
    path = _write(tmp_path, "- method: GET\n  path: /x\n  public: true\n")

    with pytest.raises(MudraIDScopesYamlError, match="top-level YAML must be a mapping"):
        load_scopes_yaml(path)


# ---- top-level schema errors ---------------------------------------------


def test_missing_platform_id_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "version: 1\nroutes: []\n")
    with pytest.raises(MudraIDScopesYamlError, match="platform_id"):
        load_scopes_yaml(path)


def test_empty_platform_id_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "platform_id: ''\nversion: 1\nroutes: []\n")
    with pytest.raises(MudraIDScopesYamlError, match="platform_id"):
        load_scopes_yaml(path)


def test_missing_version_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "platform_id: plt_a\nroutes: []\n")
    with pytest.raises(MudraIDScopesYamlError, match="version"):
        load_scopes_yaml(path)


def test_non_integer_version_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "platform_id: plt_a\nversion: '1'\nroutes: []\n")
    with pytest.raises(MudraIDScopesYamlError, match="version"):
        load_scopes_yaml(path)


def test_boolean_version_is_rejected(tmp_path: Path) -> None:
    """``True`` is technically ``int`` in Python; the loader must
    treat it as a schema error. Locks the explicit bool guard."""
    path = _write(tmp_path, "platform_id: plt_a\nversion: true\nroutes: []\n")
    with pytest.raises(MudraIDScopesYamlError, match="version"):
        load_scopes_yaml(path)


def test_zero_or_negative_version_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "platform_id: plt_a\nversion: 0\nroutes: []\n")
    with pytest.raises(MudraIDScopesYamlError, match="version"):
        load_scopes_yaml(path)


def test_missing_routes_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "platform_id: plt_a\nversion: 1\n")
    with pytest.raises(MudraIDScopesYamlError, match="routes"):
        load_scopes_yaml(path)


def test_routes_not_a_list_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "platform_id: plt_a\nversion: 1\nroutes: {}\n")
    with pytest.raises(MudraIDScopesYamlError, match="routes.*list"):
        load_scopes_yaml(path)


# ---- per-route schema errors ---------------------------------------------


def test_route_missing_method_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "platform_id: plt_a\nversion: 1\nroutes:\n  - path: /x\n    public: true\n",
    )
    with pytest.raises(MudraIDScopesYamlError, match=r"routes\[0\].*method"):
        load_scopes_yaml(path)


def test_route_invalid_method_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        (
            "platform_id: plt_a\n"
            "version: 1\n"
            "routes:\n"
            "  - method: WAT\n"
            "    path: /x\n"
            "    public: true\n"
        ),
    )
    with pytest.raises(MudraIDScopesYamlError, match=r"routes\[0\].*method"):
        load_scopes_yaml(path)


def test_route_lowercase_method_is_rejected(tmp_path: Path) -> None:
    """Locked schema enum is uppercase only. A future contributor
    must not silently uppercase callers' input — that would mask a
    portal/loader divergence."""
    path = _write(
        tmp_path,
        (
            "platform_id: plt_a\n"
            "version: 1\n"
            "routes:\n"
            "  - method: get\n"
            "    path: /x\n"
            "    public: true\n"
        ),
    )
    with pytest.raises(MudraIDScopesYamlError, match=r"routes\[0\].*method"):
        load_scopes_yaml(path)


def test_route_missing_path_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "platform_id: plt_a\nversion: 1\nroutes:\n  - method: GET\n    public: true\n",
    )
    with pytest.raises(MudraIDScopesYamlError, match=r"routes\[0\].*path"):
        load_scopes_yaml(path)


def test_route_path_without_leading_slash_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        (
            "platform_id: plt_a\n"
            "version: 1\n"
            "routes:\n"
            "  - method: GET\n"
            "    path: api/v1/items\n"
            "    public: true\n"
        ),
    )
    with pytest.raises(MudraIDScopesYamlError, match=r"routes\[0\].*path"):
        load_scopes_yaml(path)


# ---- mode-mutual-exclusion (scope / public / skip) -----------------------


def test_route_with_no_mode_is_rejected(tmp_path: Path) -> None:
    """A route with neither scope, public, nor skip is undefined
    behaviour — the loader makes the developer choose."""
    path = _write(
        tmp_path,
        "platform_id: plt_a\nversion: 1\nroutes:\n  - method: GET\n    path: /x\n",
    )
    with pytest.raises(MudraIDScopesYamlError, match="got none"):
        load_scopes_yaml(path)


def test_route_with_scope_and_public_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        (
            "platform_id: plt_a\n"
            "version: 1\n"
            "routes:\n"
            "  - method: GET\n"
            "    path: /x\n"
            "    scope: a:b\n"
            "    public: true\n"
        ),
    )
    with pytest.raises(MudraIDScopesYamlError, match="got multiple"):
        load_scopes_yaml(path)


def test_route_with_scope_and_skip_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        (
            "platform_id: plt_a\n"
            "version: 1\n"
            "routes:\n"
            "  - method: GET\n"
            "    path: /x\n"
            "    scope: a:b\n"
            "    skip: true\n"
        ),
    )
    with pytest.raises(MudraIDScopesYamlError, match="got multiple"):
        load_scopes_yaml(path)


def test_route_with_public_and_skip_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        (
            "platform_id: plt_a\n"
            "version: 1\n"
            "routes:\n"
            "  - method: GET\n"
            "    path: /x\n"
            "    public: true\n"
            "    skip: true\n"
        ),
    )
    with pytest.raises(MudraIDScopesYamlError, match="got multiple"):
        load_scopes_yaml(path)


def test_route_with_empty_scope_string_is_rejected(tmp_path: Path) -> None:
    """``scope: ""`` is meaningless — refuse rather than silently
    treat it as 'no mode chosen'."""
    path = _write(
        tmp_path,
        "platform_id: plt_a\nversion: 1\nroutes:\n  - method: GET\n    path: /x\n    scope: ''\n",
    )
    with pytest.raises(MudraIDScopesYamlError):
        load_scopes_yaml(path)


def test_route_with_non_bool_public_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        (
            "platform_id: plt_a\n"
            "version: 1\n"
            "routes:\n"
            "  - method: GET\n"
            "    path: /x\n"
            "    public: yes please\n"
        ),
    )
    with pytest.raises(MudraIDScopesYamlError, match="public"):
        load_scopes_yaml(path)


def test_route_with_public_false_falls_through_to_no_mode(tmp_path: Path) -> None:
    """``public: false`` with no other mode is the same as omitting
    public entirely — the loader rejects it as 'no mode chosen'.
    Without this, a developer who flipped a route from public to
    scoped but forgot the scope line would have a silently-broken
    route."""
    path = _write(
        tmp_path,
        (
            "platform_id: plt_a\n"
            "version: 1\n"
            "routes:\n"
            "  - method: GET\n"
            "    path: /x\n"
            "    public: false\n"
        ),
    )
    with pytest.raises(MudraIDScopesYamlError, match="got none"):
        load_scopes_yaml(path)


def test_route_not_a_mapping_is_rejected(tmp_path: Path) -> None:
    """A route entry that is somehow a bare string in YAML — caught
    with a clear per-index message rather than a Python TypeError."""
    path = _write(
        tmp_path,
        "platform_id: plt_a\nversion: 1\nroutes:\n  - 'not a mapping'\n",
    )
    with pytest.raises(MudraIDScopesYamlError, match=r"routes\[0\].*mapping"):
        load_scopes_yaml(path)


# ---- path argument forms -------------------------------------------------


def test_loader_accepts_string_path(tmp_path: Path) -> None:
    """The integrator may pass a ``str`` or ``Path`` — both work."""
    path = _write(tmp_path, VALID_YAML)
    parsed = load_scopes_yaml(str(path))
    assert parsed.platform_id == "plt_abc"
