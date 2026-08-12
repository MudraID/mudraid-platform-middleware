"""M5.3 — RouteMatcher: pattern compilation + (method, path) lookup.

The compilation is unit-tested through the public ``match()`` API to
keep the implementation free to swap regex strategies without
churning tests. Each locked behaviour from the module docstring has a
dedicated case.
"""

from __future__ import annotations

from mudraid_platform_middleware._route_matcher import RouteMatcher
from mudraid_platform_middleware._yaml_loader import RouteRule


def _route(method: str, path: str, **kwargs: object) -> RouteRule:
    """Build a RouteRule with whatever mode kwargs the test cares about.

    Most match tests don't care which mode is selected — they assert on
    "did we hit this rule?". A no-mode-set rule wouldn't pass M5.2's
    loader, but the matcher itself doesn't care about mode, only
    structure. Default to ``public=True`` so RouteRule construction
    is valid without forcing every test to spell it out."""
    if not any(k in kwargs for k in ("scope", "public", "skip")):
        kwargs["public"] = True
    return RouteRule(method=method, path=path, **kwargs)  # type: ignore[arg-type]


# ---- exact-literal matches -----------------------------------------------


def test_exact_literal_path_matches() -> None:
    matcher = RouteMatcher([_route("GET", "/api/v1/items")])

    matched = matcher.match("GET", "/api/v1/items")

    assert matched is not None
    assert matched.path == "/api/v1/items"


def test_root_path_matches() -> None:
    matcher = RouteMatcher([_route("GET", "/")])

    assert matcher.match("GET", "/") is not None


def test_no_rule_in_yaml_returns_none() -> None:
    """The middleware uses ``None`` as the 'route not covered' signal —
    M5.8 will translate it to a 404 to agents."""
    matcher = RouteMatcher([])

    assert matcher.match("GET", "/anything") is None


def test_path_not_in_yaml_returns_none() -> None:
    matcher = RouteMatcher([_route("GET", "/registered")])

    assert matcher.match("GET", "/unregistered") is None


# ---- path-param substitution ---------------------------------------------


def test_single_path_param_matches_concrete_segment() -> None:
    matcher = RouteMatcher([_route("GET", "/items/{id}")])

    assert matcher.match("GET", "/items/abc123") is not None
    assert matcher.match("GET", "/items/42") is not None


def test_multiple_path_params_match_independently() -> None:
    matcher = RouteMatcher([_route("GET", "/items/{id}/comments/{cid}")])

    assert matcher.match("GET", "/items/42/comments/7") is not None
    assert matcher.match("GET", "/items/x/comments/y") is not None


def test_path_param_does_not_cross_slash() -> None:
    """``{id}`` must match a single URL segment, not many. Without
    this lock, ``/items/{id}`` would silently match
    ``/items/abc/def``, which would route the wrong rule and
    enforce the wrong scope."""
    matcher = RouteMatcher([_route("GET", "/items/{id}")])

    assert matcher.match("GET", "/items/abc/def") is None


def test_path_param_does_not_match_empty_segment() -> None:
    """``/items/`` is NOT a match for ``/items/{id}`` because there
    is no value for ``id``. Locks the regex's ``[^/]+`` (one or
    more, not zero or more)."""
    matcher = RouteMatcher([_route("GET", "/items/{id}")])

    assert matcher.match("GET", "/items/") is None


# ---- literal segments respect URL-reserved characters --------------------


def test_dot_in_literal_segment_is_escaped() -> None:
    """``/api/v1.5/items`` must match ``/api/v1.5/items`` and NOT
    ``/api/v15/items`` — the dot in the pattern is literal, not
    regex-wildcard."""
    matcher = RouteMatcher([_route("GET", "/api/v1.5/items")])

    assert matcher.match("GET", "/api/v1.5/items") is not None
    assert matcher.match("GET", "/api/v15/items") is None


def test_plus_and_question_in_literal_segment_are_escaped() -> None:
    matcher = RouteMatcher([_route("GET", "/foo+bar")])

    assert matcher.match("GET", "/foo+bar") is not None
    assert matcher.match("GET", "/foobar") is None  # without plus
    assert matcher.match("GET", "/foooobar") is None  # without plus, repeated o


# ---- method dimension ----------------------------------------------------


def test_method_must_match_exactly() -> None:
    matcher = RouteMatcher([_route("GET", "/x")])

    assert matcher.match("GET", "/x") is not None
    assert matcher.match("POST", "/x") is None
    assert matcher.match("DELETE", "/x") is None


def test_method_match_is_case_sensitive() -> None:
    """The YAML schema locks methods to uppercase. Incoming HTTP
    methods are uppercase per RFC 7230. Don't normalise — a
    lowercase incoming method is genuinely unexpected."""
    matcher = RouteMatcher([_route("GET", "/x")])

    assert matcher.match("get", "/x") is None


def test_same_path_different_methods_get_independent_rules() -> None:
    rules = [
        _route("GET", "/items", scope="items:read", public=False),
        _route("POST", "/items", scope="items:write", public=False),
    ]
    matcher = RouteMatcher(rules)

    get = matcher.match("GET", "/items")
    post = matcher.match("POST", "/items")

    assert get is not None and get.scope == "items:read"
    assert post is not None and post.scope == "items:write"


# ---- ordering / first-match-wins -----------------------------------------


def test_literal_listed_before_parametric_wins_for_concrete_path() -> None:
    """Lets a platform owner pin a special-case literal route ahead
    of the catch-all parametric one. ``/items/new`` listed first
    matches ``/items/new`` to the literal rule; ``/items/42``
    falls through to the parametric one."""
    matcher = RouteMatcher(
        [
            _route("GET", "/items/new", scope="items:create"),
            _route("GET", "/items/{id}", scope="items:read"),
        ]
    )

    new_route = matcher.match("GET", "/items/new")
    id_route = matcher.match("GET", "/items/42")

    assert new_route is not None and new_route.scope == "items:create"
    assert id_route is not None and id_route.scope == "items:read"


def test_parametric_listed_before_literal_wins_for_both_paths() -> None:
    """The mirror case proves order matters. If a platform owner
    writes the parametric rule first by mistake, the literal special
    case is silently shadowed — and that's the developer's choice;
    we don't pick favourites."""
    matcher = RouteMatcher(
        [
            _route("GET", "/items/{id}", scope="items:read"),
            _route("GET", "/items/new", scope="items:create"),
        ]
    )

    # Both paths match the parametric rule first.
    assert matcher.match("GET", "/items/new").scope == "items:read"  # type: ignore[union-attr]
    assert matcher.match("GET", "/items/42").scope == "items:read"  # type: ignore[union-attr]


# ---- trailing slash / case sensitivity on path ---------------------------


def test_trailing_slash_is_significant() -> None:
    """``/items`` and ``/items/`` are different routes. Locks the
    'strict trailing slash' behaviour so a misconfigured route can
    fail loud, not silently route to the wrong rule."""
    matcher = RouteMatcher([_route("GET", "/items")])

    assert matcher.match("GET", "/items") is not None
    assert matcher.match("GET", "/items/") is None


def test_path_match_is_case_sensitive() -> None:
    matcher = RouteMatcher([_route("GET", "/Items")])

    assert matcher.match("GET", "/Items") is not None
    assert matcher.match("GET", "/items") is None


# ---- mode preservation across the matcher --------------------------------


def test_matched_rule_returns_full_rule_object() -> None:
    """The middleware needs the full RouteRule (method, path, scope,
    public, skip) to decide what to do with the request. The matcher
    must hand it back intact, not just the metadata used for
    matching."""
    rule = _route("GET", "/x", scope="x:y", public=False)
    matcher = RouteMatcher([rule])

    matched = matcher.match("GET", "/x")
    assert matched is rule


# ---- diagnostics ---------------------------------------------------------


def test_matcher_supports_len_for_introspection() -> None:
    matcher = RouteMatcher(
        [
            _route("GET", "/a"),
            _route("POST", "/b"),
            _route("DELETE", "/c"),
        ]
    )
    assert len(matcher) == 3
