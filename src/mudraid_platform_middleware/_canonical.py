"""Canonical JSON bytes — the exact serialization the bundle signer hashes.

A signed bundle's ``payload_digest`` and ``signature`` are computed over ONE
serialization of the payload, and a verifier that reproduces different bytes
rejects a valid bundle (or, far worse, accepts a tampered one because it
normalized the difference away). So the canonical form is not a preference here;
it is the thing being verified.

The control plane defines it in
``services/platform-integration-service/app/application/bundle_compiler.py``::

    json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")

which this module reproduces exactly. Note what that means for a Python
verifier: the signer IS Python, so canonicalization here is the reference form
rather than an imitation of it. The Kong plugin's ``canonical.lua`` is the one
carrying the burden of matching — it hand-rolls key sorting, ``\\uXXXX``
escaping and surrogate pairs to land on these same bytes.

What this module adds over a bare ``json.dumps`` is REFUSAL. A digest computed
over a "best effort" serialization would silently accept tampering, so a value
that cannot be canonicalized with certainty raises rather than being
approximated:

  - ``NaN`` / ``Infinity`` / ``-Infinity`` — accepted by Python's JSON parser by
    default and re-emitted as bare tokens that are not JSON at all. The signer
    never produces them, so their presence means the response is not a bundle
    this verifier can speak about.
  - non-integer floats — the bundle contract carries only integers (versions,
    revisions, counts), and a float has no cross-language canonical form.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["CanonicalizationError", "canonical_json_bytes", "loads_strict", "sha256_hex"]


class CanonicalizationError(ValueError):
    """A value cannot be canonicalized with certainty, so it is refused."""


def _reject_constant(token: str) -> Any:
    raise CanonicalizationError(
        f"JSON document contains the non-standard constant {token!r}; "
        "a signed bundle never carries one"
    )


def loads_strict(raw: str | bytes) -> Any:
    """Parse JSON, refusing ``NaN``/``Infinity``/``-Infinity``.

    Python's parser accepts those by default. They cannot appear in anything the
    signer produced, and they do not round-trip through any other JSON
    implementation, so a document containing one is refused at the door rather
    than carried into a digest.
    """
    return json.loads(raw, parse_constant=_reject_constant)


def _check_canonicalizable(value: Any) -> None:
    """Walk a decoded value, refusing anything without a certain canonical form."""
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        # Reached only via a literal like 1.5; NaN/Infinity are already refused
        # by loads_strict. Integer-valued floats are refused too: `1.0` serializes
        # as "1.0" here and as "1" from an int, so accepting it would mean two
        # different byte strings for one logical value.
        raise CanonicalizationError(
            f"non-integer number {value!r} has no canonical form across languages"
        )
    if isinstance(value, str):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(f"non-string object key {key!r}")
            _check_canonicalizable(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _check_canonicalizable(item)
        return
    raise CanonicalizationError(f"unsupported value type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """The exact bytes the control plane hashed and signed.

    Raises:
        CanonicalizationError: the value contains something with no certain
            canonical form. Callers treat that as an invalid bundle.
    """
    _check_canonicalizable(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(value: Any) -> str:
    """SHA-256 of the canonical bytes, lowercase hex — the signer's digest."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
