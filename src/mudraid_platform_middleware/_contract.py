"""The versioned decision contract — one name, one supported list, both ends.

The adapter and the authority have to agree on the shape of a decision exchange,
and "agree" has to mean something checkable. A wire format nobody validates is a
wire format that drifts silently until the day a field changes meaning and every
adapter keeps returning ``allow``.

So the contract is named and versioned, and BOTH directions are checked:

  - the adapter stamps :data:`REQUEST_CONTRACT` on every request, and the
    authority rejects an absent or unsupported version rather than guessing;
  - the authority stamps :data:`RESPONSE_CONTRACT` on every decision, and the
    adapter refuses to read a response that does not carry a supported one.

Request and response versions are deliberately SEPARATE names. They evolve for
different reasons — a new request field does not change how a decision is
expressed — and collapsing them into one number would force a lockstep upgrade
across two services that do not deploy together.

Everything unsupported, absent, malformed or oversized resolves to DENY. There is
no version negotiation and no best-effort parse: an adapter that cannot be sure
what a response means must not act on it.
"""

from __future__ import annotations

__all__ = [
    "DECISION_ALLOW",
    "DECISION_DENY",
    "MAX_DECISION_ID_LEN",
    "MAX_REASON_LEN",
    "MAX_RESPONSE_BYTES",
    "REQUEST_CONTRACT",
    "RESPONSE_CONTRACT",
    "SUPPORTED_REQUEST_CONTRACTS",
    "SUPPORTED_RESPONSE_CONTRACTS",
]

#: The request contract this adapter speaks. Replaces the unversioned
#: ``mudraid.enforce.decide.provisional-1`` envelope: that name described itself
#: as provisional and nothing validated it, so it could not be the thing a
#: customer-installed package depends on.
REQUEST_CONTRACT = "mudraid.enforce.decide-request/1"

#: The response envelope version this adapter accepts.
#:
#: This is the authority's OWN version string, not one this adapter invented.
#: ``enforcement-service`` stamps ``DECIDE_SCHEMA_VERSION = "2.0"`` on every
#: decision envelope (doc 08), and a client that required a different name would
#: reject every genuine response while passing every test written against its own
#: fixture — the same shape of defect as a seam that cannot carry a credential.
#: Adopt the version that exists; do not mint a parallel one.
RESPONSE_CONTRACT = "2.0"

#: Explicitly enumerated rather than range-matched. Adding a version is a
#: deliberate act with a test behind it, not a consequence of a string compare.
SUPPORTED_REQUEST_CONTRACTS: frozenset[str] = frozenset({REQUEST_CONTRACT})
SUPPORTED_RESPONSE_CONTRACTS: frozenset[str] = frozenset({RESPONSE_CONTRACT})

#: The closed decision vocabulary. Anything else — including a new word a future
#: authority might introduce — is not optimistically interpreted.
DECISION_ALLOW = "allow"
DECISION_DENY = "deny"

#: Bounds. A response is a small JSON object; anything at these scales is a
#: fault or an attack, and both are answered the same way.
MAX_RESPONSE_BYTES = 64 * 1024
MAX_DECISION_ID_LEN = 128
MAX_REASON_LEN = 256
