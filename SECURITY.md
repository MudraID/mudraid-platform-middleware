# Security policy

## Reporting a vulnerability

**Do not open a public issue for a security problem.** Report it privately to
**security@mudraid.ai**, and we will acknowledge within **2 business days**.

Please include, as far as you can establish it:

- what an attacker can do — the effect, not only the flaw;
- the version you tested, and how you installed it;
- a reproduction, or the smallest thing that shows the behaviour.

You will get a substantive reply, not only an acknowledgement: what we
reproduced, what we could not, and what we intend to do. If we disagree that
something is a vulnerability we will say so and explain why, rather than letting
the report go quiet.

We will not pursue legal action over good-faith research that stays within your
own accounts and data, does not degrade the service for others, and does not
access or retain anyone else's information.

## What is in scope

This repository — the library's own code and the wiring it documents.

The service it talks to is a separate system with the same contact address, and
a report about one is welcome under the other; we would rather route it
ourselves than have you guess which it belongs to.

**`mudraid-sdk` is a different package with its own policy.** It is the
client-side SDK an agent uses to obtain and present credentials for its own
outbound calls; it makes no authorization decision and enforces nothing. The
deny-closed model below is this package's, not that one's. Both addresses are
the same, so a misrouted report is not a lost one.

## What this library does and does not do

Worth stating plainly, because a report is often about the difference. This is
the **server-side enforcement** library — it consumes policy bundles and turns
them into allow/deny outcomes on a request path:

- It carries a **deny-closed** enforcement model. A decision that cannot be
  obtained, verified, or read within its freshness window is a denial, never an
  allow. **A path that reaches an allow without a verified decision is a
  vulnerability in this library**, and is the class of report we most want.
- The MudraID authority **does not sign decision responses** in this release,
  and this library does not claim it does. The library can *verify* an RS256
  decision signature when one is present, and can be configured to *require*
  one (`HttpDecideClient(require_signed_decisions=True)`) — but that flag
  defaults to off and is off in every environment, so decision responses are
  **not cryptographically authenticated today**. Do not build a trust
  assumption on a signature that is not there, and do not read the availability
  of mandatory mode as the protection being active — see the README for what is
  and is not authenticated today.
- It holds a credential you supply. It never writes one to disk and never logs
  one; **a credential appearing in any log line is a vulnerability**, and one we
  will treat as such even when nothing else is exploitable.

## Supported versions

Maturity and support for every published version are declared in the MudraID
adapter support matrix, which is the authority here — not this section, and not
a marketing page. It is not shipped inside this distribution; ask
security@mudraid.ai for the entry covering the version you tested.

This package is **1.x**. Report against the latest published version where you
can, and say which version you tested where you cannot.
