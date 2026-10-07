# Reproducibility and evidence limits

SPDX-License-Identifier: CC-BY-4.0

This guide gives information about source identities, reproducibility records and evidence limits.
It separates software reproduction, historical paper derivatives and fresh scientific qualification.
Use it to retain the records required for a stated result.

`uv.lock` records the target Python dependency solution. The portable reactor
prefix additionally pins Python 3.11.14 and NumPy 2.4.6 and validates the hashes
of its three packaged public inputs. Run it as described in
[examples](experiments/reactor-response/guide.md). Its report records runtime, source hashes,
relation/action identifiers, branch completion, observed delivery stages and
paired contrasts by independent unit and numerical view.

The separate [fresh campaign route](docs/fresh-experiment.md) supplies strict
target-owned authoring inputs and a no-effect preissue proof for a new target
identity. Its issued native run, if performed, has its own external controls,
task receipts, reveal record and adjudication. The example report is never a
substitute for those records. [Public source declarations](docs/sources.md)
include one bounded FAIR-MAST Zarr selection. A verified acquisition receipt
can become a prospective design input. It does not make an unqualified native
observation scientifically fit.

A complete campaign needs a clean checkout executing its tracked package and an explicit `OperatorStorageProfile`.
It also needs exact source/input identities, frozen proposal, approvals, custody authority, issued package, resource contract, guarded store and receipts. Candidate compilation
does not grant any of these. The local SQLite database is a rebuildable index,
not the scientific record. A site-packages wheel may inspect documents and run
the example, but cannot assert the source checkout's clean-Git implementation
identity for source-bound issue or execution. There is no implicit storage
mount or signing key.

The manuscript's historical primary archives, trajectories, immutable receipts
and adjudications have not been deposited in this repository.
The selected [published preprint](paper/editions/v0.60-deposit/Empirical_Lawhood_Manuscript_v0.60.pdf) is edition **0.60**.
Its deposited PDF and the separately supplied manuscript, companion and six figure assets retain their selected original bytes.
The [curated source ZIP](paper/editions/Empirical_Lawhood_v0.60_Selected_Source.zip) contains 17 unchanged current-edition members; the complete 40-member original ZIP remains unchanged in external custody.
The [member map](paper/source/canonical-inputs.json) and [preservation receipt](paper/source/source-preservation.json) distinguish that public selection from the complete original archive.
Supplied figure provenance retains its historical source/output identities; the current selected figures have their own exact member hashes and visual review.
Optional publication derivatives can be built separately, but the packaged example does **not** reproduce the manuscript's empirical evidence.
[Paper sources](paper/SOURCES.md) records which citations point to unpublished primary material.
A later evidence release needs its original code identity, configuration, native source, independent-unit roster, outputs and verification records.
Historical producing identities remain separate from the current package.

A new result should record the exact target version, command, environment,
configuration, source and seed roster, independent-unit definition, output
hashes, receipts and adjudication. Use the [provenance template](configs/templates/evidence-provenance.yaml)
as a starting point and state absent fields explicitly.

## Provenance and reviewed derivatives

The repository ships a provenance template, not historical raw trajectories,
receipts, adjudications or a paper evidence archive. The selected edition 0.60 manuscript’s
[public source register](paper/SOURCES.md) identifies the unavailable
primary records behind its reported claims. The reactor prefix writes local
demonstration output wherever the operator chooses. That output is not a
qualified law result or replacement historical evidence.

Project-generated derivatives released under the project's CC0-1.0 policy
retain that dedication unless specifically marked otherwise. See the
[licensing scopes](docs/licensing.md). For each record, identify the producing
code state, protocol and command, source and native version, configuration,
independent units, seeds, output hashes and receipts. The
[template](configs/templates/evidence-provenance.yaml) is a starting point. Remove inapplicable
fields instead of inventing values. Third-party inputs retain their own rights.

## AI usage account

This account measures recorded Codex token usage in a bounded development snapshot.
It separates measured response records from unavailable usage and scientific evidence.
Use [the aggregate](docs/ai-usage.json) to inspect the totals, coverage, source hashes and recorded model-context/task breakdowns.

### Scope and cutoff

The inventory contains 287 local project-directory exports and eight retained isolated implementation-comparison exports.
Project-directory selection uses the recorded working directory or workspace roots for the parent project and public extraction.
This scope includes review, coordination and failed attempts within the selected sessions.
It does not allocate individual requests by research topic.
Other-workspace exports are excluded.

The selected response-level subtotal contains 48,447 unique responses from 62 exports.
Its measured interval runs from `2026-09-04T21:45:19.090Z` to `2026-10-02T08:59:54.056Z`.
The cutoff is `2026-10-02T09:00:00.000Z`.
Later release activity is outside this snapshot unless a new account is selected.

Coverage is incomplete.
Of the 233 other exports, 232 contain cumulative counters alone. One has no usage metadata.
All 142 inventoried child-agent exports lack response-ID usage records and remain outside the subtotal.
Their usage, unexported work and other-client work are unknown here.
Unknown usage is not zero.
The subtotal is not a lifetime or complete-project total.

### Read-only calculation

The calculation method is `unique-response-id-subtotal`.
The maintainer packet retains the calculator and independent source audit. The aggregate binds their hashes and private source-prefix identities.
To reproduce the calculation with access to those source exports:

1. Verify each recorded source-prefix length and SHA-256 hash.
2. Select `token_usage_record` entries at or before the UTC cutoff.
3. Read each entry's `response_id` and `usage` fields.
4. Deduplicate response IDs across exports, forks and resumed sessions.
5. Reject conflicting usage for the same response ID.
6. Require nonnegative integer counts and the token identities below.
7. Sum each unique response once, then group by recorded context, task and session role.
8. Compare each first `thread_token_usage` snapshot and every later increment with the response ledger.
9. Recompute the sums directly from the raw prefixes with an independent uniqueness check.

The selected ledger has no duplicate or conflicting response IDs.
All 62 first thread snapshots and 48,385 subsequent increments match the ledger exactly.
An independent raw-prefix calculation with SQL uniqueness reproduces every subtotal and breakdown.
All 295 source-prefix hashes were rechecked.

The retained usage schema has these identities:

```text
total_tokens = input_tokens + output_tokens
0 <= cached_input_tokens <= input_tokens
0 <= reasoning_output_tokens <= output_tokens
```

Cached input and reasoning output are subsets, not additional summands.
Cache-write counts are reported separately and are zero in this selected account.
Counts come from retained usage metadata, not file sizes, commits or elapsed time.

### Counter and model limits

The `token_count` UI events are not added to the subtotal.
Some differ from the selected thread totals after explicit context-compaction events.
Other UI decreases or scope/timing differences remain recorded in the private audit.
The account selects actual response IDs and their exactly matching thread snapshots.
Counter-only exports remain outside the measured subtotal because equivalent response-level deduplication is unavailable.

Detailed model labels come from `turn_context.model`.
The response-usage record has no model field.
Special compaction or routing model differences are not separately verified.
The breakdown therefore identifies recorded model contexts, not an independently measured per-request model-routing trace.

### Retained comparison and privacy

The parent's finite-response-law preparation-screen implementation reports reuse overlapping sets of underlying responses.
The source audit compares 257 responses from four retained implementation sets with the selected ledger.
Every usage field agrees.
Those responses are already in the subtotal. The report totals are not added separately.
Other retained isolated comparison sessions contribute their distinct recorded responses through the same deduplication rule.

Raw chats, prompts, reasoning text and private source paths remain outside the public account.
The aggregate publishes totals, coverage, opaque source-prefix hashes and calculation identities.
Public statistics do not require release of private scientific payloads.
Token volume measures development activity. It does not establish software correctness or scientific qualification.
