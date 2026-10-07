# Six-matrix external inputs and parent boundaries

SPDX-License-Identifier: CC-BY-4.0

This reference gives the formats for accepted six-matrix inputs and authenticated parent handoffs.
It gives procedures for exposed development authoring and qualified-parent input validation.
Use it to construct the correct format and identify the first missing parent or grant.

The prepared, information-response and causal-response selectors below check or compile **exposed development** inputs.
Other current six-matrix operations have the separate typed boundaries described under [current modular inputs](#current-modular-inputs).
Successful decoding does not authenticate custody, qualify a source,
or permit a prospective child. Use the [six-matrix quick start](../experiments/prepared-response/guide.md) for
commands and the [inert examples](input-contracts.md) for JSON shapes.
All paths supplied as held members must be absolute regular files beneath the
selected absolute source root. Reads follow directory descriptors, reject
symlinks and concurrent changes, and enforce the limits below before native
contact. Plan/design paths are separately selected absolute files.

| Operand | Accepted shape, relation and limit |
|---|---|
| Prepared response/information-response/causal-response `--plan` | The exact frozen design-plan bytes, nonempty, at most 2 MiB. SHA-256 is over the supplied bytes, including whitespace. The public development authoring seam binds those bytes. It does not prove that arbitrary prose is an adequate scientific plan. |
| Prepared `--design-packet` | A separately retained design input, nonempty, at most 2 MiB. Its digest is bound into the authoring package. Select the reviewed packet, not a placeholder tutorial. The format at this seam is an opaque byte document. |
| Prepared `--prior-exposure` | At most 64 MiB. Exact schema `empirical-lawhood/composition/prepared-response/prepared-exposure-inspection`. Four nonempty arrays under `value`, described below. Foreign historical schemas require a separately reviewed current export. |
| information-response/causal-response `--model-bank` | At most 16 MiB. Strict canonical `InformationResponseModelBank` or `CausalResponseModelBank`, respectively. `plan_sha256` must equal the selected plan-byte digest. Prepared response forbids a bank. |
| Finite `--plan` | At most 2 MiB. Must equal the exact digest selected by `FiniteResponseLawScienceSpec`. The packaged [protocol](../src/empirical_lawhood/adapters/methods/finite_response_law/specification.md) supplies those bytes. Reformatting it changes the input identity. |
| Finite `--prior-exposure` | At most 128 MiB. Schema suffix `methods/finite-response-law/native-exposure-metadata`. The four arrays are at the top level. |
| Finite assignment | Strict `FiniteResponseLawCohortAssignment`. Stage, scientific seed table and science-plan digest must match the selector. Actual native unit and stream identities are checked against the full census. Passing this reservation check does not issue a campaign. |

The prepared example is structurally:

```json
{"schema":"empirical-lawhood/composition/prepared-response/prepared-exposure-inspection","version":"1.0.0","value":{"excluded_unit_ids":["unit.example.exposed-a"],"proposed_unit_ids":["unit.example.exposed-b"],"excluded_seed_ids":["seed.example.exposed-a"],"proposed_seed_ids":["seed.example.exposed-b"]}}
```

The finite example places the same arrays beside `schema`. Prepared arrays must
each be nonempty. Finite arrays may individually be empty, but each combined
unit/seed union must be nonempty. Entries must be sorted unique stable-ID
strings.

Both routes exclude the union of `excluded_*` and `proposed_*`. Previously proposed roots remain excluded after negative, unevaluable, unentered, interrupted or reassigned outcomes. Duplicate
mapping keys anywhere, including `schema` and `value`, are errors.

To construct a real census, inventory every available prior exposure publication. Include proposed rosters, interrupted/failed runs, development/canary roots, training roots, public examples and test assignments. Retain exact member bytes, locators, SHA-256, source schema/version,
unit IDs, native seeds, purpose-specific streams, access history, and the
inspection method/version. Union identities without dropping their source
membership.

Sort only after collecting them. Review missing archives and
unexplained aliases explicitly. A fresh JSON file or newly calculated digest
does not make incomplete history complete. The examples declare their history
incomplete.

The input-check summaries accurately retain
`prior_custody_authenticated=false`.

Caller-owned current inputs use the exact target schema and actual source identities.
The prior census requires the prepared exposure schema shown above.
Model banks require the exact route's current bank schema and current nested record schemas.
The reader performs no schema-prefix translation or root relabeling.

Foreign prior inspections stop at `PREPARED_RESPONSE_PRIOR_VERIFIED_TARGET_EXPORT_REQUIRED`.
Foreign bank schemas stop at `SOURCE_MODEL_BANK_VERIFIED_EXPORT_REQUIRED`.
Retain their original bytes, hashes, manifests, receipts, coefficients, and physical root identities unchanged.

A historical import needs a separately reviewed export before this authoring entry can consume current records.
That review must bind original hashes and interpretation to the exact target record and separate target custody.
A newly written current-schema file does not authenticate that review or transfer original grants.
This prepared/information/causal authoring entry provides no foreign-record conversion or export-verifier operation.
Stop historical-bank or foreign-census authoring until the reviewed current export and required custody are available.

For a model bank, retain the development manifest and results that actually
produced it. The [information-response model definitions](../src/empirical_lawhood/adapters/methods/information_response/models.py)
and [causal-response definitions](../src/empirical_lawhood/adapters/methods/causal_response/models.py)
are the executable formats: the bank carries `plan_sha256`,
`development_manifest_sha256`, `development_results_sha256`, and ordered
`contexts`. Each context contains its exact `PreparedBilinearModelSpec`, sixteen
ordered prepared-response `training_roots`, and the fixed model tournament. Predictor coefficient
blocks preserve their declared shapes and finite numeric values.

Export a caller-owned current bank with `canonical_bytes()` from a permitted development fit.
Historical banks require the separately reviewed current export described above.
This input seam provides no fitting CLI.
Supply the actual fitted coefficient blocks. Zero-filled placeholders do not establish a permitted development fit.
It adds every bank training-root physical ID and seed digest to exclusions.

A bank from another plan is refused before candidate compilation.

The development commands consume these files directly:

```sh
uv sync --locked --python 3.11.14 --group reactor-example
uv run --no-sync empirical-lawhood --project-root "$PWD" campaign matrix-response-author \
  --config experiments/prepared-response/prepared-source-qualification-author.json \
  --source-root "$PREPARED_RESPONSE_HELD_SOURCE_ROOT" --plan "$PREPARED_RESPONSE_PLAN" \
  --design-packet "$PREPARED_RESPONSE_DESIGN_PACKET" --prior-exposure "$PREPARED_RESPONSE_PRIOR_EXPOSURE" \
  --output-dir "$PREPARED_RESPONSE_NEW_AUTHORING_DIRECTORY"
```

For information/causal authoring, select `information-author.json` or
`causal-author.json` and add `--model-bank "$PREPARED_RESPONSE_MODEL_BANK"`. Expected output
includes `authoring.json`, extension payloads, decoder registrations,
`exposure-inspection.json`, and `candidate.json`. The summary reports no native
contact, no issued campaign, and no prospective eligibility. The finite
calibration checker validates the fixed science plan and roster. The stage selector
also needs the actual parent boundary below. See `campaign --help` and the
[per-command contracts](cli.md) for current names and flags.

For dependent refinement and fresh response calibration, and for later finite stages, a design document and bank are
insufficient. The parent input must be a qualified immutable publication. It must contain the exact source/output record, artifact manifest, task receipt, parent issue identity and authorized custody relation. The
[parent reader](../src/empirical_lawhood/adapters/composition/response_parent_reader.py)
replays those operands from the selected guarded store.

The
[parent authority store](../src/empirical_lawhood/adapters/composition/response_parent_store.py)
publishes the separate target import grants. The original producer's packet
stays in its held source tree. Placing JSON at the right pathname is not a grant
publication. The [parent-store tests](../tests/test_response_parent_store.py)
show complete synthetic field relationships and rejection of altered receipts,
manifests, authority, and stage bindings.

Their fake parents are software test
inputs, never evidence of scientific qualification.

During a live read, descriptor identity and link count must remain unchanged.
An unlink is refused even when the observed timestamps are equal.
After successful replay, consumers use the held authenticated bytes.
A later pathname change cannot redirect that verified input.

The supported historical import procedure is explicit:

1. Retain the original successful producer's terminal record, every successful
   task receipt, every output materialization, their `.manifest.json` sidecars
   and referenced publication commit markers. Include the parent scientific
   product and its stage envelope. An exact same-run recovery terminal may be
   included at its declared recovery path. Do not add unrelated files or omit
   a failed scientific conclusion from a successfully completed task.
2. Create a sorted source inventory of those **actual files**. It is a JSON
   array of objects with exactly `path` (absolute path beneath the selected
   source project), `sha256`, and `bytes`. Serialize with
   `json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()`.
   This import contract requires those exact bytes without a trailing newline.
   Retain it under the held source root. Maximums are 128 MiB for the inventory,
   100,000 entries, 300 MiB per member, 8 GiB total, and 8 MiB per decoded record.
   Hash the retained inventory for the target grants. Absolute inventory paths
   describe this selected copy. Relocating it requires a reviewed new inventory
   and new target grant identities, while original record bytes stay unchanged.
3. The accountable target operator independently supplies three
   `ResponseParentTargetGrant` records with distinct IDs and actions `CUSTODY`,
   `REVEAL`, and `ANALYSIS`. All three must agree on route, source inventory
   digest, `source_project_relative`, `terminal_relative`,
   `parent_member_relative`, `stage_envelope_relative`, original storage-root
   ID, sorted original independent-root IDs, and evidence role. Their common
   `parent` is `ResponseSourceParentIdentity(source_schema, source_version,
   physical_sha256)` over the exact retained parent bytes. Use
   `EXPOSED_SOURCE_QUALIFICATION_DEVELOPMENT_NONPROMOTABLE` for response composition and
   `EXPOSED_DEVELOPMENT_NONPROMOTABLE` for the other supported import routes.
   A source-side old grant does not supply these new target acts.
4. Compose an `ExternalArtifactPlane(GuardedExternalRoot(contract))` from the
   selected `OperatorStorageProfile` using `resolve_external_root_contract`,
   as in the [operator setup](reactor-operator.md). Instantiate
   `ResponseExternalParentAuthorityStore(plane)` and call `persist(reviewed_grant)`
   separately for each supplied grant. This writes and replays immutable
   publications at `authority/prepared-response-parent-grants/<grant_id>.json` beneath the
   profile's artifact namespace. Reuse the returned identities and exact
   canonical files. No function decides that an absent grant should exist.
5. Pass the absolute inventory path as `--parent-manifest`.
   Pass the held root as `--source-root`.
   Supply **published target-store** paths as `--custody`, `--reveal-record` and `--analysis-record`.
   Supply the top-level `--operator-profile`. The selected reader resolves grants before reading
   outcome-bearing parent bytes, replays the exact complete publication roster,
   and then applies the route's scientific-parent requirements. Any changed
   manifest, root count, receipt, stage/product binding or unsupported schema
   must stop the import.

Use the actual record emitted by the supported producer and persist its
associated manifests/receipts unchanged. Import historical evidence only through
the explicitly supported exposed-parent adapter, preserving its evidence ceiling.
The retained negative prepared-response result cannot be promoted to a qualified dependent refinement parent.
Response composition may inspect its declared negative-source input at its own exposed boundary.
This inspection does not qualify dependent refinement/fresh response calibration or reopen unentered finite milestones. Missing
qualified parents remain explicit refusals. No public file conversion can supply
missing observations, grants, or independent support.

## Supply a caller-owned numeric root census

The shipped authoring inputs use retained fixed allocations. Their empty `root_seed_census` selects only the exact bundled allocation for `source_seed_sha256`.
A different source digest requires a complete explicit census before authoring.
Changing `seed_label` or `target_prefix` does not allocate scientific randomness.

Each row is a `PreparedRootRandomness` record from the [prepared contracts](../src/empirical_lawhood/adapters/simulators/prepared_response/contracts.py).
It contains `stage`, `context`, `index`, `source_seed_sha256`, 17 ordered `stream_seed_sha256s`, and one `passive_probe_seed_sha256`.
All 18 commitments are lowercase 64-digit hexadecimal strings representing unsigned 256-bit numbers.
These are explicit numeric allocations. They need not be hashes of names or labels.

The 17 stream positions follow `PREPARED_STREAM_ROLES` exactly:

| Position | Purpose | Bridge |
|---:|---|---|
| 1 | `initial-ramp` | `False` |
| 2 | `initial-ramp` | `True` |
| 3 | `initial-post-ramp` | `False` |
| 4 | `initial-post-ramp` | `True` |
| 5 | `parent` | `False` |
| 6 | `parent` | `True` |
| 7 | `common-response` | `False` |
| 8 | `common-response` | `True` |
| 9 | `independent-response-1` | `False` |
| 10 | `independent-response-1` | `True` |
| 11 | `independent-response-2` | `False` |
| 12 | `independent-response-2` | `True` |
| 13 | `independent-response-3` | `False` |
| 14 | `independent-response-3` | `True` |
| 15 | `prospective-task` | `False` |
| 16 | `prospective-task` | `True` |
| 17 | `passive-probes` | `False` |

The separate eighteenth commitment owns the twelve-field passive-probe draw.
It differs from the `passive-probes` stream commitment. There is no passive bridge stream.
Keep every full 256-bit commitment. PCG64DXSM uses its first 128 bits, interpreted in big-endian order.
Those first 32 hexadecimal digits must be unique across all 18 commitments in every census row.
Different trailing digits do not create different PCG allocations.

Use the exact context order: all `assembling` indices, then all `prepared` indices.
Indices start at zero and increase without gaps within each context.
Every row repeats the selected stage and source digest.

| Authoring route | Census stage | Indices per context | Required rows | Selected independent roots |
|---|---|---|---:|---:|
| `source-qualification` | `qualification` | `0`–`15` | 32 | 32 |
| `information-prediction` | `prospective-evaluation` | `0`–`127` | 256 | 64 |
| `causal-response-prediction` | `prospective-evaluation` | `0`–`127` | 256 | 64 |

Information and causal authoring require the full 256-row recipe.
Their native configs select indices `0`–`31` in each context, giving 64 roots.
The other recipe rows do not add acquired roots or evaluation evidence.
The development role remains exposed even though the mechanical stage is named `prospective-evaluation`.

The following constructor writes inputs without invoking a source, sampler, fit, provider, candidate compiler, or campaign.
Run it with the selected checkout interpreter and its tracked package.
Choose a new absolute output directory outside the checkout.
For information or causal inputs, change only `route` to the required route above.

```python
from pathlib import Path
from hashlib import sha256
from secrets import token_bytes

from empirical_lawhood.cli.entry import enforce_single_thread_environment

enforce_single_thread_environment()

from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.adapters.simulators.prepared_response.contracts import (
    CONTEXTS, PREPARED_STREAM_ROLES, PreparedRootRandomness,
)
from empirical_lawhood.adapters.composition.prepared_response.native_authoring import (
    PreparedResponseNativeAuthoringInput,
)

route = "source-qualification"
family = {
    "source-qualification": "prepared-response",
    "information-prediction": "information-response",
    "causal-response-prediction": "causal-response",
}[route]
stage = "qualification" if route == "source-qualification" else "prospective-evaluation"
count = 16 if route == "source-qualification" else 128
seen_prefixes = set()


def fresh_commitment():
    while True:
        value = token_bytes(32).hex()
        if value[:32] not in seen_prefixes:
            seen_prefixes.add(value[:32])
            return value


allocation = tuple(
    tuple(fresh_commitment() for _ in range(len(PREPARED_STREAM_ROLES) + 1))
    for _ in range(len(CONTEXTS) * count)
)
numeric_bytes = canonical_json_bytes(allocation)
source_digest = sha256(numeric_bytes).hexdigest()
rows = tuple(
    PreparedRootRandomness(
        stage=stage,
        context=context,
        index=index,
        source_seed_sha256=source_digest,
        stream_seed_sha256s=allocation[context_index * count + index][:17],
        passive_probe_seed_sha256=allocation[context_index * count + index][17],
    )
    for context_index, context in enumerate(CONTEXTS)
    for index in range(count)
)
request = PreparedResponseNativeAuthoringInput(
    config_id=f"empirical-lawhood-{family}-caller-allocation",
    route=route,
    target_prefix=f"empirical-lawhood.{family}.caller-allocation.",
    seed_label=f"empirical-lawhood-{family}-caller-allocation",
    source_seed_sha256=source_digest,
    root_seed_census=rows,
    evidence_role="EXPOSED_DEVELOPMENT_NONPROMOTABLE",
)
output = Path("/absolute/new/input-directory")
output.mkdir(parents=True, exist_ok=False)
(output / "numeric-allocation.json").write_bytes(numeric_bytes)
(output / "authoring-input.json").write_bytes(request.canonical_bytes())
```

Here, `source_seed_sha256` identifies the exact saved numeric-allocation bytes.
The constructor checks its format and row agreement. It does not recompute that digest or authenticate allocation provenance.
Keep those bytes with the canonical authoring input.
Pass the generated `authoring-input.json` as `--config` in `matrix-response-author`.
Canonical serialization emits the outer `schema`, `version`, and `value` envelope, including every nested census row.

Review the complete prior exposure inventory before using the allocation.
It must include public examples, tests, failed or interrupted proposals, training roots, and their stream allocations.
Compare physical unit IDs, purpose IDs, and numeric PCG prefixes. Renaming an identity does not remove exposure.
Authoring checks selected-root collisions and adds fitted-bank training roots to exclusions.
It does not authenticate historical custody or prove that missing exposure history is complete.
All outputs remain `EXPOSED_DEVELOPMENT_NONPROMOTABLE`. Numeric commitments supply no source qualification or issue authority.

## Prepared response, information and causal authoring

Three strict target inputs select the retained prepared-response, information-response and
causal-response builders and their generated executable providers:

| Route | Input | Required held inputs | Selected independent roots |
|---|---|---|---:|
| Prepared response | [prepared-source-qualification-author.json](../experiments/prepared-response/prepared-source-qualification-author.json) | Prepared native plan, design packet, prior exposure census | 32 |
| information-response | [information-author.json](../experiments/information-response/information-author.json) | Matching information plan/design packet, prior census, fitted information-response model bank | 64 |
| causal-response | [causal-author.json](../experiments/causal-response/causal-author.json) | Matching causal plan, design packet, prior census, fitted causal-response model bank | 64 |

Run each through the same installed CLI. Supply absolute paths for the held
root and files. The prior census and model bank must be members of the held
root. The plan and design packet may be separate files. Use a new output
directory outside the target checkout. Omit `--model-bank` for prepared-response:

```sh
uv run --no-sync empirical-lawhood --project-root "$PWD" campaign matrix-response-author \
  --config experiments/information-response/information-author.json \
  --source-root "$PREPARED_RESPONSE_HELD_SOURCE_ROOT" \
  --plan "$INFORMATION_PLAN" \
  --design-packet "$INFORMATION_DESIGN_PACKET" \
  --prior-exposure "$PRIOR_EXPOSURE_INSPECTION" \
  --model-bank "$FITTED_INFORMATION_BANK" \
  --output-dir "$NEW_OUTPUT_OUTSIDE_CHECKOUT"
```

Caller-owned numeric allocation follows the [root-census recipe](#supply-a-caller-owned-numeric-root-census) above.
Information and causal inputs require 256 recipe rows for their 64 selected roots.

The plan and design packet are bounded to 2 MiB each, the prior inspection to
64 MiB and the model bank to 16 MiB. The inspection must be typed and contain
sorted excluded and proposed unit and stream arrays. The authoring entry
validates the complete supplied census for collisions.
Information-response and causal-response also reject
a bank whose plan digest differs from the supplied plan or whose training
roots reuse a proposed unit or stream. The source plan and bank are external
research inputs.

None is bundled in the wheel. A missing root stops at
`SIX_MATRIX_RESPONSE_SOURCE_ROOT_REQUIRED`, and a mismatched bank stops at
`SIX_MATRIX_RESPONSE_MODEL_PLAN_MISMATCH`, before native contact. The exported inspection retains the supplied excluded/proposed unions and adds fitted-bank training roots.
These declarations must use current schemas or separately reviewed current exports.

This is a
complete declared census, not authentication of historical custody.

The prepared source fixes a dimensionless BAOAB clock with 0.001 and 0.0005 numerical steps. Its chart has a 256-tick preparation, 16-sample pre-parent frame, five parents, 17 words and a 64-tick pulse.
Information-response and causal-response use the
declared 16-unit amplitude and separate information and causal receivers. The independent unit is a stage/context stochastic root. Each has two nested
numerical views, with parents and words nested beneath the root.

The selected task topology expands without native contact. The registered native source provider and runner construct from the exact inputs. The strict candidate compiles without native tasks, issue, reveal or authority. The command reports provider
construction separately from candidate compilation.

The shipped seed labels and experiment
prefixes are new public **development choices**. These outputs are
`EXPOSED_DEVELOPMENT_NONPROMOTABLE`. The supplied historical census is checked
for disjointness but its custody is not authenticated by this command. The
fitted banks are outcome-fitted development inputs.

The candidate is a
researcher-facing authoring check, not evidence that the proposed roots have
been acquired or are eligible for immutable prospective issue.

### Export and recover a matrix authoring bundle

Prepared-response, information-response and causal-response share this export policy.
`--output-dir` must name a **nonexistent directory outside the target checkout**.
An existing empty directory or a symlink also refuses. Supply the path without
creating that directory first; the authoring command creates it after successful
temporary candidate compilation.

A completed export contains `authoring.json`, `exposure-inspection.json`,
`payload-00.json` and subsequent numbered payloads, `decoder-00.json` and
subsequent numbered decoder registrations, `candidate.json`,
`candidate-report.json` and `selection.json`. The command prints its JSON summary
to standard output, including `authoring_dir` and the candidate identity.
Retain the exact inputs with these products. The candidate remains exposed
development and supplies no acquisition, issued campaign or prospective authority.

If input preparation or temporary compilation fails, the command has not created
the selected output directory. A later export failure reports the incomplete path
and failed stage and retains any products already written. Keep that directory and
its inputs for diagnosis; choose a fresh output path for the next attempt. A
partial export is not a completed handoff. The command does not resume, replace or
overwrite it.

Omitting `--output-dir` still compiles and prints the candidate summary, with
`authoring_dir: null`. Its temporary products are removed; there is no durable
authoring bundle to pass to a later command. Select a fresh external output path
when you need that bundle.

The [authoring check](../tests/test_response_native_authoring.py) loads all three shipped inputs and tests missing-input and malformed-census refusals.
It compiles a prepared-response candidate and its registered native provider with an explicitly synthetic prior. The
[method checks](../tests/test_response_distinct_transform_boundaries.py) use
independent feature-mask, finite-difference, signed-interval and paired-root
counterexamples. Current authoring requires the exact selected plan digest and current-schema inputs.
A retained fitted bank needs its matching plan and a separately reviewed current export.
Earlier held-input checks do not supply that export or its custody.

The source freezes its two-port frame from sixteen samples before the parent.
Information-response and causal-response retain fitted feature vectors measured at the **parent handoff**.
These include parent observations after the frame cutoff. The causal
check freezes the pre-parent frame, retains measured parent
features and validates that later future outcomes do not enter the handoff
predictions. This one-root test does not authenticate the fitted banks as
prospective evidence.

The [bounded method checks](../tests/test_response_distinct_transform_boundaries.py) test these contracts:

- The information-response snapshot mask.
- Causal-response finite-difference gain.
- Dependent-refinement whole-root folds and joint maximum loss.
- Finite-lawhood root-rank and signed-word arithmetic.
- The finite baseline-force Hessian against a central difference of the independent gradient evaluator.
- Response-composition paired/parent contrasts and whole-root folds. These synthetic numerical counterexamples check reductions.
They do not satisfy a fitted model bank, a native receiver, qualified custody
or analysis authority.

The generic issue, `issue-extensions`, package, compile, run/resume and separately
authorized reveal/adjudication commands are described in the
[CLI reference](cli.md). The candidate authoring command above does not
invoke them. Finite-lawhood and response-composition have explicit parent custody and authority input paths.
An input or binding check remains distinct from qualified parent evidence and separately authorized analysis or execution.
Historical cohorts cannot silently become new target evidence.

The [finite assignment recipe](../experiments/finite-response-law/guide.md#supply-an-explicit-finite-scientific-seed-allocation) defines the complete numeric seed table.
Labels identify cohorts without selecting scientific draws.

## Current modular inputs

The [matrix input workflow](../experiments/matrix-inputs/guide.md) supplies public fixed-operand delivery and current native/derived array exports.
OriginalF retains its original coefficients, calibration context and immutable q.
The separate frozen nomination retains exact report, coefficient and manifest identities.
Their current import publications establish custody without creating current qualification or replacing their historical provenance.

The [finite rerun workflow](../experiments/finite-response-law/guide.md) uses explicit current allocations and a complete actual exposure census.
Packet helpers bind the selected nomination and authenticated current parents.
The 32-root calibration and 64-root evaluation rosters retain two nested numerical views each.
Current parents require complete exact receipts, persisted issued plans and the selected runtime/source/authority context.

[Geometry](../experiments/matrix-geometry/guide.md), [selected events](../experiments/selected-events/guide.md) and [history analysis](../experiments/matrix-history-analysis/guide.md) have separate scientific protocols.
History input binds four families of 64 roots, paired clocks and actually consumed purpose streams.
Imported supplied histories remain exposed arrays without a native-production attestation.
The algebra and passive consumers reuse those source operands while retaining different mathematical objects and causal cutoffs.

[Tangent analysis](../experiments/matrix-tangent/guide.md) uses 1024/4096-tick prefixes, 144-tick parents and 320-tick response paths.
[Transient and baseline analysis](../experiments/matrix-transient/guide.md) instead uses 4096-tick prefixes, 400-tick parents and 192-tick responses.
Similar array headers do not make these source protocols interchangeable.
Saved analyses authenticate source manifests and member contracts before numerical work.
Outcome-visible tangent paths and measured future HOLD remain evaluator-only inputs.

[Preparation applicability](../experiments/preparation-applicability/guide.md) uses current phase allocations and explicit upstream publications.
[Preparation diagnostics](../experiments/preparation-diagnostics/guide.md) reads the resulting retained observations under their current access and receipt contracts.
Neither route accepts historical counts or schema renaming as scientific success.

## References and research

The linked source modules, command metadata and existing checks own the implemented behavior described here.
The [scientific integrity guide](scientific-integrity.md) defines its separate evidence and authority boundaries.
The [program source register](../paper/SOURCES.md) identifies bounded historical results and unavailable primary records.
