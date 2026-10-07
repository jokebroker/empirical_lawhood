# Finite response law: current forecast-mediated reuse

SPDX-License-Identifier: CC-BY-4.0

Use this workflow for current 32-root calibration, qualification of the frozen 48-root nomination, and 64-root prospective evaluation. It reuses the existing native/model/control providers and issue, package, resource, result and recovery machinery. The base locked NumPy/SciPy environment is sufficient. Original paper evidence remains preserved separately; newly run studies produce their own outcomes.

## Fixed nomination and current inputs

```bash
uv run --no-sync empirical-lawhood campaign nomination-export \
  --output-dir /work/new/nomination --nomination-id example.original-nomination
uv run --no-sync empirical-lawhood campaign nomination-check \
  --operand /work/new/nomination/nomination.canonical.json \
  --source-directory /work/new/nomination
```

This delivers the original development report, coefficient transport and manifest alongside current numerical wrappers. It authenticates the exact numerical crosswalk; it neither refits those coefficients nor treats the original 48 roots as a new development campaign.

Copy `current-calibration-allocation.json`, `current-evaluation-allocation.json` and `current-exposure.json`. Their public sample numeric seeds are permanently exposed and cannot become fresh by renaming roots or changing a role. `campaign finite-response-allocation --stage STAGE --cohort-namespace ID --master-seed INTEGER --output NEWFILE` expands an explicit numerical master using the owned 32/64-root recipe. Use namespace `empirical-lawhood.finite-response-law.calibration.NAME` or `empirical-lawhood.finite-response-law.prospective-evaluation.NAME` for the selected stage. Its default `PROPOSED_UNRUN` role is a proposal, not a freshness grant. For a new scientific study declare disjoint new numerical operands and the complete known exposure census, preserving the 32/64 counts, purpose roster, views and fixed scientific specification. Validate/prepare edited allocations with `finite-current-allocation`. The exposure record is an authenticated import, not an editable route to discard exclusions.

`campaign finite-response-packet` prepares a canonical packet from an edited allocation, the exposure census, authenticated `nomination.canonical.json` and its source directory. Supply `--stage calibration`, a distinct `--config-id`, and a new `--output`. Calibration-method and prospective-evaluation require an authenticated current `--parent`; their actual method/control operands are derived by the existing scientific owners. No user-authored hash or imported donor grant substitutes for that join.

## Author, prove and run through the existing lifecycle

```bash
uv run --no-sync empirical-lawhood --project-root /selected/clean/checkout \
  --operator-profile /work/operator-profile.json campaign finite-response-author \
  --config /work/rerun-calibration.canonical.json \
  --exposure /work/current-exposure.canonical.json \
  --nomination-directory /work/new/nomination \
  --output-dir /external/new/calibration-authoring
uv run --no-sync empirical-lawhood --project-root /selected/clean/checkout \
  --operator-profile /work/operator-profile.json campaign integration-proof \
  --authoring-dir /external/new/calibration-authoring
```

The supplied `rerun-calibration.json` is an exposed full-size authoring example. The proof reconstructs actual installed factories, output contracts, graph and resource bounds with zero scientific task execution. Select the same directory through global `--authoring-dir` for existing readiness, issue/package, run, status and resume operations; see [the shared lifecycle](../../docs/integrations.md) and [CLI reference](../../docs/cli.md).

Allow time for full-size authoring and proof even though they execute no native task.
In a branch-coverage check on an Intel i7-10700KF, Python 3.11.14 and the locked
NumPy/SciPy environment, with each numerical thread limit set to one, the current
32-root public software calls took:

| Measured operation | Elapsed time |
|---|---:|
| Author calibration, including its internal validation | 22.17 s |
| Load the saved family handoff | 22.04 s |
| Prove the loaded provider projection | 0.41 s |

The CLI `integration-proof` includes loading before proof; the proof-only time
is not its total command time. This check retained all 32 roots, 64 nested views,
705 tasks, 2,051 outputs and 705 resource cells. It did not issue or run them.
These observations are host-specific, not time limits. Evaluation has 64 roots
and much larger saved records: a separately measured 118 MB canonical decode
took 16.03 s after parse-local reuse, before the command's other work.
Native execution, qualification, protected readout and recovery add their own
costs. The complete software integrity test covering calibration through
evaluation took about 52 minutes under coverage; this is a multi-operation test,
not an individual authoring or proof command. See [testing costs](../../docs/testing.md)
for the complete suite and its retained timing logs.

After actual issue, `campaign finite-response-bind-runtime --authoring-dir DIR --context FILE` binds an operator-selected current runtime context. The context references actual current execution/reveal authorities and creates neither. Missing authority refuses before task execution. Current source identities must match the selected clean checkout.

## Current results, parent joins and recovery

`campaign finite-response-result` takes the actual current run, issued-study, execution/reveal authority, task and exact receipt IDs, a closed `--result-kind`, and a new output path. It authenticates the complete sibling publication before reading the selected scientific outcome. Kinds include native calibration, qualification, sealed closeout, authorized root reveal/readout, cohort and terminal adjudication. The bounded CLI readout preserves the complete assigned-root census, numerical/decline/cancellation dispositions and relevant q/statistical fields, with the exact custody path, hash, size and receipt of the full scientific payload. The new output file retains its authenticated selector; the complete payload stays in guarded custody. An optional development attempt retains the bounded readout without copying multi-megabyte native arrays into its report.

`campaign finite-response-parent` joins the selected authenticated native result, and where required the qualification result, into the exact current next-stage parent. Use that parent to prepare calibration-method or evaluation packets. Native and qualification runs can have separate current issued publications; both grants and receipts must authenticate. Exposed-development parents cannot authorize new prospective evaluation.

Forecasts freeze before request/future reveal. Observed prepared states may cancel delivery; they cannot refit a prediction or choose a replacement. Current calibration recomputes each boundary's q from all 32 root maxima at rank 30, including infinite scores; nonfinite q refuses that boundary. These new q values do not replace the original numerical F used by preparation applicability.

`campaign resume` retains existing scientific retry and cumulative-resource policy. Exact dependency lookup selects the bound actual receipt, including an authorized later receipt; it does not assume attempt001 or authorize a repeated scientific acquisition. Read [results and failures](../../docs/results-and-failures.md).

## Exposed diagnostics and input preflight

The remaining commands below are development, design or retained-arithmetic checks. Their completion does not supply current qualification, execute sealed evaluation or issue a lawhood verdict. The [external input formats](../../docs/response-input-formats.md) retain their own explicit role/parent gates.

## Read results and failures

Read `independent_roots`, `nested_numerical_views`, `views` and `development_only` for the native canary.
Read assignment/stage refusal codes separately from native outcomes.
The power report computes design probabilities and approximations without simulating the full prospective campaign.

Read [the common result and failure guide](../../docs/results-and-failures.md) for retained attempts,
stdout/stderr and same-identity issued recovery. The route-specific outputs and stops below remain binding.
Use `workflow show finite-response-law --format json` for static command effects and prerequisite roles.
Discovery does not inspect those prerequisites.

## Run the excluded native canary

The [excluded finite canary](finite-canary.json)
exercises the retained finite native phase engine through one prepared root,
two nested numerical views and both independent future streams:

```sh
uv run --no-sync empirical-lawhood campaign finite-response-native-check \
  --config experiments/finite-response-law/finite-canary.json \
  --output-dir /absolute/development/finite-response-law/attempt-001
```

The source fixes a dimensionless 0.001 reference tick and a half-step numerical view.
It uses ideal 11 initialization with a 256-tick linear preparation.
The causal port frame freezes at tick 4096.
The `y-positive-256` parent ends at tick 4368.
Its 192-tick future ends at tick 4560. This target canary chooses the
positive 8-unit first-port word, with its hold comparator in each future. The 64-tick pulse gives 0.512 dimensionless native-force time impulse in
both views. Requested word, accepted phase, applied kicks, realized impulse
and paired terminal displacement are reported separately.

Two futures and
two views remain nested under **one** exposed root. The
[native check](../../tests/test_finite_response_native_quickstart.py) compares the
impulse with the independent force-time integral, validates pre/post-pulse
cutoff and rejects malformed or promoted input. This bounded diagnostic
does not compile a candidate, authenticate retained prepared parent custody, or issue a
finite-lawhood verdict.

## Fixed-sample response-composition power

Run the design diagnostic without held inputs or an operator profile:

```sh
uv run --no-sync empirical-lawhood campaign response-composition-power
```

The JSON report binds the specification hash and fixes 64 evaluation roots. The joint count gate requires at least 52 successes and at most two false admissions. The report covers the specified probability grid and exact paired-use sensitivity. The separate prediction comparison is a normal approximation.

It does not simulate the percentile bootstrap or its 10% relevance requirement. Unknown dependence prevents a claim of 80% power for the full conjunction. The report reads no outcomes, draws no cohort and performs no native task.

The informative-composition stage must retain this report before fresh calibration entry.
Its identity must accompany the development readout and frozen fitted package.
Passing development gates, package freeze and technical prerequisites remain separate requirements.
The diagnostic cannot nominate or qualify that package or supply authority.
See the [frozen protocol](../../src/empirical_lawhood/adapters/methods/finite_response_law/specification.md)
and [paired-test owner](../../src/empirical_lawhood/planning/paired_power.py).

## Supply an explicit finite scientific seed allocation

`FiniteResponseLawCohortAssignment` requires the complete sorted `scientific_seeds` table.
Each row contains `(purpose, root_index, unsigned_integer_seed)`.
Calibration has 32 roots with seven purposes each.
Prospective evaluation has 64 roots with seven purposes each and one bootstrap row.

| Purpose | Scientific role and width |
|---|---|
| `prefix`, `parent`, `future-1`, `future-2` | Separate native 128-bit seed operands for each root |
| `parent-allocation` | A separate 128-bit parent-allocation operand per root |
| `calibration-request` or `prospective-evaluation-request` | A separate 128-bit request operand per root, matching the assigned stage |
| `passive-probes` | A 256-bit full random-generator commitment per root |
| `bootstrap` | Prospective evaluation only: one 128-bit seed with `root_index=-1` |

The cohort namespace is `empirical-lawhood.finite-response-law.<calibration|prospective-evaluation>.<caller-label>`.
The label identifies the cohort.
It does not choose scientific draws.
The exact numeric table passes through assignment, native config, roots, streams, requests and readout.
Fingerprints bind that table.
An incomplete, repeated, reordered or out-of-range allocation refuses.

For a reproducible new proposal, call the public allocation helper:

```python
from empirical_lawhood.adapters.composition.finite_response_law.assignment import proposed_scientific_seeds

allocation = proposed_scientific_seeds("calibration", master_seed=12345)
# Supply allocation as scientific_seeds in the reviewed assignment constructor.
```

The numeric master seed is an explicit 128-bit input. The helper uses the declared stage, purpose and root index. It does not use the cohort label. Its result is a proposed allocation, without qualification or authority.

Review and reserve every numeric operand against the complete exposure census before outcomes. The displayed master seed is public development input. It is not a fresh qualification assignment.

To preserve a historical allocation, supply the reviewed original numeric seed table explicitly.
The proposal helper supplies no historical translation or custody grant.
Retain original source bytes and qualification under their original identities.
A numeric import preserves the declared draws, not historical authority or qualification.

## Check calibration and stage inputs

Finite response-law calibration has its own request pairs, joint lower/upper receiver,
calibration and sealed evaluation rosters. Its retained missing-future supplement path requires the
exact exposed retained prepared source and eighty parent handoffs. That historical parent
cannot become a fresh unit. The [finite calibration input](finite-calibration-input.json)
selects the retained 32-root, 640-task/751,104-update response-law calibration
roster for a read-only preflight:

```sh
uv run --no-sync empirical-lawhood campaign finite-response-input-check \
  --config experiments/finite-response-law/finite-calibration-input.json \
  --source-root "$PREPARED_RESPONSE_HELD_SOURCE_ROOT" \
  --plan "$FROZEN_FINITE_PLAN" \
  --prior-exposure "$FINITE_NATIVE_EXPOSURE"
```

The native-input check requires typed finite exposure beneath the held source root.
It requires complete sorted excluded/proposed unit and stream arrays and the exact public specification bytes.
The target specification preserves thresholds, exposed nominations and unentered stages.
It has a distinct byte identity from the original historical protocol.

Missing source stops at `FINITE_RESPONSE_LAW_SOURCE_ROOT_REQUIRED`.
Missing plan or prior census stops at `FINITE_RESPONSE_LAW_PLAN_REQUIRED` or `FINITE_RESPONSE_LAW_PRIOR_EXPOSURE_REQUIRED`.
A changed plan stops at `FINITE_RESPONSE_LAW_PLAN_MISMATCH`.
The held calibration roster previously refused as exposed: 32 units and 576 streams.
Those exposed units cannot become fresh through a new label.

Supply an absolute `--assignment` path containing `FiniteResponseLawCohortAssignment` for calibration.
Use a new descriptive cohort identity, protocol hash, 32 roots and complete numeric `scientific_seeds` allocation.
The public development role is `EXPOSED_DEVELOPMENT_NONPROMOTABLE`.
Review its complete prior inventory and typed parent custody separately.
The native input check validates unit, stream, seed/state and passive-probe identities against the complete census.
It rejects the same numeric seed under another RNG label.

The [assignment tests](../../tests/test_finite_response_assignment.py) cover exposure collisions, malformed assignments and exact native seed operands.
The older reservation boundary reports `FINITE_RESPONSE_LAW_ASSIGNMENT_RESERVED_USE_STAGE_SELECTOR` with zero tasks and `provider_built=false`.
Native RNG and jumped-state identities use the source's `native_rng` records.
An independent PCG64 calculation validates the relationship.
Current assigned root/config/invocation records expand 640/1,280-task calibration/evaluation rosters with 32/64 physical roots.

The [assigned native graph check](../../tests/test_finite_response_assigned_native.py) compares both pure rosters to their external assignments and seed sets.
One excluded calibration root runs through both views, its parent and a matched HOLD/+8 future.
It produces versioned delivery, checkpoint, paired-artifact and task-result records.
This bounded result is public development input and creates no candidate.
Projection, interface, word, view and completion records retain the full physical-root denominator.

The [factory check](../../tests/test_finite_response_assigned_factory.py) builds calibration source and method providers on a synthetic nonpromotable census.
It enters no native task.
The fixed 256-pair A/B calibration request generator uses the assigned physical roots and explicit numeric inputs.
An independent seed calculation agrees for all 32 roots.
The [calibration binding check](../../tests/test_finite_response_assigned_method_records.py) validates method authoring, canonical decoding and missing/unusable custody refusals.
It supplies no authentic completion or qualified calibration parent.

The two-consumer request generator accepts a validated assigned evaluation root.
Its RNG and request identities bind the explicit numeric allocation.
The scalar readout accepts one complete ordered assigned 64-root cohort.
The explicit bootstrap row supplies its seed.
The cohort label does not choose the draw.
Source, control, projection, sealed completion and reveal bindings retain this separate evaluation cohort.

The [prospective binding check](../../tests/test_finite_response_assigned_evaluation_binding.py) authors the 1,796-step graph and binds five registered owners without tasks.
Parent effects require the persisted six-consumer lock.
Future effects require the parent join.
The same check imports 64 original prefixes into continuation, leaving 1,216 native branches on those original roots.
It also authors a three-task completion over 128 retained projections without new acquisition.
Current and historical completion formats retain separate schemas and original historical fields.

The [input tests](../../tests/test_finite_response_native_input.py) validate source, protocol, exposure and precontact refusals.
These software fixtures establish no sealed evaluation, preparation-screening result or historical continuation authority.

The seven [finite stage selections](.) use `campaign finite-response-stage-input-check`.
Select the matching input:

| Stage | Input |
|---|---|
| Calibration native source | `calibration-stage-input.json` |
| Supplemental development | `supplemental-development-input.json` |
| Informative composition | `informative-composition-input.json` |
| Calibration method | `calibration-input.json` |
| Prospective evaluation | `prospective-evaluation-input.json` |
| Prospective continuation | `prospective-continuation-input.json` |
| Preparation screening | `preparation-screen-input.json` |

Supply `--config`, held `--source-root`, frozen `--plan` and complete `--prior-exposure`.
For native calibration, supply an external 32-root `--assignment`.
For evaluation and continuation, supply the distinct 64-root evaluation assignment.
Continuation's census must contain all 64 original evaluation units and their reserved streams.
It reuses those identities and numeric commitments.
For example:

```sh
uv run --no-sync empirical-lawhood campaign finite-response-stage-input-check \
  --config experiments/finite-response-law/calibration-stage-input.json \
  --source-root "$PREPARED_RESPONSE_HELD_SOURCE_ROOT" \
  --plan "$FROZEN_FINITE_PLAN" \
  --prior-exposure "$FINITE_NATIVE_EXPOSURE" \
  --assignment "$PUBLIC_DEVELOPMENT_ASSIGNMENT"
```

The selected calibration factory reports a disjoint public development reservation.
It builds the registered assigned source, projection and completion providers.
It plans 640 native tasks across 32 physical roots. It reports
`FINITE_RESPONSE_LAW_DEVELOPMENT_BINDING_READY`, zero executed native/analysis tasks,
no compiled candidate and no issue. The remaining selectors stop at their first missing authentic parent or
authority. With `--operator-profile`, the target grant store is installed.

For missing-future supplement/calibration/prospective-evaluation/continuation/preparation-screen, `--authoring-input` supplies a strict
`FiniteResponseLawAuthoringInput` containing the exact source, primary parent digest,
prior units/streams, method/control/retention/screen configuration and any
additional independently authorized parents. All artifact operands must appear
in the authenticated inventory. The installed consumer ports resolve bounded
manifested inputs and runtime context. Missing-future supplement requires all 80 original handoffs,
preparation-screen all 24 prefixes and four qualified lower-package records.

The checker builds
the retained authoring and every registered provider/runner without execution. Continuation preserves the original 64 roots and supports a separate
nonacquiring completion. The [consumer checks](../../tests/test_finite_response_consumer_binding.py)
cover missing-future supplement and preparation-screen. The calibration/prospective-evaluation tests above cover their assigned and retained bindings.

None of these fixtures supplies an authentic qualified parent.

Preparation-screening prefix preparation remains a supplied scientific input. For all eight prepared-response and sixteen information-response roots, authenticate the successful receipts, canonical results and native payloads.
Use the existing [compact instrument](../../src/empirical_lawhood/adapters/simulators/finite_response_law/instruments.py) for both 24-channel prefix vectors.
Derive them from the common frame and 31 history samples at clock 4096. Each numerical view must end at native step
`4096 * refinement`. Declare the original instrument and source/result
identities in the typed prefix, preserving the original decimal conversion
`Decimal(str(float(value)))`.

Publish all 72 record/payload/receipt artifacts
and the four lower-package/eligibility records with their own custody. The
input check authenticates declared artifacts and builds providers. It does
not certify the derivation of supplied feature values.

Historical private-directory collectors, verifier commands and one-off repair
proofs are outside the supported CLI. Their reusable calculations remain at
the scientific owners. Reproducing a historical analysis also requires its
original status, receipt, source, authority and cumulative resource proofs.
Current input checks do not migrate those records or reset their charges.

Informative composition is a retained-only consumer, with no native provider or candidate. Its `FiniteResponseLawInformativeCompositionParent` maps 16 prepared-response and 32 information-response aliases one-to-one to original physical IDs.
It retains fixed native features, observed/valid/HOLD masks and the public scientific specification. Only the 16
complete two-future retained prepared roots evaluate decisions. Missing information-response outputs remain
masked.

After custody/reveal/analysis authentication, the stage selector checks
that exact 48-root join and selects `develop_fold`/`summarize`. The
[informative-composition checks](../../tests/test_finite_response_retained_development_input.py) exercise four outer folds,
three inner folds, unavailable-cell preservation, altered-census refusal and a
zero-response negative case. Selection itself performs no fit or native update.

The [stage tests](../../tests/test_finite_response_stage_input.py) load all seven
shipped selections and validate the precontact boundaries. A failed or noncanonical
historical donor terminal is not accepted by the current custody importer. It
requires a complete successful receipt roster and canonical stage product.
Historical continuation records are therefore not claimed to import merely
because the assigned development fixture binds successfully.

## Independent method and retained-data checks

The [finite-method counterexamples](../../tests/test_finite_response_method_counterexamples.py) test these scientific operands:

- Truth-blind signed interval choice and uncertainty refusal.
- Complete 32-root rank-30 calibration against an independent scorer.
- Two committed prospective consumer requests and 64-root joint-use/false-admission inference.
- All-nine-schedule adequacy and both-consumer actual success.
- Rank-18 and rank-23 infinity.
- All three separate preparation-screening development gates.
They use explicitly synthetic representations of original scientific operands. They do not repair the
exposed calibration/evaluation roster or import a qualified lower package.

The retained preparation-policy [saved-operand verifier](../../src/empirical_lawhood/adapters/methods/finite_response_law/retained_entry_readiness/verification.py)
independently validates normal equations, train/held root separation, rank-18
calibration, interval choices and joint success on the distinct 24-root,
nine-schedule chart. Its fixed gates and supplied report remain outcome-visible. For the original complete cache, call `validate_retained_inputs(m, trajectory,
lower)` before `check_arrays(...)`. This validates the exact finite float64 axes,
trajectory endpoints, lower forecasts and frozen normalizer/operator agreement.

Authenticate and authorize the arrays, lower qualification, prospective-evaluation eligibility and
completed preparation-screen/bridge lineage separately, and enforce the declared archive and
resource bounds before loading. These functions perform no model solve, native
work or issue. A successful arithmetic check grants no authority.

The [preparation-policy independent development calculation](../../src/empirical_lawhood/adapters/methods/finite_response_law/preparation_policy_verification.py)
retains the separate adequacy-provider screen. The [closure summary](../../src/empirical_lawhood/adapters/methods/finite_response_law/retained_entry_readiness/posthoc_summary.py)
retains all affine error components and their cross terms. The [response-composition maximum](../../src/empirical_lawhood/adapters/methods/response_composition/verification.py)
provides the original explicit per-root/per-parent reduction over both views
and the declared primary horizons. These are separate retained contracts.

From original operands, the [independent informative-composition verifier](../../src/empirical_lawhood/adapters/methods/finite_response_law/independent_informative_composition.py) reconstructs four outer folds and their three coefficient folds.
It reconstructs training-only normalizers, affine coefficients, scales, root scores and every finite-menu decision. Its linear solves belong to this
independent audit. It does not call the producer's acceptance code. The
[independent calibration verifier](../../src/empirical_lawhood/adapters/methods/finite_response_law/independent_calibration.py)
also validates the sole qualification record against rank 30, complete request
counts and the recorded eligibility gate. Authenticate and authorize all input
arrays and records before using either library. Their output is an exposed
development consistency check and grants no fresh entry or publication authority.

The [missing-future owner](../../src/empirical_lawhood/adapters/methods/finite_response_law/supplement.py)
also retains `supplement_from_evaluation`, which maps the authenticated original
16-root supplementary evaluation into the complete nine-word panel and its
observed/valid masks. It preserves the second-future clock, word order, complete
native-update census and duplicate-slot refusals. The existing
`join_missing_future` then keeps earlier observations intact. The reducer does
not authenticate receipts or the scientific adjudication on the caller's behalf.

The [independent opportunity verifiers](../../src/empirical_lawhood/adapters/methods/finite_response_law/independent_opportunity.py)
retain the distinct original single-future opportunity and two-future opportunity native-word
charts. They independently reconstruct admissibility, smallest-word selection,
aligned magnitude discrimination and equal-root request statistics, then check
the original continuation gates. Opportunity screening also requires the complete measured
supplement and verifies that the earlier retained prepared/information-response panel cells are unchanged.
Supply authenticated original arrays, request freeze and report bytes after
separate analysis authorization. These library readouts perform no native work,
publication or qualification.

The [independent evaluation audit](../../src/empirical_lawhood/adapters/methods/finite_response_law/independent_evaluation.py)
rechecks the original 64-root prospective-evaluation native losses, selected-word success and false
admissions, fixed confidence bounds, paired bootstrap and added-use comparison.
Its continuation entry point retains the separate analysis status. Its receipted
entry point verifies the frozen qualification and every root's controller-use/reveal parity
while keeping absent terminal adjudication absent. Authenticate the exact source,
receipts, revealed operands and analysis authority before entry. Any continuation
repair, retry census and cumulative elapsed accounting require their own runtime
proofs. These arithmetic functions do not perform those acts.

The [retained feature assembler](../../src/empirical_lawhood/adapters/methods/finite_response_law/retained_features.py)
keeps the original 16 retained prepared and 32 information-response roots, each with five parent payloads and two
numerical views. It assembles 24-channel prefix and handoff features with the existing compact instrument.
It validates native lineage, original root order, clocks and equal shared prefixes across parents. Authenticate the input
inventory, successful receipts, source freeze and artifact hashes separately.
The returned lineage binds supplied record identities and payload hashes. It
does not carry historical file paths or authorize fitting or native execution.

`verify_screen_readout` in the [preparation-policy verifier](../../src/empirical_lawhood/adapters/methods/finite_response_law/preparation_policy_verification.py)
validates the original complete preparation-screen screen against its unchanged lower evaluator,
all 24 panels, root-level A/J/C readouts and gates. It also reconstructs the
recorded fit dependencies and validates their exact identities. This exposed-data
audit includes independent affine fits. Authenticate the frozen operands,
receipts, retained recovery census and analysis authority first. It neither
acquires native observations nor publishes a qualification or repairs execution.

The installed [validate_retained_native_panel](../../src/empirical_lawhood/adapters/methods/finite_response_law/retained_native_validation.py) library independently validates the
original 16 retained prepared and 32 information-response retained roots before calling the existing panel
importer. It preserves all native common-start, parent-work, word, future,
clock/view, checkpoint and shared-innovation joins, plus the original sampled
endpoint comparisons. Supply the authenticated arrays and canonical native
result/payload pairs under separate analysis authority. Original inventory,
receipt, source, custody, summary/status and cumulative resource proofs remain
caller prerequisites. This readout grants no new authentication budget.

The [frozen-package owner](../../src/empirical_lawhood/adapters/methods/finite_response_law/frozen_package.py) retains `thread_settings` for observed single-thread environment/backend verification.
Its `verify_frozen_dependencies` validates original result/readout, NumPy/Python/lock, complete scientific source hashes and unchanged interval function bodies. Supply the authenticated six-file/four-family frozen scope
and its source bytes explicitly. This check does not refit coefficients or
qualify a new run. Changed scientific bytes or numerical versions require a
separate authorized freeze. CLI and worker thread limits remain enforced by
their existing bootstrap and process owners.

## Obtain the workflow inputs

Obtain this bundle from the selected repository checkout or source distribution. The guide and study inputs are repository workflow files, not wheel resources. Run each command from the checkout root in an unactivated shell. The relative paths use that root.

Keep generated data and receipts in the declared external storage. A wheel can perform its advertised inspection and native-input checks with absolute paths. Source-bound authoring and issue require a clean checkout that executes its own tracked package.

## References and research

Gareth Seneque (2026), [*Empirical Lawhood*](../../paper/manuscript.md), edition 0.60, gives the scientific formulation.
The [program source register](../../paper/SOURCES.md) identifies historical results and unavailable primary records.
The linked scientific source modules and existing tests establish only their declared implementation and development boundaries.
