# Constructing inputs and interpreting results

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for canonical input construction and scientific result interpretation.
It locates the first handoff that needs actual owner knowledge.
Use it to prepare reviewed inputs and interpret completion, support and permitted use separately.

Start with the [operator lifecycle](reactor-operator.md),
[task guide](../README.md) and [six-matrix formats](response-input-formats.md). A typed constructor validates fields. It cannot know whether a historical
inventory is complete or whether the selected units are genuinely unexposed. The first owner-knowledge handoff is the complete prior source inventory and
unit/seed ledger, before new identities or a prospective roster are selected.

Record missing provenance as unresolved. Never substitute the public example
census or seeds. Release-roster issuance needs the owner review specified in the
[release status](release-status.md).

## Canonical constructors

Run an external script with the documented `scripts/run_operator.py` launcher.
`scripts.operator_records.make_storage_profile` requires the selected external
root and actual mount, expected device/volume (explicit `None` is allowed where
the contract permits it), filesystem types, space floor, access mode and task
limit. It fixes the registered backend and strict containment policy, creates
disjoint artifact/scratch namespaces and sets `authority_granted=False`.
Export with `export_record`, which refuses a conflicting existing file. Run `doctor --help`.
Run the profile-specific diagnosis on the actual selected mount.
Structural profile validity gives no proof of available storage or write authority.

`make_assigned_reactor_profile` accepts these explicit keyword arguments:

| Argument | Meaning |
|---|---|
| `profile_id`, `experiment_id`, `public_source_sha256` | New stable identifiers and exact public source digest. |
| `census_id`, `prior_unit_ids`, `prior_seed_ids`, `source_inventory` | Complete owner-reviewed exclusions and typed `ObjectIdentity` of their actual source inventory. No generated completeness assertion. |
| `noise_seeds` | Exactly five distinct native integers, ordered by `SCENARIOS` (cooling_loss, feed_temp, fouling, kinetic_hot, nominal). No RNG/default roster is supplied. |
| `evidence_role` | Explicit `EXPOSED_DEVELOPMENT_NONPROMOTABLE` for development. `PROSPECTIVE_RELEASE_QUALIFICATION` may be supplied only after the separate review/assignment process. The string creates no authority. |

The helper constructs the census, binds its exact identity into the assignment,
validates numeric/native seed aliases and raw/unit-prefixed unit collisions, and
returns `AssignedReactorAuthoringProfile`. Obtain missing values from the owner,
not from a fitted outcome. Use `export_record(control / 'profile.json', profile)`.
`read_record` verifies canonical bytes on replay. Empty censuses, duplicate IDs,
seed collisions, wrong roster sizes and missing required arguments refuse.

[Scaffolding tests](../tests/test_operator_scaffolding.py) use an inert closed
fixture world. Its fake inventory is never a real prospective input. The
[profile example](reactor-operator.md) and
[assigned reactor route](fresh-experiment.md) show the full records and subsequent commands.

## What may be edited

| Class | Examples and consequence |
|---|---|
| Deployment | External mount/root, namespace, space floor, permitted concurrency and mirror locator. Changes storage-profile identity. Validate containment/resources again. They do not alter a scientific claim or grant authority. |
| New assignment | Experiment/unit IDs and native seeds. Bind a new assignment to the complete prior census, validate all aliases, preserve paired arms and nested views. Exposed examples remain development forever. |
| Frozen scientific operands | Denominator, clocks, action menu, causal cutoff, receiver, horizon, fitted feature order, support, loss/reduction, threshold and independent-root accounting. Do not edit a selected recipe in place. A change needs new protocol/source/config identities and independent validation. |
| Authority and outcomes | Checker signatures, grants, receipts and terminal artifacts. Obtain actual acts from permitted principals and stores. Never generate a passing default or replace a negative result. |

A new configuration identifier alone does not make old roots fresh. Changing a
schema namespace alone does not import historical authority. Canonical identities
are immutable. See [compatibility](compatibility.md).

## Six-matrix access and stopping points

Public selector JSON, schemas, finite protocol and excluded canaries are shipped. Parent outcomes, fitted banks, exposure censuses and custody/grants are held
research inputs. Request them from the producing researcher under permitted access. Retain original bytes and provenance.

Obtain separate target custody, reveal and analysis grants in the selected store. Public
canaries cannot stand in for these records. The exact formats and bounded import
procedure are in [six-matrix formats](response-input-formats.md). Commands and all route
selectors are in [six-matrix quick start](../experiments/prepared-response/guide.md).

| Route | Producing stage and accepted held operand | First material stop |
|---|---|---|
| prepared-response | Prepared native plan/design and typed complete exposure inspection | `SIX_MATRIX_RESPONSE_SOURCE_ROOT_REQUIRED`. Proposed roots/streams must be disjoint. |
| information-response / causal-response | Matching prepared-response-derived information/causal plan and canonical fitted model bank | `SIX_MATRIX_RESPONSE_MODEL_PLAN_MISMATCH` for a bank bound to different plan bytes. Exposed fits stay development. |
| dependent refinement | Qualified target-typed prepared-response evaluation with common charter, candidate costs and `PreparedResponseDependentAuthoring` | `DEPENDENT_REFINEMENT_TARGET_CUSTODY_REQUIRED`. A negative prepared-response charter stops at `DEPENDENT_REFINEMENT_NO_COMMON_QUALIFIED_CHARTER`. |
| fresh response calibration | dependent refinement nominal library nominated for fresh calibration plus matching dependent authoring | `FRESH_RESPONSE_CALIBRATION_TARGET_CUSTODY_REQUIRED`. An unnominated/negative dependent refinement library stops at `FRESH_RESPONSE_CALIBRATION_NO_QUALIFIED_DEPENDENT_REFINEMENT_LIBRARY`. |
| Finite native calibration / evaluation | Frozen finite protocol, complete native exposure and distinct 32/64-root `FiniteResponseLawCohortAssignment` | Missing source: `FINITE_RESPONSE_LAW_SOURCE_ROOT_REQUIRED`. Fixed exposed calibration: `FINITE_RESPONSE_LAW_EXPOSED_CALIBRATION_ROSTER`. |
| missing-future supplement / informative-composition | retained prepared parent. Missing-future supplement all 80 handoffs, informative-composition strict `FiniteResponseLawInformativeCompositionParent` with 48-root aliases/features/masks | `FINITE_RESPONSE_LAW_PARENT_MANIFEST_REQUIRED` precedes grant checks. Informative-composition's wrong schema stops at `FINITE_RESPONSE_LAW_INFORMATIVE_COMPOSITION_TYPED_RETAINED_PARENT_REQUIRED`. No fresh-unit claim. |
| calibration / prospective-evaluation | calibration: assigned calibration-native evaluation plus frozen method operands. Prospective-evaluation: qualified assigned calibration report and exact evaluation cohort | `FINITE_RESPONSE_LAW_TARGET_TYPED_STAGE_PARENT_REQUIRED`. A negative calibration eligibility stops prospective-evaluation at `FINITE_RESPONSE_LAW_PROSPECTIVE_EVALUATION_QUALIFIED_PARENT_REQUIRED`. |
| Continuation / preparation-screen | Original 64-root prospective-evaluation retention / prospective-evaluation formal-closeout reference, all 24 prefixes and four qualified lower-package records | `FINITE_RESPONSE_LAW_RETAINED_ROSTER_REQUIRED` for an altered continuation census. Preparation-screen's eligibility-reference mismatch stops at `FINITE_RESPONSE_LAW_PREPARATION_SCREENING_ELIGIBILITY_IDENTITY_MISMATCH`. |
| response-composition | Outcome-visible retained prepared canonical scalar cache with original 32-root census and `ResponseCompositionDevelopmentSpec` | `RESPONSE_COMPOSITION_TARGET_CUSTODY_REQUIRED`. Negative prepared-response is allowed for explicitly post-hoc nonpromotable analysis, never prospective rescue. |

Historical-schema parents refuse rather than being namespace-converted. A
successful input check constructs providers/plans with zero native or analysis
tasks. The separate issued execution/reveal route remains necessary. All parent-dependent finite selectors share the ordered missing-input stops
`FINITE_RESPONSE_LAW_PARENT_MANIFEST_REQUIRED`, `FINITE_RESPONSE_LAW_TARGET_CUSTODY_REQUIRED`,
`FINITE_RESPONSE_LAW_TARGET_REVEAL_REQUIRED`, `FINITE_RESPONSE_LAW_TARGET_ANALYSIS_REQUIRED` and
`FINITE_RESPONSE_LAW_TARGET_CUSTODY_STORE_REQUIRED`.

After authenticated parent replay, a
missing authoring packet stops at `FINITE_RESPONSE_LAW_CONSUMER_INPUT_REQUIRED` (informative-composition instead
selects its retained method directly). Calibration, evaluation and continuation
also stop at `FINITE_RESPONSE_LAW_ASSIGNMENT_REQUIRED` before parent access when their
assignment is absent. These codes identify the first unresolved handoff, not a
request to fabricate that input. Exact executable formats are in the linked
selector/consumer definitions and [six-matrix input formats](response-input-formats.md).

## Reading a terminal result

Read five separate fields/decisions together: operational completion,
evaluability, scientific support, admission, and permitted use. For example,
the historical reactor R2 run completed its 67-task program and evaluated
local pairs. Four pairs supported local laws. Nine were not supported and
three unevaluable.

Its initial domain remained uncovered, and admission/controller-use were not
attempted. Thus completion did not establish a global law or controller license. An unevaluable pair supplies no negative or positive numerical conclusion. An
evaluable `NOT_SUPPORTED` pair preserves the failed criterion.

Similarly, a
negative prepared-response common charter stops dependent refinement and fresh response calibration, even with valid receipts.
Response composition may analyze that retained negative parent under separate outcome-visible authority.
It cannot relabel the parent fresh. See [glossary](glossary.md) and
[scientific integrity](scientific-integrity.md).

The audited portable gate took roughly 20 minutes on the tested host (14 minutes
tests, five installed smoke), with a warm offline cache and several GiB of local
build/environments. Native and held runs depend on their exact envelopes. Check
space, task/update ceilings and permits before execution. These timings are
observations, not an SLA or a predicted scientific-run limit. CI profile budgets
and cache assumptions are in [CI operations](ci.md).

## References and research

The linked source modules, command metadata and existing checks own the implemented behavior described here.
The [scientific integrity guide](scientific-integrity.md) defines its separate evidence and authority boundaries.
The [program source register](../paper/SOURCES.md) identifies bounded historical results and unavailable primary records.
