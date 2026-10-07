# Read a result or recover a failure

SPDX-License-Identifier: CC-BY-4.0

This guide explains development reports, operational attempts and issued scientific outcomes.
It keeps their identities and evidence limits distinct.
Start with the readable report, then inspect its exact inputs and outcome pointers.

## Find the output

The [workflow catalogue](../experiments/README.md) identifies each task's commands, input roles and first missing prerequisite.
`workflow show ID --format json` reports static facts. It inspects no environment, source or authority.
The [CLI reference](cli.md) owns exact effects, options and exits.

| Product | Meaning and next action |
|---|---|
| Reactor example `report.md` | First readable summary. The paired contrasts describe exposed development units. |
| Example `report.json` and `panel.json` | Exact summary and canonical native episodes. Arms and numerical views remain nested within five independent units. |
| Development `report.json` | The original command's successful JSON output. Interpret its actual units, clocks and denominator using the family guide. |
| Development `report.md` | Readable operational state, exit and stage, with links to the actual JSON quantities. It supplies no qualification. |
| `invocation.json`, `diagnostics.jsonl`, `failure.json` | Operational context and diagnostics for an explicitly retained attempt. They do not replace scientific receipts. |
| Candidate authoring directory | Family-specific strict records, pure candidate/plan projections and source/resource contracts. Compilation is not issue or execution. |
| Issued manifest, artifacts, receipts and adjudication | Primary recorded acts and outcome identities under separate source, custody and authority gates. |
| Local catalog | Rebuildable metadata projection. Its absence is not absence of primary receipts. |

Keep operational output outside the source tree, except for the documented disposable demo destination.
A retained attempt is a reproduction/debugging aid, not a fresh independent scientific unit.

## Retain a development attempt

Participating stdout diagnostics accept an explicit new `--output-dir`.
For example, after [validating and preparing a copied RC model](configuration.md):

```sh
uv run --no-sync empirical-lawhood campaign rc-ladder-native-check \
  --config /absolute/development/rc-model.json \
  --output-dir /absolute/development/rc-ladder-response/attempt-001
```

The default stdout JSON and exit convention remain unchanged when the export option is omitted.
Progress and output-location messages use stderr. Raw stdout/stderr redirection remains possible,
but an empty captured stdout file is not a successful report.

The writer retains invocation context and bounded diagnostics.
`invocation.json` records `operation`, `state`, `exit_code`, `stage`, `last_completed_stage`,
`operation_completed`, `native_contact`, input/product identities and observed `runtime`.
Treat unknown contact or incomplete finalization as unknown, not as no contact.
`failure.json` normally records `exception_type`, `cause_types`, `stage`, a sanitized `diagnostic` and `next_action`.
It does not store arbitrary exception text, stack locals or hidden outcomes. A secondary export failure can retain a fuller operational failure record.
Exact returned scientific quantities remain in the original route report.
A successful computation exports its actual `report.json` and readable `report.md`.
A failed invocation exports `failure.json` when its selected destination is usable.
No success-looking `report.json` is invented for failure.
A bounded `input.json` (or supported `input.yaml`/`input.yml`) snapshot is present only for permitted small development inputs.
Large, guarded, held or authority-bearing inputs remain references/digests under their owning contracts.

Closed authoring bundles and the reactor demo retain their existing `--output-dir` products.
Use their separate `--attempt-dir` for operational records, with disjoint destinations.
Do not add operational sidecars to a closed, hash-bound scientific package.
Each family's existing empty-directory, nonexistence and partial-export rules remain in its guide.

### Recognise an actual retained directory

For the RC diagnostic above, a successfully exported attempt has this layout:

```text
attempt-001/
├── invocation.json       operation, input identities, contact and final state
├── diagnostics.jsonl     bounded progress and stage events
├── input.json            permitted small RC config snapshot for this command
├── report.json           the diagnostic's actual returned numerical quantities
└── report.md             readable operational state and links to those quantities
```

The snapshot contains the copied development model.
It contains no held or protected source.
Read `report.json` with the [RC field guide](../experiments/rc-ladder-response/guide.md#read-results-and-failures).
Voltages and solver differences are in V.
Boundary currents are in A.
Both solvers describe one model unit.

Exit 0 means that this diagnostic returned.
It does not issue a campaign or qualify a physical board.
A refused RC invocation exits 2.
If retention succeeds, its directory contains `invocation.json`, `diagnostics.jsonl`, `failure.json` and `report.md`.
`input.json` exists only if a permitted snapshot was read.
A refused invocation has no successful `report.json`.

An interrupted or failed export can leave only some of these files.
Inspect the actual final state.
Inspect completion and native contact before another operation.
Other commands retain their own products.
A closed authoring `--output-dir` or analysis `--output` is separate from its optional operational `--attempt-dir`.

| What you see | What to inspect next |
|---|---|
| Success report and final `SUCCEEDED` attempt | Read the family's quantities, units, independent denominator and scientific ceiling. Support and admission remain separate. |
| Invalid copied input or exit 2 refusal | Read the terminal refused field. Read the bounded failure/stage record. Repair the editable copy. Validate it. Select a new documented attempt path. |
| Destination refusal | Read that command's directory rules. Preserve occupied or partial products. A refused destination can retain no durable attempt. |
| Missing terminal record or interrupted export | Preserve genuine partial bytes. Determine completion and contact before further work. Missing finalization does not prove no native contact. |
| Saved analysis without completion publication | Treat it as incomplete under the family contract. Do not present its report as authenticated completion. |
| Reveal barrier | Obtain the separately authorized reveal. Use receipt recovery with the same identities. Do not repeat completed acquisition. |
| Valid negative or unevaluable scientific result | Retain the full census and declared disposition. An operational retry cannot select a different scientific result. |
| Completed doctor with warnings | Inspect availability and version fields. Select the guide's environment before native work. Diagnostic completion is not scientific readiness. |

## Work through an RC result and refusal

The [RC model guide](../experiments/rc-ladder-response/guide.md#edit-a-development-model) keeps the original edited copy and prepared input.
In a successful diagnostic, inspect these exact report fields:

| Field | Interpretation |
|---|---|
| `config_id`, `cells`, `horizon_s` | Which numerical model, node count and time horizon were computed. |
| `receiver_unit`, `boundary_current_unit` | Voltage is in V. Boundary current is in A. |
| `final_cell_voltages_v` | Ordered final node voltages. They are not independent trials. |
| `maximum_backward_euler_defect_v` | Difference between the two numerical views, not physical-board validation. |
| `native_numerical_check_executed` | Whether this disclosed numerical diagnostic completed. |
| `campaign_candidate_compiled`, `campaign_issued` | Both remain false for the native diagnostic. |

The shipped model has four cells, a one-second horizon and frozen declared components.
The edited-resistance example computes another numerical model. It does not establish a physical result or a fresh cohort.
The separate study route has its own frozen voltage/charge-rate tolerances and independent falsifier tests.

For a deliberate refusal, set one copied `capacitances_farads` Decimal value to zero.
`config validate` refuses the invalid typed model before native contact.
If an invalid input reaches the native command with a usable retained destination, read its `failure.json` and diagnostics.
The invalid attempt has no successful numerical report. Repair the copied input, validate and prepare it again,
then use a fresh attempt path. Keep the failed attempt as evidence of the earlier invocation.

## Ask six questions

1. Which system, input identity and operation produced this result?
2. Did the operation complete, stop at a prerequisite, or leave completion unknown?
3. What quantity, units, clock and independent denominator does it report?
4. What scientific claim does the evidence support, and what is its ceiling?
5. Which exact input, output, receipt or adjudication contains the underlying record?
6. What next action is implemented, and what additional input or authority does it require?

A returned quantity is not automatically an admissible choice.
Operational completion, evaluability, support, scientific verdict and admission are separate observations.
A valid negative result is not an execution failure and must not enter operational retry.

## Incomplete exports and interruption

A usable output destination permits retention of early option/input errors and later failures.
An invalid or unwritable destination can prevent durable retention. Use the terminal diagnosis in that case.
Machine loss, forced termination or storage exhaustion can leave only a start record or partial products.
Missing finalization does not prove that no native contact occurred.

An export failure after computation means completed work may exist even though its report could not be written.
Keep genuine partial products and the diagnosis. Do not automatically repeat the computation.
Development retention supplies no resume transaction. Follow the family's documented fresh-path policy.
Issued recovery uses verified receipts to avoid repeating completed native effects.

## Issued status and recovery

Use the same project, profile and authoring directory as the original issued operation.
For the assigned reactor route, the complete read-only status command is:

```sh
uv run --no-sync empirical-lawhood \
  --project-root /absolute/clean-checkout \
  --operator-profile /absolute/operator-profile.json \
  --reactor-authoring-dir /absolute/external/authoring/run-001 \
  campaign status --run run-001 --attempt-history \
  --attempt-history-limit 100 --format json
```

`--attempt-history-limit` is bounded to 1–1,000 rows.
If the report supplies a next cursor, pass it with `--attempt-history-cursor` to obtain the next page.
Use the returned cursor unchanged. This is pagination, not another execution attempt.
Status reads verified execution/attempt history without starting native tasks or revealing hidden outcomes.
Read artifact and adjudication pointers only through the existing authorized store/API owners.

An expected reveal barrier means the required reveal act has not occurred.
It is different from native calculation failure, an unevaluable adjudication or an unsupported scientific result.
Follow [the operator walkthrough](reactor-operator.md#8-resume-the-same-identities) for actual separately authorized recovery.
Preserve the original source commit, run/package/issue IDs, trust roots, approvals, grants, timestamps and store.
Source changes or unknown native completion can require the declared amendment path.

The local `.empirical-lawhood/` catalog is disposable projection state.
[Catalog recovery](compatibility.md#local-catalog-baseline-and-recovery) verifies primary receipts and authenticated projections before reconstruction.
Never delete real custody files or rewrite a receipt to make recovery succeed.

## Dependencies are not scientific readiness

`doctor --route` reports inspected dependency/version state and optional operational storage state.
A completed diagnostic exit does not mean that dependencies match or that a study is ready for issue.
Check warnings, `available`, `installed_version`, `required_version` and `version_matches` separately.
Use [environments](environments.md) for the selected locked profile and Brian2's separate interpreter.
A stale editable distribution version is a local installation mismatch, not proof of a defective wheel.

Use the [researcher guide](designing-and-running-an-experiment.md) for the next supported scientific boundary.
Historical records, authentic inputs, clean source and separate authority cannot be supplied by a green dependency check.

## Current integration results

`campaign rc-challenge-result` and `campaign finite-response-result` select exact stored receipts and authenticate their complete output publication. Protected reads require actual persisted current issue/execution/reveal authority. Their readable scientific result is distinct from `campaign status` execution state. Exported current result/parent inputs can be selected by later stages without acquiring those outcomes again. The preparation workflow similarly binds current Q and panel outputs before deriving selections or saved screen operands.

Native development exports retain their selected source/allocation before first contact.
The family's authenticated final receipt or bank manifest marks completion.
Current native matrix exports additionally require the guarded
`analysis-completion.canonical.json` publication and its complete member census.
Its absence means incomplete acquisition.
Preserve partial output and attempt diagnostics before selecting the documented recovery or fresh-path operation.
Saved operand analyses never reacquire native trajectories.
These helper attempts create no scheduler retries, qualification or study grants.

[History analysis](../experiments/matrix-history-analysis/guide.md) retains all 256 requested roots and their four family denominators.
A wholly absent source receipt/publication pair produces explicit `UNENTERED` accounting.
A partial pair, corrupt member or mismatched source refuses authentication.
Numerical invalidity remains `UNEVALUABLE` rather than disappearing from the cohort.
Complete source receipts permit read-only recovery without another native acquisition.

[Tangent, transient and baseline reports](../experiments/matrix-transient/guide.md) preserve complete cells, coefficients, masks and negative comparisons.
Their guarded `analysis-completion.canonical.json` publication binds the final
report, complete inputs, producing environment and all retained members.
A report without that completion publication is an incomplete attempt.
Report-only edits refuse authentication. Readout needs no fit or acquisition.
Interrupted directories require the documented new destination.
[Geometry](../experiments/matrix-geometry/guide.md) and [selected events](../experiments/selected-events/guide.md) retain their own cell and conditional-instance meanings.
[Preparation diagnostics](../experiments/preparation-diagnostics/guide.md) authenticates existing phase outputs instead of replaying native effects.
Historical counts and successful command exits are never substituted for actual scientific results.
See [current integration guides](integrations.md) for exact paths, access contracts and evidence ceilings.
