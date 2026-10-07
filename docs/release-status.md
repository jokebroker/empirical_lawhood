# Release identity and evidence status

SPDX-License-Identifier: CC-BY-4.0

This guide defines the external record that selects one source and its release evidence.
It joins software, paper, scientific qualification and review without changing their original records.
Use it to find the selected identity, its evidence and any remaining prerequisite.

## Prerequisites and first action

Obtain permitted access to the maintainer's external evidence directory.
Read its selected `release-selection.json`.
Compare its source and file hashes with the original packets before using a release claim.
A missing record, hash mismatch or unresolved required check stops selection.
The [artifact procedure](release-artifacts.md) gives the software packet's construction and interpretation.

Package metadata owns the software version.
The selection record owns the selected source, artifacts, evidence and release stage.
This guide does not contain a second mutable release status.
A version, tag or familiar filename alone cannot identify selected evidence.

## Software snapshots and later selection

The release runner writes `release-manifest.json` and `release-status.json` in a new external packet.
The manifest records its actual profile, source, host, checks, logs and built artifacts.
The status binds that manifest's SHA-256 and repeats its source and artifact identities.
Its scientific qualification status is `NOT_PERFORMED`.
That statement describes what the software gate performed.
It remains true when a later scientific act occurs.

Keep both files and all bound evidence unchanged.
Record a later qualification, review, tag or publication in a new external selection record.
The later record binds the software snapshot and the separately produced acts.
It does not turn software conformance or an exposed example into scientific qualification.
Never overwrite a failed packet, previous selection or negative scientific result.

## Selection record contract

The selected file is UTF-8 JSON named `release-selection.json` in a new external directory.
It uses schema `empirical-lawhood/release/selection` and format version `1.0.0`.
It is maintainer evidence, outside the source checkout and published package.
Do not write it into the source after qualification.
That write would change the source being qualified.

All fields below are required.
Use JSON `null` for an absent act and an empty list for an empty collection.
Absence does not mean success or zero scientific outcomes.
Reject duplicate keys, unknown fields and unsupported format versions.
Use strings for identifiers, versions, stages and UTC timestamps.
Use booleans for `required` and `clean`.
`limitations` is a list of explicit strings.
Unperformed review and packet collections remain empty.
Unfinished paper and source-review file bindings remain null.
Their required bindings must all exist before `SOURCE_READY`.
The record is a selection contract, not a signed scientific authorization.
An owner must review its bindings and applicable authority separately.

| Field | Required content |
|---|---|
| `schema`, `version` | The exact schema and format version stated above |
| `created_utc` | Actual record creation time in UTC |
| `selection_id` | A distinct identifier for this immutable selection |
| `previous_selection` | `null`, or a file binding for the previous selection |
| `release_stage` | `WORK_IN_PROGRESS`, `SOURCE_READY`, `LOCAL_CANDIDATE_READY` or `PUBLISHED` |
| `distribution_version` | The selected distribution metadata's version |
| `selected_source` | The exact source object from the selected portable manifest |
| `git_tree` | The Git tree object of that source commit |
| `software_packets` | Ordered packet entries for the portable gate and every selected native/held profile |
| `artifacts` | Exact wheel, sdist and any selected archive file bindings |
| `paper` | Original-build or preservation receipt, numerical/document-check and visual-review bindings, plus all selected asset bindings |
| `source_review` | Completion index, finding ledger, preservation matrix and public-history/name-check bindings |
| `qualification` | The status and act bindings described below |
| `reviews` | Actual reader, scientific-boundary and owner-review entries, each with its scope and bound materials |
| `publication` | Actual publication status, tag event, hosted evidence and published artifact bindings |
| `limitations` | Explicit evidence ceilings, absent inputs, unselected profiles and unresolved prerequisites |

Each file binding has exactly `locator`, `size_bytes` and `sha256`.
The locator is relative to the selection directory.
The byte count is a nonnegative integer.
The SHA-256 is the lowercase, 64-character digest of the exact retained bytes.
Reject missing files, unapproved path escape and mismatching counts or digests.
Preserve the external root's custody policy.
Do not copy held records, keys or private paths into public artifacts.

`selected_source` retains the manifest's exact `commit`, `tree_inventory_sha256`, `clean`, `tags_at_commit` and `uv_lock_sha256` fields.
`tree_inventory_sha256` hashes the Git tree inventory, not the Git tree object itself.
`git_tree` supplies that separate object identity.
Tags observed by the gate remain a snapshot.
A later tag event belongs under `publication` and does not change that snapshot.

Each software packet entry has `profile`, `required`, `manifest` and `status`.
The last two fields are file bindings for the original manifest and status files.
Their source objects must equal `selected_source`.
The status's `manifest_sha256` must match the bound manifest.
Every required packet must pass with its required checks present.
Its logs and environment evidence must match the manifest's hashes.
Select artifact bytes from the verified packet's actual artifact entries.
The [test profiles](testing.md) define native and held prerequisites.

`paper` has `build_manifest`, `numeric_check`, `visual_review`, `source_inputs`, `tools` and `assets`.
The first three are file bindings.
The last three are lists of file bindings for source inputs, tool/environment evidence and selected assets.

The `build_manifest` field retains its name in selection format `1.0.0`.
It binds either an original publication build manifest or an as-is preservation/provenance receipt.
For the selected edition 0.60, the [preservation owner](../paper/source/source-preservation.json) records the complete original archive identity, retained unchanged in external custody. The public [current-edition ZIP](../paper/editions/Empirical_Lawhood_v0.60_Selected_Source.zip) is a curated subset; the receipt binds its exact member-to-public-path mapping, document byte equality and absence of content or navigation edits.
Its original-build section distinguishes the supplied recipe and validation report from unavailable producing tool versions, commands and full build logs.
A preservation receipt records selection and inspection of supplied assets; it is not a new build act.
Keep the original archive, supplied manifests and historical producing source identities unchanged.

Bind the available original source and tool records in `source_inputs` and `tools`, along with the actual validation/review environment evidence.
Distinguish originally reported tools from tools used for the current inspection.
An empty original-tool inventory does not imply a tool-free build: state the concrete unavailable records in the receipt and `limitations`.
Do not invent tool versions, commands, a manuscript DOI or a producing commit from the current software source.
The `numeric_check` binding records publication inventory and available published-summary consistency checks with their actual scope and limitations.
It does not require new simulations, fits, bootstrap samples or scientific adjudication.
The visual review must cover every page of each selected PDF and every selected figure format, naming their exact output and inspected render identities.
A different derivative requires its own review.
Preserving or rebuilding publication assets cannot authenticate absent historical observations.

`source_review` has `completion_index`, `finding_ledger`, `scientific_preservation` and `public_history` file bindings.
The history record enumerates the proposed public refs and scanned reachable objects.
It also binds the package/archive/public-name checks to the selected artifact hashes.
When a new public root replaces development ancestry, record the reviewed private-to-public payload mapping and retain the previous source identities in external provenance and recovery evidence.
The new software packet binds that root itself; collapsing history does not rewrite producing identities in historical scientific records or qualify the new source.
An unresolved required finding prevents `SOURCE_READY`.
Preparing an isolated public ref does not authorize a push or replacement of existing public history.

## Scientific qualification and review

`qualification` has `status`, `inputs`, `authority`, `package_and_plans`, `task_receipts`, `adjudication` and `summary`.
Its status is `NOT_PERFORMED`, `IN_PROGRESS` or `TERMINAL_RECORDED`.
Before qualification, the first status requires null act fields and an empty `task_receipts` list.
For an attempt, `inputs`, `authority` and `package_and_plans` are lists of file bindings.
`adjudication` and `summary` are file bindings when those acts exist.
They remain null while the corresponding act is absent.
Keep that absence explicit.
Do not supply example inputs, test signers or synthetic receipts as release authority.

For an actual attempt, bind the reviewed source inventory, exposure census and assigned roster under `inputs`.
Bind separate issue, independent approval, execution and reveal acts under `authority`.
Bind the exact issued package, scientific run plan and executable plan under `package_and_plans`.
All of those identities must belong to the selected source and permitted custody.
Retain interrupted, failed and negative attempts in the source/exposure accounting.
Their presence does not qualify a corrected source or a reused cohort.

Each `task_receipts` entry has exactly `task_id`, `receipt` and `terminal_disposition`.
`task_id` identifies the planned task.
`receipt` is its actual file binding, or null when no receipt exists.
`terminal_disposition` preserves the actual engine result or states that the task is unfinished.
The summary reconciles every planned task with that list.
An unfinished task cannot become a successful receipt through a summary.
`adjudication` binds the separately authorized terminal scientific record.
`summary` binds its interpretation and the exact claimed evidence ceiling.
Keep operational completion, evaluability, support, admission and permitted use separate in that summary.
A negative or unevaluable result keeps its actual disposition.
Do not infer a controller license from successful execution.

Each review entry records `reviewer_type`, `scope`, `source`, `materials`, `result`, `record` and `created_utc`.
`source` gives the exact reviewed source object.
`materials` lists file bindings for the reviewed evidence and artifacts.
Use strings for reviewer type, scope and actual result.
The report gives that result's meaning and any unresolved findings.
`record` binds the actual review receipt or report.
A fresh-context reader checks documentation and the first genuine owner-knowledge handoff.
That reader supplies neither the owner's roster knowledge nor an independent scientific signature.
Owner final release review must cover the exact final candidate bytes.
An earlier review cannot cover a later source or history change.

## Release-stage conditions

Use `WORK_IN_PROGRESS` while a required source gate remains unresolved.
Use `SOURCE_READY` only after all required source, software, paper, history and reader gates pass.
For an as-is published paper selection, the required paper bindings identify the unchanged original bytes, their provenance and availability, and the actual consistency and visual reviews.
Regenerating the paper or rerunning its experiments is not a prerequisite for this stage.
That stage permits preparation for qualification.
It does not assert fresh qualification, final owner approval or publication.

`LOCAL_CANDIDATE_READY` also requires the actual bounded qualification and the exact local candidate/artifact review packet.
A preexecution refusal alone cannot satisfy the required native qualification.
The required personal final review and permitted publication acts remain separate.
Use `PUBLISHED` only after those actual acts and verified hosted evidence are retained.
No stage grants signing, native execution, push or publication authority.

`publication` has `status`, `tag_event`, `hosted_checks`, `published_artifacts` and `published_utc`.
`tag_event` is a file binding for the actual tag event.
`hosted_checks` and `published_artifacts` are lists of file bindings.
`published_utc` is the actual UTC publication time, or null before publication.
Use `NOT_PUBLISHED` with a null tag event, empty lists and null publication time before publication.
Use `PUBLISHED` only with the actual tag, published hashes, hosted evidence and publication time.
Never invent a DOI or publication date.
The [citation guide](../CITATION.md) owns how recipients cite the selected work.

When evidence changes, create a new selection directory and bind the previous selection.
The maintainer supplies one selected record locator to the owner-review packet.
That external locator identifies the current selection.
It does not make old software or scientific snapshots mutable.

## Historical evidence and honest limits

Historical reactor evidence reports four supported local-law pairs, nine unsupported pairs and three unevaluable pairs.
Its initial domain remained uncovered.
Admission and controller use were not attempted.
Historical prepared-response, finite-response and post-hoc studies retain their own distinct cohorts and outcomes.
No historical receipt qualifies this release's source.

The [paper source register](../paper/SOURCES.md) identifies the original result and research sources.
Primary historical archives are not deposited with the publication derivatives.
Selected paper checks establish preservation or build consistency and numerical/document/visual consistency within their stated scope.
The complete preprint experiment rerun programme follows the final software changes and freeze; its new results form separate evidence and do not replace the canonical published assets.
That programme and the fresh bounded qualification required for `LOCAL_CANDIDATE_READY` remain distinct acts.
The [scientific integrity guide](scientific-integrity.md) gives the preservation and authority boundaries.

## References and research

The [release runner](../scripts/release_check.py) owns the immutable software snapshot fields.
The [artifact guide](release-artifacts.md) owns artifact construction and installed verification.
The [operator lifecycle](reactor-operator.md) owns separately authorized execution and adjudication.
Use the [support route](compatibility.md#report-a-defect) to request permitted access to the external packet.
