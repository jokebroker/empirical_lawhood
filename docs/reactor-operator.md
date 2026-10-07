# Operate one assigned reactor campaign

SPDX-License-Identifier: CC-BY-4.0

This walkthrough gives instructions for the separately authorized lifecycle of one assigned reactor campaign.
It keeps five units and twenty native branches within the fixed public scenario conditions.
Use it to carry reviewed inputs through issue, execution, reveal and receipt recovery.

This walkthrough carries the supported five-unit, twenty-branch reactor prefix
from authoring to durable receipts and a separately authorized adjudication. The public scenario conditions remain fixed. Fresh noise seeds support a finite conditional contrast. They establish no new population, physical-reactor safety certificate or paper reproduction.

Read [the response-law example](../README.md)
first. The [software lifecycle test](../tests/reactor_lifecycle_scenario.py) exercises the same API composition with fabricated roles in disposable storage. Its declarations, seeds, signers and approvals cannot qualify your study.

Use the selected **clean editable checkout**, Linux x86-64, CPython 3.11.14,
NumPy 2.4.6, and the [locked environment](environments.md). Keep these locations distinct:

- `REPO`: the source checkout.
- `PROFILE`: your deployment-local storage profile.
- `AUTHORING`: a new directory beneath its external artifact namespace.
- `CONTROL`: a private directory outside the checkout and scientific artifact root.
- `TRUST`: the explicitly installed public checker registry in an owner-only directory. Keep checker private keys separately with
mode 0600 and their directory mode 0700. Never put them in Git or the release
packet. The small prefix usually takes seconds to minutes. Allow at least
1 GiB for control/receipt packets plus the selected profile's free-space floor.
Large exposure inventories require more space and serialization time.

## 1. Select storage and freeze the census

Create an `OperatorStorageProfile` using the complete
[storage profile example](../configs/operator-storage.example.json). Select an actual external
mount and existing artifact namespace, distinct scratch namespace, filesystem
type, free-space floor and maximum concurrency. `authority_granted` is always
false. Files beneath a checkout, home directory or `/tmp` cannot substitute for
the guarded scientific root. Inspect the actual profile before any campaign:

```sh
uv sync --locked --python 3.11.14 --group reactor-example
uv run --no-sync empirical-lawhood --project-root "$REPO" \
  --operator-profile "$PROFILE" doctor --route reactor --format json
```

Resolve every reported storage or version failure. Doctor creates no catalog
and grants no authority. A successful diagnostic exit is not a ready-to-run
verdict.

The [assigned example](../experiments/reactor-response/exposed-inputs/reactor-assigned-exposed.json) shows
the exact nested canonical format. It deliberately reports
`prospective_issue_eligible=false`. For real qualification, construct `ReactorPrefixPriorCensus` from a reviewed inventory of **all** known exposures. Include public seeds, canaries, tests, proposed rosters, interrupted/failed campaigns, development, evaluation and model-training roots.

Include native numeric aliases and purpose-specific PCG/state/stream identities. Retain the original bytes, locators, hashes and source membership used to make
each union. Include both prior `excluded_*` and `proposed_*` six-matrix entries as
explained in [six-matrix input formats](response-input-formats.md). Missing history is a
scientific stop, not an empty list.

Write an immutable inventory with an explicit schema/version and inventory ID.
Its SHA-256 must be calculated over its retained bytes. Put that identity in
`census.source_inventory`. The complete, sorted, unique unit and seed unions
become `prior_unit_ids` and `prior_seed_ids`. Use
`ObjectIdentity.from_record(census.census_id, census)` as
`assignment.prior_census`. A digest authenticates a chosen inventory's bytes.
An accountable inspection must establish its completeness.

Choose a new experiment ID. Draw five distinct integers with
`secrets.randbelow(2**63 - 1) + 1` once, before native contact. Reject every
collision with both the numeric and `seed.native-reactor.<integer>` aliases. Construct the units in the `SCENARIOS` order from
`adapters.simulators.reactor_prefix_response.panel`.

Each `unit_id` is
`<experiment_id>.cohort.<scenario-with-hyphens>`. Examine both that ID and its
`unit.`-prefixed independent identity against the census. Paired arms and both
numerical views share each unit's seed. The assignment constructor and
`check_prior_census` enforce these relationships.

Freeze the selected values.
Do not redraw after seeing an unfavorable response.

Create `AssignedReactorAuthoringProfile(profile_id, experiment_id,
public_source_sha256, assignment, prior_census)`, taking the source fingerprint
from the installed packaged source bundle (`adapters.simulators.reactor_prefix_response.packaged_source.load_packaged_reactor_source`). Use `PROSPECTIVE_RELEASE_QUALIFICATION` only after the real census and eligibility
review. Changing that string on the sample is not such a review. Serialize with
`canonical_bytes()`.

The complete census stays immutable. The authoring projection omits only numeric native-reactor aliases outside the permitted positive 63-bit seed domain. It retains the census identity, all valid aliases and source-purpose identities.

## 2. Author and prove the actual composition

```sh
uv run --no-sync empirical-lawhood --project-root "$REPO" \
  --operator-profile "$PROFILE" campaign reactor-author \
  --profile "$ASSIGNED_PROFILE" --output-dir "$AUTHORING"
uv run --no-sync empirical-lawhood --project-root "$REPO" \
  --operator-profile "$PROFILE" --reactor-authoring-dir "$AUTHORING" \
  campaign reactor-preissue-proof
```

Author emits `profile.json`, `base-authoring.json`, `authoring.json`, numbered
`payload-*.json`/`decoder-*.json`, `base-candidate.json`, `candidate.json`,
`candidate-report.json`, `source-closure.json`, `resources.json`, the two
preissue plans, and `evidence-profile.json`. Expect 21 tasks and zero native
tasks executed. Proof resolves the actual CLI provider/runners, validates all
output locators and byte ceilings, and persists/replays four control records.
All six closure sections must pass.

It neither issues nor approves the study. Keep these files unchanged. A changed source commit requires new authoring and
proof, followed by fresh review of the changed identity.

The rest of this guide uses explicit Python API calls so returned record IDs
are carried directly, without manually copying long hashes. Retain the following
blocks as `CONTROL/operator.py` **outside the clean checkout**, inserting your
reviewed acts at each handoff, and export `REPO`, `PROFILE`, `AUTHORING`, `CONTROL`
and `TRUST`. From any working directory, use this supported launcher:

```sh
uv run --project "$REPO" --no-sync python "$REPO/scripts/run_operator.py" "$CONTROL/operator.py"
```

The launcher makes this checkout's helpers importable and fixes the single-thread
numerical environment before importing numerical libraries. Run only the blocks
whose explicit review prerequisites have been completed. For an interactive
session from `REPO`, first import and call
`empirical_lawhood.cli.entry.enforce_single_thread_environment` before any other
project or numerical imports. The helper
[operator_records.py](../scripts/operator_records.py) adapts the parent
project's durable-store operations. It supplies no standing grant, signing key,
passing decision or native launch. These helpers and adapter constructors are
checkout recipes. `empirical_lawhood.api` is the public application boundary.

```python
import os
from pathlib import Path
from datetime import datetime, timezone
from empirical_lawhood.api.composition import create_cli_api
from empirical_lawhood.api.results import (
    IssueStudyRequest, IssueExtensionsRequest, AssembleExperimentPackageRequest,
    CompileCampaignRequest, RunCampaignRequest, ResumeCampaignRequest, CampaignStatusRequest,
)
from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile
from empirical_lawhood.runtime.candidate_compiler import ExecutableStudyCompilationReport
from empirical_lawhood.planning.study_issue import (
    AccountableHumanIdentity, HumanProposerAttestation, ImplementationSourceClosure,
    StudyOperationAuthority, StudyAuthorityKind,
)
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.infrastructure.study_issue import (
    PROGRAMME_ISSUE_GRANTEE_ID, PROGRAMME_EXECUTION_GRANTEE_ID, PROGRAMME_REVEAL_GRANTEE_ID,
)
from scripts.operator_records import (
    read_record, export_record, identity, open_stores, publish_reviewed_authority,
    freeze_for_review, authorize_reviewed,
)

repo, directory, local = (Path(os.environ[k]).resolve() for k in ("REPO", "AUTHORING", "CONTROL"))
profile = read_record(Path(os.environ["PROFILE"]), OperatorStorageProfile)
report = read_record(directory / "candidate-report.json", ExecutableStudyCompilationReport)
candidate = report.candidate
assert candidate is not None
base = candidate.base_candidate.base_candidate
system, policy = base.system, base.system.authority_policy
closure = read_record(directory / "source-closure.json", ImplementationSourceClosure)
run_id = base.experiment.experiment_id
def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
def require_success(result):
    if not result.succeeded:
        raise RuntimeError((result.status, result.reason_codes, result.errors))
    return result.payload
```

## 3. Install independently reviewed checker trust

The operator records the accountable human as
`AccountableHumanIdentity(human_id, role_id, identity_provider_id,
identity_subject_sha256)` in `CONTROL/human.json`. The identity provider and
subject digest must identify the actual accountable role under your policy.
An invented name or chat agent is not an attestor. Export it with
`export_record`, which refuses to replace different existing bytes.

Each checker generates or loads its own Ed25519 key using
`Ed25519ApprovalAttestationSigner.create_private_key_file(private_path)` or
`from_private_key_file(private_path)` from `infrastructure.authority`. Install
only its `verification_key_hex` in an `ApprovalCheckerRegistration`:

```python
from empirical_lawhood.planning.approval import (
    ApprovalCheckerRegistration, ApprovalCheckerRegistry, ApprovalGateKind,
    ApprovalSignatureAlgorithm, ApprovalGateAttestation, AttestationResult,
    issue_approval_gate_attestation,
)
print(policy.required_gate_ids, policy.delegate_id, policy.scope_ids)
```

For **each exact required gate ID**, construct a registration with these fields:
`checker_registration_id`, `gate_id`, `gate_kind`, `checker_id`,
`implementation_sha256`, `implementation_version`, `signature_algorithm`,
`signature_version`, `verification_key_hex`, `outcome_access`. The reactor's
`<experiment_id>.exact-production-proof` uses `ApprovalGateKind.IMPLEMENTATION`. Its other required
gate uses `ApprovalGateKind.AUTHORITY`. Record the reviewed checker implementation's
actual digest/version. The remaining values are `ED25519`, `1.0.0`, the public
key from that checker, and `OutcomeAccess.OUTCOME_BLIND`. Sort the registrations
by gate ID, construct `ApprovalCheckerRegistry(registry_id, registrations)`, and
export it to `TRUST` with mode 0600 in an owner-only directory. Trust installation
selects who may attest. It makes no passing decision.

```python
trust_path = Path(os.environ["TRUST"])
registry = read_record(trust_path, ApprovalCheckerRegistry)
human = read_record(local / "human.json", AccountableHumanIdentity)
stores = open_stores(repo_root=repo, profile=profile, trust_path=trust_path)
api = create_cli_api(repo_root=repo, operator_storage_profile=profile,
                    approval_checker_trust_path=trust_path,
                    reactor_authoring_dir=directory)
```

## 4. Proposer and custody acts, then base and extension issue

The researcher supplies two `HumanProposerAttestation` records, one for each
candidate. Use the following exact field mapping. The last six fields are
accountable assertions made **before fresh outcome access**, not defaults to
copy from a test.

| Field | Base attestation | Extension attestation |
|---|---|---|
| `candidate` | `identity(candidate.base_candidate, candidate.base_candidate.candidate_id)` | `identity(candidate, candidate.candidate_id)` |
| `raw_materialization_sha256` | `base.authoring_materialization.raw_materialization_sha256` | `report.authoring_materialization.raw_materialization_sha256` |
| `semantic_config_sha256` | `base.semantic_config_sha256` | same |
| `design_origin_id`, `design_input_ids`, `source_qualification_receipts` | `base.design_origin.origin_id`, `base.design_input_ids`, `base.source_qualification_receipts` | same |
| `attestation_id`, `proposer`, `proposer_id` | a new act ID, `identity(human, human.human_id)`, `human.human_id` | a distinct act ID, same accountable identity |
| `known_exposure_lineage_complete` | must be truthfully affirmed | must be truthfully affirmed |
| `evaluation_units_and_seeds_unexposed` | must be truthfully affirmed | must be truthfully affirmed |
| `fresh_child_outcomes_observed` | must be false | must be false |
| `codex_or_chat_is_proposer_attestor_approver_or_issuer` | must be false | must be false |
| `attested_at_utc`, `outcome_access` | actual whole-second UTC time, `OUTCOME_BLIND` | actual act time, `OUTCOME_BLIND` |

The materialization digest above is the engine's media-type-framed identity.
Do not substitute a plain file SHA-256. Export the reviewed records as
`CONTROL/base-proposer.json` and `CONTROL/extension-proposer.json`.

Each operation grant is a separate `StudyOperationAuthority`. The following
constructor recipe creates the requested record. The accountable issuer must
review and deliberately publish each act. `scope_id` is `run_id` throughout:

```python
def operation_record(authority_id, kind, subject, prerequisite=None):
    custody = kind is StudyAuthorityKind.CUSTODY_PUBLICATION
    execute = kind is StudyAuthorityKind.EXPERIMENT_EXECUTION
    reveal = kind is StudyAuthorityKind.OUTCOME_REVEAL
    return StudyOperationAuthority(
        authority_id=authority_id, kind=kind, subject=subject,
        prerequisite_authority=prerequisite, issuer=identity(human, human.human_id),
        grantee_id=PROGRAMME_ISSUE_GRANTEE_ID if custody else
                   PROGRAMME_EXECUTION_GRANTEE_ID if execute else PROGRAMME_REVEAL_GRANTEE_ID,
        scope_id=run_id, storage_root_id=stores.plane.root.contract.storage_root_id if custody else None,
        relative_root="issued-programmes" if custody else None,
        allows_source_acquisition=False, allows_external_publication=custody,
        allows_execution=execute, allows_actuation=False, allows_reveal=reveal,
        issued_at_utc=now(), expires_at_utc=None,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL if reveal else OutcomeAccess.OUTCOME_BLIND,
    )
```

Use kind `CUSTODY_PUBLICATION`, no prerequisite, and exactly the corresponding
candidate identity from the table for `base-custody` and `extension-custody`. An issuer
may choose an explicit expiry after issue. Export each reviewed act to
`CONTROL/<layer>-custody.json`, then explicitly call
`publish_reviewed_authority(stores, record)`. This installs and replays that
one immutable grant. It does not authorize execution or reveal. Changed grant
bytes need a new act ID, not replacement of the old file.

```python
from dataclasses import replace
base_request = IssueStudyRequest(
    directory / "base-authoring.json", directory / "base-candidate.json",
    local / "base-proposer.json", directory / "source-closure.json", local / "base-custody.json",
)
preview = api.issue_study(base_request)
assert preview.reason_codes == ("WRITE_CONFIRMATION_REQUIRED",)
assert preview.payload.external_bytes_written == 0
# After reviewing the planned issue and its effect:
base_issue = require_success(api.issue_study(replace(base_request, confirmed=True)))
base_publication = stores.publisher.load_study(base_issue.manifest.issue_id)
assert base_publication.publication_receipt == base_issue.publication_receipt

extension_request = IssueExtensionsRequest(
    directory / "authoring.json", tuple(sorted(directory.glob("payload-*.json"))),
    tuple(sorted(directory.glob("decoder-*.json"))), directory / "candidate.json",
    base_issue.manifest.issue_id, local / "extension-proposer.json", local / "extension-custody.json",
)
extension_preview = api.issue_extensions(extension_request)
assert extension_preview.reason_codes == ("WRITE_CONFIRMATION_REQUIRED",)
extension_issue = require_success(api.issue_extensions(replace(extension_request, confirmed=True)))
publication = stores.publisher.load_executable_study(extension_issue.manifest.issue_id)
assert publication.manifest.base == base_publication.manifest
assert publication.publication_receipt == extension_issue.publication_receipt
export_record(local / "extension-manifest.json", publication.manifest)
export_record(local / "extension-receipt.json", publication.publication_receipt)
```

The returned `manifest.issue_id` is the next stage's operand. A preview is
`BLOCKED/WRITE_CONFIRMATION_REQUIRED` with zero external bytes, not a publication
receipt. Confirmed issue writes only custody records. No native task runs.

## 5. Freeze and review each issued layer

Freeze both `base_publication` and `publication` for scientific review:

```python
for layer, selected in (("base", base_publication), ("extension", publication)):
    frozen = freeze_for_review(stores, publication=selected, system=system,
        scope_id=policy.scope_ids[0], implementation_commit=closure.implementation_commit,
        action=AuthorityAction.SIMULATION_EXECUTION)
    export_record(local / f"{layer}-frozen.json", frozen)
```

Send each exact frozen record to the independent registered checkers using your
approved review process. Each checker inspects the candidate, source closure,
exclusions, conditions and relevant implementation or authority gate. It then
loads its private signer **in its own environment** and creates an attestation:

```python
# Executed by the actual checker after review, with its selected layer and signer.
from empirical_lawhood.planning.approval import FrozenIssuedStudyApprovalProposal
frozen = read_record(local / f"{layer}-frozen.json", FrozenIssuedStudyApprovalProposal)
registration = next(r for r in registry.registrations if r.gate_id == gate_id)
attestation = issue_approval_gate_attestation(
    registration=registration, signer=signer,
    attestation_id=f"{run_id}.{layer}.{gate_id}.attestation",
    authorization_id=f"{run_id}.{layer}.approval",
    subject=identity(frozen, frozen.frozen_proposal_id),
    result=AttestationResult(input("Recorded checker decision (PASSED or FAILED): ")),
    checked_at_utc=now(), issued_at_utc=now(),
    information_cutoff=frozen.proposal.decision_cutoff,
)
export_record(local / f"{layer}-{gate_id}-attestation.json", attestation)
```

`layer`, `gate_id` and `signer` select the actual review and its own registered
key. No helper assigns `PASSED`. A failed gate must remain a failed decision.
Do not edit the attestation or relabel outcomes to make the chain continue.
The operator receives only the signed records, verifies their exact IDs and
assembles the durable authorization:

```python
from empirical_lawhood.planning.approval import DurableAuthorizationRecord, FrozenIssuedStudyApprovalProposal
for layer in ("base", "extension"):
    frozen = read_record(local / f"{layer}-frozen.json", FrozenIssuedStudyApprovalProposal)
    attestations = tuple(read_record(local / f"{layer}-{gate}-attestation.json", ApprovalGateAttestation)
                         for gate in policy.required_gate_ids)
    approval = authorize_reviewed(stores, system=system, frozen=frozen,
        authorization_id=f"{run_id}.{layer}.approval", attestations=attestations,
        approver_id=policy.delegate_id)
    export_record(local / f"{layer}-approval.json", approval)
```

The approver must be the actual delegate selected by `policy.delegate_id`.
`authorize_reviewed` validates stored signatures, required gates, frozen subject,
cutoff and policy, and replays any retained same-ID authorization. It never
manufactures a checker decision. Before you request execution, examine its decision.

## 6. Separate execution grants and package assembly

For each layer, the issuer supplies a separate `EXPERIMENT_EXECUTION` record using `operation_record`. Its subject is the exact **issued manifest** identity. Its prerequisite is the corresponding durable approval identity. The issuer explicitly publishes the reviewed act.

Use a distinct execution authority ID and export to
`CONTROL/<layer>-execution.json`. This grants no actuation and no outcome reveal. Then assemble the base package and current experiment package:

```python
from empirical_lawhood.api.models import assemble_issued_study_package
base_frozen = read_record(local / "base-frozen.json", FrozenIssuedStudyApprovalProposal)
base_approval = read_record(local / "base-approval.json", DurableAuthorizationRecord)
base_execution = read_record(local / "base-execution.json", StudyOperationAuthority)
base_package = assemble_issued_study_package(
    issued_study=base_publication.manifest, publication_receipt=base_publication.publication_receipt,
    frozen_proposal=base_frozen, scientific_approval=base_approval, execution_authority=base_execution,
    run_plan_id=run_id, grantee_id=PROGRAMME_EXECUTION_GRANTEE_ID, at_utc=now(),
)
export_record(local / "base-package.json", base_package)
assembled = require_success(api.assemble_package(AssembleExperimentPackageRequest(
    local / "base-package.json", local / "extension-manifest.json", local / "extension-receipt.json",
    local / "extension-frozen.json", local / "extension-approval.json", local / "extension-execution.json",
    directory / "resources.json", run_id, PROGRAMME_EXECUTION_GRANTEE_ID, now(),
    emit_path=Path("experiment-package.json"),
)))
assert assembled.authority_replayed and not assembled.executed
package_path = local / "experiment-package.json"
compiled = require_success(api.compile_campaign(CompileCampaignRequest(package_path)))
assert sum(map(len, compiled.parallel_task_groups)) == 21
```

`emit_path` is relative to the **base package's local directory**. The API
refuses emission inside the scientific artifact root and refuses replacement.
The experiment package binds the returned issue/receipt, frozen approval, independent
execution act, exact resource envelope and run ID. Compilation produces the
current ProtocolRunPlan/ProtocolExecutionPlan identities and 21-task topology. It executes
zero tasks. Preserve the package and returned fingerprints.

## 7. Execute sealed work, then authorize reveal

```python
preview = api.run_campaign(RunCampaignRequest(package_path))
assert preview.reason_codes == ("WRITE_CONFIRMATION_REQUIRED",)
# Explicitly confirmed native execution under the reviewed execution grant:
sealed = api.run_campaign(RunCampaignRequest(package_path, confirmed=True))
print(sealed.status, sealed.reason_codes)
status = api.campaign_status(CampaignStatusRequest(run_id, include_attempt_history=True))
print(status.to_mapping())
```

Without reveal authority, acquisition can finish and the evaluator stops at its
declared authority barrier. Inspect the 20 acquisition receipts. Do not treat
that stop as a negative scientific verdict. Review the separately frozen reveal
proposal using `freeze_for_review(..., action=AuthorityAction.EVALUATOR_REVEAL)`
and repeat the checker/authorization handoff under distinct `reveal` IDs. The issuer supplies a separate `OUTCOME_REVEAL` grant with these fields:

- Subject: the exact executable-study manifest identity.
- Prerequisite: the exact executable-study **execution grant** identity.
- Grantee: `PROGRAMME_REVEAL_GRANTEE_ID`.
- Permission: only `allows_reveal` is true.
- Outcome access: `EVALUATOR_REVEAL`.

Persist this reviewed act.
Export its canonical record.

```python
reveal = read_record(local / "reveal.json", StudyOperationAuthority)
publish_reviewed_authority(stores, reveal)
finished = require_success(api.resume_campaign(ResumeCampaignRequest(
    run_id=run_id, confirmed=True, reveal_authority_id=reveal.authority_id,
)))
print(finished.operational_status, len(finished.receipt_ids),
      finished.adjudication_evaluability, finished.scientific_status,
      finished.admission_status, finished.scientific_reason_codes)
```

Expect 21 successful task receipts only if all native tasks and adjudication
completed. Read the separately reported scientific verdict: operational success
does not imply a supported contrast, admission, or a controller license.
Negative and unevaluable outcomes remain in the terminal record. The result's
adjudication ID/fingerprint/receipt identify the actual canonical output.
`campaign status --run ... --format json` exposes verified execution and
attempt history. Inspect the proof's output locators and immutable manifests/receipts through the selected store.
Retain the complete assigned-root denominator when interpreting the scientific result.

## 8. Resume the same identities

After an interruption, keep these original items:

- Package
- Source commit
- Issue IDs
- Trust registry
- Approvals
- Operation grants
- Run ID
- External store. Restore
the same environment and API composition, then preview and confirm
`ResumeCampaignRequest(run_id, ..., reveal_authority_id=...)`. Reuse the **original grant records and timestamps** loaded from the store. Do not reconstruct the same act ID with a new clock.

The
engine replays external receipts and rebuildable local catalog state. Completed
native effects must not run again. A source change, unknown native completion or resource stop can require the declared amendment path. That condition grants no permission for a new cohort or repeated native effects.

A negative
scientific result is not a retryable execution error.

The regression gate exercises receipt recovery after publication and projection failures.
It also tests reactor completion after the reveal barrier and loss of its disposable catalog. Never delete or edit real receipt/custody files
as a recovery technique. Keep `.empirical-lawhood/` out of source identity. It
contains the rebuildable SQLite projection, while scientific custody is external.

## Read results and attempts

Use [results and failures](results-and-failures.md#issued-status-and-recovery) for the full status command,
bounded attempt pagination and the distinction between execution, evaluability and scientific verdict.
The procedure below retains its exact authoring, trust, store and authority context.
An expected reveal barrier is different from a failed native task. Resume the same identities after the declared repair.
Never rerun a completed native effect or delete primary receipts to obtain another result.


## Other retained reactor port contexts

The assigned prefix above composes its native ports from its authoring directory. The seven older prepared routes in the [reactor quick start](../experiments/reactor-response/guide.md)
instead require explicit `ReactorPortContext` publications. This is a separate
exposed-development boundary. A context records its `context_id`, route, exact `port_key`, config digest, run ID and lineage parent.

It records execution authority, resource identity, issued program and output census. It also records any required allocation, approval/reveal, phase, stage, input manifests or continuation. Use actual
producer records for those fields. The [typed context](../src/empirical_lawhood/adapters/composition/reactor_prefix_response/port_store.py)
and [publication example](../tests/test_reactor_port_store.py) show the complete
relationships and immutable replay.

After independent authority publication, publish the reviewed context through `stores.plane.write(ArtifactWriteRequest(...))`.
Use these fields:

- `logical_artifact_id=context.context_id`.
- Path: `operator/reactor-ports/<context_id>.json`.
- `payload_schema=context.SCHEMA`.
- `profile=ArtifactProfile.CANONICAL_JSON`.
- `media_type="application/json"`.
- `publication_scope_relative_root="operator/reactor-ports"`.
- A unique scope ID.
- `payload=context.canonical_bytes()`.
- `visibility_ceiling=DEVELOPMENT_ONLY`.
- `outcome_access=OUTCOME_BLIND`.
- The declared parent visibility ceilings.

`ReactorExternalPortStore(stores.plane).resolve_port(identity(context,
context.context_id))` must replay the exact publication and authority before
the binding lists that identity. A loose JSON file is insufficient. This
publishes a port context. It does not create a missing source, parent experiment,
qualified law, or fresh roster for those older routes.

## References and research

The linked source modules, command metadata and existing checks own the implemented behavior described here.
The [scientific integrity guide](scientific-integrity.md) defines its separate evidence and authority boundaries.
The [program source register](../paper/SOURCES.md) identifies bounded historical results and unavailable primary records.
