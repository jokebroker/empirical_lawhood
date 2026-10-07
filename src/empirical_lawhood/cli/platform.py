"""Thin generic CLI over :mod:`empirical_lawhood.api`."""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, TypeVar

import typer
import yaml  # type: ignore[import-untyped]

from empirical_lawhood.adapters.composition.rc_ladder_response.fresh_authoring import (
    load_rc_study,
)
from empirical_lawhood.api.rc_authoring import author_fresh_rc
from empirical_lawhood.adapters.composition.reactor_prefix_response.fresh_authoring import (
    load_fresh_reactor_profile,
)
from empirical_lawhood.api.reactor_authoring import author_fresh_reactor
from empirical_lawhood.api.reactor_preissue import prove_fresh_reactor
from empirical_lawhood.adapters.physical.mast_archive.fresh_source import (
    load_fair_mast_selection,
)
from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.synthetic_material_method_packet import (
    load_synthetic_material_response_config,
)
from empirical_lawhood.api.synthetic_material_authoring import (
    author_synthetic_material_response,
)
from empirical_lawhood.adapters.simulators.uniform_electron_gas_response.analytic_fresh_authoring import (
    load_uniform_electron_gas_analytic_config,
)
from empirical_lawhood.api.uniform_electron_gas_authoring import author_uniform_electron_gas_analytic_reference
from empirical_lawhood.api import (
    MAX_DATASET_PAGE_RECORDS,
    AcquisitionState,
    AdvanceStudyBundleRequest,
    ApiResult,
    AssembleExperimentPackageRequest,
    BindElapsedBudgetRequest,
    CampaignStatusRequest,
    CapabilityKind,
    CapabilityListRequest,
    CapabilityShowRequest,
    CatalogQueryRequest,
    CatalogRebuildRequest,
    CloseStudyBundleRequest,
    CompileCampaignRequest,
    CompileStudyRequest,
    CompileCandidateRequest,
    CompileLinkedCampaignProfileRequest,
    CompileStudyBundleRequest,
    CompileSourceProfileRequest,
    CustodyState,
    DatabaseMutationRequest,
    DatasetBindingRole,
    DatasetCatalogRecordKind,
    DatasetListRequest,
    DatasetManifestRequest,
    DatasetOperationPreviewRequest,
    DatasetProjectionRebuildRequest,
    DatasetRegistrationRequest,
    DatasetShowRequest,
    DocumentRequest,
    EmpiricalLawhoodApi,
    ExternalIdentifier,
    ExternalIdentifierKind,
    IssueStudyRequest,
    IssueExtensionsRequest,
    IssueStudyBundleRequest,
    OperationStatus,
    CheckReadinessRequest,
    StudyBundleStatusRequest,
    ResumeCampaignRequest,
    RunCampaignRequest,
    create_cli_api,
    create_inspection_api,
)
from empirical_lawhood.api.codecs import load_authoring, load_registered_authoring
from empirical_lawhood.cli.attempts import (
    ATTEMPT_DIRECTORY_OPTION,
    ATTEMPT_OUTPUT_OPTION,
    AttemptGroup,
    attempt_input,
    attempt_stage,
    emit_development_report,
    retained_command,
)
from empirical_lawhood.cli.capability_output import (
    render_capability_list,
    render_capability_show,
)
from empirical_lawhood.cli.presentation import render_campaign_status, render_doctor
from empirical_lawhood.infrastructure.artifacts import GuardedExternalRoot
from empirical_lawhood.infrastructure.source_origin import (
    ExecutingSourceMismatch,
    require_executing_target_source,
)
from empirical_lawhood.kernel.time import InformationCutoff
from empirical_lawhood.runtime.operator_profile import (
    OperatorStorageProfile,
    resolve_external_root_contract,
)


class OutputFormat(StrEnum):
    TEXT = "text"
    JSON = "json"


class DatasetShowKind(StrEnum):
    FAMILY = "family"
    RELEASE = "release"


OUTPUT_OPTION = typer.Option(
    "--format",
    case_sensitive=False,
    help="Output format: text or deterministic JSON.",
)
SPEC_OPTION = typer.Option(
    "--spec",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Strict JSON/YAML authoring document.",
)
DATASET_MANIFEST_OPTION = typer.Option(
    "--manifest",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Dataset registration, transformation, or binding manifest.",
)
DATASET_POLICY_OPTION = typer.Option(
    "--policy",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Exact DatasetOperationPolicy JSON/YAML document.",
)
DATASET_REQUEST_OPTION = typer.Option(
    "--request",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Exact DatasetOperationRequest JSON/YAML document.",
)
DATASET_AUTHORIZATION_OPTION = typer.Option(
    "--authorization",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Optional detached-signed DatasetOperationAuthorization document.",
)
DATASET_REGISTRATION_AUTHORIZATION_OPTION = typer.Option(
    "--authorization",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Required detached-signed DatasetOperationAuthorization document.",
)
DATASET_REBUILD_MANIFEST_OPTION = typer.Option(
    "--manifest",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Exact DatasetProjectionRebuildManifest JSON/YAML document.",
)
DATASET_REBUILD_POLICY_OPTION = typer.Option(
    "--policy",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Exact DatasetProjectionRebuildPolicy JSON/YAML document.",
)
DATASET_REBUILD_REQUEST_OPTION = typer.Option(
    "--request",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Exact DatasetProjectionRebuildRequest JSON/YAML document.",
)
DATASET_REBUILD_AUTHORIZATION_OPTION = typer.Option(
    "--authorization",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Required detached-signed DatasetProjectionRebuildAuthorization document.",
)
PLAN_OPTION = typer.Option(
    "--plan",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="CampaignPackage JSON/YAML document to compile and execute.",
)
EXPLORATION_PLAN_OPTION = typer.Option(
    "--plan",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help=(
        "ExplorationExecutionPackage input-only JSON/YAML document, or the "
        "read-only DualLoopPackage fixture."
    ),
)
YES_OPTION = typer.Option(
    "--yes",
    help="Confirm the write effects documented by the command preview.",
)
PROPOSER_ATTESTATION_OPTION = typer.Option(
    "--proposer-attestation",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Exact accountable-human proposer attestation document.",
)
SOURCE_CLOSURE_OPTION = typer.Option(
    "--source-closure",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Exact implementation source-closure document.",
)
CUSTODY_AUTHORITY_OPTION = typer.Option(
    "--custody-authority",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Exact custody/publication authority document.",
)
EXPECTED_CANDIDATE_OPTION = typer.Option(
    "--expected-candidate",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help='Exact StandardProgrammeCandidate emitted by campaign compile-candidate.',
)
REVEAL_AUTHORITY_ID_OPTION = typer.Option(
    "--reveal-authority-id",
    help=(
        'If the issued plan contains evaluation or reveal tasks, supply this exact trusted-store reveal/evaluator authority ID.'
    ),
)
PARENT_RECORD_OPTION = typer.Option(
    "--parent-record",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help=(
        "Exact revealed parent adjudication used to derive and bind a frozen conditional plan."
    ),
)
PARENT_INPUT_BINDING_OPTION = typer.Option(
    "--parent-input-binding",
    exists=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help=(
        "Repeat for each exact FrozenParentInputBinding already bound to the issued target."
    ),
)


ApiFactory = Callable[[], EmpiricalLawhoodApi]
CliPayloadT = TypeVar("CliPayloadT")
EnumT = TypeVar("EnumT", bound=StrEnum)
API_FACTORY: ApiFactory = create_cli_api
CLI_PROJECT_ROOT: Path | None = None
CLI_OPERATOR_PROFILE: Path | None = None
CLI_DATASET_PROJECTION_TRUST: Path | None = None
CLI_APPROVAL_CHECKER_TRUST: Path | None = None
CLI_REACTOR_AUTHORING_DIR: Path | None = None
CLI_CIRCUIT_AUTHORING_DIR: Path | None = None
CLI_ELECTRON_GAS_AUTHORING_DIR: Path | None = None
CLI_SYNTHETIC_MATERIAL_AUTHORING_DIR: Path | None = None
CLI_AUTHORING_DIR: Path | None = None


def set_cli_inputs(
    *,
    project_root: Path | None,
    operator_profile: Path | None,
    dataset_projection_trust: Path | None,
    approval_checker_trust: Path | None,
    reactor_authoring_dir: Path | None = None,
    circuit_authoring_dir: Path | None = None,
    electron_gas_authoring_dir: Path | None = None,
    synthetic_material_authoring_dir: Path | None = None,
    authoring_dir: Path | None = None,
) -> None:
    global CLI_PROJECT_ROOT, CLI_OPERATOR_PROFILE, CLI_AUTHORING_DIR
    global CLI_DATASET_PROJECTION_TRUST, CLI_APPROVAL_CHECKER_TRUST
    global \
        CLI_REACTOR_AUTHORING_DIR, \
        CLI_CIRCUIT_AUTHORING_DIR, \
        CLI_ELECTRON_GAS_AUTHORING_DIR, \
        CLI_SYNTHETIC_MATERIAL_AUTHORING_DIR
    CLI_PROJECT_ROOT = project_root
    CLI_OPERATOR_PROFILE = operator_profile
    CLI_DATASET_PROJECTION_TRUST = dataset_projection_trust
    CLI_APPROVAL_CHECKER_TRUST = approval_checker_trust
    CLI_REACTOR_AUTHORING_DIR = reactor_authoring_dir
    CLI_CIRCUIT_AUTHORING_DIR = circuit_authoring_dir
    CLI_ELECTRON_GAS_AUTHORING_DIR = electron_gas_authoring_dir
    CLI_SYNTHETIC_MATERIAL_AUTHORING_DIR = synthetic_material_authoring_dir
    CLI_AUTHORING_DIR = authoring_dir


system_app = typer.Typer(
    no_args_is_help=True,
    help="SystemSpec validation and catalog inspection.",
)
campaign_app = typer.Typer(
    cls=AttemptGroup,
    no_args_is_help=True,
    help="generic campaign DAG execution and recovery.",
)
capability_app = typer.Typer(
    no_args_is_help=True,
    help="static capability registration discovery.",
)
catalog_app = typer.Typer(
    no_args_is_help=True,
    help="bounded metadata query and projection rebuild.",
)
dataset_app = typer.Typer(
    no_args_is_help=True,
    help=(
        "dataset validation, authorized registration/rebuild and catalog inspection."
    ),
)
db_app = typer.Typer(
    no_args_is_help=True,
    help="ignored local SQLite projection lifecycle.",
)
authority_app = typer.Typer(
    no_args_is_help=True,
    help="outcome-blind authority metadata inspection.",
)
law_app = typer.Typer(no_args_is_help=True, help="response-law metadata.")
atlas_app = typer.Typer(no_args_is_help=True, help="response-atlas metadata.")
admission_app = typer.Typer(no_args_is_help=True, help="admission metadata.")
control_app = typer.Typer(
    no_args_is_help=True,
    help="controller metadata and reference conformance.",
)
source_app = typer.Typer(
    no_args_is_help=True,
    help="Explicit public-source selection, external acquisition and same-attempt recovery.",
)


@source_app.command("preview")
def source_preview(
    source_config: Annotated[
        Path,
        typer.Option("--source-config", exists=True, dir_okay=False, readable=True),
    ],
) -> None:
    """Decode a declared source and show its finite URLs and bounds without contact."""

    try:
        selection = load_fair_mast_selection(source_config)
        typer.echo(
            json.dumps(
                _api(storage_required=False).preview_fair_mast_source(selection),
                sort_keys=True,
                indent=2,
            )
        )
    except (OSError, TypeError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="--source-config") from error


def _source_effect(
    source_config: Path,
    acquisition_authority_id: str,
    custody_authority_id: str,
    *,
    recover: bool,
    yes: bool,
) -> None:
    if not yes:
        typer.echo(
            "source acquisition requires --yes after previewing its external effects",
            err=True,
        )
        raise typer.Exit(code=3)
    try:
        selection = load_fair_mast_selection(source_config)
        api = _api()
        receipt = (
            api.recover_fair_mast_source(
                selection,
                acquisition_authority_id=acquisition_authority_id,
                custody_authority_id=custody_authority_id,
            )
            if recover
            else api.acquire_fair_mast_source(
                selection,
                acquisition_authority_id=acquisition_authority_id,
                custody_authority_id=custody_authority_id,
            )
        )
    except KeyError as error:
        typer.echo(
            f"source authority is absent from the external trusted store: {error}",
            err=True,
        )
        raise typer.Exit(code=4) from error
    except PermissionError as error:
        typer.echo(f"source authority or readiness refused: {error}", err=True)
        raise typer.Exit(code=4) from error
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        typer.echo(f"source custody refused: {error}", err=True)
        raise typer.Exit(code=5) from error
    typer.echo(receipt.canonical_bytes().decode("utf-8"), nl=False)


@source_app.command("acquire")
def source_acquire(
    source_config: Annotated[
        Path,
        typer.Option("--source-config", exists=True, dir_okay=False, readable=True),
    ],
    acquisition_authority_id: Annotated[
        str, typer.Option("--acquisition-authority-id")
    ],
    custody_authority_id: Annotated[str, typer.Option("--custody-authority-id")],
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    """Acquire exactly one declared shot, array metadata and first Zarr chunk."""

    _source_effect(
        source_config,
        acquisition_authority_id,
        custody_authority_id,
        recover=False,
        yes=yes,
    )


@source_app.command("recover")
def source_recover(
    source_config: Annotated[
        Path,
        typer.Option("--source-config", exists=True, dir_okay=False, readable=True),
    ],
    acquisition_authority_id: Annotated[
        str, typer.Option("--acquisition-authority-id")
    ],
    custody_authority_id: Annotated[str, typer.Option("--custody-authority-id")],
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    """Verify a completed receipt or finish the same durable attempt."""

    _source_effect(
        source_config,
        acquisition_authority_id,
        custody_authority_id,
        recover=True,
        yes=yes,
    )


@source_app.command("design-input")
def source_design_input(
    source_config: Annotated[
        Path,
        typer.Option("--source-config", exists=True, dir_okay=False, readable=True),
    ],
    cutoff: Annotated[
        Path, typer.Option("--cutoff", exists=True, dir_okay=False, readable=True)
    ],
    input_id: Annotated[str, typer.Option("--input-id")],
    operator_id: Annotated[str, typer.Option("--operator-id")],
    authoring_at_utc: Annotated[str, typer.Option("--authoring-at-utc")],
) -> None:
    """Verify external custody and bind its receipt before declared authoring time."""

    try:
        selection = load_fair_mast_selection(source_config)
        information_cutoff = load_registered_authoring(
            cutoff, root_schemas={InformationCutoff.SCHEMA: InformationCutoff}
        )
        if not isinstance(information_cutoff, InformationCutoff):
            raise TypeError("source design input requires an InformationCutoff")
        record = _api().bind_fair_mast_design_input(
            selection,
            input_id=input_id,
            information_cutoff=information_cutoff,
            operator_id=operator_id,
            authoring_at_utc=authoring_at_utc,
        )
    except (OSError, PermissionError, TypeError, ValueError) as error:
        typer.echo(f"source design input refused: {error}", err=True)
        raise typer.Exit(code=5) from error
    typer.echo(record.canonical_bytes().decode("utf-8"), nl=False)


def _api(*, storage_required: bool = True) -> EmpiricalLawhoodApi:
    root = CLI_PROJECT_ROOT
    profile_path = CLI_OPERATOR_PROFILE
    if API_FACTORY is not create_cli_api:
        return API_FACTORY()
    if not storage_required:
        try:
            profile = (
                None
                if profile_path is None
                else load_registered_authoring(
                    profile_path,
                    root_schemas={
                        OperatorStorageProfile.SCHEMA: OperatorStorageProfile
                    },
                )
            )
            return create_inspection_api(
                repo_root=root, operator_storage_profile=profile
            )
        except (OSError, PermissionError, RuntimeError, TypeError, ValueError) as error:
            raise typer.BadParameter(
                str(error), param_hint="--operator-profile"
            ) from error
    if root is None:
        raise typer.BadParameter(
            "this command requires --project-root", param_hint="--project-root"
        )
    try:
        require_executing_target_source(root)
    except ExecutingSourceMismatch as error:
        raise typer.BadParameter(str(error), param_hint="--project-root") from error
    if profile_path is None:
        raise typer.BadParameter(
            "this command requires --operator-profile", param_hint="--operator-profile"
        )
    try:
        profile = load_registered_authoring(
            profile_path,
            root_schemas={OperatorStorageProfile.SCHEMA: OperatorStorageProfile},
        )
        if not isinstance(profile, OperatorStorageProfile):
            raise TypeError("document is not an OperatorStorageProfile")
        return create_cli_api(
            repo_root=root,
            operator_storage_profile=profile,
            dataset_projection_trust_path=CLI_DATASET_PROJECTION_TRUST,
            approval_checker_trust_path=CLI_APPROVAL_CHECKER_TRUST,
            reactor_authoring_dir=CLI_REACTOR_AUTHORING_DIR,
            circuit_authoring_dir=CLI_CIRCUIT_AUTHORING_DIR,
            electron_gas_authoring_dir=CLI_ELECTRON_GAS_AUTHORING_DIR,
            synthetic_material_authoring_dir=CLI_SYNTHETIC_MATERIAL_AUTHORING_DIR,
            authoring_dir=CLI_AUTHORING_DIR,
            maximum_parallel_tasks=profile.maximum_parallel_tasks,
        )
    except ExecutingSourceMismatch as error:
        raise typer.BadParameter(str(error), param_hint="--project-root") from error
    except (OSError, PermissionError, RuntimeError, TypeError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="--operator-profile") from error


def _reactor_port_store():
    """Select the same separately configured operator trust root as other stores."""
    store = _response_parent_store()
    if store is None:
        return None
    from empirical_lawhood.adapters.composition.reactor_prefix_response.port_store import (
        ReactorExternalPortStore,
    )

    return ReactorExternalPortStore(store.plane)


def _response_parent_store():
    """Compose the matrix-response reader from the selected operator trust root."""
    if CLI_OPERATOR_PROFILE is None:
        return None
    if CLI_PROJECT_ROOT is None:
        raise ValueError("RESPONSE_PARENT_PROJECT_ROOT_REQUIRED")
    from empirical_lawhood.adapters.composition.response_parent_store import (
        ResponseExternalParentAuthorityStore,
    )
    from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane

    profile = load_registered_authoring(
        CLI_OPERATOR_PROFILE,
        root_schemas={OperatorStorageProfile.SCHEMA: OperatorStorageProfile},
    )
    contract = resolve_external_root_contract(
        profile,
        repo_root=CLI_PROJECT_ROOT,
        home_root=Path.home(),
    )
    guard = GuardedExternalRoot(contract)
    guard.verify(for_write=False)
    return ResponseExternalParentAuthorityStore(ExternalArtifactPlane(guard))


def _exit_code(result: ApiResult[CliPayloadT]) -> int:
    if result.status is OperationStatus.SUCCEEDED:
        return 0
    if result.status is OperationStatus.INVALID:
        return 2
    if result.status is OperationStatus.NOT_FOUND:
        return 7
    if result.status is OperationStatus.CONFLICT:
        return 8
    categories = {error.category.value for error in result.errors}
    reasons = set(result.reason_codes)
    if result.status is OperationStatus.BLOCKED:
        if "AUTHORITY" in categories or result.operation.endswith(
            "authorize-nonactuating"
        ):
            return 4
        if reasons.intersection(
            {
                "CAMPAIGN_EXECUTION_FAILED",
                "RUN_EXECUTION_FAILED",
                "RUN_RECOVERY_FAILED",
            }
        ):
            return 6
        if categories.intersection({"IDENTITY", "STORAGE"}):
            return 5
        return 3
    if categories.intersection({"IDENTITY", "STORAGE"}):
        return 5
    return 6


def _emit(
    result: ApiResult[CliPayloadT],
    output_format: OutputFormat,
    *,
    text_renderer: Callable[[CliPayloadT], str] | None = None,
) -> None:
    payload = result.to_mapping()
    if output_format is OutputFormat.JSON:
        rendered = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    elif text_renderer is not None and result.succeeded and result.payload is not None:
        rendered = text_renderer(result.payload)
    else:
        rendered = yaml.safe_dump(payload, sort_keys=False).rstrip()
    typer.echo(rendered, err=not result.succeeded)
    exit_code = _exit_code(result)
    if exit_code:
        raise typer.Exit(exit_code)


def _sorted_unique(values: list[str] | None) -> tuple[str, ...]:
    return tuple(sorted(set(values or ())))


def _sorted_unique_enums(values: list[EnumT] | None) -> tuple[EnumT, ...]:
    return tuple(sorted(set(values or ()), key=lambda value: value.value))


def _external_identifiers(values: list[str] | None) -> tuple[ExternalIdentifier, ...]:
    parsed: dict[tuple[str, str, str], ExternalIdentifier] = {}
    allowed_kinds = ", ".join(value.value for value in ExternalIdentifierKind)
    for raw_value in values or ():
        parts = raw_value.split(":", 2)
        if len(parts) != 3 or any(not part for part in parts):
            raise typer.BadParameter(
                "expected KIND:NAMESPACE:VALUE with three nonempty fields",
                param_hint="--external-id",
            )
        kind_value, namespace, value = parts
        try:
            identifier = ExternalIdentifier(
                kind=ExternalIdentifierKind(kind_value),
                namespace=namespace,
                value=value,
            )
        except ValueError as error:
            raise typer.BadParameter(
                f"invalid KIND:NAMESPACE:VALUE ({allowed_kinds})",
                param_hint="--external-id",
            ) from error
        parsed[(identifier.kind.value, identifier.namespace, identifier.value)] = (
            identifier
        )
    return tuple(parsed[key] for key in sorted(parsed))


def doctor_command(
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
    route: Annotated[
        str | None,
        typer.Option(
            "--route",
            help="Read installed versions for reactor, prepared-response, open-simulators, reaction-response or grid2op.",
        ),
    ] = None,
) -> None:
    """Read-only environment, Git, catalog and external-storage diagnosis."""

    _emit(
        _api(storage_required=False).doctor(route=route), output_format,
        text_renderer=lambda summary: render_doctor(summary, project_root=CLI_PROJECT_ROOT),
    )


@capability_app.command("list")
def capability_list(
    kind: Annotated[
        CapabilityKind | None,
        typer.Option(
            "--kind",
            case_sensitive=False,
            help="Constrain registrations to one capability kind.",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=100),
    ] = 100,
    cursor: Annotated[
        str | None,
        typer.Option("--cursor", help="Opaque cursor returned by the preceding page."),
    ] = None,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """List bounded static registrations without loading or executing providers."""

    _emit(
        _api(storage_required=False).list_capabilities(
            CapabilityListRequest(limit=limit, cursor=cursor, kind=kind)
        ),
        output_format,
        text_renderer=render_capability_list,
    )


@capability_app.command("show")
def capability_show(
    capability_key: Annotated[str, typer.Argument(help="Exact capability key.")],
    capability_version: Annotated[
        str,
        typer.Option("--version", help="Exact semantic capability version."),
    ],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Show one exact static registration and its manifest."""

    _emit(
        _api(storage_required=False).show_capability(
            CapabilityShowRequest(
                capability_key=capability_key,
                capability_version=capability_version,
            )
        ),
        output_format,
        text_renderer=render_capability_show,
    )


@system_app.command("validate")
def system_validate(
    spec: Annotated[Path, SPEC_OPTION],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Strictly decode and validate one SystemSpec without creating state."""

    _emit(
        _api(storage_required=False).validate_system_document(DocumentRequest(spec)),
        output_format,
    )


@system_app.command("inspect")
def system_inspect(
    system_id: Annotated[str, typer.Option("--id", help="Catalog SystemSpec ID.")],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Inspect bounded catalog metadata and its external artifact pointer."""

    _emit(_api().inspect_system_id(system_id), output_format)


@dataset_app.command("validate")
def dataset_validate(
    manifest: Annotated[Path, DATASET_MANIFEST_OPTION],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Validate one dataset registration, transformation, or binding manifest."""

    request = DatasetManifestRequest(manifest)
    _emit(
        _api(storage_required=False).validate_dataset_manifest(request), output_format
    )


@dataset_app.command("preview")
def dataset_preview(
    manifest: Annotated[Path, DATASET_MANIFEST_OPTION],
    policy: Annotated[Path, DATASET_POLICY_OPTION],
    request: Annotated[Path, DATASET_REQUEST_OPTION],
    authorization_id: Annotated[
        str,
        typer.Option(
            "--authorization-id",
            help="Stable ID to use for the unsigned preview or exact signed authority.",
        ),
    ],
    authorization: Annotated[Path | None, DATASET_AUTHORIZATION_OPTION] = None,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    'Preview exact identity, storage, and durable authority gates. This command does no dataset work.'

    _emit(
        _api().preview_dataset_operation(
            DatasetOperationPreviewRequest(
                manifest_path=manifest,
                policy_path=policy,
                request_path=request,
                authorization_id=authorization_id,
                authorization_path=authorization,
            )
        ),
        output_format,
    )


@dataset_app.command("rebuild")
def dataset_rebuild(
    manifest: Annotated[Path, DATASET_REBUILD_MANIFEST_OPTION],
    policy: Annotated[Path, DATASET_REBUILD_POLICY_OPTION],
    request: Annotated[Path, DATASET_REBUILD_REQUEST_OPTION],
    authorization: Annotated[Path, DATASET_REBUILD_AUTHORIZATION_OPTION],
    bundle_id: Annotated[
        str,
        typer.Option("--bundle-id", help="Exact durable rebuild-authority bundle ID."),
    ],
    receipt_id: Annotated[
        str,
        typer.Option("--receipt-id", help="Stable external projection receipt ID."),
    ],
    confirmed: Annotated[bool, YES_OPTION] = False,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Install an exact unified projection after dedicated authority replay."""

    _emit(
        _api().rebuild_dataset_projection(
            DatasetProjectionRebuildRequest(
                manifest_path=manifest,
                policy_path=policy,
                request_path=request,
                authorization_path=authorization,
                bundle_id=bundle_id,
                receipt_id=receipt_id,
                confirmed=confirmed,
            )
        ),
        output_format,
    )


@dataset_app.command("register")
def dataset_register(
    manifest: Annotated[Path, DATASET_MANIFEST_OPTION],
    policy: Annotated[Path, DATASET_POLICY_OPTION],
    request: Annotated[Path, DATASET_REQUEST_OPTION],
    authorization: Annotated[Path, DATASET_REGISTRATION_AUTHORIZATION_OPTION],
    bundle_id: Annotated[
        str,
        typer.Option(
            "--bundle-id", help="Exact durable registration-authority bundle ID."
        ),
    ],
    receipt_id: Annotated[
        str,
        typer.Option("--receipt-id", help="Stable external registration receipt ID."),
    ],
    confirmed: Annotated[bool, YES_OPTION] = False,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Verify held bytes and publish their exact registration control batch."""

    _emit(
        _api().register_dataset(
            DatasetRegistrationRequest(
                manifest_path=manifest,
                policy_path=policy,
                request_path=request,
                authorization_path=authorization,
                bundle_id=bundle_id,
                receipt_id=receipt_id,
                confirmed=confirmed,
            )
        ),
        output_format,
    )


@dataset_app.command("list")
def dataset_list(
    family_ids: Annotated[
        list[str] | None,
        typer.Option("--family", help='Dataset family ID. Repeatable.'),
    ] = None,
    release_ids: Annotated[
        list[str] | None,
        typer.Option("--release", help='Dataset release ID. Repeatable.'),
    ] = None,
    provider_ids: Annotated[
        list[str] | None,
        typer.Option("--provider", help='Provider ID. Repeatable.'),
    ] = None,
    external_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--external-id",
            help='External identifier as KIND:NAMESPACE:VALUE. Repeatable.',
        ),
    ] = None,
    custody_states: Annotated[
        list[CustodyState] | None,
        typer.Option(
            "--custody", case_sensitive=False, help='Custody state. Repeatable.'
        ),
    ] = None,
    acquisition_states: Annotated[
        list[AcquisitionState] | None,
        typer.Option(
            "--acquisition",
            case_sensitive=False,
            help='Acquisition state. Repeatable.',
        ),
    ] = None,
    experiment_spec_ids: Annotated[
        list[str] | None,
        typer.Option("--experiment", help='Experiment specification ID. Repeatable.'),
    ] = None,
    roles: Annotated[
        list[DatasetBindingRole] | None,
        typer.Option(
            "--role", case_sensitive=False, help='Dataset binding role. Repeatable.'
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=MAX_DATASET_PAGE_RECORDS),
    ] = 100,
    cursor: Annotated[
        str | None,
        typer.Option("--cursor", help="Opaque cursor returned by the preceding page."),
    ] = None,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """List bounded, authenticated dataset catalog metadata."""

    request = DatasetListRequest(
        limit=limit,
        cursor=cursor,
        family_ids=_sorted_unique(family_ids),
        release_ids=_sorted_unique(release_ids),
        provider_ids=_sorted_unique(provider_ids),
        external_identifiers=_external_identifiers(external_ids),
        custody_states=_sorted_unique_enums(custody_states),
        acquisition_states=_sorted_unique_enums(acquisition_states),
        experiment_spec_ids=_sorted_unique(experiment_spec_ids),
        roles=_sorted_unique_enums(roles),
    )
    _emit(_api().list_datasets(request), output_format)


@dataset_app.command("show")
def dataset_show(
    dataset_id: Annotated[str, typer.Argument(help="Dataset family or release ID.")],
    kind: Annotated[
        DatasetShowKind | None,
        typer.Option(
            "--kind",
            case_sensitive=False,
            help="Constrain lookup to family or release.",
        ),
    ] = None,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Show one authenticated dataset family or release record."""

    record_kind = (
        None
        if kind is None
        else {
            DatasetShowKind.FAMILY: DatasetCatalogRecordKind.FAMILY,
            DatasetShowKind.RELEASE: DatasetCatalogRecordKind.RELEASE,
        }[kind]
    )
    request = DatasetShowRequest(dataset_id, kind=record_kind)
    _emit(_api().show_dataset(request), output_format)


@campaign_app.command("validate")
def campaign_validate(
    spec: Annotated[Path, SPEC_OPTION],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    'Validate a strict draft or specification. Package roots also replay authorization.'

    from empirical_lawhood.api.models import (
        CampaignPackage,
        IssuedCampaignPackage,
        IssuedStudyPackage,
        EnvelopeExperimentPackage,
        ExperimentPackage,
    )

    try:
        root = load_authoring(spec)
    except (OSError, TypeError, ValueError):
        root = None
    needs_authority = isinstance(
        root,
        (
            CampaignPackage,
            IssuedCampaignPackage,
            IssuedStudyPackage,
            EnvelopeExperimentPackage,
            ExperimentPackage,
        ),
    )
    _emit(
        _api(storage_required=needs_authority).validate_campaign_document(
            DocumentRequest(spec)
        ),
        output_format,
    )


@campaign_app.command("source-profile-compile")
def campaign_source_profile_compile(
    profile: Annotated[
        Path,
        typer.Option(
            "--profile",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Exact SourcePipelineProfile document.",
        ),
    ],
    transformation_manifest: Annotated[
        Path,
        typer.Option(
            "--transformation-manifest",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Exact predeclared DatasetTransformationManifest document.",
        ),
    ],
    dataset_request: Annotated[
        Path | None,
        typer.Option(
            "--dataset-request",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Optional already-issued DatasetOperationRequest document.",
        ),
    ] = None,
    emit: Annotated[
        Path | None,
        typer.Option(
            "--emit",
            dir_okay=False,
            help="Optional no-replace .json path relative to the profile directory.",
        ),
    ] = None,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Compile source readiness without reading a source or granting authority."""

    _emit(
        _api().compile_source_profile(
            CompileSourceProfileRequest(
                source_profile_path=profile,
                transformation_manifest_path=transformation_manifest,
                dataset_operation_request_path=dataset_request,
                emit_path=emit,
            )
        ),
        output_format,
    )


@campaign_app.command("linked-profile-compile")
def campaign_linked_profile_compile(
    profile: Annotated[
        Path,
        typer.Option(
            "--profile",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Exact LinkedCampaignProfile document.",
        ),
    ],
    source_compilation: Annotated[
        Path,
        typer.Option(
            "--source-compilation",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Exact SourcePipelineCompilation document.",
        ),
    ],
    evidence_profile: Annotated[
        Path,
        typer.Option(
            "--evidence-profile",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Exact EvidenceProfileSelection document.",
        ),
    ],
    candidate_reports: Annotated[
        list[Path],
        typer.Option(
            "--candidate-report",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help='Candidate compilation report. Repeat exactly four times.',
        ),
    ],
    emit: Annotated[
        Path | None,
        typer.Option(
            "--emit",
            dir_okay=False,
            help="Optional no-replace .json path relative to the profile directory.",
        ),
    ] = None,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Join four compiled package roles without issue, execution or reveal."""

    _emit(
        _api().compile_linked_campaign_profile(
            CompileLinkedCampaignProfileRequest(
                linked_profile_path=profile,
                source_compilation_path=source_compilation,
                evidence_profile_selection_path=evidence_profile,
                candidate_report_paths=tuple(candidate_reports),
                emit_path=emit,
            )
        ),
        output_format,
    )


@campaign_app.command("assemble-package")
def campaign_assemble_package(
    base_package: Annotated[
        Path,
        typer.Option("--base-package", exists=True, dir_okay=False, readable=True),
    ],
    issued_study: Annotated[
        Path,
        typer.Option("--issued-study", exists=True, dir_okay=False, readable=True),
    ],
    publication_receipt: Annotated[
        Path,
        typer.Option(
            "--publication-receipt", exists=True, dir_okay=False, readable=True
        ),
    ],
    frozen_proposal: Annotated[
        Path,
        typer.Option("--frozen-proposal", exists=True, dir_okay=False, readable=True),
    ],
    resource_envelope: Annotated[
        Path,
        typer.Option("--resource-envelope", exists=True, dir_okay=False, readable=True),
    ],
    run_plan_id: Annotated[str, typer.Option("--run-plan-id")],
    grantee_id: Annotated[str, typer.Option("--grantee-id")],
    at_utc: Annotated[str, typer.Option("--at-utc")],
    jit_census: Annotated[
        Path | None,
        typer.Option(
            "--jit-census",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Compilation census, required with --jit-manifest for JIT tasks.",
        ),
    ] = None,
    jit_manifest: Annotated[
        Path | None,
        typer.Option(
            "--jit-manifest",
            exists=True,
            dir_okay=False,
            readable=True,
            help='Compilation manifest. For uncompiled tasks, omit both JIT inputs.',
        ),
    ] = None,
    scientific_approval: Annotated[
        Path | None,
        typer.Option(
            "--scientific-approval", exists=True, dir_okay=False, readable=True
        ),
    ] = None,
    execution_authority: Annotated[
        Path | None,
        typer.Option(
            "--execution-authority", exists=True, dir_okay=False, readable=True
        ),
    ] = None,
    emit: Annotated[
        Path | None,
        typer.Option(
            "--emit",
            dir_okay=False,
            help="Optional no-replace .json path relative to the base package directory.",
        ),
    ] = None,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Assemble the current package from issued records and applicable JIT evidence."""

    _emit(
        _api().assemble_package(
            AssembleExperimentPackageRequest(
                base_package_path=base_package,
                issued_study_path=issued_study,
                publication_receipt_path=publication_receipt,
                frozen_proposal_path=frozen_proposal,
                scientific_approval_path=scientific_approval,
                execution_authority_path=execution_authority,
                execution_resource_envelope_spec_path=resource_envelope,
                predevelopment_jit_signature_census_path=jit_census,
                jit_graph_signature_manifest_path=jit_manifest,
                run_plan_id=run_plan_id,
                grantee_id=grantee_id,
                at_utc=at_utc,
                emit_path=emit,
            )
        ),
        output_format,
    )


@campaign_app.command("bind-elapsed-budget")
def campaign_bind_elapsed_budget(
    execution_package: Annotated[
        Path,
        typer.Option("--execution-package", exists=True, dir_okay=False, readable=True),
    ],
    campaign_elapsed_budget: Annotated[
        Path,
        typer.Option(
            "--campaign-elapsed-budget", exists=True, dir_okay=False, readable=True
        ),
    ],
    campaign_elapsed_reservations: Annotated[
        Path,
        typer.Option(
            "--campaign-elapsed-reservations",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    emit: Annotated[
        Path | None,
        typer.Option(
            "--emit",
            dir_okay=False,
            help="Optional fresh .json path relative to the execution package directory.",
        ),
    ] = None,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Bind an immutable cumulative elapsed-time contract to an authorized execution package."""

    _emit(
        _api().bind_elapsed_budget(
            BindElapsedBudgetRequest(
                execution_package_path=execution_package,
                campaign_elapsed_budget_path=campaign_elapsed_budget,
                campaign_elapsed_reservation_plan_path=campaign_elapsed_reservations,
                emit_path=emit,
            )
        ),
        output_format,
    )


@campaign_app.command("rc-ladder-native-check")
@retained_command("campaign rc-ladder-native-check", closed_output=False, native=True)
def campaign_rc_ladder_native_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/rc-ladder-response/guide.md for the accepted schema and constraints.')
    ],
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Check a strict local RC ladder against two independent numerical solvers."""

    from empirical_lawhood.adapters.simulators.rc_ladder_response.native_quickstart import (
        check_native_model,
    )
    from empirical_lawhood.infrastructure.bounded_io import BoundedFileIOError

    try:
        config = attempt_input(config, maximum_bytes=65536, copy=True, role="config")
        attempt_stage("validation_and_calculation", native_contact="unknown")
        report = check_native_model(config)
    except (BoundedFileIOError, OSError, TypeError, ValueError) as error:
        typer.echo(f"RC ladder native check refused: {error}", err=True)
        raise typer.Exit(code=2) from error
    emit_development_report(report)


@campaign_app.command("electron-gas-reference-check")
@retained_command("campaign electron-gas-reference-check", closed_output=False, native=False)
def campaign_electron_gas_reference_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/electron-gas-response/guide.md for the accepted schema and constraints.')
    ],
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Check the bounded uniform electron gas transverse-response reference without effects."""

    from empirical_lawhood.adapters.simulators.uniform_electron_gas_response.native_quickstart import (
        UniformElectronGasAnalyticReferenceConfig,
        run_analytic_development_check,
    )
    from empirical_lawhood.infrastructure.bounded_io import (
        BoundedFileIOError,
        read_bounded_bytes,
    )
    from empirical_lawhood.kernel.decoding import decode_canonical_bytes

    try:
        config = attempt_input(config, maximum_bytes=32768, copy=True, role="config")
        model = decode_canonical_bytes(
            read_bounded_bytes(config, maximum_bytes=32 * 1024),
            UniformElectronGasAnalyticReferenceConfig,
            maximum_bytes=32 * 1024,
        )
        attempt_stage("prerequisite_inspection", native_contact="none")
        report = run_analytic_development_check(model)
    except (BoundedFileIOError, OSError, TypeError, ValueError) as error:
        typer.echo(f"uniform electron gas native check refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("brian2-native-check")
@retained_command("campaign brian2-native-check", closed_output=False, native=True)
def campaign_brian2_native_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/neuron-current-response/guide.md for the accepted schema and constraints.')
    ],
    native_python: Annotated[
        Path, typer.Option("--native-python", exists=True, dir_okay=False)
    ],
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Run the bounded Brian2 LIF contract in its separately pinned native env."""

    import subprocess

    from empirical_lawhood.adapters.simulators.brian2_neuron_current_response.contracts import (
        Brian2LIFNativeConfig,
    )
    from empirical_lawhood.adapters.simulators.brian2_neuron_current_response.native_quickstart import (
        run_native_development_check,
    )
    from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
    from empirical_lawhood.kernel.decoding import decode_canonical_bytes

    try:
        config = attempt_input(config, maximum_bytes=16384, copy=True, role="config")
        model = decode_canonical_bytes(
            read_bounded_bytes(config, maximum_bytes=16 * 1024),
            Brian2LIFNativeConfig,
            maximum_bytes=16 * 1024,
        )
        attempt_stage("native_calculation", native_contact="unknown")
        report = run_native_development_check(model, native_python=native_python)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        subprocess.TimeoutExpired,
    ) as error:
        typer.echo(f"Brian2 native check refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("reaction-response-native-check")
@retained_command("campaign reaction-response-native-check", closed_output=False, native=True)
def campaign_reaction_response_native_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/reactor-flow-response/guide.md and experiments/reaction-diffusion-response/guide.md for the accepted schema and constraints.')
    ],
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Run one strict Cantera or FiPy development unit and report its native panel."""

    from empirical_lawhood.adapters.simulators.reaction_diffusion_development_input import (
        ReactionDiffusionDevelopmentInput,
        run_native_development_check,
    )

    try:
        config = attempt_input(config, maximum_bytes=32768, copy=True, role="config")
        request = load_registered_authoring(
            config,
            root_schemas={
                ReactionDiffusionDevelopmentInput.SCHEMA: ReactionDiffusionDevelopmentInput
            },
            maximum_bytes=32 * 1024,
        )
        attempt_stage("native_calculation", native_contact="unknown")
        report = run_native_development_check(request)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"Reaction response native check refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("grid2op-native-check")
@retained_command("campaign grid2op-native-check", closed_output=False, native=True)
def campaign_grid2op_native_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/grid-response-inputs/guide.md for the accepted schema and constraints.')
    ],
    source_root: Annotated[Path, typer.Option("--source-root")],
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Check one held Grid2Op chronic with reset hold/disconnect branches."""

    from importlib.metadata import PackageNotFoundError
    from zipfile import BadZipFile

    from empirical_lawhood.adapters.simulators.grid2op_response.native_quickstart import (
        Grid2OpDevelopmentInput,
        run_native_development_check,
    )

    try:
        config = attempt_input(config, maximum_bytes=16384, copy=True, role="config")
        request = load_registered_authoring(
            config,
            root_schemas={Grid2OpDevelopmentInput.SCHEMA: Grid2OpDevelopmentInput},
            maximum_bytes=16 * 1024,
        )
        attempt_stage("native_calculation", native_contact="unknown")
        report = run_native_development_check(request, source_root=source_root)
    except (
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        PackageNotFoundError,
        BadZipFile,
    ) as error:
        typer.echo(f"Grid2Op native check refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("gym-torax-native-check")
@retained_command("campaign gym-torax-native-check", closed_output=False, native=True)
def campaign_gym_torax_native_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/tokamak-control-response/guide.md for the accepted schema and constraints.')
    ],
    source_checkout: Annotated[Path, typer.Option("--source-checkout")],
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Run paired bounded Gym--TORAX native development episodes."""

    from empirical_lawhood.adapters.simulators.gym_torax_native.native_quickstart import (
        GymToraxNativeQuickstart,
        run_native_development_check,
    )

    try:
        config = attempt_input(config, maximum_bytes=16384, copy=True, role="config")
        request = load_registered_authoring(
            config,
            root_schemas={GymToraxNativeQuickstart.SCHEMA: GymToraxNativeQuickstart},
            maximum_bytes=16 * 1024,
        )
        attempt_stage("native_calculation", native_contact="unknown")
        report = run_native_development_check(request, repository_root=source_checkout)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"Gym--TORAX native check refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("torax-native-check")
@retained_command("campaign torax-native-check", closed_output=False, native=True)
def campaign_torax_native_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/tokamak-heat-response/guide.md for the accepted schema and constraints.')
    ],
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Run two bounded direct-TORAX heat actions on one preparation."""

    from importlib.metadata import PackageNotFoundError

    from empirical_lawhood.adapters.simulators.torax_native.native_quickstart import (
        NativeToraxQuickstart,
        run_native_development_check,
    )

    try:
        config = attempt_input(config, maximum_bytes=32768, copy=True, role="config")
        request = load_registered_authoring(
            config,
            root_schemas={NativeToraxQuickstart.SCHEMA: NativeToraxQuickstart},
            maximum_bytes=32 * 1024,
        )
        attempt_stage("native_calculation", native_contact="unknown")
        report = run_native_development_check(request)
    except (
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        PackageNotFoundError,
    ) as error:
        typer.echo(f"TORAX native check refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("material-workflow-input-check")
@retained_command("campaign material-workflow-input-check", closed_output=False, native=False)
def campaign_material_workflow_input_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/material-control-inputs/guide.md for the accepted schema and constraints.')
    ],
    source_root: Annotated[Path, typer.Option("--source-root")],
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Inspect ten separately custodied material-control raw workflow inputs."""

    from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.material_control_input_preflight import (
        MaterialControlInputPreflight,
        inspect_material_control_input_roster,
    )

    try:
        config = attempt_input(config, maximum_bytes=32768, copy=True, role="config")
        request = load_registered_authoring(
            config,
            root_schemas={
                MaterialControlInputPreflight.SCHEMA: MaterialControlInputPreflight
            },
            maximum_bytes=32 * 1024,
        )
        attempt_stage("prerequisite_inspection", native_contact="none")
        report = inspect_material_control_input_roster(request, source_root=source_root)
    except (ImportError, OSError, TypeError, ValueError) as error:
        typer.echo(f"Material-control input preflight refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("material-workflow-wrap-raw")
def campaign_material_workflow_wrap_raw(
    archive: Annotated[Path, typer.Option("--archive", exists=True, dir_okay=False)],
    profile_id: Annotated[str, typer.Option("--profile-id")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Wrap one bounded researcher-supplied material-control raw tar in a native HDF5 envelope."""

    from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.material_control_input_preflight import (
        inspect_material_control_tar_input,
    )
    from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.material_control_raw import (
        encode_raw_archive_hdf5,
    )

    try:
        inspected = inspect_material_control_tar_input(archive, profile_id=profile_id)
        physical_sha, physical_bytes = encode_raw_archive_hdf5(
            archive_path=archive,
            output_path=output,
            profile_id=profile_id,
            archive_sha256=str(inspected["logical_archive_sha256"]),
        )
    except (ImportError, OSError, TypeError, ValueError) as error:
        typer.echo(f"Material-control raw wrapping refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    typer.echo(
        json.dumps(
            {
                **inspected,
                "hdf5_sha256": physical_sha,
                "hdf5_bytes": physical_bytes,
                "science_frozen": False,
                "campaign_candidate_compiled": False,
                "campaign_issued": False,
            },
            sort_keys=True,
            indent=2,
        )
    )


@campaign_app.command("scale-morphism-reference-check")
@retained_command("campaign scale-morphism-reference-check", closed_output=False, native=False)
def campaign_scale_morphism_reference_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/response-method-reference/guide.md for the accepted schema and constraints.')
    ],
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Compare the physical scale-morphism method with generated reference cases."""

    from empirical_lawhood.adapters.reference_worlds.physical_scale_morphism.contracts import (
        PhysicalScaleMorphismTruthSuiteConfig,
    )
    from empirical_lawhood.adapters.reference_worlds.physical_scale_morphism.native_quickstart import (
        run_truth_method_development_check,
    )

    try:
        config = attempt_input(config, maximum_bytes=16384, copy=True, role="config")
        request = load_registered_authoring(
            config,
            root_schemas={
                PhysicalScaleMorphismTruthSuiteConfig.SCHEMA: PhysicalScaleMorphismTruthSuiteConfig
            },
            maximum_bytes=16 * 1024,
        )
        attempt_stage("prerequisite_inspection", native_contact="none")
        report = run_truth_method_development_check(request)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"Scale-morphism truth check refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("propulsion-reference-check")
@retained_command("campaign propulsion-reference-check", closed_output=False, native=False)
def campaign_propulsion_reference_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/propulsion-reliability-reference/guide.md for the accepted schema and constraints.')
    ],
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Run bounded propulsion closure and reliability reference worlds with known truth."""

    from empirical_lawhood.adapters.methods.synthetic_propulsion_development_input import (
        SyntheticPropulsionDevelopmentInput,
        run_native_development_check,
    )

    try:
        config = attempt_input(config, maximum_bytes=16384, copy=True, role="config")
        request = load_registered_authoring(
            config,
            root_schemas={
                SyntheticPropulsionDevelopmentInput.SCHEMA: SyntheticPropulsionDevelopmentInput
            },
            maximum_bytes=16 * 1024,
        )
        attempt_stage("prerequisite_inspection", native_contact="none")
        report = run_native_development_check(request)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"Propulsion native check refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("glenn-import-check")
@retained_command("campaign glenn-import-check", closed_output=False, native=False)
def campaign_glenn_import_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/laser-archive-inspection/guide.md for the accepted schema and constraints.')
    ],
    source_root: Annotated[Path | None, typer.Option("--source-root")] = None,
    preview: Annotated[bool, typer.Option("--preview")] = False,
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Inspect an exact public Glenn ZIP as retrospective, unqualified local input."""

    from empirical_lawhood.adapters.physical.glenn.quickstart import (
        GlennInputQuickstart,
        preview_public_release,
        run_local_import_check,
    )

    try:
        config = attempt_input(config, maximum_bytes=16384, copy=True, role="config")
        request = load_registered_authoring(
            config,
            root_schemas={GlennInputQuickstart.SCHEMA: GlennInputQuickstart},
            maximum_bytes=16 * 1024,
        )
        if preview:
            attempt_stage("prerequisite_inspection", native_contact="none")
            report = preview_public_release(request)
        else:
            if source_root is None:
                raise ValueError(
                    "--source-root is required for Glenn import inspection"
                )
            report = run_local_import_check(request, source_root=source_root)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"Glenn local import refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("battery-native-check")
@retained_command("campaign battery-native-check", closed_output=False, native=True)
def campaign_battery_native_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/battery-response-and-restart/guide.md for the accepted schema and constraints.')
    ],
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    'Check the bounded battery electrothermal response, battery exact restart and battery reduced observation donor contracts.'

    from empirical_lawhood.adapters.simulators.pybamm_development_input import (
        PyBaMMDevelopmentInput,
        run_native_development_check,
    )

    try:
        config = attempt_input(config, maximum_bytes=16384, copy=True, role="config")
        request = load_registered_authoring(
            config,
            root_schemas={PyBaMMDevelopmentInput.SCHEMA: PyBaMMDevelopmentInput},
            maximum_bytes=16 * 1024,
        )
        attempt_stage("native_calculation", native_contact="unknown")
        report = run_native_development_check(request)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"PyBaMM native check refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("reactor-batch-input-check")
@retained_command("campaign reactor-batch-input-check", closed_output=False, native=False)
def campaign_reactor_batch_input_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/reactor-response/guide.md for the accepted schema and constraints.')
    ],
    source_root: Annotated[
        Path | None,
        typer.Option("--source-root", help="Held upstream reactor source tree."),
    ] = None,
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Authenticate full-batch source and construct its provider without contact."""

    from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_input import (
        ReactorBatchInput,
        check_batch_input,
    )

    try:
        config = attempt_input(config, maximum_bytes=16384, copy=True, role="config")
        request = load_registered_authoring(
            config,
            root_schemas={ReactorBatchInput.SCHEMA: ReactorBatchInput},
            maximum_bytes=16 * 1024,
        )
        attempt_stage("prerequisite_inspection", native_contact="none")
        report = check_batch_input(request, source_root)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"reactor batch input check refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("reactor-prepared-input-check")
@retained_command("campaign reactor-prepared-input-check", closed_output=False, native=False)
def campaign_reactor_prepared_input_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/reactor-response/guide.md for the accepted schema and constraints.')
    ],
    source_root: Annotated[
        Path | None,
        typer.Option("--source-root", help="Held four-member reactor tree."),
    ] = None,
    discovery_file: Annotated[
        Path | None,
        typer.Option("--discovery-file", help="Held local discovery publication JSON."),
    ] = None,
    upstream_binding: Annotated[
        Path | None,
        typer.Option(
            "--upstream-binding",
            help="Typed identities for the selected route's non-source platform ports.",
        ),
    ] = None,
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Check reactor continuation inputs, binding and ports before native contact."""

    from empirical_lawhood.adapters.simulators.reactor_prefix_response.prepared_input import (
        ReactorPreparedInput,
        check_prepared_input,
    )
    from empirical_lawhood.adapters.simulators.reactor_prefix_response.reactor_binding import (
        ReactorSourceBinding,
        EnvironmentBoundReactorSourceBinding,
    )

    try:
        config = attempt_input(config, maximum_bytes=16384, copy=True, role="config")
        request = load_registered_authoring(
            config,
            root_schemas={ReactorPreparedInput.SCHEMA: ReactorPreparedInput},
            maximum_bytes=16 * 1024,
        )
        binding = (
            None
            if upstream_binding is None
            else load_registered_authoring(
                upstream_binding,
                root_schemas={
                    ReactorSourceBinding.SCHEMA: ReactorSourceBinding,
                    EnvironmentBoundReactorSourceBinding.SCHEMA: EnvironmentBoundReactorSourceBinding,
                },
                maximum_bytes=16 * 1024,
            )
        )
        attempt_stage("prerequisite_inspection", native_contact="none")
        report = check_prepared_input(
            request,
            source_root=source_root,
            discovery_file=discovery_file,
            upstream_binding=binding,
            port_store=_reactor_port_store() if binding is not None else None,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"reactor prepared input check refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("prepared-response-native-check")
@retained_command("campaign prepared-response-native-check", closed_output=False, native=True)
def campaign_prepared_response_native_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/prepared-response/guide.md for the accepted schema and constraints.')
    ],
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Run one excluded prepared-response root in both numerical views."""

    from empirical_lawhood.adapters.simulators.prepared_response.native_quickstart import (
        PreparedResponsePreparedCanary,
        run_prepared_canary,
    )

    try:
        config = attempt_input(config, maximum_bytes=16384, copy=True, role="config")
        request = load_registered_authoring(
            config,
            root_schemas={
                PreparedResponsePreparedCanary.SCHEMA: PreparedResponsePreparedCanary
            },
            maximum_bytes=16 * 1024,
        )
        attempt_stage("native_calculation", native_contact="unknown")
        report = run_prepared_canary(request)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"Prepared-response native check refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("matrix-response-author")
@retained_command("campaign matrix-response-author", closed_output=True, native=False)
def campaign_matrix_response_author(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/prepared-response/guide.md for the accepted schema and constraints.')
    ],
    source_root: Annotated[
        Path | None,
        typer.Option("--source-root", help="Held source census and model-bank root."),
    ] = None,
    plan: Annotated[
        Path | None, typer.Option("--plan", help="Exact pre-outcome native plan bytes.")
    ] = None,
    design_packet: Annotated[
        Path | None,
        typer.Option("--design-packet", help="Exact scientific design packet bytes."),
    ] = None,
    prior_exposure: Annotated[
        Path | None,
        typer.Option("--prior-exposure", help="Held prior unit and stream census."),
    ] = None,
    model_bank: Annotated[
        Path | None,
        typer.Option(
            "--model-bank", help="Held fitted bank for information or causal route."
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir", help="New output directory outside the target checkout."
        ),
    ] = None,
    attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
) -> None:
    """Compile a prepared, information or causal matrix-response development candidate."""

    from empirical_lawhood.adapters.composition.prepared_response.native_authoring import (
        PreparedResponseNativeAuthoringInput,
    )
    from empirical_lawhood.api.native_authoring import author_matrix_response

    try:
        config = attempt_input(config, maximum_bytes=1048576, copy=False, role="config")
        if CLI_PROJECT_ROOT is None:
            raise ValueError("SIX_MATRIX_RESPONSE_PROJECT_ROOT_REQUIRED: pass --project-root")
        request = load_registered_authoring(
            config,
            root_schemas={
                PreparedResponseNativeAuthoringInput.SCHEMA: PreparedResponseNativeAuthoringInput
            },
            maximum_bytes=1024**2,
        )
        attempt_stage("candidate_export", native_contact="none")
        report = author_matrix_response(
            request,
            repo_root=CLI_PROJECT_ROOT,
            source_root=source_root,
            plan=plan,
            design_packet=design_packet,
            prior_exposure=prior_exposure,
            model_bank=model_bank,
            output_dir=output_dir,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"Matrix-response native authoring refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("dependent-response-input-check")
@retained_command("campaign dependent-response-input-check", closed_output=False, native=False)
def campaign_dependent_response_input_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/prepared-response/guide.md for the accepted schema and constraints.')
    ],
    source_root: Annotated[Path | None, typer.Option("--source-root")] = None,
    parent_manifest: Annotated[Path | None, typer.Option("--parent-manifest")] = None,
    custody: Annotated[Path | None, typer.Option("--custody")] = None,
    reveal_record: Annotated[Path | None, typer.Option("--reveal-record")] = None,
    analysis_record: Annotated[Path | None, typer.Option("--analysis-record")] = None,
    authoring_input: Annotated[
        Path | None,
        typer.Option(
            "--authoring-input",
            help="Typed development source, exposure and measured dependence costs.",
        ),
    ] = None,
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Authenticate a qualified dependence or calibration parent and inspect its native binding."""

    from empirical_lawhood.adapters.composition.prepared_response.dependent_input import (
        PreparedResponseDependentInput,
        check_dependent_input,
    )

    try:
        config = attempt_input(config, maximum_bytes=16384, copy=True, role="config")
        request = load_registered_authoring(
            config,
            root_schemas={
                PreparedResponseDependentInput.SCHEMA: PreparedResponseDependentInput
            },
            maximum_bytes=16 * 1024,
        )
        attempt_stage("prerequisite_inspection", native_contact="none")
        report = check_dependent_input(
            request,
            source_root=source_root,
            parent_manifest=parent_manifest,
            custody=custody,
            reveal_record=reveal_record,
            analysis_record=analysis_record,
            authority_store=_response_parent_store(),
            authoring_input=authoring_input,
            repo_root=CLI_PROJECT_ROOT,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"Dependent-response input refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("response-composition-input-check")
@retained_command("campaign response-composition-input-check", closed_output=False, native=False)
def campaign_response_composition_input_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/causal-transfer-audit/guide.md for the accepted schema and constraints.')
    ],
    source_root: Annotated[Path | None, typer.Option("--source-root")] = None,
    parent_manifest: Annotated[Path | None, typer.Option("--parent-manifest")] = None,
    custody: Annotated[Path | None, typer.Option("--custody")] = None,
    reveal_record: Annotated[Path | None, typer.Option("--reveal-record")] = None,
    analysis_record: Annotated[Path | None, typer.Option("--analysis-record")] = None,
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Require trusted scalar-parent custody and authority before response-composition analysis."""

    from empirical_lawhood.adapters.composition.response_composition.input import (
        ResponseCompositionInput,
        check_response_composition_input,
    )

    try:
        config = attempt_input(config, maximum_bytes=16384, copy=True, role="config")
        request = load_registered_authoring(
            config,
            root_schemas={ResponseCompositionInput.SCHEMA: ResponseCompositionInput},
            maximum_bytes=16 * 1024,
        )
        attempt_stage("prerequisite_inspection", native_contact="none")
        report = check_response_composition_input(
            request,
            source_root=source_root,
            parent_manifest=parent_manifest,
            custody=custody,
            reveal_record=reveal_record,
            analysis_record=analysis_record,
            authority_store=_response_parent_store(),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"Response-composition input refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("finite-response-input-check")
@retained_command("campaign finite-response-input-check", closed_output=False, native=False)
def campaign_finite_response_input_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/finite-response-law/guide.md for the accepted schema and constraints.')
    ],
    source_root: Annotated[
        Path | None, typer.Option("--source-root", help="Held finite exposure root.")
    ] = None,
    plan: Annotated[
        Path | None, typer.Option("--plan", help="Exact frozen finite-lawhood plan.")
    ] = None,
    prior_exposure: Annotated[
        Path | None,
        typer.Option("--prior-exposure", help="Held finite native exposure census."),
    ] = None,
    assignment: Annotated[
        Path | None,
        typer.Option("--assignment", help="External target matrix-response cohort assignment."),
    ] = None,
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Inspect the retained or externally assigned root roster before native binding."""

    from empirical_lawhood.adapters.composition.finite_response_law.assignment import (
        FiniteResponseLawCohortAssignment,
    )
    from empirical_lawhood.adapters.composition.finite_response_law.native_input import (
        FiniteResponseLawCalibrationInput,
        check_finite_calibration_input,
    )

    try:
        config = attempt_input(config, maximum_bytes=16384, copy=True, role="config")
        request = load_registered_authoring(
            config,
            root_schemas={
                FiniteResponseLawCalibrationInput.SCHEMA: FiniteResponseLawCalibrationInput
            },
            maximum_bytes=16 * 1024,
        )
        assigned = None
        if assignment is not None:
            if (
                not assignment.is_absolute()
                or assignment.is_symlink()
                or not assignment.is_file()
            ):
                raise ValueError(
                    "FINITE_RESPONSE_LAW_ASSIGNMENT_REQUIRED: absolute real external file"
                )
            assigned = load_registered_authoring(
                assignment,
                root_schemas={
                    FiniteResponseLawCohortAssignment.SCHEMA: FiniteResponseLawCohortAssignment
                },
                maximum_bytes=128 * 1024,
            )
        attempt_stage("prerequisite_inspection", native_contact="none")
        report = check_finite_calibration_input(
            request,
            source_root=source_root,
            plan=plan,
            prior_exposure=prior_exposure,
            assignment=assigned,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"Finite-response input check refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("finite-response-stage-input-check")
@retained_command("campaign finite-response-stage-input-check", closed_output=False, native=False)
def campaign_finite_response_stage_input_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/finite-response-law/guide.md for the accepted schema and constraints.')
    ],
    source_root: Annotated[Path | None, typer.Option("--source-root")] = None,
    plan: Annotated[Path | None, typer.Option("--plan")] = None,
    prior_exposure: Annotated[Path | None, typer.Option("--prior-exposure")] = None,
    assignment: Annotated[Path | None, typer.Option("--assignment")] = None,
    parent_manifest: Annotated[Path | None, typer.Option("--parent-manifest")] = None,
    custody: Annotated[Path | None, typer.Option("--custody")] = None,
    reveal_record: Annotated[Path | None, typer.Option("--reveal-record")] = None,
    analysis_record: Annotated[Path | None, typer.Option("--analysis-record")] = None,
    authoring_input: Annotated[
        Path | None,
        typer.Option(
            "--authoring-input",
            help="Typed finite consumer operands and additional authenticated parents.",
        ),
    ] = None,
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Select a finite stage and refuse missing assigned roots or parent authority."""

    from empirical_lawhood.adapters.composition.finite_response_law.assignment import (
        FiniteResponseLawCohortAssignment,
    )
    from empirical_lawhood.adapters.composition.finite_response_law.stage_input import (
        FiniteResponseLawStageInput,
        check_finite_stage_input,
    )

    try:
        config = attempt_input(config, maximum_bytes=16384, copy=True, role="config")
        request = load_registered_authoring(
            config,
            root_schemas={
                FiniteResponseLawStageInput.SCHEMA: FiniteResponseLawStageInput
            },
            maximum_bytes=16 * 1024,
        )
        assigned = None
        if assignment is not None:
            if (
                not assignment.is_absolute()
                or assignment.is_symlink()
                or not assignment.is_file()
            ):
                raise ValueError(
                    "FINITE_RESPONSE_LAW_ASSIGNMENT_REQUIRED: absolute real external file"
                )
            assigned = load_registered_authoring(
                assignment,
                root_schemas={
                    FiniteResponseLawCohortAssignment.SCHEMA: FiniteResponseLawCohortAssignment
                },
                maximum_bytes=128 * 1024,
            )
        attempt_stage("prerequisite_inspection", native_contact="none")
        report = check_finite_stage_input(
            request,
            source_root=source_root,
            plan=plan,
            prior_exposure=prior_exposure,
            assignment=assigned,
            parent_manifest=parent_manifest,
            custody=custody,
            reveal_record=reveal_record,
            analysis_record=analysis_record,
            authority_store=_response_parent_store(),
            authoring_input=authoring_input,
            repo_root=CLI_PROJECT_ROOT,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"Finite-response stage input refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("response-composition-power")
@retained_command("campaign response-composition-power", closed_output=False, native=False)
def campaign_response_composition_power(
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Report fixed-sample design sensitivity without source or native contact."""

    from empirical_lawhood.adapters.methods.finite_response_law.power import (
        fixed_sample_power_report,
    )

    emit_development_report(fixed_sample_power_report(), allow_nan=False)


@campaign_app.command("finite-response-native-check")
@retained_command("campaign finite-response-native-check", closed_output=False, native=True)
def campaign_finite_response_native_check(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/finite-response-law/guide.md for the accepted schema and constraints.')
    ],
    output_dir: Annotated[Path | None, ATTEMPT_OUTPUT_OPTION] = None,
) -> None:
    """Run one excluded finite-lawhood canary through both native futures."""

    from empirical_lawhood.adapters.simulators.finite_response_law.native_quickstart import (
        FiniteResponseLawCanary,
        run_finite_canary,
    )

    try:
        config = attempt_input(config, maximum_bytes=16384, copy=True, role="config")
        request = load_registered_authoring(
            config,
            root_schemas={FiniteResponseLawCanary.SCHEMA: FiniteResponseLawCanary},
            maximum_bytes=16 * 1024,
        )
        attempt_stage("native_calculation", native_contact="unknown")
        report = run_finite_canary(request)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"Finite-response native check refused: {error}", err=True)
        raise typer.Exit(code=3) from error
    emit_development_report(report)


@campaign_app.command("reactor-author")
@retained_command("campaign reactor-author", closed_output=True, native=False)
def campaign_reactor_author(
    profile: Annotated[
        Path, typer.Option("--profile", exists=True, dir_okay=False, readable=True)
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
) -> None:
    """Emit strict reactor inputs with new identities and plan projections before issue."""

    if CLI_PROJECT_ROOT is None or CLI_OPERATOR_PROFILE is None:
        raise typer.BadParameter(
            "fresh authoring requires --project-root and --operator-profile"
        )
    try:
        profile = attempt_input(profile, maximum_bytes=67108864, copy=False, role="profile")
        storage_profile = load_registered_authoring(
            CLI_OPERATOR_PROFILE,
            root_schemas={OperatorStorageProfile.SCHEMA: OperatorStorageProfile},
        )
        if not isinstance(storage_profile, OperatorStorageProfile):
            raise TypeError("operator storage profile has another type")
        contract = resolve_external_root_contract(
            storage_profile, repo_root=CLI_PROJECT_ROOT, home_root=Path.home()
        )
        guard = GuardedExternalRoot(contract)
        guard.verify(for_write=True)
        external = Path(contract.canonical_path)
        output = output_dir.absolute()
        if output.is_symlink() or not output.is_relative_to(external):
            raise ValueError(
                "fresh authoring output must be below the guarded artifact namespace"
            )
        guard.resolve(
            str(output.relative_to(external) / "profile.json"), for_write=True
        )
        attempt_stage("candidate_export", native_contact="none")
        report = author_fresh_reactor(
            root=CLI_PROJECT_ROOT,
            profile=load_fresh_reactor_profile(profile),
            output_dir=output,
        )
    except (OSError, PermissionError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"fresh reactor authoring refused: {error}", err=True)
        raise typer.Exit(code=5) from error
    emit_development_report(report)


@campaign_app.command("circuit-author")
@retained_command("campaign circuit-author", closed_output=True, native=False)
def campaign_circuit_author(
    study: Annotated[
        Path, typer.Option("--study", exists=True, dir_okay=False, readable=True)
    ],
    experiment_id: Annotated[str, typer.Option("--experiment-id")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
) -> None:
    """Author one resistor-capacitor model through a strict candidate and plan projection."""

    if CLI_PROJECT_ROOT is None or CLI_OPERATOR_PROFILE is None:
        raise typer.BadParameter(
            "RC authoring requires --project-root and --operator-profile"
        )
    try:
        study = attempt_input(study, maximum_bytes=131072, copy=False, role="study")
        storage_profile = load_registered_authoring(
            CLI_OPERATOR_PROFILE,
            root_schemas={OperatorStorageProfile.SCHEMA: OperatorStorageProfile},
        )
        if not isinstance(storage_profile, OperatorStorageProfile):
            raise TypeError("operator storage profile has another type")
        contract = resolve_external_root_contract(
            storage_profile, repo_root=CLI_PROJECT_ROOT, home_root=Path.home()
        )
        guard = GuardedExternalRoot(contract)
        guard.verify(for_write=True)
        external = Path(contract.canonical_path)
        output = output_dir.absolute()
        if output.is_symlink() or not output.is_relative_to(external):
            raise ValueError("RC authoring output must be below guarded artifacts")
        guard.resolve(
            str(output.relative_to(external) / "profile.json"), for_write=True
        )
        attempt_stage("candidate_export", native_contact="none")
        report = author_fresh_rc(
            root=CLI_PROJECT_ROOT,
            study=load_rc_study(study),
            experiment_id=experiment_id,
            output_dir=output,
        )
    except (OSError, PermissionError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"RC authoring refused: {error}", err=True)
        raise typer.Exit(code=5) from error
    emit_development_report(report)


@campaign_app.command("electron-gas-author")
@retained_command("campaign electron-gas-author", closed_output=True, native=False)
def campaign_electron_gas_author(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/electron-gas-response/guide.md for the accepted schema and constraints.')
    ],
    experiment_id: Annotated[str, typer.Option("--experiment-id")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
) -> None:
    """Author one synthetic electron-gas acquisition through strict records without source contact."""

    if CLI_PROJECT_ROOT is None or CLI_OPERATOR_PROFILE is None:
        raise typer.BadParameter(
            "uniform electron gas authoring requires --project-root and --operator-profile"
        )
    try:
        config = attempt_input(config, maximum_bytes=131072, copy=False, role="config")
        storage_profile = load_registered_authoring(
            CLI_OPERATOR_PROFILE,
            root_schemas={OperatorStorageProfile.SCHEMA: OperatorStorageProfile},
        )
        if not isinstance(storage_profile, OperatorStorageProfile):
            raise TypeError("operator storage profile has another type")
        contract = resolve_external_root_contract(
            storage_profile, repo_root=CLI_PROJECT_ROOT, home_root=Path.home()
        )
        guard = GuardedExternalRoot(contract)
        guard.verify(for_write=True)
        external = Path(contract.canonical_path)
        output = output_dir.absolute()
        if output.is_symlink() or not output.is_relative_to(external):
            raise ValueError("uniform electron gas authoring output must be below guarded artifacts")
        guard.resolve(
            str(output.relative_to(external) / "profile.json"), for_write=True
        )
        attempt_stage("candidate_export", native_contact="none")
        report = author_uniform_electron_gas_analytic_reference(
            root=CLI_PROJECT_ROOT,
            config=load_uniform_electron_gas_analytic_config(config),
            experiment_id=experiment_id,
            output_dir=output,
        )
    except (OSError, PermissionError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"uniform electron gas authoring refused: {error}", err=True)
        raise typer.Exit(code=5) from error
    emit_development_report(report)


@campaign_app.command("synthetic-material-author")
@retained_command("campaign synthetic-material-author", closed_output=True, native=False)
def campaign_synthetic_material_author(
    config: Annotated[
        Path, typer.Option('--config', exists=True, dir_okay=False, readable=True, help='Configuration input. Read experiments/lattice-pairing-method/guide.md for the accepted schema and constraints.')
    ],
    experiment_id: Annotated[str, typer.Option("--experiment-id")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
) -> None:
    """Author one disclosed lattice-pairing method suite through strict candidate records."""

    if CLI_PROJECT_ROOT is None or CLI_OPERATOR_PROFILE is None:
        raise typer.BadParameter(
            "Lattice-pairing authoring requires --project-root and --operator-profile"
        )
    try:
        config = attempt_input(config, maximum_bytes=131072, copy=False, role="config")
        storage_profile = load_registered_authoring(
            CLI_OPERATOR_PROFILE,
            root_schemas={OperatorStorageProfile.SCHEMA: OperatorStorageProfile},
        )
        if not isinstance(storage_profile, OperatorStorageProfile):
            raise TypeError("operator storage profile has another type")
        contract = resolve_external_root_contract(
            storage_profile, repo_root=CLI_PROJECT_ROOT, home_root=Path.home()
        )
        guard = GuardedExternalRoot(contract)
        guard.verify(for_write=True)
        external = Path(contract.canonical_path)
        output = output_dir.absolute()
        if output.is_symlink() or not output.is_relative_to(external):
            raise ValueError("Lattice-pairing authoring output must be below guarded artifacts")
        guard.resolve(
            str(output.relative_to(external) / "profile.json"), for_write=True
        )
        attempt_stage("candidate_export", native_contact="none")
        report = author_synthetic_material_response(
            root=CLI_PROJECT_ROOT,
            config=load_synthetic_material_response_config(config),
            experiment_id=experiment_id,
            output_dir=output,
        )
    except (OSError, PermissionError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"Lattice-pairing authoring refused: {error}", err=True)
        raise typer.Exit(code=5) from error
    emit_development_report(report)


@campaign_app.command("reactor-preissue-proof")
def campaign_reactor_preissue_proof() -> None:
    """Resolve the production route and persist control proof without source contact."""

    if (
        CLI_PROJECT_ROOT is None
        or CLI_OPERATOR_PROFILE is None
        or CLI_REACTOR_AUTHORING_DIR is None
    ):
        raise typer.BadParameter(
            "proof requires --project-root, --operator-profile and --reactor-authoring-dir"
        )
    try:
        storage_profile = load_registered_authoring(
            CLI_OPERATOR_PROFILE,
            root_schemas={OperatorStorageProfile.SCHEMA: OperatorStorageProfile},
        )
        if not isinstance(storage_profile, OperatorStorageProfile):
            raise TypeError("operator storage profile has another type")
        report = prove_fresh_reactor(
            repo_root=CLI_PROJECT_ROOT,
            storage_profile=storage_profile,
            directory=CLI_REACTOR_AUTHORING_DIR,
            approval_checker_trust_path=CLI_APPROVAL_CHECKER_TRUST,
        )
    except (OSError, PermissionError, RuntimeError, TypeError, ValueError) as error:
        typer.echo(f"fresh reactor proof refused: {error}", err=True)
        raise typer.Exit(code=5) from error
    typer.echo(json.dumps(report, sort_keys=True, indent=2))


@campaign_app.command("compile-study")
def campaign_compile_study(
    spec: Annotated[Path, SPEC_OPTION],
    emit: Annotated[
        Path | None,
        typer.Option(
            "--emit",
            dir_okay=False,
            help="Optional no-replace .json path relative to the draft directory.",
        ),
    ] = None,
    parent_record: Annotated[Path | None, PARENT_RECORD_OPTION] = None,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Compile a strict DRAFT without issue, authority, execution or outcome access."""

    _emit(
        _api().compile_study(
            CompileStudyRequest(
                path=spec,
                emit_path=emit,
                parent_record_path=parent_record,
            )
        ),
        output_format,
    )


@campaign_app.command("compile-candidate")
def campaign_compile_candidate(
    spec: Annotated[Path, SPEC_OPTION],
    extension_payloads: Annotated[
        list[Path],
        typer.Option(
            "--extension-payload",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help='Exact extension payload. Repeat once per declared extension in roster order.',
        ),
    ],
    decoder_registrations: Annotated[
        list[Path],
        typer.Option(
            "--decoder-registration",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help='Static decoder registration. Repeat in the same order as extension payloads.',
        ),
    ],
    emit: Annotated[
        Path | None,
        typer.Option(
            "--emit",
            dir_okay=False,
            help="Optional no-replace .json path relative to the authoring document directory.",
        ),
    ] = None,
    parent_record: Annotated[Path | None, PARENT_RECORD_OPTION] = None,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Compile a candidate with extensions without issue or outcome access."""

    _emit(
        _api().compile_candidate(
            CompileCandidateRequest(
                path=spec,
                extension_payload_paths=tuple(extension_payloads),
                decoder_registration_paths=tuple(decoder_registrations),
                emit_path=emit,
                parent_record_path=parent_record,
            )
        ),
        output_format,
    )


@campaign_app.command("check-readiness")
def campaign_check_readiness(
    spec: Annotated[Path, SPEC_OPTION],
    extension_payloads: Annotated[
        list[Path], typer.Option("--extension-payload", exists=True, dir_okay=False)
    ],
    decoder_registrations: Annotated[
        list[Path], typer.Option("--decoder-registration", exists=True, dir_okay=False)
    ],
    expected_candidate: Annotated[
        Path, typer.Option("--expected-candidate", exists=True, dir_okay=False)
    ],
    source_closure: Annotated[
        Path, typer.Option("--source-closure", exists=True, dir_okay=False)
    ],
    resource_envelope: Annotated[
        Path, typer.Option("--resource-envelope", exists=True, dir_okay=False)
    ],
    run_id: Annotated[str, typer.Option("--run", help="Future frozen run-plan ID.")],
    jit_census: Annotated[
        Path | None, typer.Option("--jit-census", exists=True, dir_okay=False)
    ] = None,
    jit_manifest: Annotated[
        Path | None, typer.Option("--jit-manifest", exists=True, dir_okay=False)
    ] = None,
    campaign_elapsed_budget: Annotated[
        Path | None,
        typer.Option("--campaign-elapsed-budget", exists=True, dir_okay=False),
    ] = None,
    campaign_elapsed_reservations: Annotated[
        Path | None,
        typer.Option("--campaign-elapsed-reservations", exists=True, dir_okay=False),
    ] = None,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Prove the future runtime contract without authority, effects, or writes."""

    _emit(
        _api().preissue_readiness(
            CheckReadinessRequest(
                authoring_package_path=spec,
                extension_payload_paths=tuple(extension_payloads),
                decoder_registration_paths=tuple(decoder_registrations),
                expected_candidate_path=expected_candidate,
                source_closure_path=source_closure,
                execution_resource_envelope_spec_path=resource_envelope,
                predevelopment_jit_signature_census_path=jit_census,
                jit_graph_signature_manifest_path=jit_manifest,
                run_plan_id=run_id,
                campaign_elapsed_budget_path=campaign_elapsed_budget,
                campaign_elapsed_reservation_plan_path=campaign_elapsed_reservations,
            )
        ),
        output_format,
    )


@campaign_app.command("compile-bundle")
def campaign_compile_bundle(
    spec: Annotated[Path, SPEC_OPTION],
    emit: Annotated[
        Path | None,
        typer.Option(
            "--emit",
            dir_okay=False,
            help="Optional no-replace .json path relative to the bundle document directory.",
        ),
    ] = None,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Compile a multi-child bundle and deterministic joint fan-in without issue."""

    _emit(
        _api().compile_study_bundle(
            CompileStudyBundleRequest(path=spec, emit_path=emit)
        ),
        output_format,
    )


@campaign_app.command("issue")
def campaign_issue(
    spec: Annotated[Path, SPEC_OPTION],
    expected_candidate: Annotated[Path, EXPECTED_CANDIDATE_OPTION],
    proposer_attestation: Annotated[Path, PROPOSER_ATTESTATION_OPTION],
    source_closure: Annotated[Path, SOURCE_CLOSURE_OPTION],
    custody_authority: Annotated[Path, CUSTODY_AUTHORITY_OPTION],
    parent_record: Annotated[Path | None, PARENT_RECORD_OPTION] = None,
    yes: Annotated[bool, YES_OPTION] = False,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Preview or publish an exact outcome-blind issued-programme bundle."""

    _emit(
        _api().issue_study(
            IssueStudyRequest(
                authoring_package_path=spec,
                expected_candidate_path=expected_candidate,
                proposer_attestation_path=proposer_attestation,
                source_closure_path=source_closure,
                custody_authority_path=custody_authority,
                confirmed=yes,
                parent_record_path=parent_record,
            )
        ),
        output_format,
    )


@campaign_app.command("issue-extensions")
def campaign_issue_extensions(
    spec: Annotated[Path, SPEC_OPTION],
    extension_payloads: Annotated[
        list[Path],
        typer.Option(
            "--extension-payload",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help='Exact extension payload. Repeat once per declared extension in roster order.',
        ),
    ],
    decoder_registrations: Annotated[
        list[Path],
        typer.Option(
            "--decoder-registration",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help='Static decoder registration. Repeat in extension roster order.',
        ),
    ],
    expected_candidate: Annotated[Path, EXPECTED_CANDIDATE_OPTION],
    base_issue_id: Annotated[
        str, typer.Option("--base-issue", help="Immutable base issue ID.")
    ],
    proposer_attestation: Annotated[Path, PROPOSER_ATTESTATION_OPTION],
    custody_authority: Annotated[Path, CUSTODY_AUTHORITY_OPTION],
    parent_record: Annotated[Path | None, PARENT_RECORD_OPTION] = None,
    yes: Annotated[bool, YES_OPTION] = False,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Preview or publish a child issue with extensions without outcome access."""

    _emit(
        _api().issue_extensions(
            IssueExtensionsRequest(
                authoring_package_path=spec,
                extension_payload_paths=tuple(extension_payloads),
                decoder_registration_paths=tuple(decoder_registrations),
                expected_candidate_path=expected_candidate,
                base_issue_id=base_issue_id,
                extension_proposer_attestation_path=proposer_attestation,
                extension_custody_authority_path=custody_authority,
                confirmed=yes,
                parent_record_path=parent_record,
            )
        ),
        output_format,
    )


@campaign_app.command("issue-bundle")
def campaign_issue_bundle(
    bundle_candidate: Annotated[
        Path,
        typer.Option(
            "--bundle-candidate",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Exact ProgrammeBundleCandidate document.",
        ),
    ],
    child_issue_ids: Annotated[
        list[str],
        typer.Option(
            "--child-issue",
            help='Immutable child issue ID. Repeat exactly three times.',
        ),
    ],
    child_science: Annotated[
        list[Path],
        typer.Option(
            "--child-science",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help='World-local child science binding. Repeat exactly three times.',
        ),
    ],
    archive_protection: Annotated[
        Path,
        typer.Option(
            "--archive-protection", exists=True, dir_okay=False, readable=True
        ),
    ],
    outcome_barriers: Annotated[
        Path,
        typer.Option("--outcome-barriers", exists=True, dir_okay=False, readable=True),
    ],
    partial_morphism: Annotated[
        Path,
        typer.Option("--partial-morphism", exists=True, dir_okay=False, readable=True),
    ],
    joint_adjudication: Annotated[
        Path,
        typer.Option(
            "--joint-adjudication", exists=True, dir_okay=False, readable=True
        ),
    ],
    custody_authority: Annotated[Path, CUSTODY_AUTHORITY_OPTION],
    yes: Annotated[bool, YES_OPTION] = False,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Preview or publish one parent over three immutable children without outcome access."""

    _emit(
        _api().issue_study_bundle(
            IssueStudyBundleRequest(
                bundle_candidate_path=bundle_candidate,
                child_issue_ids=tuple(sorted(child_issue_ids)),
                child_science_paths=tuple(child_science),
                archive_outcome_protection_path=archive_protection,
                outcome_barrier_plan_path=outcome_barriers,
                partial_morphism_path=partial_morphism,
                joint_adjudication_path=joint_adjudication,
                custody_authority_path=custody_authority,
                confirmed=yes,
            )
        ),
        output_format,
    )


@campaign_app.command("transition-bundle")
def campaign_transition_bundle(
    outcome_barriers: Annotated[
        Path,
        typer.Option("--outcome-barriers", exists=True, dir_okay=False, readable=True),
    ],
    barrier_prefix: Annotated[
        Path,
        typer.Option("--barrier-prefix", exists=True, dir_okay=False, readable=True),
    ],
    barrier_id: Annotated[
        str, typer.Option("--barrier", help="Exact next barrier ID.")
    ],
    reveal_authority_id: Annotated[
        str | None,
        typer.Option(
            "--reveal-authority",
            help='Trusted-store authority ID for a reveal transition. Omit the ID to obtain AUTHORITY_REQUIRED.',
        ),
    ] = None,
    child_result: Annotated[
        Path | None,
        typer.Option(
            "--child-result",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Exact terminal child result to bind after its reveal.",
        ),
    ] = None,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    'Advance one immutable parent barrier prefix. The command never combines reveal and result bind.'

    _emit(
        _api().advance_study_bundle(
            AdvanceStudyBundleRequest(
                outcome_barrier_plan_path=outcome_barriers,
                barrier_prefix_path=barrier_prefix,
                barrier_id=barrier_id,
                reveal_authority_id=reveal_authority_id,
                child_result_path=child_result,
            )
        ),
        output_format,
    )


@campaign_app.command("bundle-status")
def campaign_bundle_status(
    outcome_barriers: Annotated[
        Path,
        typer.Option("--outcome-barriers", exists=True, dir_okay=False, readable=True),
    ],
    barrier_prefix: Annotated[
        Path,
        typer.Option("--barrier-prefix", exists=True, dir_okay=False, readable=True),
    ],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Inspect eligible reveals and bound results from one immutable parent prefix."""

    _emit(
        _api().study_bundle_status(
            StudyBundleStatusRequest(
                outcome_barrier_plan_path=outcome_barriers,
                barrier_prefix_path=barrier_prefix,
            )
        ),
        output_format,
    )


@campaign_app.command("close-bundle")
def campaign_close_bundle(
    joint_adjudication: Annotated[
        Path,
        typer.Option(
            "--joint-adjudication", exists=True, dir_okay=False, readable=True
        ),
    ],
    outcome_barriers: Annotated[
        Path,
        typer.Option("--outcome-barriers", exists=True, dir_okay=False, readable=True),
    ],
    terminal_prefix: Annotated[
        Path,
        typer.Option("--terminal-prefix", exists=True, dir_okay=False, readable=True),
    ],
    child_results: Annotated[
        list[Path],
        typer.Option(
            "--child-result",
            exists=True,
            dir_okay=False,
            readable=True,
            help='Terminal world-local result. Repeat exactly three times.',
        ),
    ],
    archive_overlap: Annotated[
        Path,
        typer.Option("--archive-overlap", exists=True, dir_okay=False, readable=True),
    ],
    morphism_verdicts: Annotated[
        list[Path],
        typer.Option(
            "--morphism-verdict",
            exists=True,
            dir_okay=False,
            readable=True,
            help='Primary-property preparation verdict. Repeat for the contacted roster.',
        ),
    ],
    morphism_controls: Annotated[
        Path,
        typer.Option("--morphism-controls", exists=True, dir_okay=False, readable=True),
    ],
    mapped_states_accepted_count: Annotated[
        int, typer.Option("--mapped-states-accepted", min=0)
    ],
    mapped_states_unevaluable_count: Annotated[
        int, typer.Option("--mapped-states-unevaluable", min=0)
    ],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Close a terminal three-child parent with deterministic non-pooling adjudication."""

    _emit(
        _api().close_study_bundle(
            CloseStudyBundleRequest(
                joint_adjudication_plan_path=joint_adjudication,
                outcome_barrier_plan_path=outcome_barriers,
                terminal_barrier_prefix_path=terminal_prefix,
                child_result_paths=tuple(child_results),
                archive_overlap_result_path=archive_overlap,
                morphism_verdict_paths=tuple(morphism_verdicts),
                morphism_control_contrast_path=morphism_controls,
                mapped_prospective_accepted_count=mapped_states_accepted_count,
                mapped_prospective_map_unevaluable_count=mapped_states_unevaluable_count,
            )
        ),
        output_format,
    )


@campaign_app.command("compile-plan")
def campaign_compile_plan(
    spec: Annotated[Path, SPEC_OPTION],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Replay stored authorization, then compile deterministic DAG identities."""

    _emit(_api().compile_campaign(CompileCampaignRequest(spec)), output_format)


@campaign_app.command("run")
def campaign_run(
    plan: Annotated[Path, PLAN_OPTION],
    parent_record: Annotated[Path | None, PARENT_RECORD_OPTION] = None,
    parent_input_bindings: Annotated[
        list[Path] | None,
        PARENT_INPUT_BINDING_OPTION,
    ] = None,
    yes: Annotated[bool, YES_OPTION] = False,
    reveal_authority_id: Annotated[
        str | None,
        REVEAL_AUTHORITY_ID_OPTION,
    ] = None,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Preview or execute a frozen package with external receipt-first writes."""

    _emit(
        _api().run_campaign(
            RunCampaignRequest(
                plan,
                confirmed=yes,
                reveal_authority_id=reveal_authority_id,
                parent_record_path=parent_record,
                parent_input_binding_paths=tuple(parent_input_bindings or ()),
            )
        ),
        output_format,
    )


@campaign_app.command("resume")
def campaign_resume(
    run_id: Annotated[str, typer.Option("--run", help="Frozen run-plan ID.")],
    parent_record: Annotated[Path | None, PARENT_RECORD_OPTION] = None,
    parent_input_bindings: Annotated[
        list[Path] | None,
        PARENT_INPUT_BINDING_OPTION,
    ] = None,
    yes: Annotated[bool, YES_OPTION] = False,
    reveal_authority_id: Annotated[
        str | None,
        REVEAL_AUTHORITY_ID_OPTION,
    ] = None,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Preview or reconcile externally persisted plans and task receipts."""

    _emit(
        _api().resume_campaign(
            ResumeCampaignRequest(
                run_id,
                confirmed=yes,
                reveal_authority_id=reveal_authority_id,
                parent_record_path=parent_record,
                parent_input_binding_paths=tuple(parent_input_bindings or ()),
            )
        ),
        output_format,
    )


@campaign_app.command("status")
def campaign_status(
    run_id: Annotated[str, typer.Option("--run", help="Frozen run-plan ID.")],
    include_attempt_history: Annotated[
        bool,
        typer.Option(
            "--attempt-history",
            help="Include bounded per-attempt operational history separately from current state.",
        ),
    ] = False,
    attempt_history_limit: Annotated[
        int,
        typer.Option(
            "--attempt-history-limit",
            min=1,
            max=1_000,
            help="Maximum attempt-history rows to return in this page.",
        ),
    ] = 100,
    attempt_history_cursor: Annotated[
        str | None,
        typer.Option(
            "--attempt-history-cursor",
            help="Opaque cursor returned by the preceding attempt-history page.",
        ),
    ] = None,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Read operational status without creating a database or run."""

    _emit(
        _api().campaign_status(
            CampaignStatusRequest(
                run_id,
                include_attempt_history=include_attempt_history,
                attempt_history_limit=attempt_history_limit,
                attempt_history_cursor=attempt_history_cursor,
            )
        ),
        output_format,
        text_renderer=render_campaign_status,
    )


@campaign_app.command("explore-detect")
def campaign_explore_detect(
    campaign: Annotated[
        Path,
        typer.Option(
            "--campaign",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Strict DualLoopPackage or ExplorationExecutionPackage document.",
        ),
    ],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Inspect the frozen anomaly/search portfolio without reading payload bytes."""

    _emit(
        _api().inspect_exploration_portfolio(DocumentRequest(campaign)), output_format
    )


@campaign_app.command("explore-propose")
def campaign_explore_propose(
    snapshot: Annotated[
        Path,
        typer.Option(
            "--snapshot",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Exploration package containing the verified snapshot and portfolio.",
        ),
    ],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """List selected/rejected registered analyses over a verified snapshot."""

    _emit(
        _api().inspect_exploration_portfolio(
            DocumentRequest(snapshot), operation="campaign.explore-propose"
        ),
        output_format,
    )


@campaign_app.command("explore-compile")
def campaign_explore_compile(
    campaign: Annotated[
        Path,
        typer.Option(
            "--campaign",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Strict DualLoopPackage or ExplorationExecutionPackage document.",
        ),
    ],
    wave: Annotated[int, typer.Option("--wave", min=1)] = 1,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Compile one read-only, non-promotable exploration execution DAG."""

    if wave != 1:
        raise typer.BadParameter(
            "the package contains exactly wave 1", param_hint="--wave"
        )
    _emit(_api().compile_exploration(DocumentRequest(campaign)), output_format)


@campaign_app.command("explore-run")
def campaign_explore_run(
    plan: Annotated[Path, EXPLORATION_PLAN_OPTION],
    yes: Annotated[bool, YES_OPTION] = False,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    'Execute input-derived exploration. DualLoopPackage fixture execution is blocked.'

    _emit(
        _api().run_exploration(RunCampaignRequest(plan, confirmed=yes)), output_format
    )


@campaign_app.command("explore-synthesize")
def campaign_explore_synthesize(
    plan: Annotated[Path, EXPLORATION_PLAN_OPTION],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Inspect the non-promotable competing hypotheses bound to a wave."""

    _emit(_api().synthesize_exploration(DocumentRequest(plan)), output_format)


@campaign_app.command("nominate-next")
def campaign_nominate_next(
    hypotheses: Annotated[
        Path,
        typer.Option(
            "--hypotheses",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Exploration package and its fresh-evidence design context.",
        ),
    ],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Nominate fresh evidence without promoting the motivating findings."""

    _emit(_api().nominate_next(DocumentRequest(hypotheses)), output_format)


@campaign_app.command("propose-next")
def campaign_propose_next(
    campaign: Annotated[
        Path,
        typer.Option(
            "--campaign",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Exploration package containing the nomination design envelope.",
        ),
    ],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Design an authority-pending fresh experiment from a nomination."""

    _emit(_api().propose_next(DocumentRequest(campaign)), output_format)


@campaign_app.command("authorize-nonactuating")
def campaign_authorize_nonactuating(
    proposal: Annotated[
        Path,
        typer.Option(
            "--proposal",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="DualLoopPackage containing the immutable generated proposal inputs.",
        ),
    ],
    policy: Annotated[
        Path,
        typer.Option(
            "--policy",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="The same package containing the bound authority policy.",
        ),
    ],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Preview the contained outcome-blind non-actuating authority boundary."""

    if proposal != policy:
        raise typer.BadParameter(
            "proposal and policy must be bound in the same immutable package"
        )
    _emit(_api().authorize_nonactuating(DocumentRequest(proposal)), output_format)


def _inspect_scientific(
    object_id: str,
    output_format: OutputFormat,
    *,
    requested_kind: str,
    operation: str,
) -> None:
    _emit(
        _api().inspect_scientific_object(
            object_id,
            requested_kind=requested_kind,
            operation=operation,
        ),
        output_format,
    )


@authority_app.command("inspect")
def authority_inspect(
    object_id: Annotated[str, typer.Option("--id")],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Inspect authority metadata without outcome or artifact access."""

    _inspect_scientific(
        object_id,
        output_format,
        requested_kind="authority",
        operation="authority.inspect",
    )


@law_app.command("inspect")
def law_inspect(
    object_id: Annotated[str, typer.Option("--id")],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Inspect a response-law catalog envelope without loading its artifact."""

    _inspect_scientific(
        object_id, output_format, requested_kind="law", operation="law.inspect"
    )


@atlas_app.command("inspect")
def atlas_inspect(
    object_id: Annotated[str, typer.Option("--id")],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Inspect a response-atlas catalog envelope."""

    _inspect_scientific(
        object_id, output_format, requested_kind="atlas", operation="atlas.inspect"
    )


@atlas_app.command("gaps")
def atlas_gaps(
    object_id: Annotated[str, typer.Option("--id")],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    'Inspect categorical atlas status. Do not infer gaps from sealed payloads.'

    _inspect_scientific(
        object_id, output_format, requested_kind="atlas", operation="atlas.gaps"
    )


@admission_app.command("explain")
def admission_explain(
    object_id: Annotated[str, typer.Option("--id")],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Explain bounded admission metadata and its external pointer."""

    _inspect_scientific(
        object_id,
        output_format,
        requested_kind="admission",
        operation="admission.explain",
    )


@control_app.command("inspect")
def control_inspect(
    object_id: Annotated[str, typer.Option("--id")],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Inspect a controller-spec catalog envelope without execution."""

    _inspect_scientific(
        object_id, output_format, requested_kind="control", operation="control.inspect"
    )


@control_app.command("validate")
def control_validate(
    object_id: Annotated[str, typer.Option("--id")],
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    'Validate controller catalog identity only. This command supplies no prospective controller use evidence.'

    _inspect_scientific(
        object_id, output_format, requested_kind="control", operation="control.validate"
    )


@catalog_app.command("query")
def catalog_query(
    object_id: Annotated[str | None, typer.Option("--id")] = None,
    kind: Annotated[str | None, typer.Option("--kind")] = None,
    categorical_status: Annotated[str | None, typer.Option("--status")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=1_000)] = 100,
    match_count_limit: Annotated[
        int,
        typer.Option("--match-count-limit", min=1, max=10_000),
    ] = 1_000,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
    relation_id: Annotated[str | None, typer.Option("--relation")] = None,
    denominator_gauge_id: Annotated[
        str | None, typer.Option("--denominator-gauge")
    ] = None,
    response_gauge_id: Annotated[str | None, typer.Option("--response-gauge")] = None,
    horizon_id: Annotated[str | None, typer.Option("--horizon")] = None,
    native_unit: Annotated[str | None, typer.Option("--unit")] = None,
    knowledge_edge_relation: Annotated[
        str | None, typer.Option("--edge-relation")
    ] = None,
    knowledge_edge_scope_id: Annotated[str | None, typer.Option("--edge-scope")] = None,
    lineage_object_id: Annotated[str | None, typer.Option("--lineage-object")] = None,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    'Query compact metadata with capped count work. This command never returns payload data.'

    _emit(
        _api().query_catalog(
            CatalogQueryRequest(
                object_id=object_id,
                kind=kind,
                categorical_status=categorical_status,
                limit=limit,
                match_count_limit=match_count_limit,
                cursor=cursor,
                relation_id=relation_id,
                denominator_gauge_id=denominator_gauge_id,
                response_gauge_id=response_gauge_id,
                horizon_id=horizon_id,
                native_unit=native_unit,
                knowledge_edge_relation=knowledge_edge_relation,
                knowledge_edge_scope_id=knowledge_edge_scope_id,
                lineage_object_id=lineage_object_id,
            )
        ),
        output_format,
    )


@catalog_app.command("rebuild")
def catalog_rebuild(
    projection: Annotated[
        Path,
        typer.Option(
            "--projection",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Canonical catalog projection below the external root.",
        ),
    ],
    yes: Annotated[bool, YES_OPTION] = False,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Preview or atomically rebuild SQLite from an external projection."""

    _emit(
        _api().rebuild_catalog(CatalogRebuildRequest(projection, confirmed=yes)),
        output_format,
    )


@db_app.command("init")
def db_init(
    yes: Annotated[bool, YES_OPTION] = False,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Preview or initialize the ignored bounded local SQLite projection."""

    _emit(_api().initialize_catalog(DatabaseMutationRequest(yes)), output_format)


@db_app.command("check")
def db_check(
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Read-only integrity/schema/payload-column audit of the local catalog."""

    _emit(_api().check_catalog(), output_format)


@db_app.command("compact")
def db_compact(
    yes: Annotated[bool, YES_OPTION] = False,
    output_format: Annotated[OutputFormat, OUTPUT_OPTION] = OutputFormat.TEXT,
) -> None:
    """Preview or incrementally compact the bounded local SQLite projection."""

    _emit(_api().compact_catalog(DatabaseMutationRequest(yes)), output_format)
