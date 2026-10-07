"""Strict, non-executable JSON/YAML codecs for canonical authoring roots."""

from __future__ import annotations

from empirical_lawhood.kernel.retrospective_prediction_experiments import RetrospectiveExperimentSpec
from empirical_lawhood.planning.study_authoring import RetrospectiveStudyDraft
from empirical_lawhood.planning.experiment_entry import RetrospectiveAuthoringBase, RetrospectiveAuthoringPackage, RetrospectiveEntryPackage
from empirical_lawhood.planning.approval import RetrospectiveApprovalProposal, FrozenRetrospectiveApproval
from empirical_lawhood.runtime.candidate_compiler import RetrospectiveStudyCandidate, RetrospectiveStandardCandidate, RetrospectiveExtensionCandidate, RetrospectiveCandidateReport, RetrospectiveStandardReport, RetrospectiveExtensionReport
from empirical_lawhood.runtime.study_issue import RetrospectiveIssuedBase, RetrospectiveIssuedStudy, RetrospectivePublicationReceipt
from empirical_lawhood.runtime.plans import ResourceBoundRetrospectiveStudyRunPlan
from empirical_lawhood.api.models import RetrospectiveCampaignBase, RetrospectiveCampaignPackage

import json
import hashlib
import os
import stat
import types
from contextvars import ContextVar
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, Union, cast, get_args, get_origin

import yaml  # type: ignore[import-untyped]
from yaml.tokens import AliasToken, AnchorToken, TagToken  # type: ignore[import-untyped]

from empirical_lawhood.infrastructure.bounded_io import MAX_RUNTIME_PLAN_JSON_BYTES
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.decoding import record_annotations
from empirical_lawhood.kernel.parsed_nodes import ParsedNodeInterner
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    CanonicalizationError,
    validate_document_shape,
)
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.campaigns import CampaignSpec
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection
from empirical_lawhood.planning.experiment_entry import ExperimentEntryPackage, StudyDefinition, ExecutableStudyDefinition
from empirical_lawhood.planning.linked_campaign import LinkedCampaignProfile
from empirical_lawhood.planning.study_authoring import StudyDraft
from empirical_lawhood.planning.study_bundles import StudyBundleSpec
from empirical_lawhood.planning.multi_world_study import ArchiveOutcomeProtectionPlan, ArchiveOverlapQualification, ArchiveToSimulatorPartialMorphismSpec, MultiWorldJointAdjudicationPlan, MorphismNegativeControlPlan, MultiWorldOutcomeBarrierPlan, MultiWorldStudyChildScientificBinding, PropertyTransportExpectation
from empirical_lawhood.planning.response_algebra import (
    ResponseAlgebraHypothesisFreeze,
    ResponseAlgebraConformanceSpec,
    ResponseAlgebraProtocolSpec,
)
from empirical_lawhood.planning.source_pipelines import SourcePipelineProfile
from empirical_lawhood.planning.approval import ApprovalGateAttestation, DurableAuthorizationRecord, FrozenApprovalProposal, FrozenIssuedStudyApprovalProposal, IssuedStudyApprovalProposal
from empirical_lawhood.planning.study_issue import AccountableHumanIdentity, HumanProposerAttestation, RetrospectiveProposerAttestation, ImplementationSourceClosure, StudyOperationAuthority
from empirical_lawhood.runtime.plans import EnvelopeExecutionPlan, ExecutionPlan, ProtocolTemplate, EnvelopeRunPlan, RunPlan, ProtocolExecutionPlan, ProtocolRunPlan
from empirical_lawhood.runtime.profile_compilation import LinkedCampaignExecutableCompilation
from empirical_lawhood.runtime.source_pipelines import SourcePipelineCompilation
from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationReport, DraftStudyCandidate, StudyCompilationReport, ExecutableStudyCompilationReport, StudyCandidate, ExecutableStudyCandidate
from empirical_lawhood.runtime.study_issue import MAX_COMPOSED_PROGRAMME_CONTROL_BYTES, IssuedDraftManifest, StudyPublicationReceipt, ExtensionPublicationReceipt, StudyExtensionDecoderRegistration, IssuedStudyManifest, IssuedExecutableStudyManifest
from empirical_lawhood.runtime.study_bundle_compiler import StudyBundleCandidate, StudyBundleCompilationInput, StudyBundleCompilationReport

from .models import CampaignPackage, DualLoopPackage, ExplorationExecutionPackage, IssuedCampaignPackage, IssuedStudyPackage, EnvelopeExperimentPackage, ExperimentPackage, ElapsedExperimentPackage


@dataclass
class _AuthoringParse:
    nodes: ParsedNodeInterner = field(default_factory=ParsedNodeInterner)
    records: dict[tuple[type[CanonicalRecord], int], CanonicalRecord] = field(default_factory=dict)


_authoring_parse: ContextVar[_AuthoringParse | None] = ContextVar("authoring_parse", default=None)


class AuthoringCodecError(ValueError):
    """A document failed syntax, schema, version or semantic validation."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


RootRecord = (
    SystemSpec
    | ExperimentSpec
    | CampaignSpec
    | ProtocolTemplate
    | FrozenApprovalProposal
    | IssuedStudyApprovalProposal
    | FrozenIssuedStudyApprovalProposal
    | ApprovalGateAttestation
    | DurableAuthorizationRecord
    | CampaignPackage
    | IssuedCampaignPackage
    | IssuedStudyPackage
    | EnvelopeExperimentPackage
    | ExperimentPackage
    | ElapsedExperimentPackage
    | DualLoopPackage
    | ExplorationExecutionPackage
    | ResponseAlgebraProtocolSpec
    | ResponseAlgebraHypothesisFreeze
    | ResponseAlgebraConformanceSpec
    | StudyDraft
    | ExperimentEntryPackage
    | StudyDefinition
    | ExecutableStudyDefinition
    | DraftStudyCandidate
    | CandidateCompilationReport
    | StudyCandidate
    | ExecutableStudyCandidate
    | StudyCompilationReport
    | ExecutableStudyCompilationReport
    | AccountableHumanIdentity
    | HumanProposerAttestation
    | ImplementationSourceClosure
    | StudyOperationAuthority
    | IssuedDraftManifest
    | IssuedStudyManifest
    | IssuedExecutableStudyManifest
    | StudyPublicationReceipt
    | ExtensionPublicationReceipt
    | StudyExtensionDecoderRegistration
    | OperatorStorageProfile
    | StudyBundleSpec
    | StudyBundleCompilationInput
    | StudyBundleCandidate
    | StudyBundleCompilationReport
    | MultiWorldStudyChildScientificBinding
    | ArchiveOutcomeProtectionPlan
    | MultiWorldOutcomeBarrierPlan
    | ArchiveToSimulatorPartialMorphismSpec
    | PropertyTransportExpectation
    | ArchiveOverlapQualification
    | MorphismNegativeControlPlan
    | MultiWorldJointAdjudicationPlan
    | EnvelopeRunPlan
    | EnvelopeExecutionPlan
    | RunPlan
    | ExecutionPlan
    | SourcePipelineCompilation
    | LinkedCampaignExecutableCompilation
    | EvidenceProfileSelection
    | SourcePipelineProfile
    | LinkedCampaignProfile
)
RecordT = TypeVar("RecordT", bound=CanonicalRecord)
MAX_AUTHORING_BYTES = 128 * 1024**2
MAX_CAMPAIGN_PACKAGE_BYTES = MAX_COMPOSED_PROGRAMME_CONTROL_BYTES
MAX_AUTHORING_ROOT_SCHEMAS = 128
MAX_AUTHORING_NESTING = 128
MAX_AUTHORING_NODES = 4_000_000
MAX_COMPOSED_CONTROL_NODES = 8_000_000


ROOT_SCHEMAS: Mapping[str, type[RootRecord]] = types.MappingProxyType(
    {
        SystemSpec.SCHEMA: SystemSpec,
        ExperimentSpec.SCHEMA: ExperimentSpec,
        RetrospectiveExperimentSpec.SCHEMA: RetrospectiveExperimentSpec,
        CampaignSpec.SCHEMA: CampaignSpec,
        ProtocolTemplate.SCHEMA: ProtocolTemplate,
        FrozenApprovalProposal.SCHEMA: FrozenApprovalProposal,
        IssuedStudyApprovalProposal.SCHEMA: IssuedStudyApprovalProposal,
        RetrospectiveApprovalProposal.SCHEMA: RetrospectiveApprovalProposal,
        FrozenIssuedStudyApprovalProposal.SCHEMA: FrozenIssuedStudyApprovalProposal,
        FrozenRetrospectiveApproval.SCHEMA: FrozenRetrospectiveApproval,
        ApprovalGateAttestation.SCHEMA: ApprovalGateAttestation,
        DurableAuthorizationRecord.SCHEMA: DurableAuthorizationRecord,
        CampaignPackage.SCHEMA: CampaignPackage,
        IssuedCampaignPackage.SCHEMA: IssuedCampaignPackage,
        IssuedStudyPackage.SCHEMA: IssuedStudyPackage,
        RetrospectiveCampaignBase.SCHEMA: RetrospectiveCampaignBase,
        EnvelopeExperimentPackage.SCHEMA: EnvelopeExperimentPackage,
        ExperimentPackage.SCHEMA: ExperimentPackage,
        RetrospectiveCampaignPackage.SCHEMA: RetrospectiveCampaignPackage,
        ElapsedExperimentPackage.SCHEMA: ElapsedExperimentPackage,
        DualLoopPackage.SCHEMA: DualLoopPackage,
        ExplorationExecutionPackage.SCHEMA: ExplorationExecutionPackage,
        ResponseAlgebraProtocolSpec.SCHEMA: ResponseAlgebraProtocolSpec,
        ResponseAlgebraHypothesisFreeze.SCHEMA: ResponseAlgebraHypothesisFreeze,
        ResponseAlgebraConformanceSpec.SCHEMA: ResponseAlgebraConformanceSpec,
        StudyDraft.SCHEMA: StudyDraft,
        RetrospectiveStudyDraft.SCHEMA: RetrospectiveStudyDraft,
        ExperimentEntryPackage.SCHEMA: ExperimentEntryPackage,
        RetrospectiveEntryPackage.SCHEMA: RetrospectiveEntryPackage,
        StudyDefinition.SCHEMA: StudyDefinition,
        RetrospectiveAuthoringBase.SCHEMA: RetrospectiveAuthoringBase,
        ExecutableStudyDefinition.SCHEMA: ExecutableStudyDefinition,
        RetrospectiveAuthoringPackage.SCHEMA: RetrospectiveAuthoringPackage,
        DraftStudyCandidate.SCHEMA: DraftStudyCandidate,
        RetrospectiveStudyCandidate.SCHEMA: RetrospectiveStudyCandidate,
        CandidateCompilationReport.SCHEMA: CandidateCompilationReport,
        RetrospectiveCandidateReport.SCHEMA: RetrospectiveCandidateReport,
        StudyCandidate.SCHEMA: StudyCandidate,
        RetrospectiveStandardCandidate.SCHEMA: RetrospectiveStandardCandidate,
        ExecutableStudyCandidate.SCHEMA: ExecutableStudyCandidate,
        RetrospectiveExtensionCandidate.SCHEMA: RetrospectiveExtensionCandidate,
        StudyCompilationReport.SCHEMA: StudyCompilationReport,
        RetrospectiveStandardReport.SCHEMA: RetrospectiveStandardReport,
        ExecutableStudyCompilationReport.SCHEMA: ExecutableStudyCompilationReport,
        RetrospectiveExtensionReport.SCHEMA: RetrospectiveExtensionReport,
        AccountableHumanIdentity.SCHEMA: AccountableHumanIdentity,
        HumanProposerAttestation.SCHEMA: HumanProposerAttestation,
        RetrospectiveProposerAttestation.SCHEMA: RetrospectiveProposerAttestation,
        ImplementationSourceClosure.SCHEMA: ImplementationSourceClosure,
        StudyOperationAuthority.SCHEMA: StudyOperationAuthority,
        IssuedDraftManifest.SCHEMA: IssuedDraftManifest,
        IssuedStudyManifest.SCHEMA: IssuedStudyManifest,
        RetrospectiveIssuedBase.SCHEMA: RetrospectiveIssuedBase,
        IssuedExecutableStudyManifest.SCHEMA: IssuedExecutableStudyManifest,
        RetrospectiveIssuedStudy.SCHEMA: RetrospectiveIssuedStudy,
        StudyPublicationReceipt.SCHEMA: StudyPublicationReceipt,
        ExtensionPublicationReceipt.SCHEMA: ExtensionPublicationReceipt,
        RetrospectivePublicationReceipt.SCHEMA: RetrospectivePublicationReceipt,
        StudyExtensionDecoderRegistration.SCHEMA: StudyExtensionDecoderRegistration,
        OperatorStorageProfile.SCHEMA: OperatorStorageProfile,
        StudyBundleSpec.SCHEMA: StudyBundleSpec,
        StudyBundleCompilationInput.SCHEMA: StudyBundleCompilationInput,
        StudyBundleCandidate.SCHEMA: StudyBundleCandidate,
        StudyBundleCompilationReport.SCHEMA: StudyBundleCompilationReport,
        MultiWorldStudyChildScientificBinding.SCHEMA: MultiWorldStudyChildScientificBinding,
        ArchiveOutcomeProtectionPlan.SCHEMA: ArchiveOutcomeProtectionPlan,
        MultiWorldOutcomeBarrierPlan.SCHEMA: MultiWorldOutcomeBarrierPlan,
        ArchiveToSimulatorPartialMorphismSpec.SCHEMA: ArchiveToSimulatorPartialMorphismSpec,
        PropertyTransportExpectation.SCHEMA: PropertyTransportExpectation,
        ArchiveOverlapQualification.SCHEMA: ArchiveOverlapQualification,
        MorphismNegativeControlPlan.SCHEMA: MorphismNegativeControlPlan,
        MultiWorldJointAdjudicationPlan.SCHEMA: MultiWorldJointAdjudicationPlan,
        EnvelopeRunPlan.SCHEMA: EnvelopeRunPlan,
        EnvelopeExecutionPlan.SCHEMA: EnvelopeExecutionPlan,
        RunPlan.SCHEMA: RunPlan,
        ResourceBoundRetrospectiveStudyRunPlan.SCHEMA: ResourceBoundRetrospectiveStudyRunPlan,
        ExecutionPlan.SCHEMA: ExecutionPlan,
        SourcePipelineCompilation.SCHEMA: SourcePipelineCompilation,
        LinkedCampaignExecutableCompilation.SCHEMA: LinkedCampaignExecutableCompilation,
        EvidenceProfileSelection.SCHEMA: EvidenceProfileSelection,
        SourcePipelineProfile.SCHEMA: SourcePipelineProfile,
        LinkedCampaignProfile.SCHEMA: LinkedCampaignProfile,
    }
)

CampaignPackageAuthoring = (
    CampaignPackage
    | IssuedCampaignPackage
    | IssuedStudyPackage
    | EnvelopeExperimentPackage
    | ExperimentPackage
    | ElapsedExperimentPackage
)
_CAMPAIGN_PACKAGE_ROOT_SCHEMAS: Mapping[
    str,
    type[CampaignPackageAuthoring],
] = types.MappingProxyType(
    {
        CampaignPackage.SCHEMA: CampaignPackage,
        IssuedCampaignPackage.SCHEMA: IssuedCampaignPackage,
        IssuedStudyPackage.SCHEMA: IssuedStudyPackage,
        RetrospectiveCampaignBase.SCHEMA: RetrospectiveCampaignBase,
        EnvelopeExperimentPackage.SCHEMA: EnvelopeExperimentPackage,
        ExperimentPackage.SCHEMA: ExperimentPackage,
        RetrospectiveCampaignPackage.SCHEMA: RetrospectiveCampaignPackage,
        ElapsedExperimentPackage.SCHEMA: ElapsedExperimentPackage,
    }
)


@dataclass(frozen=True, slots=True)
class LoadedAuthoringMaterialization:
    """One race-checked decode plus the exact raw format/byte identity."""

    record: CanonicalRecord
    media_type: str
    byte_count: int
    raw_materialization_sha256: str
    raw_bytes: bytes = field(repr=False, compare=False)


def _snapshot_root_schemas(
    root_schemas: Mapping[str, type[RecordT]],
) -> Mapping[str, type[RecordT]]:
    """Validate one small, closed, implementation-owned authoring registry."""

    if not isinstance(root_schemas, Mapping):
        raise ValueError("authoring root registry must be a mapping")
    if not root_schemas or len(root_schemas) > MAX_AUTHORING_ROOT_SCHEMAS:
        raise ValueError(
            f"authoring root registry size must be in [1, {MAX_AUTHORING_ROOT_SCHEMAS}]"
        )
    snapshot: dict[str, type[RecordT]] = {}
    for schema, record_type in tuple(root_schemas.items()):
        if (
            not isinstance(schema, str)
            or not isinstance(record_type, type)
            or not issubclass(record_type, CanonicalRecord)
            or not is_dataclass(record_type)
            or schema != record_type.SCHEMA
        ):
            raise ValueError("authoring root registry contains an invalid schema binding")
        snapshot[schema] = record_type
    if len(snapshot) != len(root_schemas):
        raise ValueError("authoring root registry changed during validation")
    return types.MappingProxyType(snapshot)


def _validate_authoring_byte_limit(
    maximum_bytes: int,
    *,
    maximum_allowed_bytes: int = MAX_AUTHORING_BYTES,
) -> int:
    if (
        not isinstance(maximum_bytes, int)
        or isinstance(maximum_bytes, bool)
        or maximum_bytes <= 0
        or maximum_bytes > maximum_allowed_bytes
    ):
        raise ValueError(f"authoring byte limit must be in [1, {maximum_allowed_bytes}]")
    return maximum_bytes


def _mapping(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise AuthoringCodecError(
            f"{where} must be a string-keyed mapping",
            field=where,
        )
    return value


def _decode_decimal(value: object, *, where: str) -> Decimal:
    raw = _mapping(value, where=where)
    if set(raw) != {"decimal"} or not isinstance(raw["decimal"], str):
        raise AuthoringCodecError(
            f"{where} must be an exact tagged decimal",
            field=where,
        )
    try:
        result = Decimal(raw["decimal"])
    except InvalidOperation as error:
        raise AuthoringCodecError(
            f"{where} contains an invalid decimal",
            field=where,
        ) from error
    if not result.is_finite():
        raise AuthoringCodecError(f"{where} decimal must be finite", field=where)
    return result


def _decode(value: object, annotation: object, *, where: str) -> object:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if annotation is Any or annotation is object:
        raise AuthoringCodecError(
            f"{where} has an unbounded authoring type",
            field=where,
        )
    if annotation is type(None):
        if value is not None:
            raise AuthoringCodecError(f"{where} must be null", field=where)
        return None
    if origin in {Union, types.UnionType}:
        errors: list[str] = []
        for option in arguments:
            try:
                return _decode(value, option, where=where)
            except (AuthoringCodecError, TypeError, ValueError) as error:
                errors.append(str(error))
        raise AuthoringCodecError(
            f"{where} does not match its closed union: {errors}",
            field=where,
        )
    if origin is tuple:
        if not isinstance(value, list):
            raise AuthoringCodecError(f"{where} must be a sequence", field=where)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(
                _decode(item, arguments[0], where=f"{where}[{index}]")
                for index, item in enumerate(value)
            )
        if len(value) != len(arguments):
            raise AuthoringCodecError(
                f"{where} has the wrong tuple length",
                field=where,
            )
        return tuple(
            _decode(item, expected, where=f"{where}[{index}]")
            for index, (item, expected) in enumerate(zip(value, arguments, strict=True))
        )
    if origin is frozenset:
        if not isinstance(value, list):
            raise AuthoringCodecError(f"{where} must be a sequence", field=where)
        return frozenset(
            _decode(item, arguments[0], where=f"{where}[{index}]")
            for index, item in enumerate(value)
        )
    if annotation is Decimal:
        return _decode_decimal(value, where=where)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if not isinstance(value, str):
            raise AuthoringCodecError(
                f"{where} enum value must be a string",
                field=where,
            )
        try:
            return annotation(value)
        except ValueError as error:
            raise AuthoringCodecError(
                f"{where} has an unknown enum value",
                field=where,
            ) from error
    if isinstance(annotation, type) and issubclass(annotation, CanonicalRecord):
        return decode_record(value, annotation, where=where)
    if annotation is bool:
        if type(value) is not bool:
            raise AuthoringCodecError(f"{where} must be boolean", field=where)
        return value
    if annotation is int:
        if type(value) is not int:
            raise AuthoringCodecError(f"{where} must be an integer", field=where)
        return value
    if annotation is str:
        if not isinstance(value, str):
            raise AuthoringCodecError(f"{where} must be a string", field=where)
        return value
    raise AuthoringCodecError(
        f"{where} uses unsupported authoring type {annotation!r}",
        field=where,
    )


def decode_record(
    document: object,
    record_type: type[RecordT],
    *,
    where: str = "document",
) -> RecordT:
    if not is_dataclass(record_type):
        raise TypeError("authoring roots must be canonical dataclasses")
    scope = _authoring_parse.get()
    cache_key = (record_type, id(document))
    owned = scope is not None and scope.nodes.owns(document)
    if owned and scope is not None and cache_key in scope.records:
        return cast(RecordT, scope.records[cache_key])
    raw = _mapping(document, where=where)
    names = frozenset(field.name for field in fields(record_type))
    try:
        values = validate_document_shape(
            raw,
            expected_schema=record_type.SCHEMA,
            expected_version=record_type.VERSION,
            field_names=names,
        )
    except (CanonicalizationError, ValueError) as error:
        raise AuthoringCodecError(f"{where}: {error}", field=where) from error
    annotations = record_annotations(record_type)
    decoded = {
        field.name: _decode(
            values[field.name],
            annotations[field.name],
            where=f"{where}.value.{field.name}",
        )
        for field in fields(record_type)
    }
    try:
        result = record_type(**decoded)
        if (
            owned and scope is not None
            and record_type.__dataclass_params__.frozen
            and result._is_cacheable()
        ):
            scope.records[cache_key] = result
        return result
    except (TypeError, ValueError) as error:
        raise AuthoringCodecError(
            f"{where} failed semantic validation: {error}",
            field=where,
        ) from error


def _parse_json(text: str, *, maximum_nodes: int | None = None) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise AuthoringCodecError(f"duplicate JSON mapping key: {key}")
            result[key] = value
        scope = _authoring_parse.get()
        return result if scope is None else scope.nodes.mapping(result)

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_float=lambda _value: (_ for _ in ()).throw(
                AuthoringCodecError("JSON binary numbers are forbidden; use tagged decimals")
            ),
            parse_constant=lambda _value: (_ for _ in ()).throw(
                AuthoringCodecError("non-finite JSON numbers are forbidden")
            ),
        )
    except json.JSONDecodeError as error:
        raise AuthoringCodecError(f"invalid JSON: {error.msg}") from error
    except (MemoryError, RecursionError) as error:
        raise AuthoringCodecError("JSON exceeds the authoring structure limits") from error
    _validate_parsed_structure(value, maximum_nodes=maximum_nodes)
    return value


class _StrictSafeLoader(yaml.SafeLoader):  # type: ignore[misc]
    """Safe YAML loader whose mapping constructor rejects duplicate meaning."""


def _construct_unique_mapping(
    loader: _StrictSafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise AuthoringCodecError("YAML mapping key is not scalar") from error
        if duplicate:
            raise AuthoringCodecError(f"duplicate YAML mapping key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _parse_yaml(text: str, *, maximum_nodes: int | None = None) -> object:
    try:
        for token in yaml.scan(text):
            if isinstance(token, (AliasToken, AnchorToken, TagToken)):
                raise AuthoringCodecError("YAML aliases, anchors and explicit tags are forbidden")
        value = yaml.load(text, Loader=_StrictSafeLoader)
    except yaml.YAMLError as error:
        raise AuthoringCodecError(f"invalid YAML: {error}") from error
    except (MemoryError, RecursionError) as error:
        raise AuthoringCodecError("YAML exceeds the authoring structure limits") from error
    _validate_parsed_structure(value, maximum_nodes=maximum_nodes)
    scope = _authoring_parse.get()
    return value if scope is None else scope.nodes.tree(value)


def _validate_parsed_structure(value: object, *, maximum_nodes: int | None = None) -> None:
    """Bound parsed syntax trees before recursive canonical-record decoding."""

    node_limit = MAX_AUTHORING_NODES if maximum_nodes is None else maximum_nodes
    stack: list[tuple[object, int]] = [(value, 1)]
    observed_nodes = 0
    while stack:
        current, depth = stack.pop()
        observed_nodes += 1
        if observed_nodes > node_limit:
            raise AuthoringCodecError("authoring document exceeds its node limit")
        if depth > MAX_AUTHORING_NESTING:
            raise AuthoringCodecError("authoring document exceeds its nesting limit")
        if isinstance(current, Mapping):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)


def loads_registered_authoring(
    text: str,
    *,
    media_type: str,
    root_schemas: Mapping[str, type[RecordT]],
    maximum_bytes: int = MAX_AUTHORING_BYTES,
) -> RecordT:
    """Decode one root from an explicit closed schema registry.

    The registry is implementation configuration, not document-controlled
    dispatch.  Supplying a smaller registry lets a public operation expose only
    its legitimate authoring roots while retaining the shared strict parser.
    """

    return _loads_registered_authoring(
        text,
        media_type=media_type,
        root_schemas=root_schemas,
        maximum_bytes=maximum_bytes,
        maximum_allowed_bytes=MAX_AUTHORING_BYTES,
    )


def _loads_registered_authoring(
    text: str,
    *,
    media_type: str,
    root_schemas: Mapping[str, type[RecordT]],
    maximum_bytes: int,
    maximum_allowed_bytes: int,
    maximum_nodes: int | None = None,
) -> RecordT:
    token = _authoring_parse.set(_AuthoringParse())
    try:
        schema_registry = _snapshot_root_schemas(root_schemas)
        byte_limit = _validate_authoring_byte_limit(
            maximum_bytes,
            maximum_allowed_bytes=maximum_allowed_bytes,
        )
        try:
            encoded_size = len(text.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise AuthoringCodecError("authoring document must be valid UTF-8 text") from error
        if encoded_size > byte_limit:
            raise AuthoringCodecError("authoring document exceeds its byte limit")
        if media_type == "application/json":
            raw = _parse_json(text, maximum_nodes=maximum_nodes)
        elif media_type in {"application/yaml", "text/yaml"}:
            raw = _parse_yaml(text, maximum_nodes=maximum_nodes)
        else:
            raise AuthoringCodecError(f"unsupported authoring media type: {media_type}")
        document = _mapping(raw, where="document")
        schema = document.get("schema")
        if not isinstance(schema, str) or schema not in schema_registry:
            raise AuthoringCodecError("unsupported authoring root schema")
        record_type = schema_registry[schema]
        if (
            issubclass(record_type, (ProtocolRunPlan, ProtocolExecutionPlan))
            and encoded_size > MAX_RUNTIME_PLAN_JSON_BYTES
        ):
            raise AuthoringCodecError("runtime plan exceeds its byte limit")
        return decode_record(document, record_type)
    finally:
        _authoring_parse.reset(token)


def loads_authoring(text: str, *, media_type: str) -> RootRecord:
    return loads_registered_authoring(
        text,
        media_type=media_type,
        root_schemas=ROOT_SCHEMAS,
    )


def load_registered_authoring(
    path: Path,
    *,
    root_schemas: Mapping[str, type[RecordT]],
    maximum_bytes: int = MAX_AUTHORING_BYTES,
) -> RecordT:
    materialization = load_registered_authoring_materialization(
        path,
        root_schemas=root_schemas,
        maximum_bytes=maximum_bytes,
    )
    return materialization.record  # type: ignore[return-value]


def load_registered_authoring_materialization(
    path: Path,
    *,
    root_schemas: Mapping[str, type[RecordT]],
    maximum_bytes: int = MAX_AUTHORING_BYTES,
) -> LoadedAuthoringMaterialization:
    return _load_registered_authoring_materialization(
        path,
        root_schemas=root_schemas,
        maximum_bytes=maximum_bytes,
        maximum_allowed_bytes=MAX_AUTHORING_BYTES,
    )


def _load_registered_authoring_materialization(
    path: Path,
    *,
    root_schemas: Mapping[str, type[RecordT]],
    maximum_bytes: int,
    maximum_allowed_bytes: int,
    maximum_nodes: int | None = None,
) -> LoadedAuthoringMaterialization:
    schema_registry = _snapshot_root_schemas(root_schemas)
    byte_limit = _validate_authoring_byte_limit(
        maximum_bytes,
        maximum_allowed_bytes=maximum_allowed_bytes,
    )
    if all(
        issubclass(record_type, (ProtocolRunPlan, ProtocolExecutionPlan))
        for record_type in schema_registry.values()
    ):
        byte_limit = min(byte_limit, MAX_RUNTIME_PLAN_JSON_BYTES)
    suffix = path.suffix.casefold()
    if suffix == ".json":
        media_type = "application/json"
    elif suffix in {".yaml", ".yml"}:
        media_type = "application/yaml"
    else:
        raise AuthoringCodecError("authoring file must use .json, .yaml or .yml")
    payload = _read_bounded_authoring_bytes(
        path,
        maximum_bytes=byte_limit,
        maximum_allowed_bytes=maximum_allowed_bytes,
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AuthoringCodecError("authoring document must be UTF-8") from error
    record = _loads_registered_authoring(
        text,
        media_type=media_type,
        root_schemas=schema_registry,
        maximum_bytes=byte_limit,
        maximum_allowed_bytes=maximum_allowed_bytes,
        maximum_nodes=maximum_nodes,
    )
    digest = hashlib.sha256(media_type.encode("ascii") + b"\x00" + payload).hexdigest()
    return LoadedAuthoringMaterialization(
        record=record,
        media_type=media_type,
        byte_count=len(payload),
        raw_materialization_sha256=digest,
        raw_bytes=payload,
    )


def load_authoring(path: Path) -> RootRecord:
    return load_registered_authoring(path, root_schemas=ROOT_SCHEMAS)


def load_campaign_package_authoring(path: Path) -> CampaignPackageAuthoring:
    """Load one closed execution-package root under its composed-control bound."""

    materialization = _load_registered_authoring_materialization(
        path,
        root_schemas=_CAMPAIGN_PACKAGE_ROOT_SCHEMAS,
        maximum_bytes=MAX_CAMPAIGN_PACKAGE_BYTES,
        maximum_allowed_bytes=MAX_CAMPAIGN_PACKAGE_BYTES,
        maximum_nodes=MAX_COMPOSED_CONTROL_NODES,
    )
    return cast(CampaignPackageAuthoring, materialization.record)


def loads_campaign_package_authoring(
    text: str,
    *,
    media_type: str,
) -> CampaignPackageAuthoring:
    """Decode one closed execution-package root under its composed-control bound."""

    return _loads_registered_authoring(
        text,
        media_type=media_type,
        root_schemas=_CAMPAIGN_PACKAGE_ROOT_SCHEMAS,
        maximum_bytes=MAX_CAMPAIGN_PACKAGE_BYTES,
        maximum_allowed_bytes=MAX_CAMPAIGN_PACKAGE_BYTES,
        maximum_nodes=MAX_COMPOSED_CONTROL_NODES,
    )


def load_standard_study_candidate(path: Path) -> ExecutableStudyCandidate:
    """Read a compiled executable study candidate under the existing composed-control bound."""
    materialization = _load_registered_authoring_materialization(
        path,
        root_schemas={
            ExecutableStudyCandidate.SCHEMA: ExecutableStudyCandidate,
            RetrospectiveExtensionCandidate.SCHEMA: RetrospectiveExtensionCandidate,
        },
        maximum_bytes=MAX_COMPOSED_PROGRAMME_CONTROL_BYTES,
        maximum_allowed_bytes=MAX_COMPOSED_PROGRAMME_CONTROL_BYTES,
        maximum_nodes=MAX_COMPOSED_CONTROL_NODES,
    )
    return cast(ExecutableStudyCandidate, materialization.record)


def load_standard_issued_study(path: Path) -> IssuedExecutableStudyManifest:
    """Read only the issued executable study root, preserving the strict authoring parser."""
    materialization = _load_registered_authoring_materialization(
        path,
        root_schemas={
            IssuedExecutableStudyManifest.SCHEMA: IssuedExecutableStudyManifest,
            RetrospectiveIssuedStudy.SCHEMA: RetrospectiveIssuedStudy,
        },
        maximum_bytes=MAX_COMPOSED_PROGRAMME_CONTROL_BYTES,
        maximum_allowed_bytes=MAX_COMPOSED_PROGRAMME_CONTROL_BYTES,
        maximum_nodes=MAX_COMPOSED_CONTROL_NODES,
    )
    return cast(IssuedExecutableStudyManifest, materialization.record)


def _read_bounded_authoring_bytes(
    path: Path,
    *,
    maximum_bytes: int = MAX_AUTHORING_BYTES,
    maximum_allowed_bytes: int = MAX_AUTHORING_BYTES,
) -> bytes:
    byte_limit = _validate_authoring_byte_limit(
        maximum_bytes,
        maximum_allowed_bytes=maximum_allowed_bytes,
    )
    if path.is_symlink():
        raise AuthoringCodecError("authoring document cannot be a symlink")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise AuthoringCodecError("authoring document must be a regular file")
    if before.st_size > byte_limit:
        raise AuthoringCodecError("authoring document exceeds its byte limit")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise AuthoringCodecError("authoring document must be a regular file")
        if _file_identity(before) != _file_identity(opened_before):
            raise AuthoringCodecError("authoring document identity changed before read")
        chunks: list[bytes] = []
        remaining = byte_limit + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.stat()
    if len(payload) > byte_limit:
        raise AuthoringCodecError("authoring document exceeds its byte limit")
    if (
        _file_identity(opened_before) != _file_identity(opened_after)
        or _file_identity(opened_after) != _file_identity(after)
        or len(payload) != opened_after.st_size
    ):
        raise AuthoringCodecError("authoring document changed during bounded read")
    return payload


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
