"""Canonical preparation-policy development screen inputs, readouts, and terminal result."""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, require_sorted_unique_ids
from empirical_lawhood.kernel.status import ScientificStatus

from .law_payloads import FiniteResponseLawLowerPayload
from .preparation_policy_projection import FiniteResponseLawPreparationPolicyProjectionConfig, FiniteResponseLawPreparationPolicyRootPanel
from .preparation_policy_screen import FiniteResponseLawPreparationPolicyUpperPayload, PreparationScreenComputation, _decimals
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_contracts import PREPARATION_POLICY_SCHEDULE_IDS

ORIGINAL_PROSPECTIVE_CLOSEOUT_DECLARED_SCHEMA = "cc1-finite-lawhood-m4-formal-closeout-v1"


@dataclass(frozen=True, slots=True)
class FiniteResponseLawRetainedProspectiveCloseoutReference(CanonicalRecord):
    "Typed reference to the preserved, historically nonconforming finite response-law evaluation JSON."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-retained-prospective-closeout-reference'
    original_artifact_id: str
    original_sha256: str
    original_size_bytes: int
    original_declared_schema: str
    formal_completion: bool
    canonical_adjudication_status: str
    preparation_policy_eligible: bool
    scientific_status: str
    independent_verification: bool
    source_commit: str

    def __post_init__(self) -> None:
        if (
            self.original_artifact_id != "cc1-finite-lawhood-v1.m4.formal-closeout"
            or len(self.original_sha256) != 64
            or self.original_size_bytes <= 0
            or self.original_declared_schema != ORIGINAL_PROSPECTIVE_CLOSEOUT_DECLARED_SCHEMA
            or self.formal_completion is not True
            or self.canonical_adjudication_status != "ADJUDICATED"
            or self.preparation_policy_eligible is not True
            or self.scientific_status != "SUPPORTED"
            or self.independent_verification is not True
            or len(self.source_commit) != 40
        ):
            raise ValueError("Finite response-law evaluation formal closeout reference changes its qualified entry facts")


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationPolicyScreenConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-preparation-policy-screen-config'
    projection: FiniteResponseLawPreparationPolicyProjectionConfig
    lower_artifact: ArtifactIdentity
    lower_identity: ObjectIdentity
    lower_qualification: ArtifactIdentity
    prospective_adjudication: ArtifactIdentity
    prospective_closeout: ArtifactIdentity

    def __post_init__(self) -> None:
        if (
            self.lower_identity.object_schema != FiniteResponseLawLowerPayload.SCHEMA
            or self.lower_artifact.sha256 != self.lower_identity.object_fingerprint
            or self.lower_qualification.artifact_id == self.prospective_adjudication.artifact_id
            or self.prospective_closeout.artifact_id
            in (self.lower_qualification.artifact_id, self.prospective_adjudication.artifact_id)
        ):
            raise ValueError("preparation-policy development screen changes lower package or entry evidence identity")

    @property
    def config_id(self) -> str:
        return f"{self.projection.native_spec.spec_id}.screen"


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationScreenRootReadout(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-preparation-screen-root-readout'
    root_id: str
    adequacy_by_schedule: tuple[bool, ...]
    adequacy_conjuncts: tuple[tuple[bool, ...], ...]
    maximum_error_ratio: tuple[Decimal, ...]
    maximum_width_ratio: tuple[Decimal, ...]
    maximum_numerical_difference: tuple[Decimal, ...]
    maximum_preservation_ratio: tuple[Decimal, ...]
    maximum_work: tuple[Decimal, ...]
    selected_schedule_id: str
    fixed_schedule_id: str
    selected_joint_successes: int
    fixed_joint_successes: int
    joined_selected_successes: int
    joined_fixed_successes: int
    request_count: int = 256

    def __post_init__(self) -> None:
        if (
            len(self.adequacy_by_schedule) != 9
            or len(self.adequacy_conjuncts) != 9
            or any(len(row) != 6 for row in self.adequacy_conjuncts)
            or any(
                len(row) != 9
                for row in (
                    self.maximum_error_ratio,
                    self.maximum_width_ratio,
                    self.maximum_numerical_difference,
                    self.maximum_preservation_ratio,
                    self.maximum_work,
                )
            )
            or self.selected_schedule_id not in PREPARATION_POLICY_SCHEDULE_IDS
            or self.fixed_schedule_id not in PREPARATION_POLICY_SCHEDULE_IDS
            or self.request_count != 256
            or any(
                type(value) is not int or not 0 <= value <= self.request_count
                for value in (
                    self.selected_joint_successes,
                    self.fixed_joint_successes,
                    self.joined_selected_successes,
                    self.joined_fixed_successes,
                )
            )
        ):
            raise ValueError("preparation-policy development root readout changes A/J/C scopes or denominator")


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationScreenResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-preparation-screen-result'
    config: ObjectIdentity
    lower_before: ObjectIdentity
    lower_after: ObjectIdentity
    panels: tuple[ObjectIdentity, ...]
    fit_dependencies: tuple[ObjectIdentity, ...]
    root_readouts: tuple[FiniteResponseLawPreparationScreenRootReadout, ...]
    headroom_roots: int
    adequacy_net_roots: int
    joint_improvement: Decimal
    gates_passed: bool
    upper_package: ObjectIdentity | None
    scientific_status: ScientificStatus
    upper_payload_eligible: bool
    unevaluable_reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_sorted_unique_ids(self.panels, attribute="object_id", field_name="panels")
        require_sorted_unique_ids(
            self.fit_dependencies, attribute="object_id", field_name="fit_dependencies"
        )
        if (
            self.config.object_schema != FiniteResponseLawPreparationPolicyScreenConfig.SCHEMA
            or self.lower_before != self.lower_after
            or self.lower_before.object_schema != FiniteResponseLawLowerPayload.SCHEMA
            or len(self.panels) != 24
            or len(self.root_readouts) not in (0, 24)
            or type(self.gates_passed) is not bool
            or self.upper_payload_eligible != self.gates_passed
            or (self.upper_package is not None) != self.gates_passed
            or self.gates_passed
            and self.scientific_status is not ScientificStatus.SUPPORTED
            or not self.gates_passed
            and self.scientific_status
            not in (ScientificStatus.NOT_SUPPORTED, ScientificStatus.UNEVALUABLE)
            or (self.scientific_status is ScientificStatus.UNEVALUABLE)
            != bool(self.unevaluable_reason_codes)
            or self.scientific_status is ScientificStatus.UNEVALUABLE
            and self.root_readouts
            or self.scientific_status is not ScientificStatus.UNEVALUABLE
            and len(self.root_readouts) != 24
        ):
            raise ValueError("preparation-policy development result changes lower identity, gates, or upper-payload eligibility")

    @property
    def result_id(self) -> str:
        return "finite-response-law.preparation-policy.screen.result"


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationPolicyUpperPayloadFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-preparation-policy-upper-payload-freeze'
    screen_result: ObjectIdentity
    package: FiniteResponseLawPreparationPolicyUpperPayload | None
    disposition: str

    def __post_init__(self) -> None:
        if (
            self.screen_result.object_schema != FiniteResponseLawPreparationScreenResult.SCHEMA
            or (self.package is not None) != (self.disposition == "FROZEN")
            or self.disposition not in ("FROZEN", "GATES_NOT_PASSED", "UNEVALUABLE")
        ):
            raise ValueError("preparation-policy upper freeze changes the preparation-policy development terminal disposition")

    @property
    def freeze_id(self) -> str:
        return "finite-response-law.preparation-policy.upper-freeze"


def freeze_preparation_screen_outputs(
    config: FiniteResponseLawPreparationPolicyScreenConfig,
    panels: tuple[FiniteResponseLawPreparationPolicyRootPanel, ...],
    lower: FiniteResponseLawLowerPayload,
    computation: PreparationScreenComputation,
) -> tuple[FiniteResponseLawPreparationScreenResult, FiniteResponseLawPreparationPolicyUpperPayload | None]:
    config_id = ObjectIdentity.from_record(config.config_id, config)
    panel_ids = tuple(
        sorted(
            (ObjectIdentity.from_record(panel.panel_id, panel) for panel in panels),
            key=lambda value: value.object_id,
        )
    )
    upper = None
    if computation.gates.passed:
        points, widths, provider, q = (
            computation.final_points,
            computation.final_widths,
            computation.final_provider,
            computation.final_q,
        )
        assert points is not None and widths is not None and provider is not None and q is not None
        upper = FiniteResponseLawPreparationPolicyUpperPayload(
            "finite-response-law.upper.preparation-policy.frozen",
            lower.identity,
            config.lower_artifact,
            config_id,
            panel_ids,
            PREPARATION_POLICY_SCHEDULE_IDS,
            PREPARATION_POLICY_SCHEDULE_IDS[provider.best_fixed],
            Decimal(format(q, ".17g")),
            _decimals(points.normalizer.center),
            _decimals(points.normalizer.scale),
            _decimals(points.operators),
            _decimals(widths.base),
            _decimals(widths.multipliers),
            _decimals(provider.operators),
        )
    readouts = []
    a = computation.adequacy
    for index, panel in enumerate(panels):
        selected = int(computation.selected[index])
        fixed = int(computation.fixed[index])
        selected_j = int(computation.selected_joint[index].sum())
        fixed_j = int(computation.fixed_joint[index].sum())
        selected_a = bool(a.event[index, selected])
        fixed_a = bool(a.event[index, fixed])
        readouts.append(
            FiniteResponseLawPreparationScreenRootReadout(
                panel.root.stage_unit,
                tuple(map(bool, a.event[index])),
                tuple(tuple(map(bool, row)) for row in a.conjuncts[index]),
                _decimals(a.maximum_error[index]),
                _decimals(a.maximum_width[index]),
                _decimals(a.maximum_numerical_difference[index]),
                _decimals(a.maximum_preservation_ratio[index]),
                _decimals(a.maximum_work[index]),
                PREPARATION_POLICY_SCHEDULE_IDS[selected],
                PREPARATION_POLICY_SCHEDULE_IDS[fixed],
                selected_j,
                fixed_j,
                selected_j if selected_a else 0,
                fixed_j if fixed_a else 0,
            )
        )
    dependencies = tuple(
        sorted(
            (
                ObjectIdentity.from_record(dependency.object_id, dependency)
                for dependency in computation.dependencies
            ),
            key=lambda value: value.object_id,
        )
    )
    result = FiniteResponseLawPreparationScreenResult(
        config_id,
        lower.identity,
        lower.identity,
        panel_ids,
        dependencies,
        tuple(readouts),
        computation.gates.headroom_roots,
        computation.gates.adequacy_net_roots,
        Decimal(format(computation.gates.joint_improvement, ".17g")),
        computation.gates.passed,
        None if upper is None else upper.identity,
        ScientificStatus.SUPPORTED
        if computation.gates.passed
        else ScientificStatus.NOT_SUPPORTED,
        computation.gates.passed,
    )
    return result, upper


def unevaluable_preparation_screen_outputs(
    config: FiniteResponseLawPreparationPolicyScreenConfig,
    panels: tuple[FiniteResponseLawPreparationPolicyRootPanel, ...],
    lower: FiniteResponseLawLowerPayload,
    reasons: tuple[str, ...],
) -> FiniteResponseLawPreparationScreenResult:
    if not reasons:
        raise ValueError("preparation-policy development unevaluable result requires an explicit scientific reason")
    return FiniteResponseLawPreparationScreenResult(
        ObjectIdentity.from_record(config.config_id, config),
        lower.identity,
        lower.identity,
        tuple(
            sorted(
                (ObjectIdentity.from_record(panel.panel_id, panel) for panel in panels),
                key=lambda value: value.object_id,
            )
        ),
        (),
        (),
        0,
        0,
        Decimal(0),
        False,
        None,
        ScientificStatus.UNEVALUABLE,
        False,
        tuple(sorted(set(reasons))),
    )
