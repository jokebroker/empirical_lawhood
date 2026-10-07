"""Aggregate Six-matrix response design, storage and issued-package contracts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.adapters.control.matrix_response_study.contracts import MatrixResponseOuterStudyAuthoringConfig, MatrixResponseProspectiveEvaluationTopology, ProtectedObservableAccessManifest
from empirical_lawhood.adapters.methods.matrix_response_study.contracts import MatrixResponseLawMethodConfig, MatrixResponsePairedPanelReducerConfig, MatrixResponseRoleEquivarianceConfig, MatrixResponseStructuralFacePlan
from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import SixMatrixResponseArtifactDataset as SixMatrixResponseArtifactDataset, SixMatrixResponseArtifactProfileConfig as SixMatrixResponseArtifactProfileConfig, SixMatrixResponseSixMatrixSourceConfig
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class MatrixResponseCampaignPackageNode(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/matrix-response-study/matrix-response-campaign-package-node'
    VERSION: ClassVar[str] = "1.0.0"

    package_id: str
    parent_package_ids: tuple[str, ...]
    product_id: str
    ordinary_stage_disposition_id: str
    required_extension_schema_ids: tuple[str, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.package_id, field_name="package_id")
        validate_stable_id(self.product_id, field_name="product_id")
        validate_stable_id(
            self.ordinary_stage_disposition_id,
            field_name="ordinary_stage_disposition_id",
        )
        require_sorted_unique_strings(
            self.parent_package_ids,
            field_name="parent_package_ids",
        )
        require_sorted_unique_strings(
            self.required_extension_schema_ids,
            field_name="required_extension_schema_ids",
            allow_empty=False,
        )
        for schema in self.required_extension_schema_ids:
            validate_schema(schema)
        if self.package_id in self.parent_package_ids:
            raise ValueError("Six-matrix response package cannot parent itself")
        if self.grants_authority:
            raise ValueError("campaign-package topology cannot grant authority")


@dataclass(frozen=True, slots=True)
class MatrixResponseCampaignPackageDag(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/matrix-response-study/matrix-response-campaign-package-dag'
    VERSION: ClassVar[str] = "1.0.0"

    dag_id: str
    nodes: tuple[MatrixResponseCampaignPackageNode, ...]
    candidate_role_ids: tuple[str, ...]
    issued_package_schema: str
    run_plan_schema: str
    execution_plan_schema: str
    cross_package_binding_schema: str
    total_catalog_loss_reconstructible: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.dag_id, field_name="dag_id")
        require_sorted_unique_ids(self.nodes, attribute="package_id", field_name="nodes")
        require_sorted_unique_strings(
            self.candidate_role_ids,
            field_name="candidate_role_ids",
            allow_empty=False,
        )
        if len(self.candidate_role_ids) != 4:
            raise ValueError("every Six-matrix response package uses the four candidate-provider roles")
        expected_schemas = (
            self.issued_package_schema,
            self.run_plan_schema,
            self.execution_plan_schema,
            self.cross_package_binding_schema,
        )
        for schema in expected_schemas:
            validate_schema(schema)
        if expected_schemas[:3] != (
            'empirical-lawhood/api/issued-compilation-package-reference',
            'empirical-lawhood/runtime/compiled-run-plan-reference',
            'empirical-lawhood/runtime/frozen-compilation-execution-plan',
        ):
            raise ValueError("Six-matrix response package DAG must use issued-package/run-plan/execution-plan")
        by_id = {value.package_id: value for value in self.nodes}
        if any(parent not in by_id for node in self.nodes for parent in node.parent_package_ids):
            raise ValueError("Six-matrix response package DAG references an absent parent")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(package_id: str) -> None:
            if package_id in visiting:
                raise ValueError("Six-matrix response package graph contains a cycle")
            if package_id in visited:
                return
            visiting.add(package_id)
            for parent in by_id[package_id].parent_package_ids:
                visit(parent)
            visiting.remove(package_id)
            visited.add(package_id)

        for package_id in by_id:
            visit(package_id)
        if not self.total_catalog_loss_reconstructible:
            raise ValueError("Six-matrix response issue topology requires exact catalog-loss reconstruction")


@dataclass(frozen=True, slots=True)
class MatrixResponseResourceEnvelope(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/matrix-response-study/matrix-response-resource-envelope'
    VERSION: ClassVar[str] = "1.0.0"

    envelope_id: str
    physical_cores: int
    worker_processes: int
    blas_threads_per_worker: int
    memory_bytes: int
    external_free_bytes_at_design: int
    maximum_output_bytes: int
    maximum_integration_steps: int
    preferred_wall_seconds: Decimal
    hard_wall_seconds: Decimal
    benchmark_forecast_rule_id: str
    over_ceiling_terminal_id: str
    contraction_allowed: bool

    def __post_init__(self) -> None:
        for name in (
            "envelope_id",
            "benchmark_forecast_rule_id",
            "over_ceiling_terminal_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.physical_cores != 8
            or self.worker_processes != 8
            or self.blas_threads_per_worker != 1
        ):
            raise ValueError("Six-matrix response baseline binds the measured eight-core execution host")
        if (
            min(
                self.memory_bytes,
                self.external_free_bytes_at_design,
                self.maximum_output_bytes,
                self.maximum_integration_steps,
            )
            < 1
        ):
            raise ValueError("Six-matrix response resource bounds must be positive")
        if self.maximum_output_bytes > self.external_free_bytes_at_design:
            raise ValueError("Six-matrix response output ceiling exceeds measured external free space")
        validate_decimal(
            self.preferred_wall_seconds,
            field_name="preferred_wall_seconds",
            minimum=Decimal(0),
        )
        validate_decimal(
            self.hard_wall_seconds,
            field_name="hard_wall_seconds",
            minimum=self.preferred_wall_seconds,
        )
        if self.preferred_wall_seconds != Decimal(64800) or self.hard_wall_seconds != Decimal(
            86400
        ):
            raise ValueError("Six-matrix response freezes the 18 h preferred and 24 h hard ceilings")
        if self.contraction_allowed:
            raise ValueError(
                "Six-matrix response design binding selects hard stop instead of outcome-driven contraction"
            )


@dataclass(frozen=True, slots=True)
class MatrixResponseDesignBinding(CanonicalRecord):
    """Exact Six-matrix response-0 scientific/platform contract; not an issue or authority."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/matrix-response-study/matrix-response-design-binding'
    VERSION: ClassVar[str] = "1.0.0"

    design_id: str
    design_version: str
    plan_id: str
    scientific_parent_sha256: str
    metatheory_contract_sha256: str
    platform_acceptance_commit: str
    source_config: SixMatrixResponseSixMatrixSourceConfig
    role_config: MatrixResponseRoleEquivarianceConfig
    law_method_config: MatrixResponseLawMethodConfig
    structural_plan: MatrixResponseStructuralFacePlan
    study_authoring: MatrixResponseOuterStudyAuthoringConfig
    prospective_topology: MatrixResponseProspectiveEvaluationTopology
    reducer: MatrixResponsePairedPanelReducerConfig
    artifact_profiles: tuple[SixMatrixResponseArtifactProfileConfig, ...]
    protected_access: ProtectedObservableAccessManifest
    package_dag: MatrixResponseCampaignPackageDag
    resource_envelope: MatrixResponseResourceEnvelope
    stable_package_names: tuple[str, ...]
    planned_write_roots: tuple[str, ...]
    external_artifact_root_id: str
    evidence_dependence_id: str
    maximum_claim_id: str
    unresolved_scientific_decisions: tuple[str, ...]
    grants_issue_authority: bool
    grants_execution_authority: bool
    grants_reveal_authority: bool

    def __post_init__(self) -> None:
        for name in (
            "design_id",
            "plan_id",
            "external_artifact_root_id",
            "evidence_dependence_id",
            "maximum_claim_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.design_version)
        validate_sha256(self.scientific_parent_sha256, field_name="scientific_parent_sha256")
        validate_sha256(self.metatheory_contract_sha256, field_name="metatheory_contract_sha256")
        if len(self.platform_acceptance_commit) != 40 or any(
            value not in "0123456789abcdef" for value in self.platform_acceptance_commit
        ):
            raise ValueError("platform acceptance commit must be a full lowercase Git identity")
        require_sorted_unique_ids(
            self.artifact_profiles,
            attribute="config_id",
            field_name="artifact_profiles",
        )
        require_sorted_unique_strings(
            self.stable_package_names,
            field_name="stable_package_names",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.planned_write_roots,
            field_name="planned_write_roots",
            allow_empty=False,
        )
        if self.unresolved_scientific_decisions:
            raise ValueError("SIX_MATRIX_RESPONSE_DESIGN_BOUND cannot retain unresolved scientific decisions")
        if (
            self.prospective_topology.possible_parent_action_word_count
            != self.study_authoring.outer_action_grammar.action_fibre_count
        ):
            raise ValueError("prospective evaluation topology and programme admission authoring use different parent charts")
        if (
            self.source_config.maximum_total_integration_steps
            != self.resource_envelope.maximum_integration_steps
        ):
            raise ValueError("source and resource integration ceilings differ")
        if self.evidence_dependence_id != "six-matrix-response.same-implementation-resample":
            raise ValueError("Six-matrix response cannot claim independent implementation recurrence")
        if any(
            (
                self.grants_issue_authority,
                self.grants_execution_authority,
                self.grants_reveal_authority,
            )
        ):
            raise ValueError("Six-matrix response design binding grants no operational authority")
