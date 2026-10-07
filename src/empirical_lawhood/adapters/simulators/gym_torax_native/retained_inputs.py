# SPDX-License-Identifier: MPL-2.0

"""Explicit predecessor identities for deferred historical Gym-TORAX compositions.

These records replace installation-specific defaults. They do not authenticate
custody, grant authority, or make the retained deterministic rosters unexposed.
"""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.adapters.composition.tokamak_control_replication.rosters import (
    GymToraxEpisodeCoordinate,
    GymToraxEpisodeStage,
    GymToraxMatchedQuartet,
    GymToraxQuartetRole,
    GymToraxRosterCell,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class GymToraxImportedProspectiveTopology(CanonicalRecord):
    """Current target topology exported from an explicit original artifact.

    The imported twenty-eight-cell parent differs from the current maximal
    roster. Its source and export references are custody inputs supplied by the
    caller; this record authenticates no external artifact or qualification.
    """
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/tokamak-control-replication/imported-prospective-topology'
    roster_id: str
    source_original: ObjectIdentity
    export_receipt: ObjectIdentity
    evaluation_primary_quartets: tuple[GymToraxMatchedQuartet, ...]
    prospective_cells: tuple[GymToraxRosterCell, ...]
    episode_coordinates: tuple[GymToraxEpisodeCoordinate, ...]
    frozen: bool
    outcome_access: OutcomeAccess
    source_qualification_transferred: bool = False
    execution_authority_transferred: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.roster_id, field_name="roster_id")
        if not isinstance(self.source_original, ObjectIdentity) or not isinstance(self.export_receipt, ObjectIdentity):
            raise ValueError("imported topology requires explicit original and external export identities")
        if self.export_receipt.object_schema != 'empirical-lawhood/composition/tokamak-control-replication/prospective-topology-export-receipt':
            raise ValueError("original topology requires its explicit external export receipt")
        if self.source_qualification_transferred or self.execution_authority_transferred:
            raise ValueError("original topology export cannot transfer qualification or authority")
        if not self.frozen or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("imported prospective topology must remain frozen and outcome-blind")
        require_sorted_unique_ids(self.evaluation_primary_quartets, attribute="quartet_id", field_name="evaluation_primary_quartets")
        require_sorted_unique_ids(self.prospective_cells, attribute="cell_id", field_name="prospective_cells")
        require_sorted_unique_ids(self.episode_coordinates, attribute="coordinate_id", field_name="episode_coordinates")
        if len(self.evaluation_primary_quartets) != 18 or any(
            quartet.role is not GymToraxQuartetRole.EVALUATION_PRIMARY
            for quartet in self.evaluation_primary_quartets
        ):
            raise ValueError("imported topology requires the original eighteen primary matched quartets")
        if len(self.prospective_cells) != 28:
            raise ValueError("imported recurrence topology requires its distinct original twenty-eight prospective cells")
        if sum(episode.stage is GymToraxEpisodeStage.MATCHED_EVALUATION_PRIMARY for episode in self.episode_coordinates) != 576:
            raise ValueError("imported topology requires its original five-hundred-seventy-six primary views")


@dataclass(frozen=True, slots=True)
class GymToraxPredecessorInputs(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-predecessor-inputs'
    protocol: ObjectIdentity
    excluded_feasibility: ObjectIdentity
    diagnostic_freeze: ObjectIdentity
    diagnostic_terminal: ObjectIdentity
    source_roster: ObjectIdentity
    hfr_contract_evidence: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        for name, schema in (
            (
                "protocol",
                'empirical-lawhood/composition/tokamak-control-replication/prospective-protocol-parent',
            ),
            (
                "excluded_feasibility",
                'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-excluded-feasibility-spec',
            ),
            ("diagnostic_freeze", 'empirical-lawhood/composition/tokamak-control-replication/excluded-diagnostic-freeze-parent'),
            (
                "diagnostic_terminal",
                'empirical-lawhood/composition/tokamak-control-replication/excluded-diagnostic-terminal-parent',
            ),
            ("source_roster", GymToraxImportedProspectiveTopology.SCHEMA),
        ):
            if getattr(self, name).object_schema != schema:
                raise ValueError(f"Gym-TORAX predecessor schema differs: {name}")
        require_sorted_unique_ids(
            self.hfr_contract_evidence,
            attribute="object_id",
            field_name="hfr_contract_evidence",
        )
        if len(self.hfr_contract_evidence) != 4 or {
            item.object_schema for item in self.hfr_contract_evidence
        } != {
            'empirical-lawhood/composition/tokamak-control-replication/finite-action-conformance-parent',
            'empirical-lawhood/planning/implementation-source-closure',
            'empirical-lawhood/planning/finite-action-occurrence-qualification-template-set',
            'empirical-lawhood/planning/finite-action-recurrence-assignment-template',
        }:
            raise ValueError("Gym-TORAX finite-action recurrence contract closure differs")


@dataclass(frozen=True, slots=True)
class GymToraxRecoveryParents(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-recovery-parents'
    freeze: ObjectIdentity
    qualification: ObjectIdentity
    terminal: ObjectIdentity
    assignment: ObjectIdentity
    law_qualification: ObjectIdentity


@dataclass(frozen=True, slots=True)
class GymToraxEvaluationParents(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-evaluation-parents'
    acquisition_freeze: ObjectIdentity
    acquisition_terminal: ObjectIdentity
    qualification: ObjectIdentity
