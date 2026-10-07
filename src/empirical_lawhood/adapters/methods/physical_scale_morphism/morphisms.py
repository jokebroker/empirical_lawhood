"""Property-indexed direct/composed defect budgets for physical scale morphism morphisms."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    require_unique_ids,
    validate_decimal,
    validate_stable_id,
)

from .contracts import PhysicalScaleMorphismMapFamily, PhysicalScaleMorphismMorphismRecord


PHYSICAL_SCALE_MORPHISM_DEFECT_IDS = (
    "calibration",
    "decision",
    "held-future-semantic",
    "interventional",
    "observational",
)


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismMorphismCompositionContract(CanonicalRecord):
    """Exact ordered map chain frozen before direct/composed scoring."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-morphism-composition-contract'

    contract_id: str
    direct_morphism: ObjectIdentity
    step_morphisms: tuple[ObjectIdentity, ...]
    map_family: PhysicalScaleMorphismMapFamily
    shared_support_ids: tuple[str, ...]
    action_correspondence_compatible: bool
    chronology_compatible: bool
    experiment_chain_compatible: bool
    frozen: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.contract_id, field_name="contract_id")
        if self.direct_morphism.object_schema != PhysicalScaleMorphismMorphismRecord.SCHEMA:
            raise ValueError("composition contract direct map has the wrong schema")
        if len(self.step_morphisms) < 2:
            raise ValueError("composition contract requires at least two ordered steps")
        if any(value.object_schema != PhysicalScaleMorphismMorphismRecord.SCHEMA for value in self.step_morphisms):
            raise ValueError("composition contract step has the wrong schema")
        if len({value.object_id for value in self.step_morphisms}) != len(self.step_morphisms):
            raise ValueError("composition contract repeats a step map")
        require_sorted_unique_strings(
            self.shared_support_ids, field_name="shared_support_ids", allow_empty=False
        )
        if not all(
            (
                self.action_correspondence_compatible,
                self.chronology_compatible,
                self.experiment_chain_compatible,
                self.frozen,
            )
        ):
            raise ValueError("invalid morphism chain cannot be frozen for composition scoring")


def freeze_morphism_composition_contract(
    *,
    contract_id: str,
    direct: PhysicalScaleMorphismMorphismRecord,
    steps: tuple[PhysicalScaleMorphismMorphismRecord, ...],
) -> PhysicalScaleMorphismMorphismCompositionContract:
    """Validate map order, experiment endpoints, support and chronology exactly."""

    require_unique_ids(steps, attribute="morphism_id", field_name="steps")
    if len(steps) < 2:
        raise ValueError("morphism composition requires at least two steps")
    step_ids = tuple(value.morphism_id for value in steps)
    if direct.direct_map_id != direct.morphism_id or direct.composed_map_ids != step_ids:
        raise ValueError("direct morphism does not freeze this exact composed map roster")
    if any(value.map_family is not direct.map_family for value in steps):
        raise ValueError("direct and step morphisms belong to different map families")
    chain_compatible = (
        direct.source_experiment == steps[0].source_experiment
        and direct.target_experiment == steps[-1].target_experiment
        and all(
            left.target_experiment == right.source_experiment
            for left, right in zip(steps, steps[1:])
        )
    )
    if not chain_compatible:
        raise ValueError("morphism experiment endpoints do not compose")
    shared_support = set(direct.support_ids).intersection(
        *(set(value.support_ids) for value in steps)
    )
    if not shared_support or set(direct.support_ids) != shared_support:
        raise ValueError("direct morphism support is not the exact common comparison domain")
    chronology_compatible = (
        len({(value.causal_cutoff_id, value.receiver_window_id) for value in (direct, *steps)}) == 1
    )
    if not chronology_compatible:
        raise ValueError("morphism composition mixes causal cutoffs or receiver windows")
    action_compatible = len({value.action_correspondence_id for value in (direct, *steps)}) == 1
    if not action_compatible:
        raise ValueError("morphism composition mixes action correspondences")
    return PhysicalScaleMorphismMorphismCompositionContract(
        contract_id=contract_id,
        direct_morphism=ObjectIdentity.from_record(direct.morphism_id, direct),
        step_morphisms=tuple(
            ObjectIdentity.from_record(value.morphism_id, value) for value in steps
        ),
        map_family=direct.map_family,
        shared_support_ids=tuple(sorted(shared_support)),
        action_correspondence_compatible=True,
        chronology_compatible=True,
        experiment_chain_compatible=True,
        frozen=True,
    )


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismNamedDefect(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-named-defect'

    defect_id: str
    value: Decimal

    def __post_init__(self) -> None:
        if self.defect_id not in PHYSICAL_SCALE_MORPHISM_DEFECT_IDS:
            raise ValueError("composition panel contains an unknown defect")
        validate_decimal(self.value, field_name="value", minimum=Decimal(0))


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismMapDefectVector(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-map-defect-vector'

    vector_id: str
    morphism_id: str
    defects: tuple[PhysicalScaleMorphismNamedDefect, ...]
    false_safe_cell_ids: tuple[str, ...]
    preserved_role_ids: tuple[str, ...]
    hold_qualified: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.vector_id, field_name="vector_id")
        validate_stable_id(self.morphism_id, field_name="morphism_id")
        if tuple(value.defect_id for value in self.defects) != PHYSICAL_SCALE_MORPHISM_DEFECT_IDS:
            raise ValueError("map defect vector lacks the exact five-defect roster")
        require_sorted_unique_strings(self.false_safe_cell_ids, field_name="false_safe_cell_ids")
        require_sorted_unique_strings(
            self.preserved_role_ids, field_name="preserved_role_ids", allow_empty=False
        )


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismCompositionPanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-composition-panel'

    panel_id: str
    composition_contract: PhysicalScaleMorphismMorphismCompositionContract
    direct: PhysicalScaleMorphismMapDefectVector
    steps: tuple[PhysicalScaleMorphismMapDefectVector, ...]
    direct_composed_defects: tuple[PhysicalScaleMorphismNamedDefect, ...]
    tolerance_by_defect: tuple[PhysicalScaleMorphismNamedDefect, ...]
    excess_by_defect: tuple[PhysicalScaleMorphismNamedDefect, ...]
    role_compatible: bool
    hold_compatible: bool
    composition_supported: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        if self.direct.morphism_id != self.composition_contract.direct_morphism.object_id:
            raise ValueError("composition direct defect vector differs from its frozen map")
        if tuple(value.morphism_id for value in self.steps) != tuple(
            value.object_id for value in self.composition_contract.step_morphisms
        ):
            raise ValueError("composition step defect vectors differ from the frozen map chain")
        require_sorted_unique_ids(self.steps, attribute="vector_id", field_name="steps")
        if len(self.steps) < 2:
            raise ValueError("composition panel requires at least two step maps")
        if tuple(value.defect_id for value in self.direct_composed_defects) != PHYSICAL_SCALE_MORPHISM_DEFECT_IDS:
            raise ValueError("direct/composed comparison lacks the five-defect roster")
        if tuple(value.defect_id for value in self.tolerance_by_defect) != PHYSICAL_SCALE_MORPHISM_DEFECT_IDS:
            raise ValueError("composition tolerance lacks the five-defect roster")
        if tuple(value.defect_id for value in self.excess_by_defect) != PHYSICAL_SCALE_MORPHISM_DEFECT_IDS:
            raise ValueError("composition excess lacks the five-defect roster")
        expected = (
            all(value.value == 0 for value in self.excess_by_defect)
            and not self.direct.false_safe_cell_ids
            and all(not value.false_safe_cell_ids for value in self.steps)
            and self.role_compatible
            and self.hold_compatible
        )
        if self.composition_supported != expected:
            raise ValueError("composition support is not derived from every property budget")


def evaluate_composition(
    *,
    panel_id: str,
    composition_contract: PhysicalScaleMorphismMorphismCompositionContract,
    direct: PhysicalScaleMorphismMapDefectVector,
    steps: tuple[PhysicalScaleMorphismMapDefectVector, ...],
    direct_composed_defects: tuple[PhysicalScaleMorphismNamedDefect, ...],
    tolerance_by_defect: tuple[PhysicalScaleMorphismNamedDefect, ...],
) -> PhysicalScaleMorphismCompositionPanel:
    require_sorted_unique_ids(steps, attribute="vector_id", field_name="steps")
    if len(steps) < 2:
        raise ValueError("composition requires at least two step maps")
    if direct.morphism_id != composition_contract.direct_morphism.object_id:
        raise ValueError("composition direct vector does not match its frozen map")
    if tuple(value.morphism_id for value in steps) != tuple(
        value.object_id for value in composition_contract.step_morphisms
    ):
        raise ValueError("composition step vectors do not match the ordered frozen chain")
    direct_composed_values = {value.defect_id: value.value for value in direct_composed_defects}
    tolerance_values = {value.defect_id: value.value for value in tolerance_by_defect}
    if set(direct_composed_values) != set(PHYSICAL_SCALE_MORPHISM_DEFECT_IDS) or set(tolerance_values) != set(
        PHYSICAL_SCALE_MORPHISM_DEFECT_IDS
    ):
        raise ValueError("composition comparison differs from the frozen defect family")
    if any({value.defect_id for value in step.defects} != set(PHYSICAL_SCALE_MORPHISM_DEFECT_IDS) for step in steps):
        raise ValueError("composition step vector differs from the frozen defect family")
    excess = tuple(
        PhysicalScaleMorphismNamedDefect(
            defect_id,
            max(
                Decimal(0),
                direct_composed_values[defect_id] - tolerance_values[defect_id],
            ),
        )
        for defect_id in PHYSICAL_SCALE_MORPHISM_DEFECT_IDS
    )
    role_sets = [set(value.preserved_role_ids) for value in steps]
    common_roles = role_sets[0].intersection(*role_sets[1:])
    role_compatible = set(direct.preserved_role_ids).issubset(common_roles)
    hold_compatible = direct.hold_qualified and all(value.hold_qualified for value in steps)
    supported = (
        all(value.value == 0 for value in excess)
        and not direct.false_safe_cell_ids
        and all(not value.false_safe_cell_ids for value in steps)
        and role_compatible
        and hold_compatible
    )
    return PhysicalScaleMorphismCompositionPanel(
        panel_id=panel_id,
        composition_contract=composition_contract,
        direct=direct,
        steps=steps,
        direct_composed_defects=direct_composed_defects,
        tolerance_by_defect=tolerance_by_defect,
        excess_by_defect=excess,
        role_compatible=role_compatible,
        hold_compatible=hold_compatible,
        composition_supported=supported,
    )


__all__ = [
    'PhysicalScaleMorphismCompositionPanel',
    'PhysicalScaleMorphismMapDefectVector',
    'PhysicalScaleMorphismMorphismCompositionContract',
    'PhysicalScaleMorphismNamedDefect',
    "PHYSICAL_SCALE_MORPHISM_DEFECT_IDS",
    "evaluate_composition",
    "freeze_morphism_composition_contract",
]
