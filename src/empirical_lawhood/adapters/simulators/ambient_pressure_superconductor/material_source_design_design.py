'Deterministic ambient pressure superconductor material source design design-basis construction.\n\nThis module contains no solver execution and consumes no candidate outcomes.\nIt turns the narrow, public, outcome-blind design decisions into canonical\nadapter records whose fingerprints can be frozen before development atlas/sealed prospective contact.\n'

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
from typing import Final

from empirical_lawhood.kernel.serialization import canonical_json_bytes

from .material_source_design_contracts import ActionStage, CalibrationAlgorithmFreeze, ExplorationDesignFreeze, FormationRouteSpec, MaterialActionSpec, MaterialFamilySpec, MaterialGateFreeze, MaterialRosterFreeze, MaterialSiteSpec, MaterialStructureSpec, MethodFormalismFreeze, RouteEvidenceStatus, ScienceDesignFreeze, SearchPolicySpec, SolverViewFreeze, SplitPartition, TruthWorldLock


D = Decimal


def _sorted_ids(*values: str) -> tuple[str, ...]:
    return tuple(sorted(values))


def _site(index: int, element: str, xyz: tuple[str, str, str]) -> MaterialSiteSpec:
    return MaterialSiteSpec(
        site_id=f'site.{index:02d}-{element.lower()}',
        element=element,
        fractional_coordinates=tuple(D(value) for value in xyz),  # type: ignore[arg-type]
        occupancy=D("1"),
    )


def _structure(
    *,
    name: str,
    formula: tuple[tuple[str, int], ...],
    family: str,
    partition: SplitPartition,
    prototype: str,
    space_group: int,
    lattice: tuple[tuple[str, str, str], ...],
    atoms: tuple[tuple[str, tuple[str, str, str]], ...],
    process_history: str,
    visibility: str,
) -> MaterialStructureSpec:
    return MaterialStructureSpec(
        structure_id=f'structure.{name}',
        candidate_id=f'candidate.{name}',
        formula=tuple(sorted(formula)),
        family_id=family,
        partition=partition,
        prototype_id=prototype,
        space_group_number=space_group,
        lattice_vectors_A=tuple(  # type: ignore[arg-type]
            tuple(D(value) for value in vector) for vector in lattice
        ),
        sites=tuple(
            sorted(
                (_site(index, element, xyz) for index, (element, xyz) in enumerate(atoms)),
                key=lambda value: value.site_id,
            )
        ),
        charge_state="neutral",
        magnetic_state="nonmagnetic-initialization",
        source_mode="source.generated-ordered-prototype",
        process_history_id=process_history,
        outcome_visibility=visibility,
    )


_FCC = (("0", "2.475", "2.475"), ("2.475", "0", "2.475"), ("2.475", "2.475", "0"))
_FCC_CU = (("0", "1.8075", "1.8075"), ("1.8075", "0", "1.8075"), ("1.8075", "1.8075", "0"))
_DIAMOND = (("0", "1.7835", "1.7835"), ("1.7835", "0", "1.7835"), ("1.7835", "1.7835", "0"))


def _cubic(a: str) -> tuple[tuple[str, str, str], ...]:
    return ((a, "0", "0"), ("0", a, "0"), ("0", "0", a))


def _fcc_primitive(half_a: str) -> tuple[tuple[str, str, str], ...]:
    return (
        ("0", half_a, half_a),
        (half_a, "0", half_a),
        (half_a, half_a, "0"),
    )


def _bcc_primitive(half_a: str) -> tuple[tuple[str, str, str], ...]:
    return (
        (f'-{half_a}', half_a, half_a),
        (half_a, f'-{half_a}', half_a),
        (half_a, half_a, f'-{half_a}'),
    )


def _hexagonal(a: str, half_a: str, root3_half_a: str, c: str) -> tuple[tuple[str, str, str], ...]:
    return ((a, "0", "0"), (f'-{half_a}', root3_half_a, "0"), ("0", "0", c))


def _b1_atoms(a: str, b: str) -> tuple[tuple[str, tuple[str, str, str]], ...]:
    return ((a, ("0", "0", "0")), (b, ("0.5", "0.5", "0.5")))


def _b2_atoms(a: str, b: str) -> tuple[tuple[str, tuple[str, str, str]], ...]:
    return _b1_atoms(a, b)


def _zincblende_atoms(a: str, b: str) -> tuple[tuple[str, tuple[str, str, str]], ...]:
    return ((a, ("0", "0", "0")), (b, ("0.25", "0.25", "0.25")))


def _a15_atoms(a: str, b: str) -> tuple[tuple[str, tuple[str, str, str]], ...]:
    return (
        (b, ("0", "0", "0")),
        (b, ("0.5", "0.5", "0.5")),
        (a, ("0", "0.5", "0.25")),
        (a, ("0", "0.5", "0.75")),
        (a, ("0.25", "0", "0.5")),
        (a, ("0.75", "0", "0.5")),
        (a, ("0.5", "0.25", "0")),
        (a, ("0.5", "0.75", "0")),
    )


def _l12_atoms(a: str, b: str) -> tuple[tuple[str, tuple[str, str, str]], ...]:
    return (
        (b, ("0", "0", "0")),
        (a, ("0", "0.5", "0.5")),
        (a, ("0.5", "0", "0.5")),
        (a, ("0.5", "0.5", "0")),
    )


def _family(
    family_id: str,
    partition: SplitPartition,
    prototype: str,
    domain: str,
    seeds: tuple[str, ...],
) -> MaterialFamilySpec:
    return MaterialFamilySpec(
        family_id=family_id,
        partition=partition,
        prototype_id=prototype,
        chemical_domain_id=domain,
        mechanism_lane_id="mechanism.conventional-electron-phonon",
        grouping_key=f"group.prototype-derivative.{family_id.removeprefix('family.')}",
        seed_structure_ids=tuple(sorted(seeds)),
        coefficient_pooling_allowed=False,
    )


def _route(
    name: str,
    route_class: str,
    p_upper: str,
    t_lower: str,
    t_upper: str,
    duration: str,
    atmosphere: str,
    equipment: str,
    status: RouteEvidenceStatus = RouteEvidenceStatus.PROPOSAL_ONLY,
) -> FormationRouteSpec:
    return FormationRouteSpec(
        route_id=f'route.{name}',
        process_history_id=f'history.{name}',
        route_class_id=route_class,
        evidence_status=status,
        pressure_lower_Pa=D("0"),
        pressure_upper_Pa=D(p_upper),
        temperature_lower_K=D(t_lower),
        temperature_upper_K=D(t_upper),
        duration_upper_s=D(duration),
        atmosphere_id=atmosphere,
        equipment_class_id=equipment,
        evidence_source_ids=(),
    )


def _action(
    name: str,
    parent: str,
    child: str,
    site_class: str,
    before: str,
    after: str,
    stage: ActionStage,
    route: str,
    *,
    action_cost: int = 1,
    compute_cost: int = 2,
    bridge: bool = False,
    kind: str = "action.ordered-site-substitution",
) -> MaterialActionSpec:
    return MaterialActionSpec(
        action_id=f'action.{name}',
        parent_structure_id=f'structure.{parent}',
        child_structure_id=f'structure.{child}',
        action_kind=kind,
        site_class_id=site_class,
        from_species=before,
        to_species=after,
        fraction=D("1"),
        stage=stage,
        route_id=f'route.{route}',
        action_cost_units=action_cost,
        compute_cost_units=compute_cost,
        discontinuous_bridge=bridge,
    )


def build_material_roster() -> MaterialRosterFreeze:
    'Construct the finite calibration/development atlas/sealed prospective roster without reading outcomes.'

    calibration_partition = (
        _structure(
            name='calibration-pb-fcc',
            formula=(("Pb", 1),),
            family='family.calibration-pb-fcc',
            partition=SplitPartition.CALIBRATION,
            prototype="prototype.fcc-a1",
            space_group=225,
            lattice=_FCC,
            atoms=(("Pb", ("0", "0", "0")),),
            process_history='history.calibration-public-bulk',
            visibility="visibility.public-control",
        ),
        _structure(
            name='calibration-mgb2-alb2',
            formula=(("B", 2), ("Mg", 1)),
            family='family.calibration-mgb2-alb2',
            partition=SplitPartition.CALIBRATION,
            prototype="prototype.alb2-c32",
            space_group=191,
            lattice=_hexagonal("3.083", "1.5415", "2.670023", "3.521"),
            atoms=(
                ("Mg", ("0", "0", "0")),
                ("B", ("0.333333333333", "0.666666666667", "0.5")),
                ("B", ("0.666666666667", "0.333333333333", "0.5")),
            ),
            process_history='history.calibration-public-bulk',
            visibility="visibility.public-control",
        ),
        _structure(
            name='calibration-cu-fcc',
            formula=(("Cu", 1),),
            family='family.calibration-cu-fcc',
            partition=SplitPartition.CALIBRATION,
            prototype="prototype.fcc-a1",
            space_group=225,
            lattice=_FCC_CU,
            atoms=(("Cu", ("0", "0", "0")),),
            process_history='history.calibration-public-bulk',
            visibility="visibility.public-control",
        ),
        _structure(
            name='calibration-c-diamond',
            formula=(("C", 2),),
            family='family.calibration-c-diamond',
            partition=SplitPartition.CALIBRATION,
            prototype="prototype.diamond-a4",
            space_group=227,
            lattice=_DIAMOND,
            atoms=(("C", ("0", "0", "0")), ("C", ("0.25", "0.25", "0.25"))),
            process_history='history.calibration-public-bulk',
            visibility="visibility.public-control",
        ),
        _structure(
            name='calibration-pb-sc-compressed',
            formula=(("Pb", 1),),
            family='family.calibration-pb-sc-invalid',
            partition=SplitPartition.CALIBRATION,
            prototype="prototype.simple-cubic-ah",
            space_group=221,
            lattice=_cubic("2.5"),
            atoms=(("Pb", ("0", "0", "0")),),
            process_history='history.calibration-hypothetical-compression',
            visibility="visibility.truth-known-control",
        ),
    )

    # Five independently grouped development families.  Every structure is an
    # ordered, integer-occupancy prototype generated from public crystallographic
    # definitions; no experimental or predicted target value enters this list.
    development_partition = (
        _structure(
            name='development-atlas-b1-nbc',
            formula=(("C", 1), ("Nb", 1)),
            family='family.development-atlas-b1-carbide',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.b1",
            space_group=225,
            lattice=_fcc_primitive("2.235"),
            atoms=_b1_atoms("Nb", "C"),
            process_history="history.carbide-sinter",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-b1-tac',
            formula=(("C", 1), ("Ta", 1)),
            family='family.development-atlas-b1-carbide',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.b1",
            space_group=225,
            lattice=_fcc_primitive("2.23"),
            atoms=_b1_atoms("Ta", "C"),
            process_history="history.carbide-sinter",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-b1-vc',
            formula=(("C", 1), ("V", 1)),
            family='family.development-atlas-b1-carbide',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.b1",
            space_group=225,
            lattice=_fcc_primitive("2.085"),
            atoms=_b1_atoms("V", "C"),
            process_history="history.carbide-sinter",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-b1-tic',
            formula=(("C", 1), ("Ti", 1)),
            family='family.development-atlas-b1-carbide',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.b1",
            space_group=225,
            lattice=_fcc_primitive("2.165"),
            atoms=_b1_atoms("Ti", "C"),
            process_history="history.carbide-sinter",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-a15-nb3sn',
            formula=(("Nb", 6), ("Sn", 2)),
            family='family.development-atlas-a15-intermetallic',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.a15",
            space_group=223,
            lattice=_cubic("5.29"),
            atoms=_a15_atoms("Nb", "Sn"),
            process_history="history.arc-melt-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-a15-v3si',
            formula=(("Si", 2), ("V", 6)),
            family='family.development-atlas-a15-intermetallic',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.a15",
            space_group=223,
            lattice=_cubic("4.72"),
            atoms=_a15_atoms("V", "Si"),
            process_history="history.arc-melt-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-a15-nb3ge',
            formula=(("Ge", 2), ("Nb", 6)),
            family='family.development-atlas-a15-intermetallic',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.a15",
            space_group=223,
            lattice=_cubic("5.17"),
            atoms=_a15_atoms("Nb", "Ge"),
            process_history="history.arc-melt-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-a15-nb3al',
            formula=(("Al", 2), ("Nb", 6)),
            family='family.development-atlas-a15-intermetallic',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.a15",
            space_group=223,
            lattice=_cubic("5.19"),
            atoms=_a15_atoms("Nb", "Al"),
            process_history="history.arc-melt-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-a15-v3ge',
            formula=(("Ge", 2), ("V", 6)),
            family='family.development-atlas-a15-intermetallic',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.a15",
            space_group=223,
            lattice=_cubic("4.79"),
            atoms=_a15_atoms("V", "Ge"),
            process_history="history.arc-melt-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-l12-ti3al',
            formula=(("Al", 1), ("Ti", 3)),
            family='family.development-atlas-l12-aluminide',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.l12",
            space_group=221,
            lattice=_cubic("4.05"),
            atoms=_l12_atoms("Ti", "Al"),
            process_history="history.arc-melt-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-l12-ti3si',
            formula=(("Si", 1), ("Ti", 3)),
            family='family.development-atlas-l12-aluminide',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.l12",
            space_group=221,
            lattice=_cubic("4.00"),
            atoms=_l12_atoms("Ti", "Si"),
            process_history="history.arc-melt-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-l12-ti3sn',
            formula=(("Sn", 1), ("Ti", 3)),
            family='family.development-atlas-l12-aluminide',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.l12",
            space_group=221,
            lattice=_cubic("4.20"),
            atoms=_l12_atoms("Ti", "Sn"),
            process_history="history.arc-melt-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-l12-al3ti',
            formula=(("Al", 3), ("Ti", 1)),
            family='family.development-atlas-l12-aluminide',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.l12",
            space_group=221,
            lattice=_cubic("3.97"),
            atoms=_l12_atoms("Al", "Ti"),
            process_history="history.arc-melt-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-l12-nb3al',
            formula=(("Al", 1), ("Nb", 3)),
            family='family.development-atlas-l12-aluminide',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.l12",
            space_group=221,
            lattice=_cubic("4.12"),
            atoms=_l12_atoms("Nb", "Al"),
            process_history="history.arc-melt-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-zb-sic',
            formula=(("C", 1), ("Si", 1)),
            family='family.development-atlas-zincblende-covalent',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.zincblende-b3",
            space_group=216,
            lattice=_fcc_primitive("2.18"),
            atoms=_zincblende_atoms("Si", "C"),
            process_history="history.cvd-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-zb-gec',
            formula=(("C", 1), ("Ge", 1)),
            family='family.development-atlas-zincblende-covalent',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.zincblende-b3",
            space_group=216,
            lattice=_fcc_primitive("2.26"),
            atoms=_zincblende_atoms("Ge", "C"),
            process_history="history.cvd-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-zb-bn',
            formula=(("B", 1), ("N", 1)),
            family='family.development-atlas-zincblende-covalent',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.zincblende-b3",
            space_group=216,
            lattice=_fcc_primitive("1.81"),
            atoms=_zincblende_atoms("B", "N"),
            process_history="history.cvd-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-zb-sige',
            formula=(("Ge", 1), ("Si", 1)),
            family='family.development-atlas-zincblende-covalent',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.zincblende-b3",
            space_group=216,
            lattice=_fcc_primitive("2.77"),
            atoms=_zincblende_atoms("Si", "Ge"),
            process_history="history.cvd-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-wz-sic',
            formula=(("C", 2), ("Si", 2)),
            family='family.development-atlas-zincblende-covalent',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.wurtzite-b4",
            space_group=186,
            lattice=_hexagonal("3.08", "1.54", "2.66736", "5.05"),
            atoms=(
                ("Si", ("0", "0", "0")),
                ("Si", ("0.666666666667", "0.333333333333", "0.5")),
                ("C", ("0", "0", "0.375")),
                ("C", ("0.666666666667", "0.333333333333", "0.875")),
            ),
            process_history="history.cvd-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-b2-val',
            formula=(("Al", 1), ("V", 1)),
            family='family.development-atlas-b2-aluminide',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.b2",
            space_group=221,
            lattice=_cubic("3.03"),
            atoms=_b2_atoms("V", "Al"),
            process_history="history.arc-melt-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-b2-tial',
            formula=(("Al", 1), ("Ti", 1)),
            family='family.development-atlas-b2-aluminide',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.b2",
            space_group=221,
            lattice=_cubic("3.18"),
            atoms=_b2_atoms("Ti", "Al"),
            process_history="history.arc-melt-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-b2-nbal',
            formula=(("Al", 1), ("Nb", 1)),
            family='family.development-atlas-b2-aluminide',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.b2",
            space_group=221,
            lattice=_cubic("3.29"),
            atoms=_b2_atoms("Nb", "Al"),
            process_history="history.arc-melt-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-b2-taal',
            formula=(("Al", 1), ("Ta", 1)),
            family='family.development-atlas-b2-aluminide',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.b2",
            space_group=221,
            lattice=_cubic("3.27"),
            atoms=_b2_atoms("Ta", "Al"),
            process_history="history.arc-melt-anneal",
            visibility="visibility.development-prefix",
        ),
        _structure(
            name='development-atlas-b2-vsi',
            formula=(("Si", 1), ("V", 1)),
            family='family.development-atlas-b2-aluminide',
            partition=SplitPartition.DEVELOPMENT_ATLAS,
            prototype="prototype.b2",
            space_group=221,
            lattice=_cubic("2.98"),
            atoms=_b2_atoms("V", "Si"),
            process_history="history.arc-melt-anneal",
            visibility="visibility.development-prefix",
        ),
    )

    prospective_partition = (
        _structure(
            name='sealed-prospective-bcc-nb',
            formula=(("Nb", 1),),
            family='family.sealed-prospective-bcc-group-v',
            partition=SplitPartition.SEALED_PROSPECTIVE,
            prototype="prototype.bcc-a2",
            space_group=229,
            lattice=_bcc_primitive("1.65"),
            atoms=(("Nb", ("0", "0", "0")),),
            process_history="history.arc-melt-anneal",
            visibility="visibility.sealed-target",
        ),
        _structure(
            name='sealed-prospective-bcc-v',
            formula=(("V", 1),),
            family='family.sealed-prospective-bcc-group-v',
            partition=SplitPartition.SEALED_PROSPECTIVE,
            prototype="prototype.bcc-a2",
            space_group=229,
            lattice=_bcc_primitive("1.515"),
            atoms=(("V", ("0", "0", "0")),),
            process_history="history.arc-melt-anneal",
            visibility="visibility.sealed-target",
        ),
        _structure(
            name='sealed-prospective-bcc-ta',
            formula=(("Ta", 1),),
            family='family.sealed-prospective-bcc-group-v',
            partition=SplitPartition.SEALED_PROSPECTIVE,
            prototype="prototype.bcc-a2",
            space_group=229,
            lattice=_bcc_primitive("1.655"),
            atoms=(("Ta", ("0", "0", "0")),),
            process_history="history.arc-melt-anneal",
            visibility="visibility.sealed-target",
        ),
        _structure(
            name='sealed-prospective-bcc-nb-strain',
            formula=(("Nb", 1),),
            family='family.sealed-prospective-bcc-group-v',
            partition=SplitPartition.SEALED_PROSPECTIVE,
            prototype="prototype.bcc-a2",
            space_group=229,
            lattice=_bcc_primitive("1.635"),
            atoms=(("Nb", ("0", "0", "0")),),
            process_history="history.arc-melt-anneal",
            visibility="visibility.sealed-target",
        ),
    )

    families = (
        _family(
            'family.calibration-c-diamond',
            SplitPartition.CALIBRATION,
            "prototype.diamond-a4",
            "domain.elemental-covalent-control",
            ('structure.calibration-c-diamond',),
        ),
        _family(
            'family.calibration-cu-fcc',
            SplitPartition.CALIBRATION,
            "prototype.fcc-a1",
            "domain.normal-metal-control",
            ('structure.calibration-cu-fcc',),
        ),
        _family(
            'family.calibration-mgb2-alb2',
            SplitPartition.CALIBRATION,
            "prototype.alb2-c32",
            "domain.boride-control",
            ('structure.calibration-mgb2-alb2',),
        ),
        _family(
            'family.calibration-pb-fcc',
            SplitPartition.CALIBRATION,
            "prototype.fcc-a1",
            "domain.elemental-superconductor-control",
            ('structure.calibration-pb-fcc',),
        ),
        _family(
            'family.calibration-pb-sc-invalid',
            SplitPartition.CALIBRATION,
            "prototype.simple-cubic-ah",
            "domain.unstable-hypothetical-control",
            ('structure.calibration-pb-sc-compressed',),
        ),
        _family(
            'family.development-atlas-a15-intermetallic',
            SplitPartition.DEVELOPMENT_ATLAS,
            "prototype.a15",
            "domain.a15-intermetallic",
            ('structure.development-atlas-a15-nb3sn',),
        ),
        _family(
            'family.development-atlas-b1-carbide',
            SplitPartition.DEVELOPMENT_ATLAS,
            "prototype.b1",
            "domain.refractory-carbide",
            ('structure.development-atlas-b1-nbc',),
        ),
        _family(
            'family.development-atlas-b2-aluminide',
            SplitPartition.DEVELOPMENT_ATLAS,
            "prototype.b2",
            "domain.b2-transition-metal-aluminide",
            ('structure.development-atlas-b2-val',),
        ),
        _family(
            'family.development-atlas-l12-aluminide',
            SplitPartition.DEVELOPMENT_ATLAS,
            "prototype.l12",
            "domain.l12-intermetallic",
            ('structure.development-atlas-l12-ti3al',),
        ),
        _family(
            'family.development-atlas-zincblende-covalent',
            SplitPartition.DEVELOPMENT_ATLAS,
            "prototype.zincblende-derived",
            "domain.covalent-ab",
            ('structure.development-atlas-zb-sic',),
        ),
        _family(
            'family.sealed-prospective-bcc-group-v',
            SplitPartition.SEALED_PROSPECTIVE,
            "prototype.bcc-a2",
            "domain.group-v-elemental",
            ('structure.sealed-prospective-bcc-nb',),
        ),
    )

    routes = (
        _route(
            "arc-melt-anneal",
            "route-class.bulk-melt-anneal",
            "101325",
            "300",
            "2000",
            "604800",
            "atmosphere.inert",
            "equipment.arc-furnace",
        ),
        _route(
            "carbide-sinter",
            "route-class.solid-state-sinter",
            "1000000000",
            "300",
            "2000",
            "604800",
            "atmosphere.inert",
            "equipment.hot-press",
        ),
        _route(
            "cvd-anneal",
            "route-class.vapor-deposition",
            "101325",
            "300",
            "1800",
            "172800",
            "atmosphere.controlled-reactive",
            "equipment.cvd-reactor",
        ),
    )

    initial = (
        _action(
            'development-atlas-a15-nb3sn-to-nb3ge',
            'development-atlas-a15-nb3sn',
            'development-atlas-a15-nb3ge',
            "site.a15-b",
            "Sn",
            "Ge",
            ActionStage.INITIAL_DEVELOPMENT,
            "arc-melt-anneal",
        ),
        _action(
            'development-atlas-a15-nb3sn-to-v3si',
            'development-atlas-a15-nb3sn',
            'development-atlas-a15-v3si',
            "site.a15-all",
            "Nb3Sn",
            "V3Si",
            ActionStage.INITIAL_DEVELOPMENT,
            "arc-melt-anneal",
        ),
        _action(
            'development-atlas-b1-nbc-to-tac',
            'development-atlas-b1-nbc',
            'development-atlas-b1-tac',
            "site.b1-metal",
            "Nb",
            "Ta",
            ActionStage.INITIAL_DEVELOPMENT,
            "carbide-sinter",
        ),
        _action(
            'development-atlas-b1-nbc-to-vc',
            'development-atlas-b1-nbc',
            'development-atlas-b1-vc',
            "site.b1-metal",
            "Nb",
            "V",
            ActionStage.INITIAL_DEVELOPMENT,
            "carbide-sinter",
        ),
        _action(
            'development-atlas-b2-val-to-nbal',
            'development-atlas-b2-val',
            'development-atlas-b2-nbal',
            "site.b2-metal",
            "V",
            "Nb",
            ActionStage.INITIAL_DEVELOPMENT,
            "arc-melt-anneal",
        ),
        _action(
            'development-atlas-b2-val-to-tial',
            'development-atlas-b2-val',
            'development-atlas-b2-tial',
            "site.b2-metal",
            "V",
            "Ti",
            ActionStage.INITIAL_DEVELOPMENT,
            "arc-melt-anneal",
        ),
        _action(
            'development-atlas-l12-ti3al-to-ti3si',
            'development-atlas-l12-ti3al',
            'development-atlas-l12-ti3si',
            "site.l12-corner",
            "Al",
            "Si",
            ActionStage.INITIAL_DEVELOPMENT,
            "arc-melt-anneal",
        ),
        _action(
            'development-atlas-l12-ti3al-to-ti3sn',
            'development-atlas-l12-ti3al',
            'development-atlas-l12-ti3sn',
            "site.l12-corner",
            "Al",
            "Sn",
            ActionStage.INITIAL_DEVELOPMENT,
            "arc-melt-anneal",
        ),
        _action(
            'development-atlas-zb-sic-to-bn',
            'development-atlas-zb-sic',
            'development-atlas-zb-bn',
            "site.zb-all",
            "SiC",
            "BN",
            ActionStage.INITIAL_DEVELOPMENT,
            "cvd-anneal",
        ),
        _action(
            'development-atlas-zb-sic-to-gec',
            'development-atlas-zb-sic',
            'development-atlas-zb-gec',
            "site.zb-cation",
            "Si",
            "Ge",
            ActionStage.INITIAL_DEVELOPMENT,
            "cvd-anneal",
        ),
    )
    eligible = (
        _action(
            'development-atlas-a15-nb3ge-to-v3ge',
            'development-atlas-a15-nb3ge',
            'development-atlas-a15-v3ge',
            "site.a15-chain",
            "Nb",
            "V",
            ActionStage.ELIGIBLE_WAVE,
            "arc-melt-anneal",
        ),
        _action(
            'development-atlas-a15-nb3sn-to-nb3al',
            'development-atlas-a15-nb3sn',
            'development-atlas-a15-nb3al',
            "site.a15-b",
            "Sn",
            "Al",
            ActionStage.ELIGIBLE_WAVE,
            "arc-melt-anneal",
        ),
        _action(
            'development-atlas-b1-nbc-to-tic',
            'development-atlas-b1-nbc',
            'development-atlas-b1-tic',
            "site.b1-metal",
            "Nb",
            "Ti",
            ActionStage.ELIGIBLE_WAVE,
            "carbide-sinter",
        ),
        _action(
            'development-atlas-b1-vc-to-tac',
            'development-atlas-b1-vc',
            'development-atlas-b1-tac',
            "site.b1-metal",
            "V",
            "Ta",
            ActionStage.ELIGIBLE_WAVE,
            "carbide-sinter",
        ),
        _action(
            'development-atlas-b2-val-to-taal',
            'development-atlas-b2-val',
            'development-atlas-b2-taal',
            "site.b2-metal",
            "V",
            "Ta",
            ActionStage.ELIGIBLE_WAVE,
            "arc-melt-anneal",
        ),
        _action(
            'development-atlas-b2-val-to-vsi',
            'development-atlas-b2-val',
            'development-atlas-b2-vsi',
            "site.b2-anion",
            "Al",
            "Si",
            ActionStage.ELIGIBLE_WAVE,
            "arc-melt-anneal",
        ),
        _action(
            'development-atlas-l12-ti3al-to-al3ti',
            'development-atlas-l12-ti3al',
            'development-atlas-l12-al3ti',
            "site.l12-sublattice-swap",
            "Ti3Al",
            "Al3Ti",
            ActionStage.ELIGIBLE_WAVE,
            "arc-melt-anneal",
            bridge=True,
            action_cost=2,
            compute_cost=3,
            kind="action.ordered-sublattice-swap",
        ),
        _action(
            'development-atlas-l12-ti3al-to-nb3al',
            'development-atlas-l12-ti3al',
            'development-atlas-l12-nb3al',
            "site.l12-face",
            "Ti",
            "Nb",
            ActionStage.ELIGIBLE_WAVE,
            "arc-melt-anneal",
        ),
        _action(
            'development-atlas-zb-bn-to-sige',
            'development-atlas-zb-bn',
            'development-atlas-zb-sige',
            "site.zb-all",
            "BN",
            "SiGe",
            ActionStage.ELIGIBLE_WAVE,
            "cvd-anneal",
            bridge=True,
            action_cost=2,
            compute_cost=3,
        ),
        _action(
            'development-atlas-zb-sic-to-wz-sic',
            'development-atlas-zb-sic',
            'development-atlas-wz-sic',
            "site.polytype",
            "B3",
            "B4",
            ActionStage.ELIGIBLE_WAVE,
            "cvd-anneal",
            bridge=True,
            action_cost=2,
            compute_cost=3,
            kind="action.polytype-bridge",
        ),
    )
    prospective = (
        _action(
            'sealed-prospective-bcc-nb-strain',
            'sealed-prospective-bcc-nb',
            'sealed-prospective-bcc-nb-strain',
            "site.homogeneous-cell",
            "Nb",
            "Nb",
            ActionStage.SEALED_PROSPECTIVE,
            "arc-melt-anneal",
            kind="action.isotropic-compressive-strain",
        ),
        _action(
            'sealed-prospective-bcc-nb-to-ta',
            'sealed-prospective-bcc-nb',
            'sealed-prospective-bcc-ta',
            "site.bcc-element",
            "Nb",
            "Ta",
            ActionStage.SEALED_PROSPECTIVE,
            "arc-melt-anneal",
        ),
        _action(
            'sealed-prospective-bcc-nb-to-v',
            'sealed-prospective-bcc-nb',
            'sealed-prospective-bcc-v',
            "site.bcc-element",
            "Nb",
            "V",
            ActionStage.SEALED_PROSPECTIVE,
            "arc-melt-anneal",
        ),
    )
    actions = tuple(sorted(initial + eligible + prospective, key=lambda value: value.action_id))
    return MaterialRosterFreeze(
        roster_id='roster.ambient-pressure-superconductor-material-source-design-finite',
        families=tuple(sorted(families, key=lambda value: value.family_id)),
        structures=tuple(sorted(calibration_partition + development_partition + prospective_partition, key=lambda value: value.structure_id)),
        actions=actions,
        formation_routes=tuple(sorted(routes, key=lambda value: value.route_id)),
        calibration_structure_ids=tuple(sorted(value.structure_id for value in calibration_partition)),
        development_initial_action_ids=tuple(sorted(value.action_id for value in initial)),
        development_eligible_wave_action_ids=tuple(sorted(value.action_id for value in eligible)),
        prospective_action_ids=tuple(sorted(value.action_id for value in prospective)),
        grouping_algorithm_id='algorithm.composition-prototype-derivative-connected-components',
        canonicalization_algorithm_id='algorithm.ordered-fractional-structure-canonical-json',
        duplicate_action_child_groups=(
            (
                'structure.development-atlas-b1-tac',
                _sorted_ids('action.development-atlas-b1-nbc-to-tac', 'action.development-atlas-b1-vc-to-tac'),
            ),
        ),
        unresolved_disorder_count=0,
        unresolved_partial_occupancy_count=0,
        family_partition_collision_count=0,
        target_outcome_values_used=False,
    )


_WORLD_KINDS: Final = (
    ("complete-empty", "case.complete-empty-graph", "hold.empty-intersection"),
    ("disconnected-target", "case.disconnected-target-island", "hold.no-reachable-path"),
    ("fidelity-reversal", "case.low-high-fidelity-reversal", "disposition.reversal-recovered"),
    ("narrow-corridor", "case.narrow-connected-corridor", "disposition.corridor-recovered"),
    (
        "requested-realized-mismatch",
        "case.requested-realized-mismatch",
        "hold.realization-mismatch",
    ),
    (
        "scalar-deceptive-ridge",
        "case.scalar-tc-deceptive-ridge",
        "disposition.deceptive-ridge-rejected",
    ),
)


def truth_world_documents(name: str) -> tuple[dict[str, object], dict[str, object]]:
    """Return the public selector view and separately privileged fixture truth."""

    public = {
        "schema": 'ambient-pressure-superconductor-truth-world-public',
        "world": name,
        "nodes": ("seed", "a", "b", "c", "target"),
        "edges": (
            {"action": "a", "cost": 1, "family": "f1", "parent": "seed"},
            {"action": "b", "cost": 1, "family": "f1", "parent": "seed"},
            {"action": "c", "cost": 2, "family": "f2", "parent": "a"},
            {"action": "target", "cost": 2, "family": "f2", "parent": "c"},
        ),
        "selector_outcomes": "withheld-until-commit",
    }
    truths: dict[str, dict[str, object]] = {
        "narrow-corridor": {
            "admissible": ("a", "c", "target"),
            "boundary": ("b",),
            "reachable": True,
        },
        "scalar-deceptive-ridge": {
            "admissible": ("a", "c"),
            "scalar_best": "b",
            "false_gate": "stability",
            "reachable": True,
        },
        "disconnected-target": {
            "admissible": ("target",),
            "disconnected": ("target",),
            "reachable": False,
        },
        "fidelity-reversal": {
            "coarse_positive": ("b",),
            "refined_positive": ("a", "c"),
            "admissible": ("c",),
            "reachable": True,
        },
        "requested-realized-mismatch": {
            "requested": "a",
            "realized": "b",
            "admissible": (),
            "reachable": False,
        },
        "complete-empty": {"admissible": (), "reachable": False},
    }
    truth = {
        "schema": 'ambient-pressure-superconductor-truth-world-privileged',
        "world": name,
        **truths[name],
    }
    return public, truth


def build_exploration_design(roster: MaterialRosterFreeze) -> ExplorationDesignFreeze:
    policies = (
        SearchPolicySpec(
            policy_id="policy.response-guided",
            method_id="method.nondominated-response-boundary-information-ordering",
            implementation_id='implementation.ambient-pressure-superconductor-response-selector',
            visible_input_ids=_sorted_ids(
                "input.compatibility",
                "input.declared-cost",
                "input.eligible-graph",
                "input.own-prefix-history",
                "input.provisional-family-atlas",
            ),
            forbidden_input_ids=_sorted_ids(
                "input.other-policy-history",
                'input.sealed-sealed-prospective-outcome',
                "input.unpublished-solver-log",
            ),
            ordering_rule_id="ordering.unresolved-overlap-sign-boundary-margin",
            tie_breaker_id="tie.canonical-family-action",
            hold_rule_id="hold.no-valid-budget-feasible-nomination",
        ),
        SearchPolicySpec(
            policy_id="policy.scalar-predicted-tc",
            method_id="method.scalar-predicted-tc-after-identical-prechecks",
            implementation_id='implementation.ambient-pressure-superconductor-scalar-comparator',
            visible_input_ids=_sorted_ids(
                "input.declared-cost",
                "input.eligible-graph",
                "input.own-prefix-history",
                "input.scalar-predicted-tc",
            ),
            forbidden_input_ids=_sorted_ids(
                "input.other-policy-history", 'input.sealed-sealed-prospective-outcome', "input.true-target-tc"
            ),
            ordering_rule_id="ordering.descending-predicted-tc-after-prechecks",
            tie_breaker_id="tie.canonical-family-action",
            hold_rule_id="hold.no-valid-budget-feasible-nomination",
        ),
        SearchPolicySpec(
            policy_id="policy.stratified-random",
            method_id="method.seeded-stratified-prototype-enumeration",
            implementation_id='implementation.ambient-pressure-superconductor-stratified-random',
            visible_input_ids=_sorted_ids(
                "input.declared-cost",
                "input.deterministic-seed",
                "input.eligible-graph",
                "input.own-prefix-history",
            ),
            forbidden_input_ids=_sorted_ids(
                "input.other-policy-history",
                "input.predicted-or-true-tc",
                'input.sealed-sealed-prospective-outcome',
            ),
            ordering_rule_id="ordering.seeded-family-stratified-permutation",
            tie_breaker_id="tie.canonical-family-action",
            hold_rule_id="hold.no-valid-budget-feasible-nomination",
        ),
    )
    worlds = []
    for name, case_kind, expected in _WORLD_KINDS:
        public, truth = truth_world_documents(name)
        worlds.append(
            TruthWorldLock(
                world_id=f'world.{name}',
                case_kind_id=case_kind,
                public_graph_sha256=sha256(canonical_json_bytes(public)).hexdigest(),
                privileged_truth_sha256=sha256(canonical_json_bytes(truth)).hexdigest(),
                expected_disposition_id=expected,
                selector_outcome_access="access.public-prefix-only",
            )
        )
    eligible = tuple(
        action
        for action in roster.actions
        if action.stage in {ActionStage.INITIAL_DEVELOPMENT, ActionStage.ELIGIBLE_WAVE}
    )
    return ExplorationDesignFreeze(
        design_id='design.ambient-pressure-superconductor-material-source-design-matched-policy',
        eligible_action_graph_sha256=sha256(canonical_json_bytes(eligible)).hexdigest(),
        policy_specs=tuple(sorted(policies, key=lambda value: value.policy_id)),
        truth_worlds=tuple(sorted(worlds, key=lambda value: value.world_id)),
        wave_count=2,
        actions_per_policy_wave=2,
        policy_action_budget=4,
        policy_compute_budget_units=8,
        family_restart_quota=1,
        bridge_action_quota_per_wave=1,
        initial_history_rule_id="history.identical-seed-prefix-per-policy",
        family_diversity_rule_id="diversity.one-family-restart-and-no-duplicate-child",
        cost_accounting_rule_id="cost.full-charge-per-policy-nomination-union-reuse-hidden",
        union_reuse_rule_id="reuse.compute-once-charge-each-policy-no-history-leakage",
        primary_estimand_ids=_sorted_ids(
            "estimand.boundary-recovery",
            "estimand.cost-to-first-complete-survivor",
            "estimand.direction-sign-calibration",
            "estimand.distinct-family-survivors",
            "estimand.false-action-rate",
            "estimand.false-admission-rate",
            "estimand.hold-specificity",
        ),
        advantage_rule_id="advantage.strict-paired-all-five-families-zero-false-admission",
        advantage_confidence_level=D("0.95"),
        advantage_minimum_relative_cost_reduction=D("0.10"),
        independent_family_unit_count=5,
        target_outcome_values_used=False,
    )


def _solver_view(
    name: str,
    role: str,
    functional: str,
    pseudo_set: str,
    k: tuple[int, int, int],
    q: tuple[int, int, int],
    fine_k: tuple[int, int, int],
    fine_q: tuple[int, int, int],
    smearing: str,
    precision: str,
    promotes: bool,
) -> SolverViewFreeze:
    return SolverViewFreeze(
        view_id=f'view.{name}',
        role_id=role,
        solver_id="solver.quantum-espresso-epw",
        solver_version="QE 7.6 + EPW 6.1 local-derived environment",
        functional=functional,
        pseudopotential_set_id=pseudo_set,
        k_mesh=k,
        q_mesh=q,
        fine_k_mesh=fine_k,
        fine_q_mesh=fine_q,
        smearing_Ry=D(smearing),
        precision=precision,
        model_closure_ids=_sorted_ids(
            "closure.harmonic-dfpt",
            "closure.migdal-eliashberg-conventional",
            "closure.norm-conserving-or-paw-sssp",
        ),
        promotes_positive_claim=promotes,
    )


_GATE_ROWS: Final = (
    (
        "authority",
        "authority",
        ("operand.authority-record",),
        "threshold.exact-authority-present",
        "reason.authority-required",
        True,
    ),
    (
        "dynamic-sink",
        "dynamics",
        ("operand.phonon-and-mechanical-stability",),
        "threshold.calibration-derived-phonon-stability",
        "reason.dynamically-unstable",
        True,
    ),
    (
        "effort",
        "effort",
        ("operand.compute-and-process-cost",),
        "threshold.frozen-resource-and-process-envelope",
        "reason.effort-exceeded",
        True,
    ),
    (
        "identity",
        "observation-validity",
        ("operand.canonical-material-identity",),
        "threshold.exact-identity-closure",
        "reason.material-identity-unresolved",
        True,
    ),
    (
        "material-constraints",
        "physical-sink",
        ("operand.hazard-scarcity-process-tolerance",),
        "threshold.frozen-manufacturing-domain",
        "reason.material-constraint-failed",
        True,
    ),
    (
        "meissner-receiver",
        "target",
        ("operand.finite-slab-shielding-interval",),
        "threshold.calibration-derived-receiver-tolerance",
        "reason.receiver-target-failed",
        True,
    ),
    (
        "metallic-state",
        "observation-validity",
        ("operand.fermi-level-density-of-states",),
        "threshold.positive-metallicity-interval",
        "reason.mechanism-state-failed",
        True,
    ),
    (
        "order-at-300k",
        "target",
        ("operand.gap-order-interval-300k",),
        "threshold.positive-lower-bound",
        "reason.order-not-supported",
        True,
    ),
    (
        "preservation",
        "baseline-preservation",
        ("operand.probe-and-time-preservation",),
        "threshold.ambient-lifetime-and-probe-envelope",
        "reason.preservation-failed",
        True,
    ),
    (
        "support",
        "atlas-native",
        ("operand.family-local-atlas-support",),
        "threshold.exact-supported-cell",
        "reason.out-of-support",
        True,
    ),
    (
        "synthesis-reachability",
        "reachability",
        ("operand.precursor-process-receipt-grid",),
        "threshold.evidence-backed-reachable-path",
        "reason.synthesis-path-required",
        True,
    ),
    (
        "target-tc",
        "target",
        ("operand.conservative-tc-interval",),
        "threshold.lower-bound-strictly-above-300k",
        "reason.tc-target-failed",
        True,
    ),
    (
        "thermodynamic-sink",
        "physical-sink",
        ("operand.decomposition-and-hull-interval",),
        "threshold.calibration-derived-ambient-sink",
        "reason.decomposition-sink",
        True,
    ),
    (
        "transverse-response",
        "target",
        ("operand.gauge-closed-transverse-stiffness",),
        "threshold.positive-gauge-consistent-lower-bound",
        "reason.transverse-operand-required",
        True,
    ),
    (
        "uncertainty",
        "uncertainty",
        ("operand.common-base-refined-independent-interval",),
        "threshold.calibration-derived-view-agreement",
        "reason.view-local-only",
        True,
    ),
    (
        "validity",
        "observation-validity",
        ("operand.convergence-sum-rule-weak-field-domain",),
        "threshold.all-model-validity-checks",
        "reason.model-validity-failed",
        True,
    ),
)


def build_science_design() -> ScienceDesignFreeze:
    views = (
        _solver_view(
            "pbe-efficiency-base",
            "role.primary-screen",
            "PBE",
            "set.sssp130-pbe-efficiency",
            (12, 12, 12),
            (4, 4, 4),
            (36, 36, 36),
            (12, 12, 12),
            "0.02",
            "base",
            False,
        ),
        _solver_view(
            "pbe-precision-refined",
            "role.refined-confirmation",
            "PBE",
            "set.sssp130-pbe-precision",
            (18, 18, 18),
            (6, 6, 6),
            (54, 54, 54),
            (18, 18, 18),
            "0.01",
            "refined",
            True,
        ),
        _solver_view(
            "pbesol-precision-independent",
            "role.independent-recurrence",
            "PBEsol",
            "set.sssp130-pbesol-precision",
            (18, 18, 18),
            (6, 6, 6),
            (54, 54, 54),
            (18, 18, 18),
            "0.01",
            "independent",
            True,
        ),
    )
    controls = _sorted_ids(
        'structure.calibration-c-diamond',
        'structure.calibration-cu-fcc',
        'structure.calibration-mgb2-alb2',
        'structure.calibration-pb-fcc',
        'structure.calibration-pb-sc-compressed',
    )
    calibrations = (
        CalibrationAlgorithmFreeze(
            "calibration.cross-view-common-interval",
            "calibration-derived.cross-view-agreement",
            controls,
            "estimator.max-paired-control-disagreement-plus-numerical-halfwidth",
            "rounding.outward-three-significant-digits",
            "native-operand-unit",
            "state.unevaluable-if-control-missing",
            True,
        ),
        CalibrationAlgorithmFreeze(
            "calibration.numerical-noise",
            "calibration-derived.numerical-noise",
            controls,
            "estimator.max-exact-repeat-and-base-refined-residual",
            "rounding.upward-three-significant-digits",
            "native-operand-unit",
            "state.unevaluable-if-control-missing",
            True,
        ),
        CalibrationAlgorithmFreeze(
            "calibration.phonon-exception",
            "calibration-derived.phonon-stability-tolerance",
            controls,
            "estimator.max-stable-control-imaginary-frequency-plus-noise",
            "rounding.upward-three-significant-digits",
            "cm-1",
            "state.no-exception-if-control-missing",
            True,
        ),
        CalibrationAlgorithmFreeze(
            "calibration.receiver-tolerance",
            "calibration-derived.receiver-tolerance",
            controls,
            "estimator.max-normal-and-insulator-absolute-shielding-residual",
            "rounding.upward-three-significant-digits",
            "dimensionless",
            "state.unevaluable-if-control-missing",
            True,
        ),
    )
    gates = tuple(
        MaterialGateFreeze(
            gate_id=f'gate.material.{name}',
            generic_gate_id=f'generic.{generic}',
            required_operand_ids=tuple(sorted(operands)),
            threshold_rule_id=threshold,
            failure_reason_id=reason,
            support_required=support,
        )
        for name, generic, operands, threshold, reason, support in _GATE_ROWS
    )
    formalisms = (
        MethodFormalismFreeze(
            formalism_id="formalism.full-spectrum-xi-stability",
            source_ids=('source.ambient-pressure-superconductor.semenok-2407-12922v2', 'source.ambient-pressure-superconductor.xi-code-v1-1'),
            applicability_ids=_sorted_ids(
                "applicability.good-metal",
                "applicability.migdal-eliashberg",
                "applicability.phonon-mediated",
            ),
            equation_id="equation.xi-max-t-integral-g-alpha2f-over-omega",
            promotion_role_id="role.hard-necessary-stability-gate",
            hard_rule_id="rule.upper-xi-interval-strictly-less-than-one",
            diagnostic_rule_ids=("diagnostic.xi-c-half-nonuniversal-engineering-margin",),
            nonuniversal_parameter_ids=("parameter.xi-c-equals-one-half",),
        ),
        MethodFormalismFreeze(
            formalism_id="formalism.strong-coupling-frequency-bound",
            source_ids=('source.ambient-pressure-superconductor.semenok-2407-12922v2',),
            applicability_ids=_sorted_ids(
                "applicability.good-metal",
                "applicability.migdal-eliashberg",
                "applicability.phonon-mediated",
            ),
            equation_id="equation.tc-less-than-0p18-sqrt-lambda-omega2",
            promotion_role_id="role.assumption-tagged-screening-bound",
            hard_rule_id="rule.never-promote-without-full-spectrum-and-receiver-gates",
            diagnostic_rule_ids=_sorted_ids(
                "diagnostic.tc-less-than-0p20-omega-max-at-xi-half",
                "diagnostic.tc-less-than-0p32-omega-max",
            ),
            nonuniversal_parameter_ids=("parameter.xi-c-equals-one-half",),
        ),
        MethodFormalismFreeze(
            formalism_id="formalism.mcmillan-low-coupling-diagnostic",
            source_ids=('source.ambient-pressure-superconductor.gao-2502-18281v1',),
            applicability_ids=_sorted_ids(
                "applicability.lambda-at-most-one-point-five", "applicability.phonon-mediated"
            ),
            equation_id="equation.mcmillan-tc-lambda-mustar-omega-log",
            promotion_role_id="role.nonpromoting-diagnostic",
            hard_rule_id="rule.outside-applicability-is-unevaluable-not-fail",
            diagnostic_rule_ids=("diagnostic.lambda-omega-log-tradeoff",),
            nonuniversal_parameter_ids=("parameter.coulomb-pseudopotential",),
        ),
        MethodFormalismFreeze(
            formalism_id="formalism.superfluid-weight-receiver-bridge",
            source_ids=('source.plan.ambient-pressure-superconductor-literature-method-integration',),
            applicability_ids=_sorted_ids(
                "applicability.material-linked-gap-available",
                "applicability.temperature-extrapolation-declared",
            ),
            equation_id="equation.superfluid-weight-tensor-conventional-plus-geometric",
            promotion_role_id='role.nonpromoting-material-linked-receiver-receiver-screen',
            hard_rule_id='rule.material-linked-receiver-cannot-fill-missing-gauge-covariant-response-gauge-closure',
            diagnostic_rule_ids=_sorted_ids(
                "diagnostic.local-nonlocal-correction", "diagnostic.penetration-depth-tensor"
            ),
            nonuniversal_parameter_ids=_sorted_ids(
                "parameter.sample-quality", "parameter.strong-coupling-correction"
            ),
        ),
        MethodFormalismFreeze(
            formalism_id="formalism.gauge-closed-transverse-kubo",
            source_ids=('source.plan.uniform-electron-gas-transverse-receiver-screen',),
            applicability_ids=_sorted_ids(
                "applicability.finite-q-weak-field", "applicability.normal-comparator-available"
            ),
            equation_id="equation.kt-odd-current-over-vector-potential",
            promotion_role_id='role.required-admission-receiver-operand',
            hard_rule_id="rule.positive-common-stiffness-with-ward-and-normal-cancellation",
            diagnostic_rule_ids=("diagnostic.penetration-depth-from-stiffness",),
            nonuniversal_parameter_ids=(),
        ),
        MethodFormalismFreeze(
            formalism_id="formalism.finite-slab-receiver-admittance",
            source_ids=('source.plan.uniform-electron-gas-transverse-receiver-screen',),
            applicability_ids=_sorted_ids(
                "applicability.linear-meissner-regime",
                "applicability.slab-thickness-one-millimeter",
            ),
            equation_id="equation.slab-field-cosh-penetration-profile",
            promotion_role_id="role.finite-receiver-map",
            hard_rule_id="rule.probe-at-most-one-tenth-conservative-hc1",
            diagnostic_rule_ids=("diagnostic.geometry-resolution-sensitivity",),
            nonuniversal_parameter_ids=(),
        ),
    )
    return ScienceDesignFreeze(
        design_id='design.ambient-pressure-superconductor-material-source-design-science',
        operating_temperature_K=D("300"),
        operating_pressure_Pa=D("101325"),
        pressure_tolerance_Pa=D("1000"),
        probe_field_T=D("0.0001"),
        probe_field_fraction_of_hc1_upper=D("0.1"),
        slab_thickness_m=D("0.001"),
        required_ambient_lifetime_s=D("86400"),
        maximum_process_pressure_Pa=D("1000000000"),
        maximum_process_temperature_K=D("2000"),
        maximum_process_duration_s=D("604800"),
        mechanism_lane_id="mechanism.conventional-electron-phonon",
        physical_independent_unit_id="unit.one-realized-material-preparation",
        solver_views=tuple(sorted(views, key=lambda value: value.view_id)),
        calibration_algorithms=tuple(sorted(calibrations, key=lambda value: value.algorithm_id)),
        material_gates=tuple(sorted(gates, key=lambda value: value.gate_id)),
        method_formalisms=tuple(sorted(formalisms, key=lambda value: value.formalism_id)),
        compatibility_map_ids=_sorted_ids(
            "compatibility.design-to-realization",
            "compatibility.material-action-to-native-word",
            "compatibility.material-gates-to-nine-generic-gates",
            "compatibility.response-law-to-receiver-admittance",
        ),
        receiver_level_ids=_sorted_ids(
            'receiver.counterfactual', 'receiver.material-linked', 'receiver.gauge-covariant'
        ),
        receiver_survivor_limit=2,
        refinement_rule_ids=_sorted_ids(
            "refine.coarse-support-only",
            "refine.fidelity-precommitted",
            "refine.no-repeat-to-favorable-answer",
            "refine.no-physical-gate-already-failed",
            "refine.within-budget",
        ),
        stop_rule_ids=_sorted_ids(
            "stop.any-mandatory-gate-fail-hold",
            'stop.calibration-control-failure-before-development-atlas',
            "stop.environment-closure-incomplete",
            'stop.no-transverse-operand-at-admission',
            "stop.source-custody-failure",
            "stop.truth-world-conformance-failure",
        ),
        only_calibration_derived_ids=_sorted_ids(
            "calibration-derived.cross-view-agreement",
            "calibration-derived.numerical-noise",
            "calibration-derived.phonon-stability-tolerance",
            "calibration-derived.receiver-tolerance",
        ),
        target_outcome_values_used=False,
    )


__all__ = [
    "build_exploration_design",
    "build_material_roster",
    "build_science_design",
    "truth_world_documents",
]
