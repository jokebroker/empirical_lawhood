"""Identity-only obstruction atlas construction over compact terminals."""

from __future__ import annotations

from dataclasses import dataclass

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.metatheory import LegitimateNextAct, MetatheoryCellDisposition, MetatheoryObstructionKind
from empirical_lawhood.planning.obstruction_atlas import ObstructionAtlasBuildStopKind, ObstructionAtlasBuildStop, ObstructionAtlasSpec, ObstructionAtlas, ObstructionCell, ObstructionClaimEffect, ObstructionSourceBinding


_EFFECT_RANK = {
    ObstructionClaimEffect.NO_SCIENTIFIC_EFFECT: 0,
    ObstructionClaimEffect.PREREQUISITE_NONATTEMPT: 1,
    ObstructionClaimEffect.UNEVALUABLE: 2,
    ObstructionClaimEffect.OPPOSES: 3,
}


@dataclass(frozen=True, slots=True)
class ObstructionAtlasBuilder:
    def build(
        self,
        *,
        spec: ObstructionAtlasSpec,
        sources: tuple[ObstructionSourceBinding, ...],
    ) -> ObstructionAtlas | ObstructionAtlasBuildStop:
        spec_identity = ObjectIdentity.from_record(spec.spec_id, spec)
        source_refs = tuple(
            sorted(
                (ObjectIdentity.from_record(value.source_id, value) for value in sources),
                key=lambda value: value.object_id,
            )
        )
        source_ids = [value.source_id for value in sources]
        expected_ids = [value.expected_cell_id for value in sources]
        if len(set(source_ids)) != len(source_ids) or len(set(expected_ids)) != len(expected_ids):
            return self._stop(
                spec_identity,
                source_refs,
                ObstructionAtlasBuildStopKind.INPUT_INCOMPATIBLE,
                ("OBSTRUCTION_SOURCE_OR_EXPECTED_CELL_REPEATED",),
            )
        if any(
            value.terminal.object_schema not in spec.accepted_terminal_schemas for value in sources
        ):
            return self._stop(
                spec_identity,
                source_refs,
                ObstructionAtlasBuildStopKind.INPUT_INCOMPATIBLE,
                ("OBSTRUCTION_TERMINAL_SCHEMA_NOT_ACCEPTED",),
            )
        expected = {value.expected_cell_id: value for value in spec.expected_cells}
        if any(
            value.expected_cell_id not in expected
            or value.stage is not expected[value.expected_cell_id].stage
            or value.target_id != expected[value.expected_cell_id].target_id
            for value in sources
        ):
            return self._stop(
                spec_identity,
                source_refs,
                ObstructionAtlasBuildStopKind.INPUT_INCOMPATIBLE,
                ("OBSTRUCTION_SOURCE_OUTSIDE_EXPECTED_ROSTER",),
            )
        missing = tuple(
            sorted(
                value.expected_cell_id
                for value in spec.expected_cells
                if value.required and value.expected_cell_id not in set(expected_ids)
            )
        )
        if missing and not spec.missing_expected_as_obstruction:
            return self._stop(
                spec_identity,
                source_refs,
                ObstructionAtlasBuildStopKind.EXPECTED_TERMINAL_MISSING,
                tuple(f"EXPECTED_TERMINAL_MISSING.{value}" for value in missing),
            )
        mapping = {
            (value.terminal_schema, value.reason_code): value
            for value in spec.mapping_registry.mappings
        }
        cells = []
        unobstructed = []
        evidence: dict[str, ObjectIdentity] = {}
        effects = [ObstructionClaimEffect.NO_SCIENTIFIC_EFFECT]
        for source in sorted(sources, key=lambda value: value.source_id):
            source_identity = ObjectIdentity.from_record(source.source_id, source)
            if source.scientific_disposition is MetatheoryCellDisposition.SUPPORTED:
                unobstructed.append(source.terminal)
                continue
            for reason in source.source_reason_codes:
                selected = mapping.get((source.terminal.object_schema, reason))
                next_acts: tuple[LegitimateNextAct, ...]
                if selected is None:
                    if not spec.allow_unknown_as_unresolved:
                        return self._stop(
                            spec_identity,
                            source_refs,
                            ObstructionAtlasBuildStopKind.UNKNOWN_REASON,
                            (f"UNKNOWN_OBSTRUCTION_REASON.{reason}",),
                        )
                    kind = MetatheoryObstructionKind.UNRESOLVED
                    effect = ObstructionClaimEffect.UNEVALUABLE
                    next_acts = (LegitimateNextAct.PRESERVE_UNEVALUABLE,)
                else:
                    kind = selected.obstruction_kind
                    effect = selected.claim_effect
                    next_acts = selected.allowed_next_acts
                effects.append(effect)
                for next_act in next_acts:
                    cells.append(
                        ObstructionCell(
                            cell_id=(
                                f"obstruction.{source.source_id}.{reason.lower()}"
                                f".{next_act.value.lower()}"
                            ),
                            source_bindings=(source_identity,),
                            obstruction_kind=kind,
                            claim_effect=effect,
                            legitimate_next_act=next_act,
                            overlap_group_id=f"overlap.{source.expected_cell_id}",
                            maximum_ordinary_evidence_ceiling=(
                                spec.maximum_ordinary_evidence_ceiling
                            ),
                            maximum_structural_evidence_ceiling=(
                                spec.maximum_structural_evidence_ceiling
                            ),
                            grants_authority=False,
                        )
                    )
            for link in source.evidence_links:
                evidence[link.object_id] = link
        for expected_id in missing:
            effects.append(ObstructionClaimEffect.UNEVALUABLE)
            cells.append(
                ObstructionCell(
                    cell_id=f"obstruction.missing.{expected_id}",
                    source_bindings=(),
                    obstruction_kind=MetatheoryObstructionKind.OPERAND_ABSENT,
                    claim_effect=ObstructionClaimEffect.UNEVALUABLE,
                    legitimate_next_act=LegitimateNextAct.MEASURE_MISSING_OPERAND,
                    overlap_group_id=f"overlap.{expected_id}",
                    maximum_ordinary_evidence_ceiling=spec.maximum_ordinary_evidence_ceiling,
                    maximum_structural_evidence_ceiling=(spec.maximum_structural_evidence_ceiling),
                    grants_authority=False,
                )
            )
        maximum_effect = max(effects, key=lambda value: _EFFECT_RANK[value])
        return ObstructionAtlas(
            atlas_id=f"atlas.{spec.spec_id}",
            atlas_spec=spec_identity,
            source_bindings=tuple(sorted(sources, key=lambda value: value.source_id)),
            cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
            missing_expected_cell_ids=missing,
            unobstructed_terminal_refs=tuple(
                sorted(unobstructed, key=lambda value: value.object_id)
            ),
            operational_statuses=tuple(
                sorted(
                    {value.operational_status for value in sources},
                    key=lambda value: value.value,
                )
            ),
            scientific_dispositions=tuple(
                sorted(
                    {value.scientific_disposition for value in sources},
                    key=lambda value: value.value,
                )
            ),
            evidence_links=tuple(evidence[key] for key in sorted(evidence)),
            maximum_claim_effect=maximum_effect,
            grants_authority=False,
            executed_or_revealed=False,
        )

    @staticmethod
    def _stop(
        spec: ObjectIdentity,
        sources: tuple[ObjectIdentity, ...],
        kind: ObstructionAtlasBuildStopKind,
        reasons: tuple[str, ...],
    ) -> ObstructionAtlasBuildStop:
        return ObstructionAtlasBuildStop(
            stop_id=f"stop.{spec.object_id}.{kind.value.lower()}",
            atlas_spec=spec,
            stop_kind=kind,
            source_binding_refs=sources,
            reason_codes=tuple(sorted(reasons)),
            atlas_constructed=False,
        )


__all__ = ["ObstructionAtlasBuilder"]
