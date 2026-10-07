# SPDX-License-Identifier: MPL-2.0

"""Retained D1/preparation-policy development graphs select every real owner; fixtures cannot execute."""

from tests.prepared_seed_fixtures import fixture_seed_census

from dataclasses import dataclass
from decimal import Decimal

import pytest

from empirical_lawhood.adapters.composition.finite_response_law.consumer_input import FiniteResponseLawAuthoringInput, build_consumer_authoring
from empirical_lawhood.adapters.composition.finite_response_law.stage_input import inspect_consumer_providers
from empirical_lawhood.adapters.methods.finite_response_law.law_payloads import FiniteResponseLawLowerPayload
from empirical_lawhood.adapters.methods.finite_response_law.science import FiniteResponseLawScienceSpec
from empirical_lawhood.adapters.methods.finite_response_law.preparation_policy_projection import FiniteResponseLawPreparationPolicyProjectionConfig
from empirical_lawhood.adapters.methods.finite_response_law.preparation_policy_provider import PREPARATION_POLICY_EVIDENCE_PORT
from empirical_lawhood.adapters.methods.finite_response_law.preparation_screen_results import FiniteResponseLawRetainedProspectiveCloseoutReference, FiniteResponseLawPreparationPolicyScreenConfig
from empirical_lawhood.adapters.simulators.finite_response_law.contracts import FiniteResponseLawNativeConfig
from empirical_lawhood.adapters.simulators.finite_response_law.discovery import SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.finite_response_law.provider import RETAINED_SOURCE_PORT
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_screen.discovery import SOURCE_CAPABILITY as PREPARATION_POLICY_SOURCE
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_contracts import FiniteResponseLawPreparationPolicyNativeConfig, FiniteResponseLawPreparationPolicyRetainedPrefix
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_provider import PREPARATION_POLICY_RETAINED_SOURCE_PORT
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PARENTS, PreparedNativeSpec, prepared_native_member, prepared_numerical_view
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.planning.source_qualification import SourceQualificationRetainedPredecessor
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.executable_bindings import ExecutablePlatformPort


def identity(name, schema):
    return ObjectIdentity(name, schema, "1.0.0", "0" * 64)


def artifact(name, schema):
    return ArtifactIdentity(
        name, "synthetic-development", schema, "0" * 64, "application/json", 1
    )


def prepared(stage):
    return PreparedNativeSpec(
        stage,
        "0" * 64,
        "a" * 64,
        "b" * 64,
        identity("synthetic.code", 'empirical-lawhood/synthetic/code'),
        prepared_native_member(),
        tuple(prepared_numerical_view(r) for r in (1, 2)),
        None if stage == 'qualification' else Decimal(8),
        root_seed_census=fixture_seed_census(stage, '0' * 64),
    )


def declaration(root, segment, end, views):
    return SourceQualificationRetainedPredecessor(
        segment,
        root.physical_unit_id,
        'finite-response-law.reference-clock',
        Decimal(end),
        views,
        (artifact(f"{segment}.result", PreparedNativeTaskResult.SCHEMA),),
        artifact(f"{segment}.receipt", CanonicalTaskReceipt.SCHEMA),
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        VisibilityCeiling.OUTCOME_VISIBLE,
    )


@dataclass
class GuardedSyntheticSources:
    artifacts: tuple

    def create_source(self, artifact):
        raise AssertionError("Synthetic binding fixture must not read or acquire")


def stage_packet(stage):
    if stage == 'supplemental-development':
        original = prepared('qualification')
        rows = tuple(
            sorted(
                (
                    declaration(
                        root,
                        f"{root.root_id}.{parent}.parent.native",
                        4368,
                        tuple(f"{root.root_id}.flh-project.r{r}" for r in (1, 2)),
                    )
                    for root in original.roots
                    if root.context == "prepared"
                    for parent in PARENTS
                ),
                key=lambda row: row.segment_id,
            )
        )
        source = FiniteResponseLawNativeConfig(
            stage,
            FiniteResponseLawScienceSpec(),
            ObjectIdentity.from_record(
                SOURCE_CAPABILITY.capability_key, SOURCE_CAPABILITY
            ),
            original,
            rows,
        )
        packet = FiniteResponseLawAuthoringInput(
            stage, source, "0" * 64, ("prior-unit",), ("prior-stream",)
        )
        inventories = {
            RETAINED_SOURCE_PORT: tuple(
                a for row in rows for a in (*row.artifacts, row.task_receipt)
            )
        }
    else:
        rows = []
        for cohort, old_stage, count, offset in (
            ('prepared-response', 'qualification', 8, 0),
            ('information-response-prediction', "prospective-evaluation", 16, 8),
        ):
            original = prepared(old_stage)
            for root in (
                r for r in original.roots if r.context == "prepared" and r.index < count
            ):
                segment = f"{root.root_id}.prefix.native"
                prior = declaration(
                    root,
                    segment,
                    4096,
                    (f"preparation-screening.r{root.index + offset:03d}.project",),
                )
                rows.append(
                    FiniteResponseLawPreparationPolicyRetainedPrefix(
                        prior,
                        ObjectIdentity.from_record(original.spec_id, original),
                        identity(
                            f"{segment}.result", PreparedNativeTaskResult.SCHEMA
                        ),
                        ((Decimal(0),) * 24,) * 2,
                        identity(
                            "synthetic.instrument",
                            'empirical-lawhood/simulators/finite-response-law/finite-response-law-reference-instrument',
                        ),
                    )
                )
        source = FiniteResponseLawPreparationPolicyNativeConfig(
            'preparation-screening',
            FiniteResponseLawScienceSpec(),
            ObjectIdentity.from_record(PREPARATION_POLICY_SOURCE.capability_key, PREPARATION_POLICY_SOURCE),
            tuple(sorted(rows, key=lambda row: row.declaration.segment_id)),
        )
        lower = artifact("synthetic.lower", FiniteResponseLawLowerPayload.SCHEMA)
        screen = FiniteResponseLawPreparationPolicyScreenConfig(
            FiniteResponseLawPreparationPolicyProjectionConfig(source),
            lower,
            identity(lower.artifact_id, lower.payload_schema),
            artifact(
                "synthetic.qualification",
                'empirical-lawhood/synthetic/qualification',
            ),
            artifact(
                "synthetic.adjudication", 'empirical-lawhood/synthetic/adjudication'
            ),
            artifact("synthetic.closeout", FiniteResponseLawRetainedProspectiveCloseoutReference.SCHEMA),
        )
        packet = FiniteResponseLawAuthoringInput(
            stage,
            source,
            "0" * 64,
            ("prior-unit",),
            ("prior-stream",),
            screen=screen,
            prospective_eligibility=identity(
                "synthetic.closeout", FiniteResponseLawRetainedProspectiveCloseoutReference.SCHEMA
            ),
        )
        inventories = {
            PREPARATION_POLICY_RETAINED_SOURCE_PORT: tuple(
                a
                for row in rows
                for a in (*row.declaration.artifacts, row.declaration.task_receipt)
            ),
            PREPARATION_POLICY_EVIDENCE_PORT: (
                screen.lower_artifact,
                screen.lower_qualification,
                screen.prospective_adjudication,
                screen.prospective_closeout,
            ),
        }
    return packet, tuple(
        ExecutablePlatformPort(
            key,
            GuardedSyntheticSources(tuple(sorted(values, key=lambda a: a.artifact_id))),
        )
        for key, values in sorted(inventories.items())
    )


@pytest.mark.parametrize(
    ("stage", "roots", "tasks"), (('supplemental-development', 16, 720), ('preparation-screening', 24, 4104))
)
def test_retained_consumer_graphs_build_real_providers_without_read_or_execution(
    stage, roots, tasks
):
    packet, ports = stage_packet(stage)
    assert (
        decode_canonical_bytes(
            packet.canonical_bytes(), type(packet), maximum_bytes=8 * 1024**2
        )
        == packet
    )
    if stage == "preparation-screening":
        from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_contracts import preparation_policy_native_invocations

        assert len(packet.source.roots) == roots
        assert len(preparation_policy_native_invocations(packet.source)) == tasks
        with pytest.raises(ValueError, match="FINITE_RESPONSE_LAW_CURRENT_PREPARATION_ELIGIBILITY_EXPORT_REQUIRED"):
            build_consumer_authoring(
                packet,
                implementation_sha256="0" * 64,
                exposure=ObjectIdentity.from_record("synthetic.nonpromotable", packet),
                current_units=tuple(sorted(("prior-unit", *(root.physical_unit_id for root in packet.source.roots)))),
                current_seeds=("prior-stream",),
            )
        return
    bundle = build_consumer_authoring(
        packet,
        implementation_sha256="0" * 64,
        exposure=ObjectIdentity.from_record("synthetic.nonpromotable", packet),
        current_units=tuple(
            sorted(
                ("prior-unit", *(root.physical_unit_id for root in packet.source.roots))
            )
        ),
        current_seeds=("prior-stream",),
    )
    providers = inspect_consumer_providers(bundle, ports)
    assert len(packet.source.roots) == roots
    assert len(providers) == 3
    assert sum(p["native_tasks_planned"] for p in providers) == tasks
    with pytest.raises(ValueError, match="port"):
        inspect_consumer_providers(bundle, ())
