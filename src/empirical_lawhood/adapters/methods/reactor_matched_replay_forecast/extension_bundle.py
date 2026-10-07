"""Noncontact discovery of the task's finite-action campaign binding."""

from empirical_lawhood.adapters.simulators.reactor_matched_replay_forecast.transport import ReactorTraceEnvelope
from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import FINITE_ACTION_MANIFEST
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from .science import ReactorForecastDesign
from .records import ReactorForecastResult


INPUT_TYPES = (
    ReactorForecastDesign,
    ReactorTraceEnvelope,
    ReactorForecastResult,
)
OUTPUT_TYPES = (
    ReactorForecastResult,
    ScientificAdjudicationRecord,
)
CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="reactor-matched-replay-forecast.forecast-qualification",
    kind=CapabilityKind.EVALUATOR,
    maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    permissions=tuple(
        sorted(
            (
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
            ),
            key=lambda p: p.value,
        )
    ),
    config_schema=ReactorForecastDesign.SCHEMA,
    config_schema_sha256=sha256(ReactorForecastDesign.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(sorted(t.SCHEMA for t in INPUT_TYPES)),
    output_schema_ids=tuple(sorted(t.SCHEMA for t in OUTPUT_TYPES)),
    implementation_sha256=sha256(
        b"reactor-matched-replay-forecast:matched-input-absolute-profile-and-separate-recovery-operands"
    ).hexdigest(),
    runtime_id="cpython-3.11-reactor-matched-replay-forecast",
    resource_ceiling=ResourceBudget(1, 4 * 1024**3, 0, 1800, 8 * 1024**3, 16 * 1024**2),
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "forecast-qualification",
    ExtensionContributionKind.METHOD,
    CAPABILITY,
    (ReactorForecastDesign,),
    None,
    namespace="reactor-matched-replay-forecast",
)
