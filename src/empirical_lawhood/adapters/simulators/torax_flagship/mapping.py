"""Outcome-blind partial mapping from MAST causal states into TORAX preparations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_decimal,
    validate_stable_id,
)
from empirical_lawhood.adapters.physical.mast_archive.contracts import (
    MastCausalState,
    MastFeatureValue,
    MastShotRole,
)

from .contracts import (
    TORAX_HORIZON_S,
    ToraxAction,
    ToraxActionSpec,
    ToraxActionStage,
    ToraxFieldClassification,
    ToraxFieldOrigin,
    ToraxMappedField,
    ToraxMappingDisposition,
    ToraxMappingRejectionCode,
    ToraxMappingResult,
    ToraxNumericalView,
    ToraxPreparation,
    ToraxPreparationEnsemble,
    ToraxRuntimeIdentity,
    ToraxThetaMember,
    ToraxViewRole,
)


@dataclass(frozen=True, slots=True)
class ToraxMappingConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-mapping-config'

    config_id: str
    theta_members: tuple[ToraxThetaMember, ...]
    primary_view: ToraxNumericalView
    sensitivity_view: ToraxNumericalView
    electron_density_feature_id: str
    electron_temperature_feature_id: str
    plasma_current_feature_id: str
    source_qualified: bool
    transport_qualified: bool
    boundary_qualified: bool
    composition_qualified: bool
    development_only: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        require_sorted_unique_ids(
            self.theta_members, attribute="theta_id", field_name="theta_members"
        )
        if not self.theta_members:
            raise ValueError("TORAX mapping requires a nonempty Theta ensemble")
        for name, value in (
            ("electron_density_feature_id", self.electron_density_feature_id),
            ("electron_temperature_feature_id", self.electron_temperature_feature_id),
            ("plasma_current_feature_id", self.plasma_current_feature_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.primary_view.role is not ToraxViewRole.PRIMARY:
            raise ValueError("primary TORAX view has the wrong role")
        if self.sensitivity_view.role is not ToraxViewRole.SENSITIVITY:
            raise ValueError("sensitivity TORAX view has the wrong role")
        if not self.development_only:
            raise ValueError("Phase 5 mapping is development-only and unissued")


@dataclass(frozen=True, slots=True)
class ToraxActionPanelConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-flagship/torax-action-panel-config'

    config_id: str
    down_power_w: Decimal
    hold_power_w: Decimal
    up_power_w: Decimal
    duration_s: Decimal
    source_model_id: str
    current_drive_model_id: str
    source_is_proxy: bool
    development_only: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.source_model_id, field_name="source_model_id")
        validate_stable_id(self.current_drive_model_id, field_name="current_drive_model_id")
        for name, value in (
            ("down_power_w", self.down_power_w),
            ("hold_power_w", self.hold_power_w),
            ("up_power_w", self.up_power_w),
            ("duration_s", self.duration_s),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
        if not 0 < self.down_power_w < self.hold_power_w < self.up_power_w:
            raise ValueError("TORAX absolute proxy power must order DOWN < HOLD < UP")
        if self.duration_s != TORAX_HORIZON_S:
            raise ValueError("TORAX action panel duration differs from 40 ms")
        if self.source_model_id != "generic-ion-electron-gaussian-proxy":
            raise ValueError("TORAX action panel source differs")
        if not self.source_is_proxy or not self.development_only:
            raise ValueError("TORAX source must remain an explicit development proxy")


def decode_torax_mapping_config(payload: bytes) -> ToraxMappingConfig:
    return decode_canonical_bytes(payload, ToraxMappingConfig, maximum_bytes=512 * 1024)


def decode_torax_action_panel_config(payload: bytes) -> ToraxActionPanelConfig:
    return decode_canonical_bytes(payload, ToraxActionPanelConfig, maximum_bytes=64 * 1024)


def map_mast_state_to_torax(
    *,
    state: MastCausalState,
    config: ToraxMappingConfig,
    runtime: ToraxRuntimeIdentity,
) -> ToraxMappingResult:
    state_identity = ObjectIdentity.from_record(state.state_id, state)
    reasons: set[ToraxMappingRejectionCode] = set()
    hard_rejection = False
    if state.role not in {
        MastShotRole.DEVELOPMENT,
        MastShotRole.PROTECTED_EVALUATION,
    }:
        reasons.add(ToraxMappingRejectionCode.OUTCOME_ACCESS_FORBIDDEN)
        hard_rejection = True
    if (
        state.role is MastShotRole.PROTECTED_EVALUATION
        and state.outcome_access is not OutcomeAccess.OUTCOME_BLIND
    ):
        reasons.add(ToraxMappingRejectionCode.OUTCOME_ACCESS_FORBIDDEN)
        hard_rejection = True
    if not config.source_qualified:
        reasons.add(ToraxMappingRejectionCode.UNQUALIFIED_SOURCE)
        hard_rejection = True
    if not config.transport_qualified:
        reasons.add(ToraxMappingRejectionCode.UNQUALIFIED_TRANSPORT)
        hard_rejection = True
    if not config.boundary_qualified:
        reasons.add(ToraxMappingRejectionCode.UNQUALIFIED_BOUNDARY)
        hard_rejection = True
    if not config.composition_qualified:
        reasons.add(ToraxMappingRejectionCode.UNQUALIFIED_COMPOSITION)
        hard_rejection = True
    density: MastFeatureValue | None
    temperature: MastFeatureValue | None
    current: MastFeatureValue | None
    try:
        density = state.feature(config.electron_density_feature_id)
        temperature = state.feature(config.electron_temperature_feature_id)
        current = state.feature(config.plasma_current_feature_id)
    except KeyError:
        reasons.add(ToraxMappingRejectionCode.INSUFFICIENT_PROFILE)
        density = temperature = current = None
    if density is not None and temperature is not None and current is not None:
        if density.native_unit != "m^-3" or temperature.native_unit != "eV":
            reasons.add(ToraxMappingRejectionCode.INSUFFICIENT_PROFILE)
        if current.native_unit != "A":
            reasons.add(ToraxMappingRejectionCode.INSUFFICIENT_PROFILE)
        if (
            not density.radial_coordinates
            or density.radial_coordinates != temperature.radial_coordinates
            or any(density.missing_mask)
            or any(temperature.missing_mask)
            or any(current.missing_mask)
            or density.quality_reason_codes
            or temperature.quality_reason_codes
            or current.quality_reason_codes
        ):
            reasons.add(ToraxMappingRejectionCode.INSUFFICIENT_PROFILE)
    if reasons:
        return ToraxMappingResult(
            result_id=f"torax-mapping-{state.shot_id}",
            mast_state=state_identity,
            disposition=(
                ToraxMappingDisposition.REJECTED
                if hard_rejection
                else ToraxMappingDisposition.UNEVALUABLE
            ),
            ensemble=None,
            rejection_codes=tuple(sorted(reasons)),
        )
    assert density is not None and temperature is not None and current is not None
    runtime_identity = ObjectIdentity.from_record(runtime.runtime_id, runtime)
    marginalized = len(config.theta_members) > 1
    assumption_origin = ToraxFieldOrigin.MARGINALIZED if marginalized else ToraxFieldOrigin.ASSUMED
    preparations: list[ToraxPreparation] = []
    for theta in config.theta_members:
        ion_temperature = tuple(value * theta.ti_over_te for value in temperature.values)
        fields = (
            ToraxMappedField(
                field_id="electron-density",
                origin=ToraxFieldOrigin.OBSERVED,
                values=density.values,
                radial_coordinates=density.radial_coordinates,
                native_unit=density.native_unit,
                source_feature_id=density.feature_id,
                assumption_id=None,
            ),
            ToraxMappedField(
                field_id="electron-temperature",
                origin=ToraxFieldOrigin.OBSERVED,
                values=temperature.values,
                radial_coordinates=temperature.radial_coordinates,
                native_unit=temperature.native_unit,
                source_feature_id=temperature.feature_id,
                assumption_id=None,
            ),
            ToraxMappedField(
                field_id="ion-temperature",
                origin=ToraxFieldOrigin.DERIVED,
                values=ion_temperature,
                radial_coordinates=temperature.radial_coordinates,
                native_unit="eV",
                source_feature_id=temperature.feature_id,
                assumption_id=None,
            ),
            ToraxMappedField(
                field_id="plasma-current",
                origin=ToraxFieldOrigin.OBSERVED,
                values=current.values,
                radial_coordinates=current.radial_coordinates,
                native_unit=current.native_unit,
                source_feature_id=current.feature_id,
                assumption_id=None,
            ),
        )
        classifications = tuple(
            sorted(
                (
                    _classification(
                        "electron-density", ToraxFieldOrigin.OBSERVED, density.feature_id, None
                    ),
                    _classification(
                        "electron-temperature",
                        ToraxFieldOrigin.OBSERVED,
                        temperature.feature_id,
                        None,
                    ),
                    _classification(
                        "ion-temperature", ToraxFieldOrigin.DERIVED, temperature.feature_id, None
                    ),
                    _classification(
                        "plasma-current", ToraxFieldOrigin.OBSERVED, current.feature_id, None
                    ),
                    *(
                        _classification(
                            field_id, assumption_origin, None, f"{theta.theta_id}.{field_id}"
                        )
                        for field_id in (
                            "boundary",
                            "composition",
                            "current-convention",
                            "geometry",
                            "radial-mapping",
                            "source",
                            "transport",
                        )
                    ),
                ),
                key=lambda value: value.field_id,
            )
        )
        preparations.append(
            ToraxPreparation(
                preparation_id=f"torax-preparation-{state.shot_id}-{theta.theta_id}",
                mast_state=state_identity,
                theta=theta,
                fields=fields,
                classifications=classifications,
                runtime=runtime_identity,
                stock_iter_hybrid=False,
                outcome_blind_mapping=True,
            )
        )
    ensemble = ToraxPreparationEnsemble(
        ensemble_id=f"torax-ensemble-{state.shot_id}",
        mast_state=state_identity,
        preparations=tuple(preparations),
    )
    return ToraxMappingResult(
        result_id=f"torax-mapping-{state.shot_id}",
        mast_state=state_identity,
        disposition=ToraxMappingDisposition.ACCEPTED,
        ensemble=ensemble,
        rejection_codes=(),
    )


def build_torax_action_specs(config: ToraxActionPanelConfig) -> tuple[ToraxActionSpec, ...]:
    powers = {
        ToraxAction.DOWN: config.down_power_w,
        ToraxAction.HOLD: config.hold_power_w,
        ToraxAction.UP: config.up_power_w,
    }
    return tuple(
        ToraxActionSpec(
            action_id=f"torax-action-{action.value.lower()}",
            action=action,
            stages=tuple(
                ToraxActionStage(
                    stage=stage,
                    power_w=powers[action],
                    native_unit="W",
                    coordinate_s=Decimal(0),
                )
                for stage in ActionDeliveryStage
            ),
            duration_s=config.duration_s,
            horizon_s=TORAX_HORIZON_S,
            source_model_id=config.source_model_id,
            source_is_proxy=config.source_is_proxy,
            current_drive_model_id=config.current_drive_model_id,
        )
        for action in ToraxAction
    )


def _classification(
    field_id: str,
    origin: ToraxFieldOrigin,
    source_feature_id: str | None,
    assumption_id: str | None,
) -> ToraxFieldClassification:
    return ToraxFieldClassification(
        field_id=field_id,
        origin=origin,
        source_feature_id=source_feature_id,
        assumption_id=assumption_id,
    )


__all__ = [
    "ToraxActionPanelConfig",
    "ToraxMappingConfig",
    "build_torax_action_specs",
    "decode_torax_action_panel_config",
    "decode_torax_mapping_config",
    "map_mast_state_to_torax",
]
