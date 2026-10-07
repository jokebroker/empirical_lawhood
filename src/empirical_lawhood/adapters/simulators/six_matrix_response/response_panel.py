"Bind independent units to the integrity and supervised qualification panels.\n\nThe current source config, native slots and explicit numerical inputs determine\neach panel's units. Binding does not transfer qualification or authority.\n"

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from .response_qualification import ResponseGeometryAssayNativeConfig, ResponseGeometryAssayNativeRoot


INTEGRITY_QUALIFICATION_PANEL_ID = "response-geometry-integrity-qualification"
INTEGRITY_QUALIFICATION_SOURCE_CONFIG_ID = f"{INTEGRITY_QUALIFICATION_PANEL_ID}.source-config"
INTEGRITY_QUALIFICATION_SEED_SHA256 = "c08b7f2a99f6150403585da24fd8119a438509488d24756e5d88f3cdca0f1ad0"
SUPERVISED_QUALIFICATION_PANEL_ID = "response-geometry-supervised-qualification"
SUPERVISED_QUALIFICATION_SOURCE_CONFIG_ID = f"{SUPERVISED_QUALIFICATION_PANEL_ID}.source-config"
SUPERVISED_QUALIFICATION_SEED_SHA256 = "24f3c8c95aa00144a0b9a9abe2323d323a4ad947500b994ca94462d3a9e97a28"


def assay_physical_unit_id(config: ResponseGeometryAssayNativeConfig, root: ResponseGeometryAssayNativeRoot) -> str:
    'Resolve a current slot within its declared scientific panel.'
    if root not in config.roots:
        raise ValueError("assay unit binding selects a slot outside its source configuration")
    panels = {
        INTEGRITY_QUALIFICATION_SOURCE_CONFIG_ID: (INTEGRITY_QUALIFICATION_PANEL_ID, INTEGRITY_QUALIFICATION_SEED_SHA256),
        SUPERVISED_QUALIFICATION_SOURCE_CONFIG_ID: (SUPERVISED_QUALIFICATION_PANEL_ID, SUPERVISED_QUALIFICATION_SEED_SHA256),
    }
    if config.config_id not in panels:
        return root.root_id
    panel, seed = panels[config.config_id]
    if config.seed_root_sha256 != seed:
        raise ValueError("assay follow-up changes its predeclared fresh seed root")
    return f"{panel}.{root.context}.t{root.landmark_tick}.r{root.index}"


@dataclass(frozen=True, slots=True)
class ResponsePreparationSlotInstance(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/response-preparation/response-preparation-slot-instance'

    native_slot: ResponseGeometryAssayNativeRoot
    physical_independent_unit_id: str
    preparation_instance_id: str
    rng_purpose_prefix: str

    def __post_init__(self) -> None:
        for name in (
            "physical_independent_unit_id",
            "preparation_instance_id",
            "rng_purpose_prefix",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.preparation_instance_id != f"{self.physical_independent_unit_id}.instance":
            raise ValueError("assay slot binding changes its preparation instance")
        if self.rng_purpose_prefix != self.physical_independent_unit_id:
            raise ValueError("assay slot binding changes its native RNG purpose namespace")


@dataclass(frozen=True, slots=True)
class IntegrityResponsePanelBinding(CanonicalRecord):
    'Complete current integrity qualification slot and instance binding.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/response-preparation/integrity-response-panel-binding'

    binding_id: str
    source_config: ResponseGeometryAssayNativeConfig
    source_identity: ObjectIdentity
    slots: tuple[ResponsePreparationSlotInstance, ...]

    def __post_init__(self) -> None:
        if self.binding_id != f"{INTEGRITY_QUALIFICATION_PANEL_ID}.slot-instance-binding":
            raise ValueError("assay follow-up panel has another binding identity")
        if self.source_config.config_id != INTEGRITY_QUALIFICATION_SOURCE_CONFIG_ID:
            raise ValueError("assay follow-up binding requires its distinct source identity")
        if self.source_identity != ObjectIdentity.from_record(
            self.source_config.config_id, self.source_config
        ):
            raise ValueError("assay follow-up binding changes its source configuration")
        expected = tuple(
            ResponsePreparationSlotInstance(root, unit, f"{unit}.instance", unit)
            for root in self.source_config.roots
            for unit in (assay_physical_unit_id(self.source_config, root),)
        )
        if self.slots != expected:
            raise ValueError("assay follow-up binding changes the exact slot/instance/RNG map")

    @property
    def physical_unit_ids(self) -> tuple[str, ...]:
        return tuple(sorted(value.physical_independent_unit_id for value in self.slots))


def bind_integrity_response_panel(config: ResponseGeometryAssayNativeConfig) -> IntegrityResponsePanelBinding:
    return IntegrityResponsePanelBinding(
        f"{INTEGRITY_QUALIFICATION_PANEL_ID}.slot-instance-binding",
        config,
        ObjectIdentity.from_record(config.config_id, config),
        tuple(
            ResponsePreparationSlotInstance(root, unit, f"{unit}.instance", unit)
            for root in config.roots
            for unit in (assay_physical_unit_id(config, root),)
        ),
    )


@dataclass(frozen=True, slots=True)
class SupervisedResponsePanelBinding(IntegrityResponsePanelBinding):
    'Complete current supervised qualification slot and instance binding.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/response-preparation/supervised-response-panel-binding'

    def __post_init__(self) -> None:
        if self.binding_id != f"{SUPERVISED_QUALIFICATION_PANEL_ID}.slot-instance-binding":
            raise ValueError("assay supervised panel has another binding identity")
        if self.source_config.config_id != SUPERVISED_QUALIFICATION_SOURCE_CONFIG_ID:
            raise ValueError("assay supervised binding requires its distinct source identity")
        if self.source_identity != ObjectIdentity.from_record(
            self.source_config.config_id, self.source_config
        ):
            raise ValueError("assay supervised binding changes its source configuration")
        expected = tuple(
            ResponsePreparationSlotInstance(root, unit, f"{unit}.instance", unit)
            for root in self.source_config.roots
            for unit in (assay_physical_unit_id(self.source_config, root),)
        )
        if self.slots != expected:
            raise ValueError("assay supervised binding changes the exact slot/instance/RNG map")


def bind_supervised_response_panel(config: ResponseGeometryAssayNativeConfig) -> SupervisedResponsePanelBinding:
    return SupervisedResponsePanelBinding(
        f"{SUPERVISED_QUALIFICATION_PANEL_ID}.slot-instance-binding",
        config,
        ObjectIdentity.from_record(config.config_id, config),
        tuple(
            ResponsePreparationSlotInstance(root, unit, f"{unit}.instance", unit)
            for root in config.roots
            for unit in (assay_physical_unit_id(config, root),)
        ),
    )
