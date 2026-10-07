"""Full bootstrap inputs independent of scientific record names and hashes."""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.adapters.independent_source_exports import IndependentSourceExport
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class FreeGsnkeMetricBootstrapInputs(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/simulators/freegsnke/metric-bootstrap-inputs"
    input_id: str
    full_seed: int
    unit_ids: tuple[str, ...]
    exchange_ids: tuple[str, ...]
    replications: int
    familywise_alpha: Decimal
    reference_export: IndependentSourceExport
    evaluation_export: IndependentSourceExport
    verified_numerical_inputs: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        for name in ("unit_ids", "exchange_ids"):
            require_sorted_unique_strings(
                getattr(self, name), field_name=name, allow_empty=False
            )
        validate_decimal(
            self.familywise_alpha,
            field_name="familywise_alpha",
            minimum=Decimal(0),
        )
        if (
            type(self.full_seed) is not int
            or not 0 <= self.full_seed < 2**256
            or type(self.replications) is not int
            or self.replications < 2
            or not Decimal(0) < self.familywise_alpha < Decimal(1)
            or type(self.reference_export) is not IndependentSourceExport
            or type(self.evaluation_export) is not IndependentSourceExport
            or self.evaluation_export.target_physical_unit_ids != self.unit_ids
            or type(self.verified_numerical_inputs) is not ObjectIdentity
            or self.verified_numerical_inputs.object_schema
            != "empirical-lawhood/migration/verified-numerical-inputs"
            or self.verified_numerical_inputs.object_version != "1.0.0"
        ):
            raise ValueError(
                "FreeGSNKE metric bootstrap requires its full numerical input, exact unit/exchange order and separately verified original export custody"
            )

    def validate_census(
        self,
        *,
        reference: ObjectIdentity,
        evaluation: ObjectIdentity,
        unit_ids: tuple[str, ...],
        exchange_ids: tuple[str, ...],
        replications: int,
        familywise_alpha: Decimal,
    ) -> None:
        if (
            self.reference_export.target_source != reference
            or self.evaluation_export.target_source != evaluation
            or self.unit_ids != unit_ids
            or self.exchange_ids != exchange_ids
            or self.replications != replications
            or self.familywise_alpha != familywise_alpha
        ):
            raise ValueError(
                "FreeGSNKE metric bootstrap input is detached from its exact current source/census"
            )
