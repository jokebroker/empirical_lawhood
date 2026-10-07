"""Exact operands for the analytical common-input RC construction.

SPDX-License-Identifier: MPL-2.0
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from fractions import Fraction
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)


class RCInformationInstanceKind(StrEnum):
    BASELINE = "BASELINE"
    VARIANT = "VARIANT"


def _pair(
    values: tuple[Decimal, Decimal], name: str, *, positive: bool = False
) -> None:
    if type(values) is not tuple or len(values) != 2:
        raise ValueError(f"{name} must contain exactly two Decimal operands")
    for value in values:
        validate_decimal(value, field_name=name)
        if positive and value <= 0:
            raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class RCInformationConfig(CanonicalRecord):
    """Two equally weighted RC states under one declared predictor input.

    The two branches have the same capacitance, so their arithmetic mean is
    also their capacitance-weighted receiver. State coordinates multiply V0.
    Predictor-input digests identify the same complete input in both cases;
    their equality is a declared analytical premise, not source qualification.
    """

    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/rc-information/config"

    config_id: str
    instance_kind: RCInformationInstanceKind
    voltage_scale_volts: Decimal
    equal_branch_capacitance_farads: Decimal
    time_constants_seconds: tuple[Decimal, Decimal]
    initial_state_a_v0: tuple[Decimal, Decimal]
    initial_state_b_v0: tuple[Decimal, Decimal]
    predictor_input_a_sha256: str
    predictor_input_b_sha256: str
    receiver_id: str = "equal-capacitance-mean-voltage"
    voltage_unit: str = "V"
    time_unit: str = "s"
    capacitance_unit: str = "F"

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if not isinstance(self.instance_kind, RCInformationInstanceKind):
            raise ValueError("instance_kind must be an RCInformationInstanceKind")
        for name in ("voltage_scale_volts", "equal_branch_capacitance_farads"):
            value = getattr(self, name)
            validate_decimal(value, field_name=name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        _pair(self.time_constants_seconds, "time_constants_seconds", positive=True)
        _pair(self.initial_state_a_v0, "initial_state_a_v0")
        _pair(self.initial_state_b_v0, "initial_state_b_v0")
        for name in ("predictor_input_a_sha256", "predictor_input_b_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if self.predictor_input_a_sha256 != self.predictor_input_b_sha256:
            raise ValueError(
                "the identical-input bound requires the same predictor input"
            )
        # Fraction preserves the supplied Decimal values without ambient rounding.
        if sum(map(Fraction, self.initial_state_a_v0)) != sum(
            map(Fraction, self.initial_state_b_v0)
        ):
            raise ValueError(
                "the common-input premise requires equal initial receiver means"
            )
        if self.receiver_id != "equal-capacitance-mean-voltage":
            raise ValueError(
                "this construction requires the common mean-voltage receiver"
            )
        if (self.voltage_unit, self.time_unit, self.capacitance_unit) != (
            "V",
            "s",
            "F",
        ):
            raise ValueError("RC information operands require V, s and F")
        if self.instance_kind is RCInformationInstanceKind.BASELINE and (
            self.time_constants_seconds != (Decimal(1), Decimal(4))
            or self.initial_state_a_v0 != (Decimal(1), Decimal(1))
            or self.initial_state_b_v0 != (Decimal("0.5"), Decimal("1.5"))
        ):
            raise ValueError(
                "changed RC time constants or states require a VARIANT instance"
            )


@dataclass(frozen=True, slots=True)
class RCInformationResult(CanonicalRecord):
    """Numerical evaluations of an exact analytical construction.

    Approximate Decimal values are not certified interval endpoints. The
    analytical lower bound follows from the stated exact expression and
    identical-input premise, not from a sampled confidence interval.
    """

    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/rc-information/result"

    config: ObjectIdentity
    maximizing_time_approx_seconds: Decimal
    homogeneous_response_a_approx_volts: Decimal
    homogeneous_response_b_approx_volts: Decimal
    maximum_response_separation_approx_volts: Decimal
    identical_input_minimax_lower_bound_approx_volts: Decimal
    maximizing_time_expression: str
    maximum_separation_expression: str
    minimax_lower_bound_expression: str
    evaluation_precision_digits: int = 80
    analytical_construction: bool = True
    numerical_interval_certified: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.config, ObjectIdentity)
            or self.config.object_schema != RCInformationConfig.SCHEMA
        ):
            raise ValueError("RC information result requires its exact input identity")
        for name in (
            "maximizing_time_approx_seconds",
            "maximum_response_separation_approx_volts",
            "identical_input_minimax_lower_bound_approx_volts",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        for name in (
            "homogeneous_response_a_approx_volts",
            "homogeneous_response_b_approx_volts",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        if (
            type(self.evaluation_precision_digits) is not int
            or self.evaluation_precision_digits != 80
        ):
            raise ValueError(
                "RC analytical evaluation has its declared 80-digit context"
            )
        if (
            self.analytical_construction is not True
            or self.numerical_interval_certified is not False
        ):
            raise ValueError(
                "RC illustration cannot claim sampled or certified interval evidence"
            )
        if not all(
            type(value) is str and value
            for value in (
                self.maximizing_time_expression,
                self.maximum_separation_expression,
                self.minimax_lower_bound_expression,
            )
        ):
            raise ValueError(
                "RC information result requires its analytical expressions"
            )


@dataclass(frozen=True, slots=True)
class RCInformationBoundCheck(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/rc-information/bound-check"

    result: ObjectIdentity
    checker_precision_digits: int
    input_identity_matches: bool
    stationary_or_degenerate_maximum: bool
    numerical_equations_agree: bool
    identical_input_bound_agrees: bool
    analytical_expressions_agree: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.result, ObjectIdentity)
            or self.result.object_schema != RCInformationResult.SCHEMA
        ):
            raise ValueError(
                "bound check requires the exact analytical result identity"
            )
        if (
            type(self.checker_precision_digits) is not int
            or self.checker_precision_digits != 110
        ):
            raise ValueError(
                "RC independent checker has its declared 110-digit context"
            )
        for name in (
            "input_identity_matches",
            "stationary_or_degenerate_maximum",
            "numerical_equations_agree",
            "identical_input_bound_agrees",
            "analytical_expressions_agree",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be an exact boolean")

    @property
    def passed(self) -> bool:
        return all(
            (
                self.input_identity_matches,
                self.stationary_or_degenerate_maximum,
                self.numerical_equations_agree,
                self.identical_input_bound_agrees,
                self.analytical_expressions_agree,
            )
        )
