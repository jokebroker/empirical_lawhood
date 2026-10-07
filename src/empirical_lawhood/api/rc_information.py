"""Public analytical RC calculation with its independent equation check.

SPDX-License-Identifier: MPL-2.0
"""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.rc_information.analytic import (
    evaluate_rc_information,
)
from empirical_lawhood.adapters.methods.rc_information.checker import (
    check_rc_information_bound,
)
from empirical_lawhood.adapters.methods.rc_information.contracts import (
    RCInformationBoundCheck,
    RCInformationConfig,
    RCInformationResult,
)
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.provenance import ObjectIdentity


@dataclass(frozen=True, slots=True)
class RCInformationReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/api/rc-information-report"

    input: RCInformationConfig
    result: RCInformationResult
    check: RCInformationBoundCheck

    def __post_init__(self) -> None:
        if not isinstance(self.input, RCInformationConfig):
            raise ValueError("RC information report requires its typed input")
        if not isinstance(
            self.result, RCInformationResult
        ) or self.result.config != ObjectIdentity.from_record(
            self.input.config_id, self.input
        ):
            raise ValueError("RC information report changed its input/result binding")
        if not isinstance(
            self.check, RCInformationBoundCheck
        ) or self.check.result != ObjectIdentity.from_record(
            f"{self.input.config_id}.result", self.result
        ):
            raise ValueError("RC information report changed its result/check binding")


def calculate_rc_information(config: RCInformationConfig) -> RCInformationReport:
    """Calculate one analytical instance; no issue, native effects or custody.

    The shared configuration and development-retention owners supply path
    loading, exact preparation and optional report export at the CLI boundary.
    """
    result = evaluate_rc_information(config)
    check = check_rc_information_bound(config, result)
    if not check.passed:
        raise ArithmeticError("RC information independent analytical check failed")
    return RCInformationReport(config, result, check)
