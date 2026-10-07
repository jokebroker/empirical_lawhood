"""Explicit bootstrap allocations for the receiver and response estimands.

These numerical commitments preserve reviewed streams independently of labels.
They confer no historical scientific qualification, custody or authority.
"""

from enum import StrEnum
from types import MappingProxyType
from typing import Final


class BootstrapRole(StrEnum):
    RECEIVER_ORDER = "receiver-order"
    TOTAL_COVARIANCE = "total-covariance"
    RESPONSE_COORDINATE = "response-coordinate"


BOOTSTRAP_SEEDS: Final = MappingProxyType({
    BootstrapRole.RECEIVER_ORDER: 14153480271473371821,
    BootstrapRole.TOTAL_COVARIANCE: 10096632014295185039,
    BootstrapRole.RESPONSE_COORDINATE: 5580106335060441336,
})


def bootstrap_seed(role: BootstrapRole) -> int:
    if not isinstance(role, BootstrapRole):
        raise ValueError("bootstrap scientific role is undeclared")
    return BOOTSTRAP_SEEDS[role]


__all__ = ["BootstrapRole", "bootstrap_seed"]
