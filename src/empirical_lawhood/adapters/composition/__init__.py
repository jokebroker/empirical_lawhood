"""Closed adapter compositions for additive platform successors."""

from .acquisition_recurrence_margin_contracts import BOUNDED_ACQUISITION_CAPABILITY_KEY, GATE_MARGIN_CAPABILITY_KEY, MAXIMIN_CONTINUATION_CAPABILITY_KEY, RECURRENCE_CAPABILITY_KEY, ACQUISITION_RECURRENCE_MARGIN_CAPABILITY_KEYS, AcquisitionRecurrenceMarginContractRegistry, build_acquisition_recurrence_margin_contract_registry, acquisition_recurrence_margin_capability_manifests

__all__ = [
    "BOUNDED_ACQUISITION_CAPABILITY_KEY",
    'AcquisitionRecurrenceMarginContractRegistry',
    "GATE_MARGIN_CAPABILITY_KEY",
    "MAXIMIN_CONTINUATION_CAPABILITY_KEY",
    "RECURRENCE_CAPABILITY_KEY",
    "ACQUISITION_RECURRENCE_MARGIN_CAPABILITY_KEYS",
    'build_acquisition_recurrence_margin_contract_registry',
    'acquisition_recurrence_margin_capability_manifests',
]
