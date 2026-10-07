"""Pure shared evidence-geometry services for the sole controller compiler."""

from __future__ import annotations

from empirical_lawhood.planning.controller_study import ImplementationBinding, ImplementationRole
from empirical_lawhood.planning.evidence_geometry import AtlasAdmissionSpec, ReceiptAdmissionSpec, AtlasReachabilitySpec, ControlledMapReachabilitySpec, ControlledMapReachabilityComparison, derive_atlas_admission_comparison, derive_receipt_admission_comparison, derive_atlas_reachability_comparison, derive_controlled_map_reachability_comparison
from empirical_lawhood.planning.finite_admission import FiniteCertificateAdmissionSpec, FiniteCertificateReachabilitySpec, FiniteCertificateReachabilityComparison, derive_finite_certificate_admission_comparison, derive_finite_certificate_reachability_comparison
from empirical_lawhood.planning.geometry import AdmissionComparison, ReachabilityComparison


def _require_role(binding: ImplementationBinding, role: ImplementationRole) -> None:
    if binding.role is not role:
        raise ValueError(f"service binding must have role {role.value}")


class AtlasAdmissionDeriver:
    """Derive the complete noncompensating admission intersection."""

    def __init__(self, implementation_binding: ImplementationBinding) -> None:
        _require_role(implementation_binding, ImplementationRole.ADMISSION_DERIVER)
        self.implementation_binding = implementation_binding

    def derive(self, spec: AtlasAdmissionSpec) -> AdmissionComparison:
        return derive_atlas_admission_comparison(spec)


class AtlasReachabilityDeriver:
    """Derive complete memberwise and robust viable/reachable geometry."""

    def __init__(self, implementation_binding: ImplementationBinding) -> None:
        _require_role(implementation_binding, ImplementationRole.REACHABILITY_DERIVER)
        self.implementation_binding = implementation_binding

    def derive(self, spec: AtlasReachabilitySpec) -> ReachabilityComparison:
        return derive_atlas_reachability_comparison(spec)


class ReceiptAdmissionDeriver:
    """Replay the complete raw-receipt member/version admission intersection."""

    def __init__(self, implementation_binding: ImplementationBinding) -> None:
        _require_role(implementation_binding, ImplementationRole.ADMISSION_DERIVER)
        self.implementation_binding = implementation_binding

    def derive(self, spec: ReceiptAdmissionSpec) -> AdmissionComparison:
        return derive_receipt_admission_comparison(spec)


class ReceiptReachabilityDeriver:
    """Replay complete member/version reachability from the same raw corpus."""

    def __init__(self, implementation_binding: ImplementationBinding) -> None:
        _require_role(implementation_binding, ImplementationRole.REACHABILITY_DERIVER)
        self.implementation_binding = implementation_binding

    def derive(self, spec: ControlledMapReachabilitySpec) -> ControlledMapReachabilityComparison:
        return derive_controlled_map_reachability_comparison(spec)


class FiniteCertificateAdmissionDeriver:
    def __init__(self, implementation_binding: ImplementationBinding) -> None:
        _require_role(implementation_binding, ImplementationRole.ADMISSION_DERIVER)
        self.implementation_binding = implementation_binding

    def derive(self, spec: FiniteCertificateAdmissionSpec) -> AdmissionComparison:
        return derive_finite_certificate_admission_comparison(spec)


class FiniteCertificateReachabilityDeriver:
    def __init__(self, implementation_binding: ImplementationBinding) -> None:
        _require_role(implementation_binding, ImplementationRole.REACHABILITY_DERIVER)
        self.implementation_binding = implementation_binding

    def derive(self, spec: FiniteCertificateReachabilitySpec) -> FiniteCertificateReachabilityComparison:
        return derive_finite_certificate_reachability_comparison(spec)


__all__ = [
    'AtlasAdmissionDeriver',
    'AtlasReachabilityDeriver',
    'ReceiptAdmissionDeriver',
    'ReceiptReachabilityDeriver',
]
