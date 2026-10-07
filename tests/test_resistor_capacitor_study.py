"""Shipped RC study: native SI behavior, nested views and real counterexamples."""

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from empirical_lawhood.adapters.simulators.rc_ladder_response.campaign import VIEWS, check_native_views, run_native_view
from empirical_lawhood.adapters.simulators.rc_ladder_response.contracts import ResistorCapacitorLadderStudyConfig
from empirical_lawhood.adapters.simulators.rc_ladder_response.executable_binding import BINDING, ResistorCapacitorLadderNativeFactory
from empirical_lawhood.adapters.simulators.rc_ladder_response.extension_bundle import CAPABILITY
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.runtime.capabilities import CapabilityRegistry

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/rc-ladder-response/study.json"


def _study() -> ResistorCapacitorLadderStudyConfig:
    return decode_canonical_bytes(
        read_bounded_bytes(STUDY, maximum_bytes=65536),
        ResistorCapacitorLadderStudyConfig,
        maximum_bytes=65536,
    )


def test_shipped_study_has_one_unit_two_views_and_finite_source_impedance() -> None:
    study = _study()
    panels = tuple(run_native_view(study, view) for view in VIEWS)
    check = check_native_views(study, panels)
    assert check.converged
    assert check.independent_unit_count == 1
    assert check.nested_view_count == 2
    assert all(panel.unit_id == study.unit_id for panel in panels)
    assert all(
        panel.times_seconds == study.model.output_times_seconds for panel in panels
    )
    exact = next(panel for panel in panels if panel.view_id == "matrix-exponential")
    last_voltage = exact.node_voltages_volts[-1]
    # Independent DC series-network reference: source, three interiors and
    # termination carry 1 V / (50 + 300 + 100) ohm.  The 1 s transient has
    # advanced toward, but cannot overshoot, those four divider voltages.
    dc_current = Decimal(1) / Decimal(450)
    dc_nodes = tuple(
        Decimal(1) - dc_current * resistance
        for resistance in (Decimal(50), Decimal(150), Decimal(250), Decimal(350))
    )
    assert all(
        Decimal(0) < transient < dc
        for transient, dc in zip(last_voltage, dc_nodes, strict=True)
    )
    # Native emitted left current must retain the source resistor, not treat
    # the requested 1 V as an ideal voltage imposed on the first capacitor.
    assert abs(
        exact.left_boundary_currents_amperes[1]
        - (Decimal(1) - exact.node_voltages_volts[1][0]) / Decimal(50)
    ) < Decimal("1e-15")
    assert abs(
        exact.right_boundary_currents_amperes[1]
        + exact.node_voltages_volts[1][-1] / Decimal(100)
    ) < Decimal("1e-15")
    assert check.maximum_view_defect_volts < study.maximum_view_defect_volts
    assert (
        check.maximum_charge_rate_defect_amperes
        < study.maximum_charge_rate_defect_amperes
    )


def test_stricter_prospective_limit_and_wrong_native_roster_refuse() -> None:
    study = _study()
    panels = tuple(run_native_view(study, view) for view in VIEWS)
    tighter = replace(study, maximum_view_defect_volts=Decimal("0.00001"))
    assert not check_native_views(tighter, panels).converged
    with pytest.raises(ValueError, match="two exact solver views"):
        check_native_views(study, (panels[0], panels[0]))
    with pytest.raises(ValueError, match="unsupported RC numerical view"):
        run_native_view(study, "physical-board")
    with pytest.raises(ValueError, match="another type|exact issued study"):
        ResistorCapacitorLadderNativeFactory().build_provider(
            registry=CapabilityRegistry("rc-study-registry", (CAPABILITY,)),
            records=(study.model,),
            platform_ports=(),
        )


def test_issued_native_binding_selects_exact_local_study_decoder() -> None:
    study = _study()
    registry = CapabilityRegistry("rc-study-registry", (CAPABILITY,))
    provider = ResistorCapacitorLadderNativeFactory().build_provider(
        registry=registry, records=(study,), platform_ports=()
    )
    assert BINDING.required_issued_payload_schemas == (ResistorCapacitorLadderStudyConfig.SCHEMA,)
    assert provider.runners(registry)[0].manifest == CAPABILITY
    with pytest.raises(ValueError, match="external platform port"):
        ResistorCapacitorLadderNativeFactory().build_provider(
            registry=registry, records=(study,), platform_ports=(object(),)
        )
