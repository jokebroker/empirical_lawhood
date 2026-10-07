"""Explicit one-sampling-interval native software conformance; no response/Q claim."""

from pathlib import Path
from decimal import Decimal
import platform
import numpy as np

from empirical_lawhood.adapters.methods.preparation_applicability.conformance import PreparationApplicabilityNativeConformance,current_native_implementation
from empirical_lawhood.api.authoring_handoff import preflight_output_directory, write_exclusive_record
from empirical_lawhood.api.authoring_output import report_incomplete_output


def preparation_native_conformance(*, roots, output_dir: Path, conformance_id: str):
    """Exercise the actual native marcher on three explicit source kinds/two views.

    This bounded software check does not acquire a4096prefix, prepare a4496
    handoff, assay responses, qualify a constructor, issue or publish a law.
    """
    if tuple(root.cohort for root in roots)!=("q2","cir1","constructed"):
        raise ValueError("native conformance requires explicit q2/cir1/constructed allocations")
    if any(root.source_prefix is not None for root in roots):
        raise ValueError("native conformance cannot import a state")
    output_dir=preflight_output_directory(directory=output_dir)
    from empirical_lawhood.adapters.simulators.prepared_response.source import _Branch,march_native_intervals
    from empirical_lawhood.adapters.simulators.prepared_response.contracts import prepared_native_member,prepared_numerical_view
    from empirical_lawhood.adapters.simulators.six_matrix_response.model import ideal_state
    from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import derive_probe_roster
    from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import ResponseGeometryNativePassiveObserver
    from empirical_lawhood.adapters.simulators.preparation_applicability.records import PreparationApplicabilityStream
    from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.records import ConstructedPreparationStream
    from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.native import nominal_digest
    implementation=current_native_implementation()
    cells=[]
    for root in roots:
        for refinement in (1,2):
            initial=0.0 if root.cohort=="cir1" else 8.0
            state=ideal_state(q=2,alpha_tilde_x=initial,alpha_tilde_y=initial,constitution="00" if root.cohort=="cir1" else "11")
            roster=derive_probe_roster(config_fingerprint=root.fingerprint(),scientific_seed=root.seed_for("passive-probes"),rule_id="cc1-applicability-v1.passive-probes")
            branch=_Branch(state,ResponseGeometryNativePassiveObserver(roster=roster,y=state.positions[1],y_velocity=state.momenta[1],refinement=refinement),[0],[state.positions],[state.momenta])
            def stream(name):
                if root.cohort=="constructed":
                    return ConstructedPreparationStream(root.seed_for(name),nominal_digest(name),Decimal(".002"))
                return PreparationApplicabilityStream(root.seed_for(name))
            parent,bridge=stream("prefix"),stream("prefix-bridge")
            ramp=(np.arange(16*refinement)+1)/(256*refinement)
            schedule=initial+ramp[:,None]*(np.asarray((2/3,22/3))-initial)
            raw=march_native_intervals(branch=branch,start=0,end=16,refinement=refinement,
                member=prepared_native_member(),view=prepared_numerical_view(refinement),schedule=schedule,
                word=None,frame=None,rng=parent.generator(),stream=parent,bridge=bridge.generator(),bridge_stream=bridge,
                accumulate_parent_work=False,force_parent_ticks=400,progress=None)
            if (raw.disposition!="COMPLETE" or raw.completed!=16*refinement or tuple(raw.ticks)!=(0,16)
                or not np.isfinite(raw.positions).all() or not np.isfinite(raw.momenta).all()
                or raw.streams!=(parent,bridge)):
                raise ValueError("native software conformance failed its actual complete finite cell")
            cells.append((root.cohort,refinement,raw.completed,raw.disposition))
    if current_native_implementation()!=implementation:
        raise ValueError("native implementation changed during conformance")
    report=PreparationApplicabilityNativeConformance(conformance_id,implementation,platform.python_version(),np.__version__,tuple(cells),tuple(root.fingerprint() for root in roots),tuple(roots))
    output_dir.mkdir(parents=True,exist_ok=False)
    with report_incomplete_output(output_dir):
        write_exclusive_record(output_dir,"native-conformance.canonical.json",report)
    return report
