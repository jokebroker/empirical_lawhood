"""Frozen standalone export from installed numerical code and pinned actuator block."""

from __future__ import annotations
from dataclasses import asdict
import inspect
import json
from . import controller, numerical
from .numerical import FrozenFit
from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator


def export_source(
    model: FrozenFit, q: float, *, arm: str = "EL", scheduled_source: bytes | None = None
) -> bytes:
    from empirical_lawhood.adapters.simulators.reactor_causal_response.actuator_statements import project_stages

    if arm not in ("EL", "F0", "F1", "MARGIN", "FIXED") or (arm == "FIXED") != (
        scheduled_source is not None
    ):
        raise ValueError("undeclared empirical export arm/source")
    extraction = inspect.getsource(project_stages)
    # Standard-library/NumPy-only module. No platform storage or plant equations.
    source = inspect.getsource(numerical)
    controller_source = inspect.getsource(controller)
    controller_source = "\n".join(
        line
        for line in controller_source.splitlines()
        if not line.startswith("from __future__") and not line.startswith("from .numerical")
    )
    projection = (
        inspect.getsource(Actuator.project) + "\n" + inspect.getsource(Actuator.project_request)
    )
    projector = "class ExportActuator:\n" + projection
    projector += "\n    project_stages = staticmethod(project_stages)\n"
    actuator_params = vars(Actuator().params)
    payload = json.dumps(asdict(model), sort_keys=True, allow_nan=False)
    trailer = f"""
from types import SimpleNamespace
_FIT = json.loads({payload!r})
_MODEL = FrozenFit(_FIT['family'],_FIT['penalty'],tuple(_FIT['mean']),tuple(_FIT['scale']),tuple(map(tuple,_FIT['operator'])),tuple(SupportCell(c['clock_stratum'],c['action'],tuple(c['roots']),tuple(c['lower']),tuple(c['upper'])) for c in _FIT['support']),_FIT['training_digest'],_FIT['rows'],tuple(_FIT['columns']))
class Controller(NumericalController):
    def __init__(self):
        actuator = ExportActuator()
        actuator.params = SimpleNamespace(**{actuator_params!r})
        super().__init__(_MODEL,{q!r},actuator)
"""
    output = source + "\n" + controller_source + "\n" + extraction + "\n" + projector + trailer
    if arm != "EL":
        from .comparators import DiagnosticController

        output += "\nfrom dataclasses import replace\n" + inspect.getsource(DiagnosticController)
        if scheduled_source is not None:
            from hashlib import sha256
            from .config import COMPARATOR_SOURCES

            expected = next(d for name, _, d in COMPARATOR_SOURCES if name == "SCHEDULED_BACKOFF_HALF")
            if sha256(scheduled_source).hexdigest() != expected:
                raise ValueError("FIXED export scheduled source differs")
            output += "\n" + "\n".join(
                line
                for line in scheduled_source.decode().splitlines()
                if not line.startswith("from __future__")
            ).replace("class Controller:", "class ScheduledProposal:")
        output += f"""
class Controller(DiagnosticController):
    def __init__(self):
        actuator = ExportActuator()
        actuator.params = SimpleNamespace(**{actuator_params!r})
        super().__init__(_MODEL,{q!r},actuator,{arm!r},{"ScheduledProposal(back_off=0.5)" if arm == "FIXED" else "None"})
"""
    return output.encode()
