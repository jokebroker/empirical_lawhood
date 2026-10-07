"""Constructed mean-waveform wrappers over the shared current native marcher."""

from functools import partial
from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.records import nominal_digest as nominal_digest
from empirical_lawhood.adapters.simulators.preparation_applicability.native import acquire_phase as _acquire_phase, acquire_prefix as _acquire_prefix
from empirical_lawhood.adapters.simulators.preparation_applicability.contracts import WORDS as WORDS, SCHEDULES as SCHEDULES

acquire_phase = partial(_acquire_phase, constructed=True)
acquire_prefix = partial(_acquire_prefix, constructed=True)
