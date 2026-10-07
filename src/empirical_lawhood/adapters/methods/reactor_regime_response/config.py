"Unissued fixed science and allocation record for the reactor regime-response study."

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord

PREFIX = "reactor-regime-law-identification"
ROOT_BLOCKS = (
    ("fit", 32, 98000),
    ("nomination", 16, 98100),
    ("calibration", 32, 98200),
    ("qualification", 64, 98300),
    ("prospective", 64, 98400),
)
ROOTS = tuple(
    (f"reactor-regime-response-{role}-{index:03d}", role, index, first_seed + index)
    for role, count, first_seed in ROOT_BLOCKS
    for index in range(count)
)


@dataclass(frozen=True, slots=True)
class ReactorRegimeResponseDesign(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/reactor-regime-response-design'

    config_id: str = "reactor-regime-and-coefficient-identification"
    roots: tuple[tuple[str, str, int, int], ...] = ROOTS
    preparation_phase: int = 0
    baseline_windows_s: tuple[tuple[int, int], ...] = ((600, 1200), (7200, 9000), (14400, 16200))
    prepared_window_s: tuple[int, int] = (600, 6000)
    feed_words_kg_s: tuple[D, ...] = (D(0), D(".016"), D(".032"))
    requests_K: tuple[D, ...] = (D(".0004"), D(".0008"), D(".0012"), D(".0016"))
    probe_blocks_kg_s: tuple[D, ...] = (D(0), D(".032"), D(".032"), D(0), D(".032"), D(0))
    probe_block_s: int = 50
    constant_feed_kg_s: D = D(".016")
    active_preparation_s: int = 300
    assay_endpoint_offset_s: int = 330
    persistence_endpoint_offset_s: int = 930
    fixed_return_s: int = 940
    temperature_limit_K: D = D("356.2")
    maximum_predicted_cooling_K: D = D(".01")
    temperature_precision_K: D = D(".25")
    cooling_scale_floor_K: D = D(".00005")
    cooling_scale_slope: D = D(".5")
    temperature_numerical_padding_K: D = D(".01")
    cooling_numerical_padding_K: D = D(".000001")
    dose_numerical_padding_kg: D = D(".000001")
    scalar_residual_limit_K: D = D(".00001")
    split_canonical_grid: D = D(".000000001")
    split_perturbation: D = D(".00000001")
    split_crossing_fraction_limit: D = D(".01")
    local_max_leaves: int = 3
    local_max_depth: int = 2
    local_min_fit_roots_per_child: int = 12
    local_min_nomination_roots_per_leaf: int = 8
    local_min_relative_gain: D = D(".10")
    local_min_absolute_gain_K2: D = D(".000000000001")
    model_penalties: tuple[D, ...] = (D(".000001"), D(".001"), D(".1"))
    rbf_length_multipliers: tuple[D, ...] = (D(".5"), D(1), D(2))
    calibration_min_contact_roots: int = 29
    comparison_min_contact_roots: int = 48
    qualification_joint_floor: D = D(".90")
    prospective_joined_floor: D = D(".70")
    prospective_false_admission_ceiling: D = D(".10")
    confidence: D = D(".95")
    confirmatory_bootstrap_draws: int = 20000
    confirmatory_bootstrap_seed: int = 20260924
    descriptive_bootstrap_seed: int = 20260925
    phase_B_native_call_cap: int = 2592
    phase_C_native_call_cap: int = 5184
    phase_D_native_call_cap: int = 1152
    native_call_cap: int = 8928
    acquisition_wall_seconds: int = 12 * 3600
    acquisition_cpu_seconds: int = 36 * 3600
    maximum_workers: int = 4
    maximum_numerical_threads_per_worker: int = 1
    maximum_worker_bytes: int = 8 * 1024**3
    maximum_new_artifact_bytes: int = 256 * 1024**3
    minimum_external_reserve_bytes: int = 100 * 1024**3

    def __post_init__(self) -> None:
        for name, field in self.__dataclass_fields__.items():
            if name != "SCHEMA" and (
                getattr(self, name) != field.default
                or type(getattr(self, name)) is not type(field.default)
            ):
                raise ValueError(f"unamended reactor follow-up design required: {name}")
        if (
            len(self.roots) != 208
            or len({row[0] for row in self.roots}) != 208
            or len({row[3] for row in self.roots}) != 208
            or sum(
                (
                    self.phase_B_native_call_cap,
                    self.phase_C_native_call_cap,
                    self.phase_D_native_call_cap,
                )
            )
            != self.native_call_cap
        ):
            raise ValueError("reactor regime-response study independent-unit or resource roster differs")
