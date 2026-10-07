"""Authorized direct-TORAX execution boundary and complete-panel orchestrator."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from decimal import Decimal
from importlib.metadata import version
from typing import Protocol

from empirical_lawhood.kernel.authority import AuthorityAction, ResourceBudget
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity

from .contracts import (
    ToraxAction,
    ToraxActionSpec,
    ToraxMatchedPanel,
    ToraxMappedField,
    ToraxNumericalView,
    ToraxPanelDisposition,
    ToraxPreparation,
    ToraxRolloutResult,
    ToraxRolloutState,
    ToraxRuntimeIdentity,
    ToraxValidityChecks,
)


@dataclass(frozen=True, slots=True)
class ToraxKernelOutput:
    """Ephemeral direct-kernel output before conversion to an immutable receipt."""

    state: ToraxRolloutState
    endpoint_delta_te_core_ev: Decimal | None
    preservation_margin: Decimal | None
    effort_j: Decimal | None
    validity: ToraxValidityChecks
    state_artifact: ArtifactIdentity | None
    action_artifact: ArtifactIdentity | None
    runtime_artifacts: tuple[ArtifactIdentity, ...]
    reason_codes: tuple[str, ...]


class DirectToraxKernel(Protocol):
    """Version-specific TORAX binding; it must not be backed by Gym--TORAX."""

    @property
    def runtime_id(self) -> str: ...

    def execute(
        self,
        preparation: ToraxPreparation,
        action: ToraxActionSpec,
        view: ToraxNumericalView,
    ) -> ToraxKernelOutput: ...


class ToraxExecutionAuthorityVerifier(Protocol):
    def require(
        self,
        authorization: ObjectIdentity,
        action: AuthorityAction,
        scope_id: str,
        budget: ResourceBudget,
    ) -> None: ...


class DirectToraxExecutionProvider:
    """Strict adapter-local provider around the installed direct TORAX kernel API."""

    def __init__(self, kernel: DirectToraxKernel) -> None:
        if kernel.runtime_id != "torax-1.4.2-cpu-x64-flagship":
            raise ValueError("direct TORAX kernel runtime identity differs")
        self._kernel = kernel

    def execute(
        self,
        preparation: ToraxPreparation,
        action: ToraxActionSpec,
        view: ToraxNumericalView,
    ) -> ToraxRolloutResult:
        if preparation.runtime.object_id != self._kernel.runtime_id:
            raise ValueError("preparation runtime differs from the direct TORAX kernel")
        output = self._kernel.execute(preparation, action, view)
        return ToraxRolloutResult(
            rollout_id=(
                f"rollout.{preparation.preparation_id}.{view.view_id}.{action.action.value.lower()}"
            ),
            preparation=ObjectIdentity.from_record(preparation.preparation_id, preparation),
            theta_id=preparation.theta.theta_id,
            view=view,
            action=action,
            state=output.state,
            endpoint_delta_te_core_ev=output.endpoint_delta_te_core_ev,
            preservation_margin=output.preservation_margin,
            effort_j=output.effort_j,
            validity=output.validity,
            state_artifact=output.state_artifact,
            action_artifact=output.action_artifact,
            runtime_artifacts=output.runtime_artifacts,
            reason_codes=output.reason_codes,
        )


class ToraxMatchedPanelRunner:
    def __init__(
        self,
        *,
        provider: DirectToraxExecutionProvider,
        authority: ToraxExecutionAuthorityVerifier,
    ) -> None:
        self._provider = provider
        self._authority = authority

    def run(
        self,
        *,
        preparation: ToraxPreparation,
        view: ToraxNumericalView,
        actions: tuple[ToraxActionSpec, ...],
        simulation_authorization: ObjectIdentity,
        write_authorization: ObjectIdentity,
        budget: ResourceBudget,
    ) -> ToraxMatchedPanel:
        if simulation_authorization == write_authorization:
            raise ValueError("simulation and external-write authority must be distinct")
        self._authority.require(
            simulation_authorization,
            AuthorityAction.SIMULATION_EXECUTION,
            preparation.preparation_id,
            budget,
        )
        self._authority.require(
            write_authorization,
            AuthorityAction.DATASET_TRANSFORMATION,
            preparation.preparation_id,
            budget,
        )
        if tuple(value.action for value in actions) != tuple(ToraxAction):
            raise ValueError("TORAX matched panel requires exact DOWN/HOLD/UP action order")
        realized = tuple(value.stages[-1].power_w for value in actions)
        if not realized[0] < realized[1] < realized[2]:
            raise ValueError("TORAX matched panel proxy powers must order DOWN < HOLD < UP")
        rollouts: list[ToraxRolloutResult] = []
        for action in actions:
            result = self._provider.execute(preparation, action, view)
            rollouts.append(result)
            if result.state is ToraxRolloutState.INTERRUPTED:
                break
        complete = len(rollouts) == 3 and all(
            value.state is ToraxRolloutState.COMPLETE for value in rollouts
        )
        if complete:
            disposition = ToraxPanelDisposition.COMPLETE
            reasons: tuple[str, ...] = ()
        else:
            disposition = ToraxPanelDisposition.INCOMPLETE
            reasons = ("MATCHED_PANEL_INCOMPLETE",)
        return ToraxMatchedPanel(
            panel_id=f"panel.{preparation.preparation_id}.{view.view_id}",
            preparation=ObjectIdentity.from_record(preparation.preparation_id, preparation),
            theta_id=preparation.theta.theta_id,
            view=view,
            rollouts=tuple(sorted(rollouts, key=lambda value: value.rollout_id)),
            disposition=disposition,
            independent_preparation_count=0,
            reason_codes=reasons,
        )


def inspect_installed_torax_runtime(
    *,
    runtime_artifacts: tuple[ArtifactIdentity, ...] = (),
) -> ToraxRuntimeIdentity:
    """Inspect package/platform identity without initializing a device or simulation."""

    return ToraxRuntimeIdentity(
        runtime_id="torax-1.4.2-cpu-x64-flagship",
        python_version=".".join(str(value) for value in sys.version_info[:3]),
        torax_version=version("torax"),
        jax_version=version("jax"),
        jaxlib_version=version("jaxlib"),
        numpy_version=version("numpy"),
        scipy_version=version("scipy"),
        platform_id=f"{platform.system().lower()}-{platform.machine().lower()}",
        backend="cpu",
        device_id="cpu-0",
        precision="float64",
        x64_enabled=True,
        runtime_artifacts=runtime_artifacts,
    )


def build_torax_config(
    preparation: ToraxPreparation,
    action: ToraxActionSpec,
    view: ToraxNumericalView,
) -> object:
    """Build the exact installed direct-TORAX config without executing it.

    This late import is deliberate: ordinary contract tests and mapping do not
    initialize JAX.  Production execution remains behind the separately
    authorized kernel port above.
    """

    from torax._src.torax_pydantic import model_config  # type: ignore[import-untyped]

    fields = {value.field_id: value for value in preparation.fields}
    theta = preparation.theta
    te = fields["electron-temperature"]
    ti = fields["ion-temperature"]
    ne = fields["electron-density"]
    current = fields["plasma-current"]
    if not (te.radial_coordinates == ti.radial_coordinates == ne.radial_coordinates):
        raise ValueError("direct TORAX profiles do not share a radial domain")
    if len(current.values) != 1:
        raise ValueError("direct TORAX fixed-current preparation requires one total current")
    main_ion_symbols = {"deuterium": "D", "tritium": "T"}
    impurity_symbols = {"carbon": "C", "neon": "Ne", "argon": "Ar"}
    if theta.main_ion not in main_ion_symbols or theta.impurity not in impurity_symbols:
        raise ValueError("TORAX species identifier has no qualified direct-package symbol")

    def profile(
        field: ToraxMappedField,
        *,
        scale: Decimal = Decimal(1),
    ) -> dict[float, dict[float, float]]:
        return {
            0.0: {
                float(radius): float(value * scale)
                for radius, value in zip(
                    field.radial_coordinates,
                    field.values,
                    strict=True,
                )
            }
        }

    config = {
        "profile_conditions": {
            "Ip": float(current.values[0]),
            "T_i": profile(ti, scale=Decimal("0.001")),
            "T_e": profile(te, scale=Decimal("0.001")),
            "n_e": profile(ne),
            "T_i_right_bc": float(ti.values[-1] * Decimal("0.001")),
            "T_e_right_bc": float(te.values[-1] * Decimal("0.001")),
            "n_e_right_bc": float(ne.values[-1]),
            "n_e_nbar_is_fGW": False,
            "n_e_right_bc_is_fGW": False,
            "normalize_n_e_to_nbar": False,
            "initial_j_is_total_current": True,
            "initial_psi_from_j": True,
            "initial_psi_mode": "j",
        },
        "plasma_composition": {
            "main_ion": main_ion_symbols[theta.main_ion],
            "impurity": {
                "impurity_mode": "fractions",
                "species": {impurity_symbols[theta.impurity]: 1.0},
            },
            "Z_eff": float(theta.zeff),
        },
        "numerics": {
            "t_initial": 0.0,
            "t_final": float(view.horizon_s),
            "exact_t_final": True,
            "fixed_dt": float(view.timestep_s),
            "adaptive_dt": False,
            "evolve_ion_heat": True,
            "evolve_electron_heat": True,
            "evolve_current": False,
            "evolve_density": False,
        },
        "geometry": {
            "geometry_type": "circular",
            "n_rho": view.radial_cells,
            "R_major": float(theta.major_radius_m),
            "a_minor": float(theta.minor_radius_m),
            "B_0": float(theta.toroidal_field_t),
            "elongation_LCFS": float(theta.elongation_lcfs),
        },
        "sources": {
            "generic_current": {"mode": "ZERO"},
            "generic_heat": {
                "model_name": "gaussian",
                "gaussian_width": float(theta.source_width),
                "gaussian_location": float(theta.source_radial_location),
                "P_total": float(action.stages[-1].power_w),
                "electron_heat_fraction": float(theta.electron_heat_fraction),
                "absorption_fraction": float(theta.absorbed_power_fraction),
                "mode": "MODEL_BASED",
            },
            "ei_exchange": {"mode": "MODEL_BASED", "Qei_multiplier": 1.0},
        },
        "neoclassical": {
            "bootstrap_current": {"model_name": "zeros"},
            "transport": {"model_name": "zeros"},
        },
        "pedestal": {"model_name": "no_pedestal"},
        "transport": {
            "model_name": "constant",
            "chi_i": float(theta.chi_i_m2_s),
            "chi_e": float(theta.chi_e_m2_s),
            "D_e": float(theta.particle_diffusivity_m2_s),
            "V_e": float(theta.particle_convection_m_s),
        },
        "solver": {
            "solver_type": "linear",
            "theta_implicit": 1.0,
            "use_predictor_corrector": False,
            "use_pereverzev": False,
            "implicit_solver_type": "thomas",
        },
        "time_step_calculator": {"calculator_type": "fixed"},
    }
    return model_config.ToraxConfig(**config)


__all__ = [
    "DirectToraxExecutionProvider",
    "DirectToraxKernel",
    "ToraxExecutionAuthorityVerifier",
    "ToraxKernelOutput",
    "ToraxMatchedPanelRunner",
    'build_torax_config',
    "inspect_installed_torax_runtime",
]
