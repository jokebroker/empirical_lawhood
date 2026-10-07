"""Authentic retained source/discovery preflight for four reactor successors."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_selected_action_response.config import ClassicalDesign as SelectedActionDesign
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.config import ClassicalDesign as StagedPulseDesign
from empirical_lawhood.adapters.methods.reactor_causal_response.config import COMPARATOR_SOURCES, EmpiricalRecipe
from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.config import FeedQualificationDesign
from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.config import freeze_discovery as freeze_feed_discovery
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.config import WORDS, FrontierDesign
from empirical_lawhood.adapters.methods.reactor_matched_replay_forecast.science import forecast_design
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.config import LocalQualificationDesign
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.config import freeze_discovery as freeze_local_discovery
from empirical_lawhood.adapters.methods.reactor_regime_response.config import ReactorRegimeResponseDesign
from empirical_lawhood.adapters.simulators.reactor_selected_action_response.config import ClassicalNativeConfig as SelectedActionNativeConfig
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.config import ClassicalNativeConfig as StagedPulseNativeConfig
from empirical_lawhood.adapters.simulators.reactor_causal_response.config import EmpiricalNativeConfig
from empirical_lawhood.adapters.simulators.reactor_prepared_feed_qualification.config import FeedNativeConfig
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.config import FrontierNativeConfig
from empirical_lawhood.adapters.simulators.reactor_local_domain_qualification.config import LocalNativeConfig
from empirical_lawhood.adapters.simulators.reactor_regime_response.config import ReactorRegimeNativeConfig
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from empirical_lawhood.adapters._bounded_files import AdapterFileBoundError, read_bounded_contained

from .batch_input import _ReactorSourceSnapshot, load_batch_source
from .reactor_binding import ReactorTargetPortStore, ReactorSourceBinding, inspect_reactor_binding


@dataclass(frozen=True, slots=True)
class ReactorPreparedInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-prepared-input'
    config_id: str
    route: str
    evidence_role: str = "EXPOSED_DEVELOPMENT_NONPROMOTABLE"

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if (
            self.route
            not in (
                "matched-replay-history",
                "local",
                "feed",
                "regime",
                "finite-control-frontier",
                "selected-action-response",
                "staged-pulse-response",
                "causal-response-study",
            )
            or self.config_id != f"empirical-lawhood-reactor-{self.route}-input"
            or self.evidence_role != "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
        ):
            raise ValueError(
                "reactor prepared input changes its route or development role"
            )


def check_prepared_input(
    config: ReactorPreparedInput,
    *,
    source_root: Path | None,
    discovery_file: Path | None,
    upstream_binding: ReactorSourceBinding | None = None,
    port_store: ReactorTargetPortStore | None = None,
) -> dict[str, object]:
    """Build route-specific typed configs without importing old results as authority."""

    if source_root is None:
        raise ValueError("REACTOR_SOURCE_ROOT_REQUIRED")
    snapshot = _ReactorSourceSnapshot(source_root)
    source = load_batch_source(source_root, _snapshot=snapshot)
    needs_discovery = config.route not in ("matched-replay-history", "causal-response-study")
    if not needs_discovery and discovery_file is not None:
        raise ValueError("REACTOR_DISCOVERY_NOT_USED_FOR_ROUTE")
    raw = None
    atlas = prepared = domain = None
    if needs_discovery:
        if discovery_file is None:
            raise ValueError("REACTOR_DISCOVERY_REQUIRED")
        if (
            not discovery_file.is_absolute()
            or discovery_file.is_symlink()
            or not discovery_file.is_file()
        ):
            raise ValueError(
                "REACTOR_DISCOVERY_REQUIRED: absolute real publication file"
            )
        try:
            raw = read_bounded_contained(
                discovery_file.parent, discovery_file, maximum_bytes=4 * 1024**2
            )
        except AdapterFileBoundError as error:
            reason = "REACTOR_DISCOVERY_TOO_LARGE" if "byte limit" in str(error) else "REACTOR_DISCOVERY_REQUIRED"
            raise ValueError(reason) from error
        atlas = freeze_local_discovery(raw)
        prepared = freeze_feed_discovery(raw) if config.route != "local" else None
        domain = prepared.domains[0] if prepared is not None else None
    if config.route == "matched-replay-history":
        design = forecast_design()
        native = design.native
        actions = ("upstream-reference", "jacket-minus-1-K-at-7200-to-7800-s")
        receivers = design.receiver_ids
        roots = native.scenarios
    elif config.route == "local":
        assert atlas is not None
        design = LocalQualificationDesign(atlas)
        native = LocalNativeConfig(design)
        actions = tuple(
            sorted({action for item in atlas.candidates for action in item.actions})
        )
        receivers = tuple(
            sorted({r for item in atlas.candidates for r in item.nominated_receivers})
        )
        roots = design.assigned_roots
    elif config.route == "feed":
        assert prepared is not None and domain is not None
        design = FeedQualificationDesign(prepared)
        native = FeedNativeConfig(design)
        actions, receivers = domain.actions, domain.nominated_receivers
        roots = design.assigned_roots
    elif config.route == "regime":
        assert domain is not None
        design = ReactorRegimeResponseDesign()
        native = ReactorRegimeNativeConfig(design, domain)
        actions, receivers = domain.actions, domain.nominated_receivers
        roots = design.roots
    elif config.route == "finite-control-frontier":
        assert domain is not None
        design = FrontierDesign()
        native = FrontierNativeConfig(design, domain)
        actions = tuple(word.word_id for word in WORDS)
        receivers = ("temperature-K", "realized-feed-mass-kg", "guard-safety")
        roots = design.assigned_roots
    elif config.route == "selected-action-response":
        assert domain is not None
        design = SelectedActionDesign()
        native = SelectedActionNativeConfig(design, domain)
        actions, receivers = domain.actions, domain.nominated_receivers
        roots = design.assigned_roots
    elif config.route == "staged-pulse-response":
        assert domain is not None
        design = StagedPulseDesign()
        native = StagedPulseNativeConfig(design, domain)
        actions = tuple(
            word.request_id for word in (*design.local_requests, *design.pair_requests)
        )
        receivers = ("selected-action-local-cooling-K", "induced-history-cooling-K")
        roots = design.roots
    else:
        design = EmpiricalRecipe()
        native = EmpiricalNativeConfig()
        for arm, relative, expected in COMPARATOR_SOURCES:
            try:
                raw_comparator = snapshot.read_member(relative, 1024**2)
            except AdapterFileBoundError as error:
                raise ValueError(f"REACTOR_COMPARATOR_REQUIRED: {arm} {relative}") from error
            if sha256(raw_comparator).hexdigest() != expected:
                raise ValueError(f"REACTOR_COMPARATOR_PIN_MISMATCH: {arm} {relative}")
        actions = design.arms
        receivers = tuple(sorted({endpoint for _, _, endpoint in design.comparisons}))
        roots = tuple(
            (role, i) for role, count, _ in design.roles for i in range(count)
        )
    return {
        "config_id": config.config_id,
        "route": config.route,
        "source_sha256": source.fingerprint(),
        "discovery_sha256": None if raw is None else sha256(raw).hexdigest(),
        "design_sha256": design.fingerprint(),
        "native_config_sha256": native.fingerprint(),
        "prepared_domain_sha256": None if domain is None else domain.fingerprint(),
        "comparator_sources_authenticated": config.route == "causal-response-study",
        "independent_roots_in_retained_roster": len(roots),
        "nested_plant_views_per_root": 2,
        "actions": actions,
        "nominated_receivers": receivers,
        "retained_roster_exposed": True,
        "prior_discovery_outcome_visible": needs_discovery,
        "evidence_role": config.evidence_role,
        **inspect_reactor_binding(
            config.route,
            native,
            source,
            source_root=source_root,
            upstream_binding=upstream_binding,
            port_store=port_store,
            _snapshot=snapshot,
        ),
        "native_tasks_executed": 0,
        "campaign_candidate_compiled": False,
        "campaign_issued": False,
    }


__all__ = ['ReactorPreparedInput', "check_prepared_input"]
