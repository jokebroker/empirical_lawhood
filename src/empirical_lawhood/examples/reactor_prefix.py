"""Run the pinned, two-decision native reactor prefix as a local demonstration.

This uses the existing scientific records and reactor bridge. It creates no
campaign issue, execution receipt, law qualification or admission decision.
"""

# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
from typing import Any

from empirical_lawhood.adapters.methods.reactor_prefix_response.design import reactor_science_design, reactor_system
from empirical_lawhood.adapters.methods.reactor_prefix_response.words import action_words
from empirical_lawhood.adapters.simulators.reactor_prefix_response.contracts import PARAMS_SHA256, PLANT_SHA256, PUBLIC_SCENARIOS_SHA256, UPSTREAM_COMMIT
from empirical_lawhood.adapters.simulators.reactor_prefix_response.packaged_source import load_packaged_reactor_source
from empirical_lawhood.adapters.simulators.reactor_prefix_response.panel import BRANCHES, SCENARIOS, VIEWS, ReactorPrefixEpisode, ReactorPrefixPanel, ReactorSourceBundle, acquire_prefix_branch


CLAIM_LABEL = "demonstration only / unqualified local measurement"


class ReactorPrefixRunFailed(RuntimeError):
    """A genuine branch failed; a partial record was written without a full panel."""


def _json_record(record: ReactorPrefixEpisode) -> dict[str, Any]:
    return json.loads(record.canonical_bytes())


def _write_json(path: Path, document: dict[str, Any]) -> None:
    # An existing run is not overwritten or mistaken for this invocation.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _source_hashes(source: ReactorSourceBundle) -> dict[str, str]:
    return {
        "tests/plant.py": sha256(source.plant_source.encode()).hexdigest(),
        "environment/spec/plant_params.json": sha256(
            source.plant_params.encode()
        ).hexdigest(),
        "environment/spec/scenarios_public.json": sha256(
            source.public_scenarios.encode()
        ).hexdigest(),
    }


def _delivery_stages(episode: ReactorPrefixEpisode) -> list[dict[str, Any]]:
    stages = []
    for delivery in episode.deliveries:
        stages.append(
            {
                "decision_id": delivery.command.decision_id,
                "time_s": str(delivery.command.time_s),
                "requested": {
                    "feed_kg_s": str(delivery.command.feed_kg_s),
                    "jacket_k": str(delivery.command.jacket_k),
                },
                "accepted": {
                    "feed_kg_s": str(delivery.accepted_feed_kg_s),
                    "jacket_k": str(delivery.accepted_jacket_k),
                },
                "applied": {
                    "feed_kg_s": str(delivery.applied_feed_kg_s),
                    "jacket_k": str(delivery.applied_jacket_k),
                },
                "realized_exposures": [
                    {
                        "time_s": str(exposure.time_s),
                        "duration_s": str(exposure.duration_s),
                        "feed_kg_s": str(exposure.feed_kg_s),
                        "jacket_k": str(exposure.jacket_k),
                    }
                    for exposure in delivery.exposures
                ],
                "next_callback": {
                    "time_s": str(delivery.next_measurement.time_s),
                    "temperature_k": str(delivery.next_measurement.t_reactor_k),
                },
            }
        )
    return stages


def _partial(
    output_dir: Path,
    *,
    branch: tuple[str, str, str],
    episodes: list[ReactorPrefixEpisode],
    error: str,
    source_hashes: dict[str, str],
) -> None:
    _write_json(
        output_dir / "partial.json",
        {
            "label": CLAIM_LABEL,
            "status": "incomplete",
            "failed_branch": list(branch),
            "failure": error,
            "completed_branches": sum(e.failure_code is None for e in episodes),
            "observed_deliveries": sum(len(e.deliveries) for e in episodes),
            "recorded_episodes": [_json_record(e) for e in episodes],
            "source_sha256": source_hashes,
        },
    )


def run_reactor_prefix(output_dir: Path) -> dict[str, Any]:
    """Acquire the predeclared 20 branches and write one truthful local report."""
    if output_dir.is_symlink():
        raise ValueError("reactor example output directory must not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(
        (output_dir / name).exists()
        for name in ("panel.json", "report.json", "report.md", "partial.json")
    ):
        raise FileExistsError(
            "reactor example output already exists; choose a fresh directory"
        )

    source = load_packaged_reactor_source()
    hashes = _source_hashes(source)
    if hashes != {
        "tests/plant.py": PLANT_SHA256,
        "environment/spec/plant_params.json": PARAMS_SHA256,
        "environment/spec/scenarios_public.json": PUBLIC_SCENARIOS_SHA256,
    }:
        raise ValueError("packaged reactor input digest differs")
    design = reactor_science_design()
    system = reactor_system(design)
    words = action_words(design)
    episodes: list[ReactorPrefixEpisode] = []
    for branch in BRANCHES:
        try:
            branch_panel = acquire_prefix_branch(design.native, source, branch)
        except Exception as error:
            _partial(
                output_dir,
                branch=branch,
                episodes=episodes,
                error=f"{type(error).__name__}: {error}",
                source_hashes=hashes,
            )
            raise ReactorPrefixRunFailed(
                f"native branch {branch} failed; see partial.json"
            ) from error
        episode = branch_panel.episodes[0]
        episodes.append(episode)
        if episode.failure_code is not None:
            _partial(
                output_dir,
                branch=branch,
                episodes=episodes,
                error=episode.failure_code,
                source_hashes=hashes,
            )
            raise ReactorPrefixRunFailed(
                f"native branch {branch} failed; see partial.json"
            )

    panel = ReactorPrefixPanel(design.native, tuple(episodes))
    paired = []
    for scenario in SCENARIOS:
        for view, _step in VIEWS:
            pair = {
                e.arm_id: e
                for e in episodes
                if e.scenario_id == scenario and e.view_id == view
            }
            comparator = pair["comparator"].deliveries[1].next_measurement
            pulse = pair["pulse"].deliveries[1].next_measurement
            if comparator.time_s != Decimal(20) or pulse.time_s != Decimal(20):
                raise ValueError("reactor receiver is not the 20 s callback")
            paired.append(
                {
                    "unit": scenario,
                    "view": view,
                    "comparator_temperature_k": str(comparator.t_reactor_k),
                    "pulse_temperature_k": str(pulse.t_reactor_k),
                    "pulse_minus_comparator_k": str(
                        pulse.t_reactor_k - comparator.t_reactor_k
                    ),
                }
            )
    means = {
        view: str(
            sum(
                (
                    Decimal(row["pulse_minus_comparator_k"])
                    for row in paired
                    if row["view"] == view
                ),
                Decimal(0),
            )
            / Decimal(len(SCENARIOS))
        )
        for view, _step in VIEWS
    }
    report = {
        "label": CLAIM_LABEL,
        "scope": "native two-decision prefix; no production campaign issue or qualification",
        "source_upstream_commit": UPSTREAM_COMMIT,
        "source_sha256": hashes,
        "runtime": {
            "python": platform.python_version(),
            "numpy": metadata.version("numpy"),
            "package": metadata.version("empirical-lawhood"),
        },
        "relation_id": system.relation.relation_id,
        "system_id": system.system_id,
        "science_design_schema": design.SCHEMA,
        "action_word_ids": [word.word_id for word in words],
        "receiver": "20 s callback temperature, carrying the delayed 10 s state",
        "independent_units": len(SCENARIOS),
        "views": [view for view, _step in VIEWS],
        "complete_branches": len(episodes),
        "observed_deliveries": sum(len(e.deliveries) for e in episodes),
        "delivery_stages": [
            {
                "unit": e.scenario_id,
                "view": e.view_id,
                "arm": e.arm_id,
                "decisions": _delivery_stages(e),
            }
            for e in episodes
        ],
        "paired_by_unit_and_view": paired,
        "descriptive_mean_pulse_minus_comparator_k_by_view": means,
    }
    with (output_dir / "panel.json").open("xb") as stream:
        stream.write(panel.canonical_bytes())
    _write_json(output_dir / "report.json", report)
    lines = [
        "# Reactor prefix: local demonstration",
        "",
        f"**{CLAIM_LABEL}**",
        "",
        f"Complete native branches: {len(episodes)}; observed deliveries: {report['observed_deliveries']}.",
        f"Independent units: {len(SCENARIOS)}; two views and two arms stay nested within each unit.",
        "",
        "| Unit | View | Comparator K | Pulse K | Pulse − comparator K |",
        "|---|---|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            f"| {row['unit']} | {row['view']} | {row['comparator_temperature_k']} | "
            f"{row['pulse_temperature_k']} | {row['pulse_minus_comparator_k']} |"
        )
    lines.extend(
        ("", "Descriptive mean of five paired unit contrasts, separately by view:", "")
    )
    for view, _step in VIEWS:
        lines.append(f"- {view}: {means[view]} K (n = 5 units)")
    lines.extend(
        (
            "",
            "The typed panel and report.json retain requested, accepted, applied and realized stages.",
            'This demonstration makes no law qualification, admission, safety, prospective success or production workflow claim.',
            "",
        )
    )
    with (output_dir / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return report
