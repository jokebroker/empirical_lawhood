"""Shipped reactor native-contract inputs against a held publication and source."""

import json
import os
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from empirical_lawhood.adapters.simulators.reactor_prefix_response import prepared_input
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.config import freeze_discovery as freeze_local_discovery
from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.config import FeedQualificationDesign, freeze_discovery as freeze_feed_discovery
from empirical_lawhood.adapters.simulators.reactor_regime_response.config import OLD_PREPARED_DESIGN_SHA256, OLD_PREPARED_DOMAIN_SHA256
from empirical_lawhood.adapters.simulators.reactor_prefix_response.prepared_input import ReactorPreparedInput, check_prepared_input
from empirical_lawhood.api.codecs import load_registered_authoring

ROOT = Path(__file__).parents[1]
ROUTES = (
    "matched-replay-history",
    "local",
    "feed",
    "regime",
    "finite-control-frontier",
    "selected-action-response",
    "staged-pulse-response",
    "causal-response-study",
)


def _selection(route: str) -> ReactorPreparedInput:
    return load_registered_authoring(
        ROOT / f"experiments/reactor-response/{route}-input.json",
        root_schemas={ReactorPreparedInput.SCHEMA: ReactorPreparedInput},
        maximum_bytes=16 * 1024,
    )


@pytest.mark.parametrize("route", ROUTES)
def test_shipped_selection_requires_held_source_and_refuses_promotion(
    route: str,
) -> None:
    selection = _selection(route)
    with pytest.raises(ValueError, match="REACTOR_SOURCE_ROOT_REQUIRED"):
        check_prepared_input(selection, source_root=None, discovery_file=None)
    with pytest.raises(ValueError, match="development role"):
        replace(selection, evidence_role="PROSPECTIVE")


@pytest.mark.held
def test_held_inputs_build_eight_distinct_native_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_name = os.environ.get("REACTOR_HELD_SOURCE_ROOT")
    discovery_name = os.environ.get("REACTOR_LOCAL_DISCOVERY")
    if not source_name or not discovery_name:
        pytest.skip("authentic held reactor source and discovery are not mounted")
    source, discovery = Path(source_name), Path(discovery_name)
    raw_discovery = discovery.read_bytes()
    feed_atlas = freeze_feed_discovery(raw_discovery)
    feed_domain = feed_atlas.domains[0]
    assert feed_domain.fingerprint() == OLD_PREPARED_DOMAIN_SHA256
    assert FeedQualificationDesign(feed_atlas).fingerprint() == OLD_PREPARED_DESIGN_SHA256
    changed_domain = replace(
        feed_domain, mean=(feed_domain.mean[0] + Decimal("0.1"), *feed_domain.mean[1:])
    )
    with pytest.raises(ValueError, match="feed candidate support differs"):
        FeedQualificationDesign(replace(feed_atlas, domains=(changed_domain,)))
    changed_nomination = json.loads(raw_discovery)
    changed_nomination["model"]["domains"][0]["fit"]["mean"][0] += 0.1
    with pytest.raises(ValueError, match="changes the target nomination"):
        freeze_local_discovery(json.dumps(changed_nomination).encode())
    outputs = tuple(
        check_prepared_input(
            _selection(route),
            source_root=source,
            discovery_file=discovery
            if route not in ("matched-replay-history", "causal-response-study")
            else None,
        )
        for route in ROUTES
    )
    assert [entry["independent_roots_in_retained_roster"] for entry in outputs] == [
        42,
        64,
        64,
        208,
        192,
        96,
        480,
        96,
    ]
    assert [entry["actions"] for entry in outputs[1:4]] == [
        tuple(range(9)),
        (1, 4, 7),
        (1, 4, 7),
    ]
    assert outputs[5]["actions"] == (1, 4, 7)
    assert outputs[-1]["comparator_sources_authenticated"]
    assert len({entry["native_config_sha256"] for entry in outputs}) == 8
    assert outputs[0]["provider_built"] is True
    assert outputs[0]["runner_selected"] == 'ReactorBatchRunner'
    assert outputs[0]["missing_port_keys"] == ()
    assert all(
        entry["status"] == "UPSTREAM_PORTS_REQUIRED"
        and entry["provider_built"] is False
        and entry["missing_port_keys"] == entry["required_port_keys"]
        for entry in outputs[1:]
    )
    assert all(
        entry["retained_roster_exposed"]
        and entry["native_contact"] is False
        and not entry["campaign_candidate_compiled"]
        and not entry["campaign_issued"]
        and entry["native_tasks_executed"] == 0
        for entry in outputs
    )
    with pytest.raises(ValueError, match="publication identity differs"):
        check_prepared_input(
            _selection("feed"),
            source_root=source,
            discovery_file=ROOT / "experiments/reactor-response/guide.md",
        )
    with pytest.raises(ValueError, match="DISCOVERY_NOT_USED_FOR_ROUTE"):
        check_prepared_input(
            _selection("matched-replay-history"), source_root=source, discovery_file=discovery
        )
    monkeypatch.setattr(
        prepared_input, "COMPARATOR_SOURCES", (("ALT", "absent.py", "0" * 64),)
    )
    with pytest.raises(ValueError, match="COMPARATOR_REQUIRED"):
        check_prepared_input(
            _selection("causal-response-study"),
            source_root=source,
            discovery_file=None,
        )
