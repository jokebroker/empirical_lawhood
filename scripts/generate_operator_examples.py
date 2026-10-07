# SPDX-License-Identifier: MPL-2.0
"""Generate inert, explicitly exposed record examples, never fresh authority."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

from empirical_lawhood.adapters.composition.reactor_prefix_response.fresh_authoring import AssignedReactorAuthoringProfile
from empirical_lawhood.adapters.simulators.reactor_prefix_response.assigned import ReactorPrefixAssignedUnit, ReactorPrefixAssignment, ReactorPrefixPriorCensus
from empirical_lawhood.adapters.simulators.reactor_prefix_response.panel import SCENARIOS
from empirical_lawhood.adapters.simulators.reactor_prefix_response.packaged_source import load_packaged_reactor_source
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes


def examples() -> dict[str, bytes]:
    fields = {
        "excluded_unit_ids": ("unit.example.exposed-a",),
        "proposed_unit_ids": ("unit.example.exposed-b",),
        "excluded_seed_ids": ("seed.example.exposed-a",),
        "proposed_seed_ids": ("seed.example.exposed-b",),
    }
    prepared = canonical_json_bytes({
        "schema": 'empirical-lawhood/composition/prepared-response/prepared-exposure-inspection',
        "version": "1.0.0", "value": fields,
    })
    finite = canonical_json_bytes({
        "schema": 'empirical-lawhood/methods/finite-response-law/native-exposure-metadata', **fields,
    })
    inventory = canonical_json_bytes({
        "schema": 'empirical-lawhood/examples/exposed-inventory', "version": "1.0.0",
        "value": {
            "inventory_id": "example.exposed-inventory",
            "evidence_role": "SYNTHETIC_EXPOSED_NONPROMOTABLE",
            "history_complete": False,
            "members": (
                {"locator": 'prepared-response-prior-exposure.json', "sha256": sha256(prepared).hexdigest()},
                {"locator": 'finite-response-prior-exposure.json', "sha256": sha256(finite).hexdigest()},
            ),
        },
    })
    census = ReactorPrefixPriorCensus(
        "example.prior-census",
        ("unit.example.exposed-a", "unit.example.exposed-b"),
        ("seed.example.exposed-a", "seed.example.exposed-b"),
        ObjectIdentity("example.exposed-inventory", 'empirical-lawhood/examples/exposed-inventory',
                       "1.0.0", sha256(inventory).hexdigest()),
    )
    stem = "example-reactor-exposed"
    cohort = f"{stem}.cohort"
    assignment = ReactorPrefixAssignment(
        cohort,
        tuple(ReactorPrefixAssignedUnit(f"{cohort}.{scenario.replace('_', '-')}", scenario, 910_000_000 + index)
              for index, scenario in enumerate(SCENARIOS)),
        ObjectIdentity.from_record(census.census_id, census),
        "EXPOSED_DEVELOPMENT_NONPROMOTABLE",
    )
    profile = AssignedReactorAuthoringProfile(
        "example.reactor-assigned-profile", stem, load_packaged_reactor_source().fingerprint(), assignment, census,
    )
    return {
        'prepared-response-prior-exposure.json': prepared,
        'finite-response-prior-exposure.json': finite,
        "exposed-inventory.json": inventory,
        "reactor-assigned-exposed.json": profile.canonical_bytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[1] / "experiments/reactor-response/exposed-inputs"
    if not args.check:
        directory.mkdir(exist_ok=True)
    for name, payload in examples().items():
        path = directory / name
        if args.check:
            if path.read_bytes() != payload:
                raise SystemExit(f"operator example drifted: {name}")
        else:
            path.write_bytes(payload)


if __name__ == "__main__":
    main()
