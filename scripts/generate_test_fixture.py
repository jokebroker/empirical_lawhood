# SPDX-License-Identifier: MPL-2.0
"""Rebuild the target-owned, nonpromotable reference lifecycle fixture."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from empirical_lawhood.api.models import CampaignPackage
from tests.runtime_platform.conftest import build_protocol_fixture


def fixture_bytes() -> bytes:
    fixture = build_protocol_fixture()
    return CampaignPackage(
        package_id="test.reference-campaign",
        run_plan_id=fixture.run_plan.run_plan_id,
        execution_plan_id=fixture.execution_plan.execution_plan_id,
        implementation_commit="a" * 40,
        system=fixture.system,
        experiment=fixture.experiment,
        campaign=fixture.campaign,
        frozen_proposal=fixture.frozen_proposal,
        authorization=fixture.authorization,
        protocol=fixture.template,
        registry=fixture.registry,
    ).canonical_bytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "tests/fixtures/reference-campaign.json"
    payload = fixture_bytes()
    if args.check:
        if path.read_bytes() != payload:
            raise SystemExit("synthetic reference fixture differs; regenerate and review")
    else:
        path.write_bytes(payload)


if __name__ == "__main__":
    main()
