"""Render only the shared workflow table in the experiment directory entry.

SPDX-License-Identifier: MPL-2.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from empirical_lawhood.api.workflows import list_workflows
from empirical_lawhood.cli.workflows import workflow_details

START = "<!-- BEGIN GENERATED WORKFLOWS -->"
END = "<!-- END GENERATED WORKFLOWS -->"
ROOT = Path(__file__).resolve().parents[1]

# Presentation only: scientific limits remain in the workflow facts.
DISPLAY_STATES = {
    'matrix-geometry': '🟢 Current operation',
    'selected-events': '🟢 Current operation',
    'matrix-history-analysis': '🟢 Current acquisition / analysis; 🟠 Exposed-array import',
    'matrix-tangent': '🟢 Current acquisition / analysis',
    'matrix-transient': '🟢 Current retained analysis',
    'preparation-diagnostics': '🟢 Current retained analysis',
    'rc-challenges': '🟢 Current issued phases',
    'matrix-inputs': '🟢 Current source export; 🟠 Operand checks / transforms',
    'preparation-applicability': '🟢 Current phases; 🟡 Development screens',
    'rc-information': '🟢 Analytical operation',
    'reactor-response': '🟢 Assigned prefix lifecycle; 🔵 Development authoring; 🟠 Other input checks',
    'prepared-response': '🔵 Development authoring; 🟡 Native check; 🟠 Parent preflight',
    'information-response': '🔵 Development authoring',
    'causal-response': '🔵 Development authoring',
    'finite-response-law': '🟢 Current C32/E64 phases; 🟡 Development diagnostics; 🟠 Input checks',
    'causal-transfer-audit': '🟠 Authenticated parent preflight',
    'rc-ladder-response': '🔵 Development authoring; 🟡 Native check',
    'electron-gas-response': '🔵 Development authoring; 🟡 Reference check',
    'lattice-pairing-method': '🔵 Development authoring',
    'battery-response-and-restart': '🟡 Development check',
    'tokamak-heat-response': '🟡 Development check',
    'tokamak-control-response': '🟡 Development check',
    'neuron-current-response': '🟡 Development check',
    'reactor-flow-response': '🟡 Development check',
    'reaction-diffusion-response': '🟡 Development check',
    'grid-response-inputs': '🟡 Development check',
    'material-control-inputs': '🟠 Input check',
    'laser-archive-inspection': '🟠 Retrospective import / inspection',
    'response-method-reference': '🟡 Development check',
    'propulsion-reliability-reference': '🟡 Development check',
}


def render_table() -> str:
    workflows = list_workflows()
    if set(DISPLAY_STATES) != {workflow.workflow_id for workflow in workflows}:
        raise ValueError("Workflow presentation states must cover the exact closed inventory")
    lines = [
        "| Workflow / boundary | Current state | Commands and input roles | Environment / first stop / evidence ceiling |",
        "|---|---|---|---|",
    ]
    for workflow in workflows:
        facts = workflow_details(workflow)
        commands = "<br>".join(
            f"`{entry['command']}` ({entry['boundary']})"
            for entry in facts["operations"]
        )
        inputs = "; ".join(
            f"`{Path(item.path).name}`: {item.role}" for item in workflow.inputs
        )
        guide = workflow.guide.removeprefix("experiments/")
        lines.append(
            f"| [{workflow.title}]({guide})<br>{workflow.kind} | "
            f"{DISPLAY_STATES[workflow.workflow_id]} | {commands}<br>{inputs} | {workflow.environment}<br>"
            f"{workflow.first_missing_prerequisite}<br>{workflow.scientific_ceiling} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "experiments/README.md"
    text = path.read_text(encoding="utf-8")
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("Experiment entry must have exactly one workflow table region")
    before, remainder = text.split(START)
    _, after = remainder.split(END)
    generated = before + START + "\n\n" + render_table() + "\n" + END + after
    if args.check:
        if generated != text:
            print("Workflow index differs; run scripts/generate_workflow_index.py")
            return 1
    else:
        path.write_text(generated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
