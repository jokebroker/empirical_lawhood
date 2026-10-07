# SPDX-License-Identifier: MPL-2.0
"""Generate the public CLI reference from the installed command tree and metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from typer.main import get_command

from empirical_lawhood.cli.app import app
from empirical_lawhood.cli.introspection import CliCommandFact, canonical_click_tree
from empirical_lawhood.cli.metadata import (
    GROUP_METADATA,
    COMMAND_METADATA,
    invocation_contracts,
)
from empirical_lawhood.cli.reference_contract import join_command_metadata


HEADER = "# CLI reference\n\nSPDX-License-Identifier: CC-BY-4.0\n\nThis reference gives information about the installed `empirical-lawhood` commands.\nIt gives command prerequisites and limits for the operator workflow.\nUse it to find the required inputs for each command.\n\n## Form an invocation\n\nUse the [workflow catalogue](../experiments/README.md) to select a task procedure,\nits input files and environment. This reference supplies command inputs,\neffects, output conventions and refusal exits.\n\n```sh\nempirical-lawhood --help\nempirical-lawhood workflow show rc-ladder-response\nempirical-lawhood config validate --config /absolute/work/model.json --consumer rc-ladder-model --format json\n```\n\nFrom the selected locked checkout environment, prefix each invocation with\n`uv run --no-sync`. The model path above must be your copied RC input; follow\n[configuration](configuration.md) to obtain and prepare it. Use\n`empirical-lawhood <group> <verb> --help` for conditional options.\nPut global options before the group and command options after the verb:\n\n```text\nempirical-lawhood [global options] <group> <verb> [command options]\n```\n\nCommands have different output conventions: some print text or canonical\nreports, and API commands can emit a JSON envelope with `--format json`.\nRetained products require the selected command's explicit destination options.\nThe [results and exit codes](#results-and-exit-codes) tables below identify each\nconvention; [results and failures](results-and-failures.md) explains the files\nand the permitted recovery action. Operational success and scientific support\nare separate outcomes.\n\n## Global inputs\n\n`empirical-lawhood` is the only installed command.\nIf a command needs `--project-root` or `--operator-profile`, put these options before its group.\nPut each optional trust or authoring option before the command group:\n\n- `--dataset-projection-trust`\n- `--approval-checker-trust`\n- `--reactor-authoring-dir`\n- `--circuit-authoring-dir`\n- `--electron-gas-authoring-dir`\n- `--synthetic-material-authoring-dir`\n- `--authoring-dir` (closed current integration handoff)\n\nThese inputs do not grant scientific authority.\nThe CLI has no implicit home directory default for a trust path.\n\n## Portable and source requirements\n\nThe wheel supplies help, the portable reactor example, document validation, static inspection, and family checks marked `none` below.\nInstall each selected native stack separately.\n`doctor` can inspect an explicit storage profile without catalog creation.\nThe profile can be read-only.\n\nThe following checks have conditional requirements:\n\n- prepared-response parent checks\n- finite-response stage checks\n- response-composition checks\n- reactor prepared checks\n\nTheir preliminary refusals are portable.\nFor authenticated parent or upstream reads, use the selected project and guarded store.\nMatrix-response child authoring also needs the target source.\nDraft `campaign validate` is portable.\nAn executable package requires authority replay from the configured store.\n\nCommands marked `target checkout` require that checkout to run its own tracked `src/empirical_lawhood` package.\nA wheel has no claim to that checkout's clean Git implementation identity.\nThe clean tree, exact stored approvals, applicable native inputs, and authority are separate gates.\n\n## Capability and evidence limits\n\nA discoverable static capability descriptor can still require source data, native software, or authority for its executable binding.\nThe `list` and `show` commands do not read source bytes or grant readiness.\nThe reactor-prefix example is an unqualified native demonstration.\nThe separate RC-information example is an analytical construction with an\nindependent equation check, without sampled confirmation.\nResistor-capacitor, electron-gas and disclosed lattice-pairing method workflows\nretain their target-owned candidate/provider routes.\nThe ambient-pressure material programme remains unqualified.\n\nThe [current paper integrations](integrations.md) include these implemented\noperations. Their guides identify actual input producers and the next stage:\n\n| Route | Current software operation | Scientific and access boundary |\n|---|---|---|\n| [RC challenges](../experiments/rc-challenges/guide.md) | Four independently authored canary, nomination, development and evaluation phases; issued execution, exact results and recovery | Numeric-source custody and exact preceding phase results; separate current issue, execution and applicable reveal authority |\n| [Finite response law](../experiments/finite-response-law/guide.md) | Frozen nomination, current 32-root calibration/qualification and 64-root evaluation authoring, issued lifecycle and result joins | Fixed scientific operands, complete exposure census and current parents/grants; supplied sample seeds remain exposed |\n| [Preparation applicability](../experiments/preparation-applicability/guide.md) | Ordinary D32/E64 screens, constructed Q8/E32 authoring/lifecycle and saved 24-root readiness | Unchanged OriginalF; exact current Q prerequisite for constructed E; protected reads require current grants |\n| [Matrix scans and saved analyses](integrations.md) | Geometry and conditional-event development operations; current history/tangent acquisition; transient, baseline and preparation diagnostics | Development or outcome-visible analysis as stated by each family; exact complete inputs and custody, with current grants before protected diagnostics |\n\nAuthoring and no-contact provider/graph proof execute no scientific tasks and\ncreate no authority. An implemented issued route still requires its actual\nsource, cohort, records and separately authorized operations. Current software\nintegration does not reproduce historical paper results or establish fresh\nscientific qualification. Full scientific reruns remain separate work.\n\nThe following reference worlds supply only bounded native or truth-known reference checks:\n\n- Brian2\n- Cantera\n- FiPy\n- PyBaMM\n- Grid2Op\n- TORAX\n- physical scale-morphism\n- propulsion-reliability\n\nThe reactor batch command is a held-source preflight.\nThe prepared-response, information-response and causal-response held-input\ncandidate routes retain their nonpromotable development boundary. Prepared and\nfinite excluded canaries and finite calibration/stage input checks remain\nbounded diagnostics beside the current integration routes above. An accepted\npreflight or a new label supplies no missing scientific eligibility.\n\nGlenn has retrospective local import and preview only.\nMaterial control has held input wrapping and preflight only.\nThe Gym--TORAX native development check needs the target source checkout.\nCandidate context is not automatically available for arbitrary substrates.\n\n## Exact command prerequisites\n\n`Required config / records / ID` lists required Click arguments and options.\nIf you supply a path, its file must contain a strict document.\nUse per-command `--help` and the guide links below to find accepted inputs and their constraints.\nSome commands also accept optional records.\nThe API validates the exact relationships between those records.\n`Source checkout` means a clean target checkout that runs its own package for a source-identity claim.\nOther store operations also use that explicit project root.\n`Profile` is a typed `OperatorStorageProfile` path.\nIt is not a storage default.\n\n| Command | Project | Storage | Required config / records / ID | Native software | Authority and effects | Status |\n|---|---|---|---|---|---|---|\n"

FOOTER = '\n## Installed options and help\n\nThe generator converts the installed Click option states and help strings for this table.\nThe table shows required inputs and defaults.\nTo read the full invocation and conditional options, use `empirical-lawhood <group> <verb> --help`.\n\n| Path | Kind | Options and arguments | Help |\n|---|---|---|---|\n{structure}\n\n## Results and exit codes\n\nCommands with the `api` convention return a structured `ApiResult`.\nWhen a command returns an `ApiResult`, `--format json` writes sorted compact JSON.\nArgument and input preconditions can refuse before an `ApiResult` exists.\nThese API commands write their failures to stderr.\nNative checks, authoring helpers, source operations, and the demonstration use the separate conventions in the table.\nOperational success does not imply scientific support.\nA result can report a negative or unevaluable claim.\n\n| Command | Output | Error convention |\n|---|---|---|\n{contracts}\n\nAll commands use exit 2 for Click argument or option errors.\nThe following list gives the conventions:\n\n- `api`: the common result-to-exit mapping in the following table.\n- `refusal-3`: handled schema, input, prerequisite or native-check errors exit 3.\n- `refusal-5`: handled authoring, source, integrity or storage errors exit 5.\n- `invalid-2`: handled document or numerical-check input errors exit 2.\n- `source`: missing `--yes` exits 3. Source authority failures exit 4. Custody or storage failures exit 5.\n- `example-1`: demonstration failures, including an existing output directory, exit 1.\n\nThe following table applies only to the **common API envelope**:\n\n| Exit | Meaning |\n|---:|---|\n| 0 | operation completed. Inspect its scientific verdict separately |\n| 2 | invalid authoring or schema |\n| 3 | missing readiness, prerequisite or write confirmation |\n| 4 | authority missing or refused |\n| 5 | identity, custody, integrity or storage guard refused |\n| 6 | execution or recovery failed |\n| 7 | requested object absent |\n| 8 | immutable identity or concurrent-write conflict |\n\nThe `example reactor-prefix` command has its own simple text result.\nIt writes only to the required output directory.\nIt reports demonstration measurements.\nIt does not issue a campaign, an evidence receipt, or a law qualification.\n\n## Maintain this reference\n\nThe generator reads the installed Typer/Click tree and exact command metadata.\nFrom the selected locked environment, run\n`uv run --no-sync python scripts/generate_cli_reference.py` to regenerate;\nadd `--check` to compare with the saved reference. Change the generator or\ncommand metadata owner, then inspect the generated diff. The\n[contribution guide](../CONTRIBUTING.md#documentation-policy) governs maintained\nand generated documentation.\n'


def escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def required_inputs(fact: CliCommandFact) -> str:
    return (
        ", ".join(
            f"`{parameter.declarations[0]}`"
            for parameter in fact.parameters
            if parameter.required
        )
        or "none"
    )


def render() -> str:
    facts = canonical_click_tree(get_command(app))
    entries = join_command_metadata(facts, GROUP_METADATA, COMMAND_METADATA)
    by_path = {entry.fact.command: entry for entry in entries}
    contracts = invocation_contracts()
    leaves = [fact for fact in facts if fact.command_kind == "command"]
    lines = [HEADER.rstrip()]
    for fact in leaves:
        command = fact.command
        metadata = by_path[command].metadata
        contract = contracts[command]
        project, storage = contract.project, contract.storage
        if metadata.native_software is None or metadata.native_status is None:
            raise ValueError(f"missing native prerequisite/status for {command}")
        native, status = metadata.native_software, metadata.native_status
        authority = metadata.authority + ". " + metadata.effects
        lines.append(
            f"| `{command}` | {project} | {storage} | {required_inputs(fact)} | {escape(native)} | {escape(authority)} | {escape(status)} |"
        )
    structure = []
    for fact in facts:
        parameters = []
        for parameter in fact.parameters:
            label = ", ".join(parameter.declarations)
            required = (
                "required"
                if parameter.required
                else f"default {parameter.default or 'none'}"
            )
            parameters.append(
                f"`{escape(label)}` ({required}: {escape(parameter.help)})"
            )
        structure.append(
            f"| `{fact.command}` | {fact.command_kind} | {'<br>'.join(parameters) or 'none'} | {escape(fact.help) or '—'} |"
        )
    output_contracts = "\n".join(
        f"| `{fact.command}` | {escape(contracts[fact.command].output)} | `{contracts[fact.command].errors}` |"
        for fact in leaves
    )
    lines.append(
        FOOTER.format(
            structure="\n".join(structure), contracts=output_contracts
        ).strip()
    )
    return "\n".join(lines[:-1]) + "\n\n" + lines[-1] + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    destination = Path(__file__).resolve().parents[1] / "docs" / "cli.md"
    expected = render()
    if args.check:
        if not destination.is_file() or destination.read_text() != expected:
            raise SystemExit("generated CLI reference is stale")
        return 0
    destination.write_text(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
