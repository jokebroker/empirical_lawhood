# Contributing

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for contributions of code, scientific procedures and documentation.
It gives the required checks, evidence limits and license for a proposed change.
Use it to prepare a change for review.

## Documentation policy

The technical writing standard is [ASD-STE100 Issue 9](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf), dated 15 January 2025.
Use this policy for maintained documents, experiment guides, source guides and generated prose.
The manuscript uses its abstract for orientation and retains its scientific structure.

Start each document with three short sentences about its task, purpose and reader action.
Use the dictionary's approved word meanings, parts of speech and verb forms.
Use [defined scientific and software terms](docs/glossary.md) in their stated contexts.
Classify technical nouns under rule 1.5 and technical verbs under rule 1.12.
Technical words still obey the standard's grammar and word-use rules.
If an approved word has the same meaning, use that word.
Use one term for each meaning.
In Issue 9, `CHECK` is an approved noun.
Use “make sure” or the precise glossary verb for the action.
Use `CONNECT` for a physical connection.
For record relationships, use “bind” in its defined technical meaning.
Do not expand a technical verb into a general dictionary exception.
Write code identifiers exactly.

Write active instructions with one action per sentence.
Put a necessary condition before the instruction.
Write procedural sentences with 20 words or fewer.
Write descriptive sentences with 25 words or fewer.
Use the standard's counting rules in section 8.

Count identifiers, quoted text, numbers with units and hyphenated terms as single items where that rule applies.
Count text inside parentheses separately as well as one item in the enclosing sentence.
In a vertical list, count the introduction and each item separately.

Use only simple verb forms and the dictionary’s approved forms.
Use an -ing form only within a defined technical noun.
Avoid contractions, omitted articles and ambiguous phrasal verbs.
Give an object's meaning before its details.
Keep each paragraph about one topic, with at most six sentences.
Write two sentences instead of using a semicolon.
Use a vertical list for a complex set of conditions.
Use a note for information alone.
For an actual safety instruction, identify the risk level and explain the consequence.

Put prerequisites and the first action before background.
Give the expected output, stops and next action.
Keep evidence limits beside the claims and actions they constrain.
Add **References and research** when the document depends on external work.
Change the owning template or command metadata for generated text.
Keep mathematical operands, units, cutoffs, independent units and negative findings during prose edits.

### Authoring template

```markdown
# Scientific task

This guide gives instructions for <the task>.
It supports <the workflow and purpose>.
Use it to <the first useful action>.

## Prerequisites

<Required software, editable inputs, permissions and input identity.>

## First action

<One instruction.>
<Exact command.>

## Output and stops

<Expected files or records, their interpretation and the first honest stop.>
<Evidence limit and next action.>

## References and research

<Primary source, DOI or URL, version and its role in this procedure.>
```

### Review and exceptions

Examine sentence structure, dictionary meanings and glossary usage as well as length.
Record each document's review coverage and unresolved findings in the external maintainer packet.
[Software aids](https://www.asd-ste100.org/STEsoftware.html) assist review.
A link check, word count or AI rewrite alone does not establish ASD-STE100 compliance.

These exceptions apply only to the identified content:

| Surface | Permitted exception |
|---|---|
| Command blocks, paths, returned values and code identifiers | Exact executable spelling and syntax |
| Equations and scientific notation | Exact mathematical symbols, operands and units |
| Quoted text and bibliography entries | Exact source wording, names and titles |
| License texts, contribution-license terms and upstream notices | Exact legal or required attribution wording |
| README mission metaphor | The requested Neptune, Toyota and civilisational-engine framing |
| Changelog and bibliography organization | Chronological entries or reference structure instead of the procedure template |

Surrounding explanations still follow this policy.
An exception does not change scientific meaning or excuse an inherited internal program name.
Keep frozen historical documents in their original source archive.
Label original record names as historical provenance when a maintained guide must cite them.
Use scientific task names for current actions.

## References and research

ASD Simplified Technical English Maintenance Group (2025), [*ASD-STE100 Simplified Technical English*](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf), Issue 9, 15 January 2025.
This specification gives the technical writing rules used in this policy.

ASD (undated), [*STE software*](https://www.asd-ste100.org/STEsoftware.html).
This source gives information about software aids for authors.
A software aid does not establish compliance with the specification.

## Development and release checks

Use the [workflow index](experiments/README.md), [configuration guide](docs/configuration.md)
and [results guide](docs/results-and-failures.md) as the public contributor conventions.
The root [agent instructions](AGENTS.md) give the same source map and preservation rules.
For a small adapter or method change, follow the [worked RC recipe](docs/extending-the-engine.md#worked-rc-extension).
Keep local results and the additional issued-provider work distinct.
For a descriptive analysis route, reuse bounded source and array contracts with its own declared protocol and complete result census.
Retain source/environment identities, failures and independent scientific falsifiers.
Its completion receipt is development custody, not a study grant or qualification.

Run the affected focused tests with the already-selected locked environment:

```sh
uv run --no-sync pytest -m 'not native and not held' --fail-on-skip tests/test_workflow_discovery.py tests/test_worked_rc_extension.py
uv run --no-sync python scripts/generate_workflow_index.py --check
uv run --no-sync python scripts/generate_config_schemas.py --check
uv run --no-sync python scripts/check_documentation.py
```

The extension guide gives aggregate generator order. Inspect every generated diff.
Use [test profiles](docs/testing.md) for required optional/held boundaries with genuine prerequisites.
After shared runtime changes settle, the [release procedure](docs/release-artifacts.md) runs the complete portable gate on that source.
Do not repeat the full suite for a later prose-only edit. Bind that edit and its affected checks separately.
Preserve failed checks and genuine scientific negatives. Tests and synthetic authority establish software behavior, not scientific qualification.


Use the [tested Linux environment](docs/environments.md). From an unactivated
shell in the checkout:

```sh
uv sync --locked --python 3.11.14 --group reactor-example
uv run --no-sync ruff check src tests scripts
uv run --no-sync python scripts/generate_extension_bundle_aggregate.py --check
uv run --no-sync python scripts/generate_executable_binding_aggregate.py --check
uv run --no-sync python scripts/generate_cli_reference.py --check
uv run --no-sync python scripts/generate_test_fixture.py --check
uv run --no-sync python scripts/generate_operator_examples.py --check
uv run --no-sync python scripts/check_documentation.py
```

During editing, use the existing focused tests for the changed behavior.
For a single precommit portable suite, settle and stage the reviewed source.
Run the candidate checker before the final commit:

```sh
uv run --no-sync python scripts/release_candidate_check.py --output-dir /path/to/new-candidate-packet
```

Require passing tests, coverage and unchanged candidate bytes.
Commit that exact staged tree locally.
Use its evidence for the clean committed source:

```sh
uv run --no-sync python scripts/release_check.py --profile portable \
  --portable-evidence /path/to/new-candidate-packet/precommit-portable-evidence.json \
  --output-dir /path/to/new-release-packet
```

The artifact runner authenticates the original complete portable gate without repeating its tests.
It installs the locked environment and does generator and static-error checks.
It builds the wheel through the sdist and exercises the installed entry point outside the checkout.
Its manifest binds source, any selected tag, environment, logs, package inventory and artifact hashes.
The runner publishes no artifacts and changes no tags.

Keep the selected packet.
Select artifacts by their recorded hashes.
The [artifact guide](docs/release-artifacts.md) also gives the combined gate and failed-packet procedure.

[Native and held-input profiles](docs/testing.md) use separate jobs.
A selected profile fails when its required dependency or input is absent.
Portable results give evidence for only the portable profile.
Synthetic approvals under `tests/` provide software conformance fixtures.
They grant no operator authority.
For schema or provider changes, follow the [extension recipe](docs/extending-the-engine.md).

## Licensing of contributions

Unless explicitly agreed otherwise before submission, a contribution is made under the license applying to the destination file or directory:

- software and executable experimental machinery: **MPL-2.0**;
- project-authored documentation, manuscript material, and figures: **CC-BY-4.0**;
- project-generated evidence and data contributed to `evidence/`: **CC0-1.0**.

By submitting a contribution, you represent that you created it or otherwise have sufficient rights to contribute it under the applicable license.

Do not add third-party material with unknown provenance or license.
Make sure that its terms let you use it as intended.

## Source-file notices

New MPL-covered source files should include:

```
SPDX-License-Identifier: MPL-2.0
```

Where practical, they may also include the MPL Exhibit A notice.

## Reproducibility and evidence

Document the environment, entry point and required inputs for executable additions.
Read the [reproducibility guidance](REPRODUCIBILITY.md).

Routine scientific primary outputs, receipts and authorization records remain in guarded external custody.
Keep these records outside the source repository.
A released derivative requires review, complete provenance, permitted redistribution and an explicit evidence ceiling.
Record its exact producing source, configuration, inputs, environment and verification records.
Keep negative outcomes and causal cutoffs.
Keep private keys and raw held archives outside the repository.
Read the [release status](docs/release-status.md) and [scientific integrity guide](docs/scientific-integrity.md).

The template at `configs/templates/evidence-provenance.yaml` may be adapted for a released evidence tranche.

## Third-party dependencies

For vendored or copied third-party material, record at minimum:

- upstream project or source
- source URL or identifier
- version or commit
- upstream license
- local path
- modifications, if any

See the [licensing and provenance policy](docs/licensing.md).

## Registry and reference generators

The first two tools regenerate installed static registries from the package's closed adapter descriptor roots.
They need no external scientific input or service.
Install the Python dependencies declared in `pyproject.toml`.
From the target source checkout, use these commands:

```sh
uv run --no-sync python scripts/generate_extension_bundle_aggregate.py
uv run --no-sync python scripts/generate_executable_binding_aggregate.py
```

Use that order.
The first tool reads allowed `extension_bundle.py` files and writes `src/empirical_lawhood/adapters/composition/generated_extension_bundles.py`.
The second reads allowed `executable_binding.py` files and the first aggregate.
It writes `src/empirical_lawhood/adapters/composition/generated_executable_bindings.py`.
For a drift check without writes, use:

```sh
uv run --no-sync python scripts/generate_extension_bundle_aggregate.py --check
uv run --no-sync python scripts/generate_executable_binding_aggregate.py --check
```

The CLI reference generator reads the installed Click tree and exact metadata.
It writes `docs/cli.md`.
It needs the target package and declared Python dependencies.
It needs no external scientific data or service.
Use:

```sh
uv run --no-sync python scripts/generate_cli_reference.py
uv run --no-sync python scripts/generate_cli_reference.py --check
```

`scripts/generate_test_fixture.py` regenerates the synthetic target reference campaign.
Its `--check` option compares the existing fixture.
The generator uses target typed constructors and test-only approvals.
`scripts/check_documentation.py` validates local Markdown and HTML destinations against shipped files.
It also validates the unactivated-shell convention.
Remote URLs and section anchors require separate review.

`generate_operator_examples.py --check` verifies the four inert, explicitly
exposed [input examples](docs/input-contracts.md).
`operator_records.py` supplies the reviewed-record persistence and replay
helpers used by the [operator walkthrough](docs/reactor-operator.md).
It supplies no authority decision, signing key, or automatic native execution.
