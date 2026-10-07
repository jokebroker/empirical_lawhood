# Truth-known circuit-method reference

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for a truth-known circuit-method reference.
It separates generated synthetic cases from physical board measurements.
Use it to run the disclosed method suite and inspect the energy-collision counterexample.

## Read results and failures

Read `independent_truth_cases`, `fixture_count`, `nested_variants_per_fixture`, `passed_case_count` and `failed_case_ids`.
The thirty generated cases are fifteen fixtures with two variants.
`physical_board_claim` and `claim_promotion_allowed` prevent interpreting method conformance as physical qualification.

Read [the common result and failure guide](../../docs/results-and-failures.md) for retained attempts,
stdout/stderr and same-identity issued recovery. The route-specific outputs and stops below remain binding.
Use `workflow show response-method-reference --format json` for static command effects and prerequisite roles.
Discovery does not inspect those prerequisites.

## Prerequisites and first action

Use the checkout root and the [tested environment](../../docs/environments.md).
Obtain the colocated strict input from the selected source archive.
Install the profile shown below.
This command runs an exposed development diagnostic.
It grants no scientific qualification or execution authority.

```sh
uv sync --locked --python 3.11.14
uv run --no-sync empirical-lawhood campaign scale-morphism-reference-check \
  --config experiments/response-method-reference/config.json \
  --output-dir /absolute/development/response-method-reference/attempt-001
```

## Expected output and scientific scope

The [strict development input](config.json)
runs the retained truth-blind scale-morphism method against 15 fixed challenges in
two variants each. The 30 generated cases are independent **synthetic method
units**. No physical RC board, laboratory action, or prospective cohort is
represented. This is separate from the [four-cell numerical RC route](../rc-ladder-response/guide.md).

The source fixed the 15 challenge definitions, deterministic/noisy variants,
datum units and truth-blind oracle split before validation. The target chooses
a new disclosed development seed pair `20260930`/`20261001` and new case IDs
under `case.empirical-lawhood-physical-scale-morphism-method-reference`. The source's
`42017`/`90173` validation pair is outcome visible and refused by this command. The seeds and cases here are development references, not hidden evaluation.

The generated data include cell capacitance in F and voltage in V. Their
challenge coordinates, code labels and seed values otherwise have no SI clock
or physical action stages. The method must not turn them into requested,
accepted, applied or realized board actions.



The single CLI strictly decodes the input and generates the cases.
It passes their datum records to the truth-blind method.
It compares returned codes with the separate privileged oracle. It reports case and fixture counts, failure IDs,
one input digest, the nonpromotable evidence ceiling and false candidate/issue
flags. To inspect a different **development** draw, copy the config and change
both sorted `validation_seeds` and its target-owned `config_id`. A nonzero
physical-outcome count, changed fixture roster, wrong access role, historical
seed pair or non-target ID refuses before generation.

The [focused check](../../tests/test_circuit_truth_known_development_input.py) reads the
shipped config and verifies all 30 cases. It independently shows that two
16-cell voltage fields have the same 1 V mean but unequal squared-voltage
sums, 32 versus 16 V². The method must separate this energy collision. A
changed fine-gate input then produces a different fingerprint and a failed
oracle score rather than a false-safe pass.

These cases test synthetic method
semantics and label isolation. They do not measure board delivery or a physical
receiver. No native runtime beyond base NumPy/SciPy is needed.

The ceiling is `NON_PROMOTABLE`. This quick start has no selected executable
production provider/candidate, qualified physical source, action journal,
independent board roster or sealed authority. The physical RC-board route was
explicitly descoped by the owner. The [general experiment guide](../../docs/designing-and-running-an-experiment.md)
describes generic candidate, preissue, issue, run/resume and evaluator-reveal
operations and their typed prerequisites. This truth-world config is not an
input for that sequence. The [register](../../docs/substrate-route-register.md) keeps
the numerical RC and distinct scale-morphism contracts separate.

## Obtain the workflow inputs

Obtain this bundle from the selected repository checkout or source distribution. The guide and study inputs are repository workflow files, not wheel resources. Run each command from the checkout root in an unactivated shell. The relative paths use that root.

Keep generated data and receipts in the declared external storage. A wheel can perform its advertised inspection and native-input checks with absolute paths. Source-bound authoring and issue require a clean checkout that executes its own tracked package.

## References and research

Gareth Seneque (2026), [*Empirical Lawhood*](../../paper/manuscript.md), edition 0.60, gives the scientific formulation.
The [program source register](../../paper/SOURCES.md) identifies historical results and unavailable primary records.
The linked scientific source modules and existing tests establish only their declared implementation and development boundaries.
