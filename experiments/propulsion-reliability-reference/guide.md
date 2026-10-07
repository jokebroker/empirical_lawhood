# Propulsion closure and reliability reference

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for a truth-known propulsion-reliability reference.
It tests finite selection and causal cycle behavior without a physical machine.
Use it to run the development worlds and inspect the corrected oracle counterexample.

## Read results and failures

Read `independent_generated_preparations`, `unique_truth_hypotheses`, `false_promotion_count` and `corrected_oracle_unresolved_count`.
Decision transcripts and cycles remain nested observations.
`physical_icf_claim=false` and `oracle_deployable=false` preserve the reference-world ceiling.

Read [the common result and failure guide](../../docs/results-and-failures.md) for retained attempts,
stdout/stderr and same-identity issued recovery. The route-specific outputs and stops below remain binding.
Use `workflow show propulsion-reliability-reference --format json` for static command effects and prerequisite roles.
Discovery does not inspect those prerequisites.

## Prerequisites and first action

Use the checkout root and the [tested environment](../../docs/environments.md).
Obtain the colocated strict input from the selected source archive.
Install the profile shown below.
This command runs an exposed development diagnostic.
It grants no scientific qualification or execution authority.

```sh
uv sync --locked --python 3.11.14
uv run --no-sync empirical-lawhood campaign propulsion-reference-check \
  --config experiments/propulsion-reliability-reference/config.json \
  --output-dir /absolute/development/propulsion-reliability-reference/attempt-001
```

## Expected output and scientific scope

The [strict development input](config.json)
runs the retained truth-known ICF-to-propulsion **method reference**, not a
physical ICF or propulsion simulator. The finite chart has eight binary mechanism axes, ten interface experiments and four selectors.
It permits at most eight acts and cost 12 per world.
Its 24 nested zero-based cycles have a fault at cycle 8 and possible repair at cycle 13. One generated target-and-machine
preparation is one independent unit. Its four decision transcripts and 24
cycle observations are nested views. Requested, accepted, applied and
realized choices and cycle drives are reported separately. Cycle numbers and
drives are dimensionless simulated coordinates, not SI machine measurements.

The source fixed those equations, chart and limits before its propulsion-reference attempts.
The target input chooses a **new disclosed development canary** of 16 worlds,
seed `20260929` and `world.empirical-lawhood-` identities. Those are not the
source's 48 exposed worlds or a fresh sealed evaluation roster. The oracle
uses the evaluator-known truth and is explicitly nondeployable. The corrected oracle searches for a least-cost experiment set that resolves the full
observable equivalence class. The earlier greedy arm could exhaust its cost
budget without doing so.



The one CLI strictly decodes the input and runs all 16 development worlds,
their four nested selectors and 24-cycle reliability trajectories. It reports
the first world's exact decision and cycle stages, counts, truth retention,
false promotions and corrected-oracle unresolved count. A researcher can copy
the config and choose a new `generation_seed` and `world_id_prefix` for a
different **development** demonstration. The input refuses an evaluation
split, changed method chart or changed act/cost/cycle budget. It compiles no
candidate and issues no run.

The [focused scientific check](../../tests/test_synthetic_propulsion_development_input.py) reads the
shipped input, validates one preparation versus nested transcripts/cycles and
the prior-cycle acceptance cutoff. It independently computes the symmetry interface's `2 × asymmetry + parity(delivery + diagnostic)` outcome.
A decisive truth assignment leaves the old greedy oracle unresolved at cost 12.
The corrected oracle resolves it at cost 9. At the fault step the independent balance is
`new integrity = prior integrity − applied drive × degradation rate − fault severity`.
A cycle's acceptance uses the previous observed integrity and thermal
load. A later receiver must not decide an earlier action. A truth-loss,
false promotion, cost overrun, earlier action depending on a future cycle or
oracle claim of deployability would falsify this bounded reference behavior.

This is a truth-known method ceiling. The retained propulsion-reference world has no selected
executable production provider or candidate, no physical source binding and no
prospective machine roster. Measurement conformance and response/selection
are illustrated within a truth-known world. Physical-source qualification,
order-relation, local law, admission and controller-use claims are not established.
The [general experiment guide](../../docs/designing-and-running-an-experiment.md) gives the generic `campaign compile-candidate`, `check-readiness`, `issue-extensions`, `run`/`resume` sequence and its separate evaluator reveal authority gate.
Typed source, proposer, custody, scientific and execution prerequisites are necessary for the sequence.
This propulsion-reference config is **not**
a compiled input for that sequence. The [route register](../../docs/substrate-route-register.md)
keeps the provider and physical-claim limits visible.

## Obtain the workflow inputs

Obtain this bundle from the selected repository checkout or source distribution. The guide and study inputs are repository workflow files, not wheel resources. Run each command from the checkout root in an unactivated shell. The relative paths use that root.

Keep generated data and receipts in the declared external storage. A wheel can perform its advertised inspection and native-input checks with absolute paths. Source-bound authoring and issue require a clean checkout that executes its own tracked package.

## References and research

Gareth Seneque (2026), [*Empirical Lawhood*](../../paper/manuscript.md), edition 0.60, gives the scientific formulation.
The [program source register](../../paper/SOURCES.md) identifies historical results and unavailable primary records.
The linked scientific source modules and existing tests establish only their declared implementation and development boundaries.
