# Synthetic lattice-pairing method development

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for development authoring of a synthetic lattice-pairing method suite.
It tests signed-current and validity gates without qualifying a material preparation.
Use it to compile the method candidate and prove its selected providers without contact.

The [target-owned strict config](authoring.json)
binds the gauge-covariant transverse-response implementation to **one disclosed
synthetic method suite**. Its nine fixtures and base/refined meshes are nested
conditions, not nine independent preparations. This route tests the method's
signed-current, normal-state subtraction, finite-q, Ward and numerical-view
falsifiers. It does not supply the five material controls, a 300 K material
pairing state, a Wannier position matrix or an SI current map. A passing method
suite has no material admission/controller-use or superconductivity claim.

## Read results and failures

Read the emitted candidate, source closure, resource envelope and pure execution-plan projections.
The synthetic producer/evaluator path supplies no physical Hamiltonian or superconducting-material qualification.
A partial authoring export is not a completed handoff; preserve it and choose a fresh output path.

Read [the common result and failure guide](../../docs/results-and-failures.md) for retained attempts,
stdout/stderr and same-identity issued recovery. The route-specific outputs and stops below remain binding.
Use `workflow show lattice-pairing-method --format json` for static command effects and prerequisite roles.
Discovery does not inspect those prerequisites.

## Source trace and fixed chart

The [historical ambient-pressure material review](../../docs/sources/ambient-pressure-material-historical-review.md) traces
the historical Pb solver inspection, staged material design, method amendment and material controls.
The method amendment froze a finite-temperature lattice-BCS method after the initial design inspection.
It preceded development and prospective target-material contact.
The target reuses the gauge-covariant response producer without copying historical run identities or outcomes. The suite uses 300 K, 101,325 Pa, hopping 1 eV, chemical potential −1 eV and
gap 0.06 eV for the positive method fixture.

Its signed Peierls bond-phase
probe is 0.0005. The derivative step is 0.00005. The base mesh is 24×48 and
the refined mesh 32×64. The receiver is the signed transverse lattice current
in eV/link after finite-temperature BdG evaluation.

The pre-action cutoff is
stage 0. Request, acceptance, application and receiver stages are 0, 1, 2,
3. The denominator is the *synthetic clean lattice-BCS state*. Neither
eV/link nor lattice spacing is silently identified with A/m² or a material
preparation.

The three accepting fixtures are positive response, normal state and band
insulator. Six rejection canaries independently challenge wrong sign, missing
diamagnetic term, nonlinear response, finite-q drift, Ward failure and
base/refined disagreement. The method evaluator consumes the sealed producer
panel, validates each observation's digest and scientific disposition against
the frozen chart, and emits a method-only adjudication. The
[focused check](../../tests/test_synthetic_material_response_method.py) validates the shipped config through strict candidate compilation and selected provider factories.
It tests positive current direction and normal cancellation.
It constructs a sign-reversed observation with a repaired digest and requires evaluator refusal.

## Author and prove without contact

From a clean committed target checkout, install the base profile and provide
an explicit [operator storage profile](../../configs/operator-storage.example.json)
whose guarded external artifact root is writable. Set `ROOT` and `PROFILE` to
their absolute paths and `OUT` to a new empty directory under that root's
`artifacts` namespace. The single installed CLI handles both operations:

```sh
uv sync --locked --python 3.11.14
uv run --no-sync empirical-lawhood --project-root "$ROOT" --operator-profile "$PROFILE" \
  campaign synthetic-material-author \
  --config "$ROOT/experiments/lattice-pairing-method/authoring.json" \
  --experiment-id my-lattice-pairing-method-001 --output-dir "$OUT"

uv run --no-sync empirical-lawhood --project-root "$ROOT" --operator-profile "$PROFILE" \
  --synthetic-material-authoring-dir "$OUT" campaign check-readiness \
  --spec "$OUT/authoring.json" \
  --extension-payload "$OUT/payload-0.json" \
  --extension-payload "$OUT/payload-1.json" \
  --decoder-registration "$OUT/decoder-0.json" \
  --decoder-registration "$OUT/decoder-1.json" \
  --expected-candidate "$OUT/candidate.json" \
  --source-closure "$OUT/source-closure.json" \
  --resource-envelope "$OUT/resources.json" \
  --run my-lattice-pairing-method-001 --format json
```

`synthetic-material-author` strictly compiles an executable candidate and projects a run and
execution plan. It performs no fixture computation. The preissue operation validates the selected fixture producer, attached evaluator, exact issued configs and external inputs.
It validates output/adjudication locators, resource reservations, source closure and separate future authority gates. The
clean-checkout proof passed all six closure sections with two runners, three
external inputs, three output locators and four future authority gates. It
performed no issue, native task, outcome read or source contact. The public suite is
exposed method-development input and cannot become a fresh independent
prospective material unit merely by changing its ID.

The subsequent generic commands are `campaign issue-extensions`, `campaign
assemble-package`, `campaign compile-plan`, `campaign run`/`campaign resume` and
the separate reveal operation. They require a newly frozen eligible roster,
typed proposer and materialization attestations, source custody,
authorization and reveal records described in the
[researcher lifecycle guide](../../docs/designing-and-running-an-experiment.md). The
shipped public method suite carries no new evaluation-unit attestation and
must not be issued as a material discovery experiment.

## Material program boundary

Pb SCF solver qualification is distinct from staged material design and the five-material control gate.
Each needs fresh source and roster identities.
Historical operator scripts provide no current issue path.
Before new material-control work, an authorized researcher must supply these legally usable inputs under separate custody:

- Exact preparation identities and QE/EPW and Wannier versions.
- Same-gauge `*_hr.dat` and `*_r.dat`.
- 300 K pairing and normal comparators.
- Lattice, reciprocal and SI maps.
- Requested, accepted, applied and realized preparation.
- Bounded resource and receiver records.
- Both nested SSSP views for each of five independent controls.

The [material operand assessment](../../docs/sources/ambient-pressure-material-operands.md)
lists the native `tar.gz` member contract and validity gates. The retained
reducer refuses the authenticated R11 MgB₂ tutorial archive at its absent
`mgb2_r.dat` **before** producing a material transverse-response result.
No supplied material source has passed the complete control and compatibility contract.
The current route is therefore limited to method development.

A new material-control candidate/provider/no-effect
path requires those qualified inputs and is deferred for the first release.

## Obtain the workflow inputs

Obtain this bundle from the selected repository checkout or source distribution. The guide and study inputs are repository workflow files, not wheel resources. Run each command from the checkout root in an unactivated shell. The relative paths use that root.

Keep generated data and receipts in the declared external storage. A wheel can perform its advertised inspection and native-input checks with absolute paths. Source-bound authoring and issue require a clean checkout that executes its own tracked package.

## Recover an incomplete authoring export

A later authoring failure reports the incomplete output path and failed stage.
Keep that directory and its inputs for diagnosis. Use a fresh output path for the next attempt.
A partial export is not an issued campaign or a completed authoring handoff.
The command does not resume, replace or overwrite the incomplete directory.
An existing empty output directory remains accepted. An existing nonempty directory refuses.

## References and research

EPW Developers (2026 documentation), [*Superconducting properties*](https://docs.epw-code.org/tutorials/tutorial_04/index.html).
The tutorial supplies solver recipes. The synthetic method suite supplies no material-control qualification.

Gareth Seneque (2026), [*Empirical Lawhood*](../../paper/manuscript.md), edition 0.60, gives the scientific formulation.
The [program source register](../../paper/SOURCES.md) identifies historical results and unavailable primary records.
The linked scientific source modules and existing tests establish only their declared implementation and development boundaries.
