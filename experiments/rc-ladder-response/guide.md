# Numerical RC ladder development study

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for a numerical resistor-capacitor ladder development study.
It keeps solver views nested within one model unit.
Use it to run the diagnostic or compile the permitted development candidate.

## Read results and failures

Read `config_id`, `cells`, `horizon_s`, `final_cell_voltages_v` and `maximum_backward_euler_defect_v`.
The receiver is in V and the boundary current in A. The two solvers share one model unit.
The native diagnostic reports `campaign_candidate_compiled=false` and `campaign_issued=false`.
Candidate authoring has separate records, tolerances and no-contact readiness gates.

Read [the common result and failure guide](../../docs/results-and-failures.md) for retained attempts,
stdout/stderr and same-identity issued recovery. The route-specific outputs and stops below remain binding.
Use `workflow show rc-ladder-response --format json` for static command effects and prerequisite roles.
Discovery does not inspect those prerequisites.

## Prerequisites and first action

Use the checkout root and the [tested environment](../../docs/environments.md).
Obtain the colocated strict input from the selected source archive.
Install the profile shown below.
This command runs an exposed development diagnostic.
It grants no scientific qualification or execution authority.

```sh
uv sync --locked --python 3.11.14 --group reactor-example
uv run --no-sync empirical-lawhood campaign rc-ladder-native-check \
  --config /path/to/clean-target/experiments/rc-ladder-response/model.json \
  --output-dir /absolute/development/rc-ladder-response/attempt-001
```

## Edit a development model

Keep the shipped input unchanged. Create a copied input outside the checkout:

```sh
EL_DEV=/absolute/path/to/new-rc-development
mkdir "$EL_DEV"
uv run --no-sync python -m json.tool \
  experiments/rc-ladder-response/model.json "$EL_DEV/model.edit.json"
```

Change only the copied `left_source_resistance_ohms` from `{"decimal":"50"}` to `{"decimal":"75"}`.
Its unit remains Ω. Positive component values, roster lengths, chronological action intervals and output times remain constructor requirements.
Keep Decimal tags and the declared units. A config ID change alone does not create fresh evidence.

```sh
uv run --no-sync empirical-lawhood config validate \
  --config "$EL_DEV/model.edit.json" --consumer rc-ladder-model --format json
uv run --no-sync empirical-lawhood config prepare \
  --config "$EL_DEV/model.edit.json" --consumer rc-ladder-model \
  --output-file "$EL_DEV/model.json" --format json
uv run --no-sync empirical-lawhood campaign rc-ladder-native-check \
  --config "$EL_DEV/model.json" --output-dir "$EL_DEV/attempt-001"
```

Validation is no-contact. Preparation writes a new canonical file and reports original/prepared identities.
It does not overwrite the edited copy. A formatted document can describe the same valid model while remaining unready for the exact-byte native consumer.
The native command retains its actual report and bounded operational diagnostics in the new attempt directory.
Open `report.md`, then inspect the fields listed above in `report.json`.
This is a different numerical model, not a new physical board or qualified scientific cohort.

For an invalid copy, set one `capacitances_farads` value to `{"decimal":"0"}`.
Validate that copy before any native call. It refuses the nonpositive capacitance.
If a native invocation itself refuses a bad input, its selected usable attempt directory retains the failure, without a success report.
Keep that attempt, repair the copied model and use a fresh output path.
See [configuration](../../docs/configuration.md) for offline editor schemas, semantic checks and import limits.

## Expected output and scientific scope

The [strict study](study.json) is a
locally generable **synthetic development** circuit, derived from the owner's
four-cell numerical default. Its four capacitors are 0.001 F. Its three
interior resistors are 100 Ω. The finite left source is 50 Ω and the right
termination is 100 Ω.

All node voltages start at 0 V. The requested left
boundary is 1 V and the right boundary 0 V over 0–1 s. The native receiver is
each node voltage in V and both boundary currents in A every 0.05 s. The
matrix-exponential and backward-Euler solvers are **two nested views of one
model unit**.

The latter has a 0.001 s maximum step. The newly selected
development falsifiers are a 0.005 V maximum solver difference and a 0.002 A
maximum output-grid charge-rate residual. These are not historical result
thresholds. Component metrology is frozen in the model before response.

The
causal cutoff is the 0 s pre-action boundary.

Use CPython 3.11.14 and the base locked NumPy 2.4.6/SciPy 1.17.1 stack. The
[model input](model.json) supports a
native diagnostic.



With no output option, the diagnostic writes no files and reports `campaign_candidate_compiled: false`.
The strict candidate route uses the study and the adapter-owned source and
numerical-evaluator bindings. Set up a typed
[operator storage profile](../../configs/operator-storage.example.json)
with your own external mount and artifact namespace. From a **clean target
commit**, run:

```sh
uv run --no-sync empirical-lawhood --project-root /path/to/clean-target \
  --operator-profile /path/to/operator-profile.json \
  campaign circuit-author \
  --study /path/to/clean-target/experiments/rc-ladder-response/study.json \
  --experiment-id my-rc-development-001 \
  --output-dir /your/external-root/artifacts/authoring/my-rc-development-001

uv run --no-sync empirical-lawhood --project-root /path/to/clean-target \
  --operator-profile /path/to/operator-profile.json \
  --circuit-authoring-dir /your/external-root/artifacts/authoring/my-rc-development-001 \
  campaign check-readiness \
  --spec /your/external-root/artifacts/authoring/my-rc-development-001/authoring.json \
  --extension-payload /your/external-root/artifacts/authoring/my-rc-development-001/payload-0.json \
  --extension-payload /your/external-root/artifacts/authoring/my-rc-development-001/payload-1.json \
  --decoder-registration /your/external-root/artifacts/authoring/my-rc-development-001/decoder-0.json \
  --decoder-registration /your/external-root/artifacts/authoring/my-rc-development-001/decoder-1.json \
  --expected-candidate /your/external-root/artifacts/authoring/my-rc-development-001/candidate.json \
  --source-closure /your/external-root/artifacts/authoring/my-rc-development-001/source-closure.json \
  --resource-envelope /your/external-root/artifacts/authoring/my-rc-development-001/resources.json \
  --run my-rc-development-001 --format json
```

`circuit-author` writes a strict extension authoring packet, selected issued config
payloads/decoders, candidate, source closure, resource envelope and pure
CandidateRunPlan/CandidateExecutionPlan projections. The generic preissue command has no
native effects. It validates the production API/provider path, exact shared native config and source-study inputs.
It validates two selected source tasks and the evaluator config/runner.
It validates output/adjudication locators, resource envelope and separate authority gates. The checked-in
study and its model unit are public development inputs. They cannot support
an attestation that the evaluation unit was previously unexposed, and should
not be issued as a fresh independent confirmation.

For a new prospective numerical unit, freeze a distinct model/component and
action roster, its model-preparation ID, thresholds and evaluation split
before any response. Supply it as a strict `ResistorCapacitorLadderStudyConfig` JSON to
the same `--study` option. The command recomposes the native contract from
those exact bytes and refuses malformed units, grids and intervals. A new ID
alone on the same deterministic model is not independent evidence.

The
generic `campaign issue`, `issue-extensions`, `assemble-package`, `compile-plan`,
`run`/`resume` and separate reveal operations then require their documented
typed proposer, custody, approval, execution and reveal authorities. See
the [researcher lifecycle guide](../../docs/designing-and-running-an-experiment.md). No source contact, issue or authority is supplied by this quick start.

The [candidate integration test](../../tests/test_resistor_capacitor_candidate.py)
validates that the shipped study compiles through the generated registry and
selects the three planned runners and their external input contracts. The [scientific tests](../../tests/test_resistor_capacitor_study.py) load the shipped study and compare native current to Ohm's law.
They compare its transient to an independent DC divider limit.
They falsify a tighter frozen tolerance and a wrong view roster. The separate
[one-cell and charge-balance tests](../../tests/test_resistor_capacitor_native_science.py)
challenge the solver against an analytic solution and a source-impedance
counterexample. Measurement numerical-method conformance is the only claim here.
Order, response, local law, admission and controller use and physical-board response are outside this synthetic route. A
physical board needs a separate authentic pre-response component metrology
and source-custody route, rather than relabeling this model as measurement.

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

Gareth Seneque (2026), [*Empirical Lawhood*](../../paper/manuscript.md), edition 0.60, gives the scientific formulation.
The [program source register](../../paper/SOURCES.md) identifies historical results and unavailable primary records.
The linked scientific source modules and existing tests establish only their declared implementation and development boundaries.
