# Gym–TORAX native development input

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for a Gym–TORAX current-response development check.
It keeps matched episodes and time views within one numerical preparation.
Use it to run the input and inspect the causal cutoff and unavailable operator API.

## Read results and failures

Read `independent_units`, `nested_action_episodes`, `native_state_clocks`, delivery stages and observation reason codes.
`controlled_io_operator_available=false` identifies the missing controlled snapshot.
Keep the paired action episodes nested in one preparation; the report supplies no candidate or fresh qualification.

Read [the common result and failure guide](../../docs/results-and-failures.md) for retained attempts,
stdout/stderr and same-identity issued recovery. The route-specific outputs and stops below remain binding.
Use `workflow show tokamak-control-response --format json` for static command effects and prerequisite roles.
Discovery does not inspect those prerequisites.

## Prerequisites and first action

Use the checkout root and the [tested environment](../../docs/environments.md).
Obtain the colocated strict input from the selected source archive.
Install the profile shown below.
This command runs an exposed development diagnostic.
It grants no scientific qualification or execution authority.

```sh
uv sync --locked --python 3.11.14 --group open-simulators
uv run --no-sync empirical-lawhood campaign gym-torax-native-check \
  --config experiments/tokamak-control-response/config.json \
  --source-checkout "$PWD" \
  --output-dir /absolute/development/tokamak-control-response/attempt-001
```

## Adapt a development input and retain its report

Copy the [config](config.json) to a fresh external directory. Replace the example
absolute path with a new directory whose parent already exists; keep the shipped
config unchanged.

```sh
TOKAMAK_CONTROL_DEV=/absolute/path/to/new-tokamak-control-development
mkdir "$TOKAMAK_CONTROL_DEV"
cp experiments/tokamak-control-response/config.json "$TOKAMAK_CONTROL_DEV/config.json"
```

Edit the copied JSON before running it. Keep the outer schema
`empirical-lawhood/simulators/gym-torax-native/gym-torax-native-quickstart`,
the nested preparation schema, version `1.0.0` and their `value` envelopes.
Decimal fields retain their `{"decimal":"..."}` representation; seed and output
size remain JSON integers. Unknown fields, duplicate keys and bare fractional
numbers refuse. The [preparation contract](../../src/empirical_lawhood/adapters/simulators/gym_torax_native/diagnostic_contracts.py)
and [quickstart contract](../../src/empirical_lawhood/adapters/simulators/gym_torax_native/native_quickstart.py)
permit only these input choices:

| Field | Permitted choice |
|---|---|
| `value.config_id` | Stable ID beginning `empirical-lawhood-`. |
| `value.preparation.value.preparation_id` | Stable ID beginning `preparation.empirical-lawhood-`. |
| `value.preparation.value.physical_independent_unit_id` | Stable ID beginning `unit.empirical-lawhood-`. |
| `value.preparation.value.environment_seed` | Nonnegative integer. Both matched episodes use this same seed and preparation. |
| `value.preparation.value.initial_temperature_scale` | Finite Decimal in [0.990, 1.020]. |
| `value.preparation.value.initial_density_nbar` | Finite Decimal in [0.848, 0.852]. |
| `value.preparation.value.bootstrap_multiplier` | Finite Decimal in [0.990, 1.005]. |
| `value.preparation.value.inner_transport_scale` | Finite Decimal in [0.990, 1.010]. |

Stable IDs use lowercase letters, digits, `.`, `_` or `-`, beginning with a letter
or digit. Record the chosen identities and seed; renaming an exposed unit or
choosing another seed does not establish a fresh qualified roster.
`numerical_member_id` must remain `member.tokamak-control.primary` and
`maximum_output_bytes` must remain 67108864. The native hold/future words, action at
request 111, 121 state clocks and scalar `Q_fusion` receiver are fixed by the
retained check, not editable JSON fields. The input file is bounded to 16 KiB.

Use the previously installed optional runtime and retain the output with the
edited input. `--source-checkout` remains the absolute target checkout root:

```sh
uv run --no-sync empirical-lawhood campaign gym-torax-native-check \
  --config "$TOKAMAK_CONTROL_DEV/config.json" \
  --source-checkout "$PWD" \
  > "$TOKAMAK_CONTROL_DEV/report.json" 2> "$TOKAMAK_CONTROL_DEV/diagnostics.log"
```

On success, `report.json` is the CLI's JSON standard output. Check the selected
identities, one independent unit, two nested episodes, 121 clocks, equal pre-action
receivers, post-action change and the four delivery stages. Inspect observation
dispositions and the controlled-I/O availability field separately; a successful
state capture does not supply an unavailable operator API. The CLI has no
report-file export option. A handled failure exits 3 and writes
`Gym--TORAX native check refused: ...` to standard error; an empty or incomplete
captured stdout file is not a successful report. CLI option errors may fail earlier.
Keep failed inputs and diagnostics and use a new external directory for another
development attempt. This command still produces no candidate, issued campaign
or source qualification.

## Expected output and scientific scope

The [strict target config](config.json)
selects one new, exposed development preparation under the retained
Gym–TORAX native state/receiver capture. The adapter's source chart has 120 request clocks
and 121 state clocks. This bounded check runs the retained native-hold and
future-only current words on the **same** preparation and seed. The two
episodes and 121 time views do not become two or 121 independent units.

The
source-native absolute plasma-current action is in A. The checked `Q_fusion`
receiver has native unit `1` and its declared scalar frame. Requested,
accepted, applied and realized stages are kept separately.

The target config's new preparation and independent-unit IDs are disclosed **development canaries**.
Its numerical canaries are:

- Seed 2718
- Temperature scale 1
- Density `nbar=0.85`
- Bootstrap multiplier 1
- Inner transport scale 1. They are not a
fresh prospective roster or a source-owned evaluation choice. The checked
numerical member is the retained primary member. The future-only word first differs at request clock 111 with a 12.6 MA request.
State-clock receivers 0–111 must remain identical to native hold.
The request can first affect state 112.

The native check requires a changed
post-action `Q_fusion` receiver as well as an identical pre-action prefix. The [focused test](../../tests/test_gym_torax_native_quickstart.py) exercises the shipped input in the pinned runtime.
It validates realized current against applied current to 10⁻⁶ A.
It validates the causal cutoff, units, clocks and one-unit count. Existing [source contract checks](../../tests/test_gym_torax_native_contract.py)
cover metadata dimension drift, native action decoding and source-identity
precontact refusal.



This source-backed check needs the target checkout because it hashes the
retained adapter/source file roster before building the native environment. It refuses an absent checkout and request/manifest divergence before native
reset. This development command does not certify a clean Git source closure. The
`open-simulators` group pins Gym–TORAX 1.1.1, TORAX 1.4.2, JAX/JAXLIB 0.10.2,
NumPy 2.4.6, SciPy 1.17.1 and xarray 2026.7.0 on tested Linux x86-64 CPU
float64.

The local result has 121 state clocks, complete native observations,
a complete 12.6 MA action at request 111, and a `Q_fusion` change after the
cutoff. It is a nonpromotable metadata-canary/development check. It makes no
plasma or controller-law claim.

The installed Gym–TORAX API did not expose controlled I/O operator snapshots
and the report says `controlled_io_operator_available=false` with
`OPERATOR_API_UNAVAILABLE`. That limits controlled-I/O, source assessment and controller composition even though state/receiver capture succeeds. A fresh source
qualification, disjoint preparation roster, selected provider/candidate,
typed authority and the distinct action/receiver contracts for observation conditions remain
open. The command reports `campaign_candidate_compiled=false` and
`campaign_issued=false`.

## Obtain the workflow inputs

Obtain this bundle from the selected repository checkout or source distribution. The guide and study inputs are repository workflow files, not wheel resources. Run each command from the checkout root in an unactivated shell. The relative paths use that root.

Keep generated data and receipts in the declared external storage. A wheel can perform its advertised inspection and native-input checks with absolute paths. Source-bound authoring and issue require a clean checkout that executes its own tracked package.

## References and research

Citrin, J. et al. (2024), [*TORAX: A fast and differentiable tokamak transport simulator in JAX*](https://arxiv.org/abs/2406.06718v1), arXiv:2406.06718v1.
The [versioned author manuscript](https://arxiv.org/abs/2406.06718v1) gives the upstream transport model.
This workflow uses TORAX 1.4.2, as pinned in [pyproject.toml](../../pyproject.toml).
The paper does not qualify this target check or establish transfer to a physical tokamak.

Mouchamps, A., Malherbe, A., Bolland, A. and Ernst, D. (2026), [*Gym-TORAX: Open-source software for integrating reinforcement learning with plasma control simulators in tokamak research*](https://doi.org/10.1016/j.simpa.2026.100829). *Software Impacts* **27**, 100829.
The [upstream project](https://github.com/antoine-mouchamps/gymtorax) gives this citation for its reinforcement-learning environment.
This workflow uses Gym–TORAX 1.1.1, as pinned in [pyproject.toml](../../pyproject.toml).
The research paper and software pin have separate roles.
They do not qualify this target check or supply the missing controlled-I/O operator API.

Gareth Seneque (2026), [*Empirical Lawhood*](../../paper/manuscript.md), edition 0.60, gives the scientific formulation.
The [program source register](../../paper/SOURCES.md) identifies historical results and unavailable primary records.
The linked scientific source modules and existing tests establish only their declared implementation and development boundaries.
