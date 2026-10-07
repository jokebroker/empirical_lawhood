# Direct TORAX native heat check

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for a direct TORAX heat-response development check.
It tests declared numerical heat and core response within one synthetic preparation.
Use it to run the input and inspect the bounded energy report.

## Read results and failures

Read `independent_units`, preparation identity and the native requested/accepted/applied/realized heat actions.
Inspect eV receiver changes and the 0–0.04 s clock. The two heat actions are nested in one preparation.
A native report supplies no source qualification or candidate.

Read [the common result and failure guide](../../docs/results-and-failures.md) for retained attempts,
stdout/stderr and same-identity issued recovery. The route-specific outputs and stops below remain binding.
Use `workflow show tokamak-heat-response --format json` for static command effects and prerequisite roles.
Discovery does not inspect those prerequisites.

## Prerequisites and first action

Use the checkout root and the [tested environment](../../docs/environments.md).
Obtain the colocated strict input from the selected source archive.
Install the profile shown below.
This command runs an exposed development diagnostic.
It grants no scientific qualification or execution authority.

```sh
uv sync --locked --python 3.11.14 --group open-simulators
uv run --no-sync empirical-lawhood campaign torax-native-check \
  --config experiments/tokamak-heat-response/config.json \
  --output-dir /absolute/development/tokamak-heat-response/attempt-001
```

## Adapt a development input and retain its report

Copy the [config](config.json) to a fresh external directory. Replace the example
absolute path with a new directory whose parent already exists; keep the shipped
config unchanged.

```sh
TOKAMAK_HEAT_DEV=/absolute/path/to/new-tokamak-heat-development
mkdir "$TOKAMAK_HEAT_DEV"
cp experiments/tokamak-heat-response/config.json "$TOKAMAK_HEAT_DEV/config.json"
```

Edit the copied JSON before running it. Keep schema
`empirical-lawhood/simulators/torax-native/native-torax-quickstart`, all nested
record schemas, version `1.0.0` and the `value` envelopes. Decimal fields use
`{"decimal":"..."}`; `radial_cells` is an integer. Unknown fields, duplicate keys
and bare fractional numbers refuse. The [typed records](../../src/empirical_lawhood/adapters/simulators/torax_native/contracts.py)
and [bounded quickstart](../../src/empirical_lawhood/adapters/simulators/torax_native/native_quickstart.py)
permit these changes within one synthetic development preparation:

| Fields | Required relation |
|---|---|
| `config_id`, preparation/view/action IDs | Stable IDs, with prefixes `empirical-lawhood-`, `preparation.empirical-lawhood-`, `view.empirical-lawhood-` and `action.empirical-lawhood-` respectively. |
| Preparation `radial_coordinates`, `electron_temperature_ev`, `ion_temperature_ev`, `electron_density_m3` | One common grid of at least three ordered coordinates, starting at 0 and ending at 1; each profile has the same length, with finite positive values in its declared units. |
| Preparation `plasma_current_a`, `zeff`, `major_radius_m`, `minor_radius_m`, `toroidal_field_t`, `elongation_lcfs`, `source_width`, `chi_i_m2_s`, `chi_e_m2_s`, `particle_diffusivity_m2_s`, `maximum_core_temperature_ev` | Finite and positive; minor radius cannot exceed major radius. |
| Preparation `source_radial_location`, `electron_heat_fraction`, `absorbed_power_fraction` | Finite Decimal in [0, 1]. |
| Preparation `particle_convection_m_s` | Any finite Decimal, including signed values. |
| Preparation `main_ion`, `impurity`, `assumption_ids` | Main ion is `deuterium` or `tritium`; impurity is `argon`, `carbon` or `neon`; assumptions are a nonempty sorted unique string array. Keep the assumptions about the actual supplied synthetic input explicit. |
| View `radial_cells`, `timestep_s`, `horizon_s` | 4–24 cells; finite positive, representable native timestep and horizon; horizon at most 0.04 s; update and history admission limits below. |
| Low/high action `duration_s` and stage `power_w`, `coordinate_s` | Duration equals view horizon; all four stages of each action have the same power at coordinate 0; 0 < low power < high power ≤ 1100000 W. |

Stable IDs use lowercase letters, digits, `.`, `_` or `-`, beginning with a letter
or digit. The view retains solver `linear-theta-fully-implicit`, linear solver
`thomas` and precision `float64`; `closure_id` is a stable declared identity.
The actions retain labels `down`/`up`, all four delivery stages in their original
order and source model `generic-ion-electron-gaussian-proxy`. The input file is
bounded to 32 KiB. A config satisfying these input relations can still refuse
during native execution or fail the response-order check; decoding is not a
promise of a successful simulation.

The [operational admission owner](../../src/empirical_lawhood/adapters/simulators/_native_admission.py)
permits at most 100,000 updates per arm and 1,000,000 retained radial history
cells. A partial final timestep counts as a full update; the initial sample is
also retained. With `S` admitted samples and `R` radial cells, the history
allowance is `S × (512 × R + 1024)` bytes, capped at 64 MiB per arm. The 512-byte
cell allowance covers numeric profile histories and copies; the 1024-byte
sample allowance covers reduced receivers and output buffers. These are
conservative operational estimates, not measured worst-case process RSS:
native compiler and runtime overhead is separate. Admission checks both the
exact Decimal grid and the representable, advancing float64 native clock before
TORAX import or construction. These limits do not select scientific accuracy
or change the shipped 0.001 s timestep.

Record edited values and identities and retain JSON standard output with the
input using the previously installed optional runtime:

```sh
uv run --no-sync empirical-lawhood campaign torax-native-check \
  --config "$TOKAMAK_HEAT_DEV/config.json" \
  > "$TOKAMAK_HEAT_DEV/report.json" 2> "$TOKAMAK_HEAT_DEV/diagnostics.log"
```

On success, inspect the one independent preparation, two nested action views,
reported native clock counts, low/high powers and delivery stages, effort in J,
core-temperature changes in eV and trajectory digests. For an edited horizon,
effort is power × that horizon and the shipped 41-point and 20,000/44,000 J results
below need not apply. The CLI has no report-file export option. A handled failure
exits 3 and writes `TORAX native check refused: ...` to standard error; an empty or
incomplete captured stdout file is not a successful report. CLI option errors may
fail earlier. Keep refused inputs and diagnostics, and use a new external directory
for the next attempt. Changed IDs do not supply independent machine shots,
physical-source qualification, a campaign candidate or issue authority.

## Expected output and scientific scope

The [strict target input](config.json)
uses the retained direct TORAX 1.4.2 adapter. It fixes one disclosed
synthetic preparation, a 24-cell circular primary numerical view and two
nested heat actions. The source's pre-outcome design used a 0.04 s horizon,
0.001 s fixed timestep, fully implicit Thomas float64 solver and
0.5/0.8/1.1 MW down/hold/up Gaussian heat chart. This bounded target check
uses only down (0.5 MW) and up (1.1 MW), with **declared stage-exact**
requested, accepted, applied and realized powers at 0 s.

The direct adapter
sets TORAX's Gaussian heat input from that declared realized power. It does
not recover a separate hardware or simulator-delivery journal. The source converts input electron and ion
temperature from eV to TORAX keV and reports core temperature changes in eV. It keeps the exact native 0–0.04 s clock.

Each trajectory has 41 time points.

The declared core-temperature ceiling applies to every retained core-electron
temperature sample, including the initial state. Equality is accepted. The
reported safety margin is the ceiling minus the retained peak, while the
response remains the endpoint change. An interior overshoot retains its
trajectory and an invalid episode with no scoreable outputs. Sampled compliance
does not establish a continuous-time or physical safety bound between samples.

The target config's preparation ID is part of an **exposed synthetic development canary**.
Its other canary inputs are:

- Three-point 1000/800/600 eV electron profile
- 900/750/550 eV ion profile
- Density
- Geometry
- Transport parameters
- 0.8 MA plasma current.
They are not a new machine shot or an evaluation unit. Both action views
share this one independent preparation. The higher power must raise the
core-temperature response relative to lower power. The adapter's calculated
heat effort must equal declared power × 0.04 s: 20,000 J and 44,000 J
respectively. The [focused test](../../tests/test_torax_native_quickstart.py) loads the shipped config through the one CLI in the pinned optional runtime.
It validates the independent energy equation, native response ordering, units, clocks and all four action stages.
It refuses horizon/resource drift before native contact.



The command uses the installed TORAX 1.4.2 runtime and outputs a bounded
development report. On the shipped input the native down/up core-temperature
changes are 581.095/1590.775 eV, with no solver error. Measurement, order and response local numerical
preparation, action and response are demonstrated for this synthetic model. The real MAST/flagship source context, native plasma measurement and
MAST-to-TORAX transport remain separate.

A new researcher can change the synthetic preparation within the typed contract.
This change gives no physical-source qualification or authority for local law, admission or controller use. The selected provider/candidate and fresh
roster are still absent. The command reports
`campaign_candidate_compiled=false` and `campaign_issued=false`.

## Obtain the workflow inputs

Obtain this bundle from the selected repository checkout or source distribution. The guide and study inputs are repository workflow files, not wheel resources. Run each command from the checkout root in an unactivated shell. The relative paths use that root.

Keep generated data and receipts in the declared external storage. A wheel can perform its advertised inspection and native-input checks with absolute paths. Source-bound authoring and issue require a clean checkout that executes its own tracked package.

## References and research

Citrin, J. et al. (2024), [*TORAX: A fast and differentiable tokamak transport simulator in JAX*](https://arxiv.org/abs/2406.06718v1), arXiv:2406.06718v1.
The [versioned author manuscript](https://arxiv.org/abs/2406.06718v1) gives the upstream transport model.
This workflow uses TORAX 1.4.2, as pinned in [pyproject.toml](../../pyproject.toml).
The paper does not qualify this target check or establish transfer to a physical tokamak.

Gareth Seneque (2026), [*Empirical Lawhood*](../../paper/manuscript.md), edition 0.60, gives the scientific formulation.
The [program source register](../../paper/SOURCES.md) identifies historical results and unavailable primary records.
The linked scientific source modules and existing tests establish only their declared implementation and development boundaries.
