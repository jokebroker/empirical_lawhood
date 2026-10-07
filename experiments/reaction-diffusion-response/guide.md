# Reaction-diffusion field development check

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for a numerical reaction-diffusion development check.
It tests source integration and field balance within one synthetic preparation.
Use it to run the disclosed input and understand the missing campaign binding.

## Read results and failures

Read each action stage, receiver value/unit, horizon and complete-unit disposition in the native panel.
Keep nested cells/views inside the declared one-unit denominator.
A local native result does not supply a selected shared provider or source qualification.

Read [the common result and failure guide](../../docs/results-and-failures.md) for retained attempts,
stdout/stderr and same-identity issued recovery. The route-specific outputs and stops below remain binding.
Use `workflow show reaction-diffusion-response --format json` for static command effects and prerequisite roles.
Discovery does not inspect those prerequisites.

## Prerequisites and first action

Use the checkout root and the [tested environment](../../docs/environments.md).
Obtain the colocated strict input from the selected source archive.
Install the profile shown below.
This command runs an exposed development diagnostic.
It grants no scientific qualification or execution authority.

```sh
uv sync --locked --python 3.11.14 --group reaction-response-simulators
uv run --no-sync empirical-lawhood campaign reaction-response-native-check \
  --config experiments/reaction-diffusion-response/config.json \
  --output-dir /absolute/development/reaction-diffusion-response/attempt-001
```

## Expected output and scientific scope

The [strict development input](config.json)
retains the owner's outcome-blind FiPy independent-recurrence scientific design with new
target-owned config, design and independent-unit IDs. The adapter's semantic
`target_id` is fixed by its current validator. It is not an issued experiment
ID. FiPy 4.0.3 is acquired from the pinned optional package group. No private
field file or historical digest is needed for this local synthetic source.

The native equation is transient one-dimensional diffusion with linear decay
and a signed localized source. Its domain coordinate runs from 0 to 1 in 60 equal cells. The code declares no physical length calibration. Its diffusivities 0.01 and 0.08 are in domain-coordinate²/s.

The timestep is
0.010 s, decay rate 0.15 s⁻¹ and solver ceiling 500 steps. Each independent
unit deterministically draws a cleared or residual field preparation from its
predeclared ID. The source acts only on cell centers in [0.15, 0.25] for at
most 0.10 s. Supported requested rates are −0.25, 0 and +0.25 field
amplitude/s. −0.50 is outside support and is rejected to hold.

Applied rate
is field amplitude/s, while the realized integral is field mass. Short and
long receiver horizons are 0.10 and 0.50 s. Receivers are downstream mean,
field minimum/maximum (field amplitude), total field mass, boundary flux and
mass-balance error. The 2 diffusivity × 2 history × 4 action × 2 horizon
cells are nested views of **one** independent preparation.



The installed CLI strictly decodes the config and validates FiPy's version.
It executes 32 native conditions and refuses a solver stop.
It reports action stages, receiver units, horizon clock and one-unit count. To inspect another local
development preparation, copy the strict config and assign a fresh
`independent_unit_id` before execution. The packaged ID is exposed and cannot
be an evaluation unit.

The [scientific test](../../tests/test_independent_substrate_native_contracts.py) validates the
integrated source against rate × supported region × action time, field mass
balance, positive downstream response, and the outside-support hold
counterexample. The analysis views distinguish active action, diffusivity,
history and horizon dependence, history-response invariance and the support
boundary. These outcomes falsify this bounded contract:

- A realized source integral inconsistent with the stage rate and clock.
- Balance error above 0.002 field mass.
- A changed receiver after a rejected outside action. Measurement, order and response can be explored in
this development canary. Local law, admission and controller use require separate candidate, transport and
authority qualification.

**Development boundary. Further integration deferred for first release:**
this command produces local native output only. The
independent-recurrence candidate authoring path still requires construct review and
method/source completion envelopes, and the installed CLI has no provider
selection for this new config. A strict candidate and no-effect production
proof are therefore unavailable.

This is an integration gap, not missing
external input. For a prospective route, predeclare fresh independent-unit IDs. Bind this design to an executable provider. Before `campaign compile-candidate` and `check-readiness`, obtain the typed construction records.

This guide supplies no authority or issued result.

## Obtain the workflow inputs

Obtain this bundle from the selected repository checkout or source distribution. The guide and study inputs are repository workflow files, not wheel resources. Run each command from the checkout root in an unactivated shell. The relative paths use that root.

Keep generated data and receipts in the declared external storage. A wheel can perform its advertised inspection and native-input checks with absolute paths. Source-bound authoring and issue require a clean checkout that executes its own tracked package.

## References and research

Jonathan E. Guyer, Daniel Wheeler and James A. Warren (2009), [*FiPy: Partial Differential Equations with Python*](https://doi.org/10.1109/MCSE.2009.52).
The adopted solver is FiPy 4.0.3. The project owns the bounded reaction-field contract and its falsifiers.

Gareth Seneque (2026), [*Empirical Lawhood*](../../paper/manuscript.md), edition 0.60, gives the scientific formulation.
The [program source register](../../paper/SOURCES.md) identifies historical results and unavailable primary records.
The linked scientific source modules and existing tests establish only their declared implementation and development boundaries.
