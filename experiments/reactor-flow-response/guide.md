# Stirred-flow reactor development check

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for a numerical stirred-flow reactor development check.
It tests flow delivery and chemical balance within one preparation.
Use it to run the disclosed input and understand the missing campaign binding.

## Read results and failures

Read each action stage and receiver in the native panel, including its kg/s delivery and horizon.
Keep all 32 cells within the one complete development unit. Mass-flow/element closure and the hold counterexample remain decisive.
A local response does not supply a selected shared provider or source qualification.

Read [the common result and failure guide](../../docs/results-and-failures.md) for retained attempts,
stdout/stderr and same-identity issued recovery. The route-specific outputs and stops below remain binding.
Use `workflow show reactor-flow-response --format json` for static command effects and prerequisite roles.
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
  --config experiments/reactor-flow-response/config.json \
  --output-dir /absolute/development/reactor-flow-response/attempt-001
```

## Expected output and scientific scope

The [strict development input](config.json)
retains the owner's outcome-blind Cantera independent-recurrence scientific design with a new
target-owned config, design and independent-unit ID. The adapter's semantic
`target_id` remains fixed by its current validator. It is not an issued
experiment ID. No historical source digest, authority or outcome is imported.
Cantera 3.2.0 supplies the public `gri30.yaml` mechanism from its installed
distribution, so this local input needs no separate file acquisition.

The native system is an energy-enabled, non-isothermal ideal-gas CSTR with
finite-rate GRI 3.0 chemistry, an inlet, pressure controller and wall heat
loss. Each complete unit deterministically draws its own inlet, bath, initial
hot/cold state, equivalence ratio, heat-transfer scale and 0.00090–0.00110 m³
reactor volume from its predeclared ID. The two denominator coefficients are
0.14 and 0.26 W/(m² K), multiplied by the unit's heat-transfer scale. The
base residence time is 0.12 s.

Solver ceilings are 20,000 steps and 30 s per
unit. Flow multipliers 0.75, 1.00 and 1.25 are supported. A requested 1.60
is rejected to the hold multiplier 1.00. The requested and accepted stages are
dimensionless, the applied stage is kg/s and the realized mass is kg.

Short
and long receiver horizons are 0.05 and 0.60 s. Receivers are terminal and
peak temperature in K, methane conversion, CO mole fraction and element
closure as dimensionless fractions. The 2 denominator × 2 history × 4 action
× 2 horizon cells are nested views of **one** independent preparation.



The command strictly decodes the config and validates the installed solver version. It executes all 32 native cells and refuses any stopped condition. It reports every requested/accepted/applied/realized action and receiver with native units and clocks. For a new development preparation, copy the strict config.

Before execution, assign a fresh `independent_unit_id`. This ID seeds the independent bounded draws. Do not reuse this
publicly exposed canary as a prospective evaluation unit.

The [scientific test](../../tests/test_independent_substrate_native_contracts.py) validates mass-flow integration against applied kg/s and the horizon. It tests element closure, temperature response to changed flow and the outside-support counterexample against the hold branch. The native panel reports local development
observations. It is not a qualified physical reactor measurement.

The check
exercises preparation, measurement conformance and a bounded response. It does not establish order-relation, local law, admission or controller-use
validity. Those claims require separate evidence and authority. An element
closure above 10⁻⁶, a different outside/hold receiver, or a realized mass
inconsistent with flow × time would
falsify this bounded contract.

**Development boundary. Further integration deferred for first release:**
the native check does not compile a strict candidate
or resolve a no-effect production plan. The independent-recurrence authoring path requires a
construct review and method/source completion envelopes. The installed CLI
does not bind this new config to a campaign provider/runner.

This is an
integration gap, not a request for external data. A prospective route needs
fresh unit IDs frozen before response, a provider binding for this design and
the typed construction records before `campaign compile-candidate` and
`check-readiness`. No issue, execution authority or sealed result is
asserted here.

## Obtain the workflow inputs

Obtain this bundle from the selected repository checkout or source distribution. The guide and study inputs are repository workflow files, not wheel resources. Run each command from the checkout root in an unactivated shell. The relative paths use that root.

Keep generated data and receipts in the declared external storage. A wheel can perform its advertised inspection and native-input checks with absolute paths. Source-bound authoring and issue require a clean checkout that executes its own tracked package.

## References and research

David G. Goodwin, Harry K. Moffat, Ingmar Schoegl, Raymond L. Speth and Bryan W. Weber (2025), [*Cantera*](https://doi.org/10.5281/zenodo.17620923), version 3.2.0.
The upstream toolkit supplies the finite-rate chemistry and installed mechanism. It does not qualify this development preparation.

Gareth Seneque (2026), [*Empirical Lawhood*](../../paper/manuscript.md), edition 0.60, gives the scientific formulation.
The [program source register](../../paper/SOURCES.md) identifies historical results and unavailable primary records.
The linked scientific source modules and existing tests establish only their declared implementation and development boundaries.
