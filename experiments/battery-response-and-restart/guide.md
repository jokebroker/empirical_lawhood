# Battery response, state restart and reduced-observation checks

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for the disclosed battery response, exact-restart and reduced-observation checks.
It provides a bounded native development workflow across three distinct scientific contracts.
Use it to run the checks and interpret their separate results and limits.

## Read results and failures

Read `independent_preparations` and the separate `electrothermal_response`, `exact_restart` and `reduced_observation` sections.
Nested words and restart routes are not new preparations.
An unsupported reduced description is retained as a result; changing labels or retrying does not make it supported.

Read [the common result and failure guide](../../docs/results-and-failures.md) for retained attempts,
stdout/stderr and same-identity issued recovery. The route-specific outputs and stops below remain binding.
Use `workflow show battery-response-and-restart --format json` for static command effects and prerequisite roles.
Discovery does not inspect those prerequisites.

## Prerequisites and first action

Use a clean Linux x86-64 CPU checkout with the selected source and its [tested environment](../../docs/environments.md).
Obtain this bundle from that checkout or the selected source archive.
Its guide and [strict input](config.json) are workflow files, outside the wheel's package resources.
Use the checkout root as the working directory in an unactivated shell.
For a wheel installation, supply the input's absolute path.
Source-bound authoring and issue require a checkout that executes its own tracked package.

Install the locked optional simulator profile:

```sh
uv sync --locked --python 3.11.14 --group open-simulators
```

Run the native development check:

```sh
uv run --no-sync empirical-lawhood campaign battery-native-check   --config experiments/battery-response-and-restart/config.json \
  --output-dir /absolute/development/battery-response-and-restart/attempt-001
```

The check uses PyBaMM 26.6.2.0 and its installed `Chen2020` parameter set.
It needs no downloaded battery data.
The inputs are disclosed development preparations with fixed initial states.
They are already exposed.
A different ID alone cannot make the same preparation prospective.

## Expected report and stops

The report gives `independent_preparations=5` across the three scientific contracts.
Read its `electrothermal_response`, `exact_restart` and `reduced_observation` sections separately.
`nested_electrothermal_words=3` and `nested_restart_routes=2` describe nested operations.
They do not add independent units.
`campaign_candidate_compiled=false` confirms that this command constructed no campaign candidate.

The electrothermal section compares capacity and voltage after a current action.
The restart section reports defects against the native same-object continuation.
The reduced-observation target is outside the frozen donor support.
Its correct method response is `OUTSIDE_FROZEN_LOCAL_SUPPORT`, with no adequacy verdict.
An unsupported result is retained.
It must not become a positive result through retries or changed labels.

Wrong native versions, malformed fields, invalid strata and incomplete continuations refuse.
The command returns a local outcome-visible development diagnostic.
It creates no immutable issue, execution receipt, law qualification, admission or controller-use verdict.
Selected provider/candidate integration and guarded issued execution remain deferred.
The [route register](../../docs/substrate-route-register.md) owns that integration ceiling.

## Edit the disclosed development input

Use the exact typed field names in `config.json`.
The [input owner](../../src/empirical_lawhood/adapters/simulators/pybamm_development_input.py) validates their native meanings and ranges.

| Field | Scientific role |
|---|---|
| `electrothermal_unit_id` | One preparation with three nested current words |
| `restart_unit_id`, `restart_history_id` | A different preparation and the exact source history for state restart |
| `reduced_observation_unit_ids` | Three distinct preparations: two donors and one target |
| `reduced_observation_initial_socs`, `reduced_observation_initial_temperatures_k` | Ordered donor/target initial states in state-of-charge and K |
| `reduced_observation_history_id`, `reduced_observation_chart_id` | The fixed source history and disclosed reduced-coordinate chart |

For future independent work, freeze the actual preparation, initial states and exposure role before native contact.
Bind its complete prior source/unit inventory and applicable review.
Keep actions, numerical views and reconstruction routes nested within each preparation.
The three contracts do not form one pooled evidence population.
Changing a frozen scientific operand requires a new scientific identity and validation.

## Scientific contracts and numerical limits

### Electrothermal response

The selected denominator is SPMe, isothermal, CasADi with a 20-point coarse view. Its nominal output interval is 10 s. Its relative and absolute tolerances are `1e-5` and `1e-6`. The wider source chart has eight nested denominators: SPMe/DFN, isothermal/lumped and CasADi/IDAKLU.

The full thermal model remains excluded until its missing heat-transfer parameter is qualified. This starter validates one denominator. It does not qualify the other seven.

Hold, +1 A and future-only current words are nested on one preparation over 0–150 s.
The 150 s receiver is discharged capacity in A·h and terminal voltage in V.
The future current begins at 600 s.
It must leave the earlier receiver unchanged.
A +1 A, 150 s step changes capacity by `1 × 150 / 3600` A·h.
The 150 s receiver supplies the pre-600 s causal cutoff.

### Exact state restart

Restart uses a second preparation. Its +1 A prefix reaches a 300 s checkpoint, followed by a 300 s hold. A rebuilt model must recover the native same-object continuation from exact state and the checkpoint source-input ledger. The receivers are capacity in A·h, volume-mean temperature in K and terminal voltage in V.

Read them at 300, 450 and 600 s. The source localization chart has five history words and eight denominators. This starter validates one history/denominator. Its selected reconstruction placement followed source localization and remains a disclosed development diagnostic.

### Reduced-observation prediction

Reduced observation uses two low-state-of-charge/cool donors and a high-state-of-charge/warm target. All three are different development preparations. Their histories reach a 300 s checkpoint. The future receiver is a hold through 600 s.

Native same-object and exact-restart traces must agree before coordinate comparison. The retained four-coordinate chart was selected after source development outcomes. Its use here remains diagnostic. The target lies outside the frozen donor support radius.

That stop establishes no adequacy or prospective qualification.

## Existing verification and next route

The [native science checks](../../tests/test_battery_native_science.py) compare capacity with the current integral and reconstruct native state continuations.
The [shipped-input checks](../../tests/test_battery_development_input.py) verify the three contracts and refusal before native contact.
With the selected native prerequisite installed, use those existing checks:

```sh
uv run --no-sync pytest -q --fail-on-skip   tests/test_battery_native_science.py tests/test_battery_development_input.py
```

These checks establish their stated software and numerical boundaries.
They do not qualify the wider source campaign or supply a fresh chart.
The [general experiment guide](../../docs/designing-and-running-an-experiment.md) gives candidate, issue, execution, recovery and reveal boundaries.
That workflow requires a selected family binding and its actual custody and authority inputs.
Keep generated data and receipts in the declared external storage.

## References and research

PyBaMM Team (undated), [*Parameter sets*](https://docs.pybamm.org/en/v26.6.2.0/source/api/parameters/parameter_sets.html#chen2020), documentation version 26.6.2.0.
This source identifies the installed `Chen2020` parameter set.
The [version-pinned Chen2020 source](https://github.com/pybamm-team/PyBaMM/blob/v26.6.2.0/src/pybamm/input/parameters/lithium_ion/Chen2020.py) supplies the parameter definitions and research citation.

Chen, C.-H., Brosa Planella, F., O’Regan, K., Gastol, D., Widanage, W. D. and Kendrick, E. (2020), [*Development of Experimental Techniques for Parameterization of Multi-scale Lithium-ion Battery Models*](https://doi.org/10.1149/1945-7111/ab9050). *Journal of The Electrochemical Society* **167**, 080534.
This study gives the LG M50 parameterization adopted by `Chen2020`.
These sources do not qualify this target workflow or its wider scientific claims.

The target implementations own the [electrothermal](../../src/empirical_lawhood/adapters/simulators/pybamm_electrothermal_hierarchy.py), [restart](../../src/empirical_lawhood/adapters/simulators/pybamm_exact_restart.py) and [reduced-observation](../../src/empirical_lawhood/adapters/simulators/pybamm_reduced_coordinate_sufficiency.py) contracts.
The [input owner](../../src/empirical_lawhood/adapters/simulators/pybamm_development_input.py) supplies the shared command handoff.
The [scientific integrity guide](../../docs/scientific-integrity.md) states the independent-unit and authority boundaries.
