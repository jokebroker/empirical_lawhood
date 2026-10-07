# Leaky integrate-and-fire neuron development check

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for a native leaky integrate-and-fire neuron development check.
It uses a separate numerical environment for one independently reset trial block.
Use it to run the current-step check and interpret the bounded spike report.

## Read results and failures

Read `config_id`, `independent_unit_id`, `independent_units` and the `native` report.
Inspect the reported runtime, requested/accepted/applied/realized pA values and native spike clocks.
Two reset arms share one block. The analytic LIF clock is an independent falsifier, not another unit.

Read [the common result and failure guide](../../docs/results-and-failures.md) for retained attempts,
stdout/stderr and same-identity issued recovery. The route-specific outputs and stops below remain binding.
Use `workflow show neuron-current-response --format json` for static command effects and prerequisite roles.
Discovery does not inspect those prerequisites.

## Prerequisites and first action

Use the checkout root and the [tested environment](../../docs/environments.md).
Obtain the colocated strict input from the selected source archive.
Install the profile shown below.
This command runs an exposed development diagnostic.
It grants no scientific qualification or execution authority.

```sh
uv sync --locked --python 3.11.14 --group reactor-example
uv sync --locked --project experiments/neuron-current-response/native-env --python 3.11.14
uv run --no-sync empirical-lawhood campaign brian2-native-check \
  --config experiments/neuron-current-response/config.json \
  --native-python experiments/neuron-current-response/native-env/.venv/bin/python \
  --output-dir /absolute/development/neuron-current-response/attempt-001
```

## Adapt a development input and retain its report

Copy the [config](config.json) to a fresh external directory. Replace the example
absolute path below with a new directory whose parent already exists. Keep the
shipped config unchanged and record your edited input alongside the result.

```sh
NEURON_DEV=/absolute/path/to/new-neuron-development
mkdir "$NEURON_DEV"
cp experiments/neuron-current-response/config.json "$NEURON_DEV/config.json"
```

Edit the copied JSON before running it. Retain schema
`empirical-lawhood/simulators/brian2-neuron-current-response/brian2-lif-native-config`,
version `1.0.0` and the `value` envelope. Decimal fields use tagged strings such as
`{"decimal":"0.1"}`, never bare fractional JSON numbers. The
[strict config](../../src/empirical_lawhood/adapters/simulators/brian2_neuron_current_response/contracts.py)
permits these development edits:

| Fields inside `value` | Required relation |
|---|---|
| `config_id`, `independent_unit_id` | Stable IDs containing lowercase letters, digits, `.`, `_` or `-`, beginning with a letter or digit. Record the selected development identities. Renaming a block does not prove independence or erase prior exposure. |
| `timestep_ms`, `baseline_ms`, `horizon_ms` | Finite, positive and representable in the native clock; baseline and horizon must each be an exact integer multiple of the timestep; baseline plus horizon must be at most 1000 ms; at most 100,000 total updates per arm. |
| `membrane_tau_ms`, `membrane_resistance_mohm`, `maximum_accepted_pa` | Finite and positive. The accepted step is the smaller of the requested step and maximum accepted current. |
| `rest_mv`, `reset_mv`, `threshold_mv` | Finite; reset must equal rest and threshold must exceed rest. |
| `requested_step_pa` | Finite and positive. |

There is no configurable seed field. Hold/step reset arms, Euler execution and the
voltage/spike receivers remain the check's fixed structure. A different permitted
input is another exposed development check; the shipped seven-spike result below
does not predict its result.

The [operational admission owner](../../src/empirical_lawhood/adapters/simulators/_native_admission.py)
checks exact tick counts before worker launch and repeats the check in the
standalone worker before Brian2 import or array allocation. The estimated
history allowance is 128 bytes per update, capped at 16 MiB per arm, covering
the float64 stimulus, voltage/time monitors, spike times and working copies.
The two reset arms run sequentially. This conservative admission estimate is
not a proven process RSS bound; interpreter and native-runtime overhead is
separate. Decimal times must preserve their integer tick counts and advance in
the float64 native clock. The shipped 0.1 ms timestep and 1,000 updates remain
unchanged. The 60-second worker timeout is additional containment, not memory
admission or scientific qualification.

Brian2's CLI requires exact canonical bytes, so ordinary editor formatting alone
is insufficient. Validate and serialize the edited copy through the existing
typed codec before native contact. This step executes no simulator:

```sh
uv run --no-sync python - "$NEURON_DEV/config.json" <<'PY'
from pathlib import Path
import sys
from empirical_lawhood.api.codecs import load_registered_authoring
from empirical_lawhood.adapters.simulators.brian2_neuron_current_response.contracts import Brian2LIFNativeConfig

path = Path(sys.argv[1])
config = load_registered_authoring(
    path,
    root_schemas={Brian2LIFNativeConfig.SCHEMA: Brian2LIFNativeConfig},
    maximum_bytes=16 * 1024,
)
path.write_bytes(config.canonical_bytes())
PY
```

If validation succeeds, use the previously installed native environment and retain
standard output and diagnostics outside the checkout:

```sh
uv run --no-sync empirical-lawhood campaign brian2-native-check \
  --config "$NEURON_DEV/config.json" \
  --native-python experiments/neuron-current-response/native-env/.venv/bin/python \
  > "$NEURON_DEV/report.json" 2> "$NEURON_DEV/diagnostics.log"
```

On success, `report.json` is the CLI's JSON standard output: check runtime, one
independent unit, both native arms, current delivery stages, mean voltage and spike
count. The CLI has no report-file export option. A handled input or native-check
failure exits 3 and writes `Brian2 native check refused: ...` to standard error;
an empty or incomplete captured stdout file is not a successful report. CLI option
errors may fail earlier. Retain the refusal and input rather than adjusting values
to conceal it. Neither result produces a candidate or issued campaign; use a new
external directory for a subsequent development attempt.

## Expected output and scientific scope

The original nomination describes one independently reset neuron trial block and a bounded current step in pA.
Its receivers are membrane potential (mV) and spike count.
It fixes an 80 ms post-step horizon and a pre-step causal cutoff. It did not
freeze a runnable native config. The [config](config.json) is a **new development canary** with these values:

- Rest: 20 ms.
- Clock-driven Euler step: 0.1 ms.
- Membrane time constant: 10 ms.
- Resistance: 100 MΩ.
- Rest/reset: −70 mV.
- Threshold: −50 mV.
- Requested step: 1000 pA, capped at 300 pA.
- Post-step receiver window: 80 ms.
 Hold and step arms are reset and nested in one
trial block. They are not two independent units. The realized current is read
from the native TimedArray at the neuron clock. The post-step mean membrane
potential and spike count are separate receivers. A realized current different from the accepted action falsifies this claim.
So does a spike count that violates the reset LIF threshold clock at this bounded timestep. Preparation
and measurement conformance support this bounded native response check.
Order relation, local law, admission and controller use require separate claims and evidence.

Brian2 2.9.0 raises `AttributeError: numpy.ndarray.ptp` under the main
package's NumPy 2.4.6. The [separate pinned native environment](native-env/pyproject.toml)
uses CPython 3.11.14, Brian2 2.9.0 and NumPy 1.26.4. Its lock is independent of the main package lock.
It installs no second console command.



The installed CLI strictly decodes the input and calls the internal worker in its native environment.
It reports requested, accepted, applied and realized current in pA.
It reports post-step receiver units, one independent block and exact runtime. The [scientific check](../../tests/test_brian2_native_science.py) compares the native spike count to the independent analytic LIF threshold clock.
It validates the hold counterexample and refusal of malformed input before contact. The reported seven spikes are a local simulator result for
the development canary, not a sealed evaluation outcome.

**Development boundary. Further integration deferred for first release:**
this native check does not yet select a Brian2
provider/runner in the shared candidate compiler. It produces no
`ExecutableStudyDefinition`, candidate or no-effect production plan. The
historical independent-recurrence source-qualification record continues to describe its old
NumPy 2.x import failure.

This new native environment does not rewrite that
record or import its authority and source hashes. For a future candidate, predeclare fresh trial-block IDs and seeds. Bind the strict native config and worker as an executable capability. Before issue, prove output locators, resources and typed authority through `campaign compile-candidate` and `check-readiness`.

No authority or prospective
attestation is supplied here.

## Obtain the workflow inputs

Obtain this bundle from the selected repository checkout or source distribution. The guide and study inputs are repository workflow files, not wheel resources. Run each command from the checkout root in an unactivated shell. The relative paths use that root.

Keep generated data and receipts in the declared external storage. A wheel can perform its advertised inspection and native-input checks with absolute paths. Source-bound authoring and issue require a clean checkout that executes its own tracked package.

## References and research

Marcel Stimberg, Romain Brette and Dan F. M. Goodman (2019), [*Brian 2, an intuitive and efficient neural simulator*](https://doi.org/10.7554/eLife.47314).
The adopted simulator is Brian2 2.9.0. The separate environment owns its exact numerical dependencies.

Gareth Seneque (2026), [*Empirical Lawhood*](../../paper/manuscript.md), edition 0.60, gives the scientific formulation.
The [program source register](../../paper/SOURCES.md) identifies historical results and unavailable primary records.
The linked scientific source modules and existing tests establish only their declared implementation and development boundaries.
