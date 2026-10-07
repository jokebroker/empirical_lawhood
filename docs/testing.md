# Verification profiles

SPDX-License-Identifier: CC-BY-4.0

This guide gives the portable, native and held-input verification profiles.
It separates software conformance from scientific qualification.
Use it to select existing checks for a change or release claim.

The portable suite includes independent scientific counterexamples and shared
engine conformance. All lifecycle authorities, signatures, sources and cohorts
under `tests/` are synthetic software fixtures. Their records cannot attest a
real prior census, authorize an operator, or qualify a scientific claim.

The focused lifecycle tests in `runtime_platform/`, `api/` and their support
modules were adapted from the historical response-law source project. They retain its issue,
approval, scheduler, artifact and recovery behavior while using this package's
namespace and strict source-origin contract. The target reference campaign is
rebuilt by `scripts/generate_test_fixture.py`. It is not a copied historical
approval or qualification packet. Keep fixture signatures confined to tests.

`test_reactor_public_lifecycle.py` creates a clean temporary Git checkout of the
current source, a real isolated Linux tmpfs storage profile and ephemeral test
signers. It traverses base and extension issue, experiment-package assembly, execution-plan compilation and twenty native acquisitions. It then exercises the separate reveal gate, adjudication and receipt recovery after loss of its disposable catalog. A second case retains a synthetic census of 20,000 units and 230,000 streams through public authoring/proof/issue preview.
It validates the real 64 MiB member boundary. Positive fixtures model prospective
eligibility only inside their closed synthetic test world.

They cannot establish
eligibility for a real study. A third case supplies an explicitly exposed
assignment and contradictory freshness attestations: both issue paths refuse
before writes. Allow several minutes and a few hundred MiB of temporary storage.

## Independent checks for current outputs

The coverage inventory requires executed statements in the declared critical owners.
It checks measurement presence, not correctness, complete branch coverage or qualification.
Use these independent checks when changing the corresponding scientific or custody boundary:

| Boundary / output | Independent check and refusal owner |
|---|---|
| History algebra and passive forecasts | [Spectrum and causal-history oracles](../tests/test_matrix_history_science.py); the pinned `lambda3 / lambda4` convention and complete history denominator |
| Tangent, transfer and support analyses | [Differential comparisons](../tests/test_matrix_preparation_differential.py) and [public tangent](../tests/test_matrix_tangent_public.py); whole-root units and fixed pre-outcome ranks |
| Retained reports and current result custody | [Retained-analysis mutations](../tests/test_retained_analysis_custody.py) and [issued output/receipt joins](../tests/test_current_result_custody.py); exact complete members and protected-read authority |
| Actual producing source and numerical environment | [Provenance refusals](../tests/test_matrix_numerical_provenance.py); actual lock/runtime/source identities rather than caller labels |
| RC response and extension | [Independent RC equation checks](../tests/test_rc_information.py), [native ladder checks](../tests/test_resistor_capacitor_native_science.py) and the [worked extension](extending-the-engine.md); analytical falsifiers remain separate from reducers |
| Native clocks, retained histories and temperature ceilings | [Pre-contact admission probes](../tests/test_native_operational_admission.py) and [TORAX peak history](../tests/test_torax_native_quickstart.py); operational capacity is separate from accuracy |
| Full finite authoring/proof | [C32/E64 lifecycle](../tests/test_finite_response_rerun_authoring.py), [integrity refusals](../tests/test_finite_response_rerun_integrity.py) and [parse/latency preservation](../tests/test_finite_response_rerun_latency.py); public fixtures remain software-only evidence |

```sh
uv sync --locked --python 3.11.14 --group reactor-example
uv run --no-sync pytest -q -ra --fail-on-skip -m 'not native and not held'
```

The release runners fix all five BLAS/OpenMP thread variables before starting
their selected interpreters. The test session repeats that setup before
numerical collection. For focused numerical checks, set those variables to `1`
before starting pytest when an installed plugin imports numerical libraries early.
The installed-wheel smoke additionally validates the real console
script sets these values before its first NumPy import, with conflicting
values deliberately supplied by the caller.

| Profile | What it requires | What it verifies |
|---|---|---|
| `portable` | Base + `dev`, `build`, `reactor-example` | Portable scientific tests, source, authority, issue, execution/recovery contracts, generators/static errors, wheel-through-sdist and installed-wheel checks |
| `native-open` | Also `open-simulators`, `reaction-response-simulators` | Selected installed-source/native checks for Gym–TORAX, TORAX, PyBaMM, Cantera and FiPy |
| `native-brian2` | Separate checked-in Brian2 native environment | The pinned LIF check against its independent analytic spike clock |
| `held-reactor` | `REACTOR_HELD_SOURCE_ROOT`, `REACTOR_LOCAL_DISCOVERY` | Explicitly selected authentic reactor preflight, ports and native development checks |
| `held-response` | Held source, matching plans, design packets, exposure files and route model banks listed below | Selected prepared-response, information-response, causal-response and finite-input contracts, no implicit qualification |

Run a profile from a clean committed checkout:

```sh
uv run --no-sync python scripts/release_check.py --profile portable --output-dir /path/to/new-release-packet
uv run --no-sync python scripts/release_check.py --profile native-open --output-dir /path/to/new-native-check-packet
```

Each invocation clones the exact clean commit, creates a fresh locked
environment and preserves logs plus a machine-readable result. Use `--offline`
only when the required interpreter, wheels and build dependencies are already
cached. The runner never reuses an existing packet directory. Budget hours for
the current complete portable suite. A pre-finish run on an Intel i7-10700KF,
Python 3.11.14 and the locked environment, under branch coverage with each
numerical thread limit set to one, took 2 h 35 min. Its complete C32/E64 integrity
case took 52 min 30 s; its complete 9,126-row geometry/custody case took
28 min 14 s. Both large cases passed, but that run ended with two stale CLI
fixture failures; it is not a passing final-source release gate.
The [finite workflow](../experiments/finite-response-law/guide.md) gives measured
individual authoring, loading and proof costs. These command timings and the
complete test durations describe different work.
The suite reports its ten slowest test durations so the selected source's actual
cost is retained with the gate, rather than inferred from progress dots.

Budget several GiB for its checkout, two environments and temporary fixtures. Optional native
compilation can take longer. Authentic-input sizes are determined by their
manifested files and declared limits, not a fixed test-size promise.

`pytest` without a selector remains convenient for a configured development
machine and may skip optional inputs. **Every selected release profile uses
`--fail-on-skip`.** A missing dependency, file or collection-time import produces
a failed job. A successful portable job says nothing about unselected profiles.
An existing check linked from the route register is not a claim that its optional
native or held profile passed on the final source. Historical development
observations keep their original producing identity and evidence limit.
TORAX's advertised heat check verifies declared power-times-duration effort and
native core-response ordering; it does not independently integrate deposited heat
or establish a full thermal balance.
The skip-policy regression tests exercise both collection and runtime skips.

For six-matrix checks, supply the explicit environment variables used by the selected checks:

- `PREPARED_RESPONSE_HELD_SOURCE_ROOT`
- `PREPARED_RESPONSE_SOURCE_QUALIFICATION_PLAN`
- `PREPARED_RESPONSE_SOURCE_QUALIFICATION_DESIGN_PACKET`
- `PREPARED_RESPONSE_SOURCE_QUALIFICATION_PRIOR_EXPOSURE`
- `INFORMATION_RESPONSE_PREDICTION_PLAN`
- `INFORMATION_RESPONSE_PREDICTION_PRIOR_EXPOSURE`
- `INFORMATION_RESPONSE_PREDICTION_MODEL_BANK`
- `CAUSAL_RESPONSE_PREDICTION_BANK_MATCHING_PLAN`
- `CAUSAL_RESPONSE_PREDICTION_DESIGN_PACKET`
- `CAUSAL_RESPONSE_PREDICTION_PRIOR_EXPOSURE`
- `CAUSAL_RESPONSE_PREDICTION_MODEL_BANK`. Finite checks also use
`EL_FINITE_RESPONSE_LAW_FROZEN_PLAN` and `EL_FINITE_RESPONSE_LAW_PRIOR_EXPOSURE`. Before you select a held profile, examine the exact `_external(...)` calls in its test files.
Supply the actual permitted input paths.
Do not synthesize absent historical evidence.

No profile contacts a paid instrument or resumes an owner's old scientific run.
Passed regression tests preserve their synthetic/development ceiling. A new
scientific qualification needs the independently authorized fresh cohort,
complete exposure census and exact source identity described in the
[operator walkthrough](reactor-operator.md).

The portable packet retains branch coverage for the critical seams listed in
`.coveragerc`, including custody/reader joins, canonical encoding, SQL and public
API boundaries. It measures the parent test process. Child lifecycle/native
processes are tested but not included in that coverage numerator. Missing
decisions are retained in `coverage.json`.

Deliberately unmeasured or unselected
surfaces include real owner signatures/held archives, physical controllers,
deferred adapters and unentered scientific programs. The focused SQLite
regressions exercise family writes/query/corruption and read concurrency. They
do not independently exhaust all six dataset record kinds or evidence-verifier
routing. No repository-wide coverage threshold or scientific qualification is
inferred.

See [CI profiles](ci.md) and [release status](release-status.md).

## Current paper-integration checks

Current integration tests exercise public input, provider proof, custody, result readout and recovery boundaries.
The finite-response fixtures retain the actual 32-root calibration and 64-root evaluation graphs with their two nested views.
Typed plan tests exercise the dedicated 64 MiB limit without increasing compact-control or receipt limits.
Synthetic issue, approval and qualification records remain confined to those software fixtures.

The [history science tests](../tests/test_matrix_history_science.py) use independent algebra and differential-equation controls.
The [history API tests](../tests/test_matrix_history_public_api.py) authenticate supplied arrays, full requested rosters and retained results.
They also check one paired native step, cancellation, corrupt bytes, wrong clocks and recovery without reacquisition.
Related geometry, selected-event, tangent, transient and preparation tests retain their independent falsifiers and scientific failure dispositions.
Each guide identifies its operative protocol and evidence ceiling.

Bounded numerical checks do not complete the full geometry scan, history cohort or preparation campaign.
Missing historical custody remains unresolved rather than replaced by synthetic records.
Portable conformance and a selected release gate establish software behavior only.
Full scientific campaigns require their own explicit inputs, source evidence and authority where the protocol requires it.

## References and research

The linked source modules, command metadata and existing checks own the implemented behavior described here.
The [scientific integrity guide](scientific-integrity.md) defines its separate evidence and authority boundaries.
The [program source register](../paper/SOURCES.md) identifies bounded historical results and unavailable primary records.
