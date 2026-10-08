# Installation and tested environments

SPDX-License-Identifier: CC-BY-4.0

This guide gives installation procedures and the tested native environments.
It distinguishes dependency installation from source qualification and scientific authority.
Use it to select the required profile before a documented command.

The supported release-check host is **Linux x86-64, CPython 3.11.14, CPU**.
The package uses POSIX file descriptors, `fcntl`, `pwd`, Linux process inspection
and `findmnt` for guarded execution/storage. Windows, macOS, ARM and GPU execution
are not qualified by this release gate. A platform-independent wheel filename
does not establish runtime portability.

`requires-python >=3.11,<3.12` and the broad dependency ranges describe installation
compatibility. A native scientific route can require a narrower environment.
The reactor and six-matrix native checks require **CPython 3.11.14 and NumPy 2.4.6** and
refuse drift. Keep that refusal: a successful import with another patch/version
is not a numerical qualification.

| Task or route | Selected environment | Contact and evidence boundary |
|---|---|---|
| Help, doctor, validation, static discovery | Locked base environment. Tested on the host above | Reads metadata/documents. No native source readiness or authority implied |
| Reactor prefix, assigned reactor, six-matrix | `reactor-example` group. CPython 3.11.14, NumPy 2.4.6, other base dependencies from `uv.lock` | Public canaries are exposed development measurements. Issued work has separate input, source and authority gates |
| Gym–TORAX, TORAX, PyBaMM | `open-simulators`: Gymtorax 1.1.1, TORAX 1.4.2, JAX/JAXlib 0.10.2, NumPy 2.4.6, SciPy 1.17.1, xarray 2026.7.0, PyBaMM 26.6.2.0 | Optional bounded native checks. No general campaign qualification |
| Cantera and FiPy | `reaction-response-simulators`: Cantera 3.2.0, FiPy 4.0.3 | Bounded development contracts |
| Grid2Op | `grid2op-held`: Grid2Op 1.12.5, LightSim2Grid 0.13.1 | Also needs an explicitly held offline chronic. Dependency presence alone is insufficient |
| Brian2 | Separate [native environment](../experiments/neuron-current-response/native-env/pyproject.toml): CPython 3.11.14, Brian2 2.9.0, NumPy 1.26.4 | Runs through the explicit native Python subprocess. Do not combine its NumPy pin with the main environment |
| Release packaging | `dev`, `build`, `reactor-example`. Hatchling 1.32.4 and its transitive dependencies in `uv.lock` | Builds and tests software artifacts. No scientific requalification |

## Frozen RC numerical input profile

The retained RC reproduction uses NumPy `X86_V3` dispatch and the OpenBLAS Haswell kernel.
A different CPU dispatch can change exact descriptor bytes with the same dependency versions.
On an AVX2/FMA host, select this profile before NumPy import:

```sh
export OPENBLAS_CORETYPE=HASWELL
export NPY_DISABLE_CPU_FEATURES=X86_V4,AVX512_ICL,AVX512_SPR
uv run --no-sync python scripts/check_frozen_rc_inputs.py
```

The portable CI job selects these variables before its first numerical import.
The release runner separately retains its single-thread limits.
The check requires exact hashes for all eighteen frozen exposed development descriptors and their original fibre seeds.
A mismatch refuses before the expensive portable tests.
It performs no native acquisition or scientific qualification.

Other numerical environments retain their own source and verification requirements.
Keep the exact original descriptor hashes and numerical records unchanged.

## Discover and interpret readiness

`workflow show ID --format json` lists the selected task's environment and input roles without checking either.
Use [configuration](configuration.md) to validate supported copied inputs before native contact.
Use the existing `doctor --route` inspection to check the selected dependency versions.
A successful doctor exit means inspection completed. Missing dependencies, version drift and storage state remain separate report fields.
Neither installed packages nor writable storage establish an eligible cohort, qualified source or scientific authority.
An editable installation's stale distribution metadata is a local mismatch, not proof that a selected wheel is defective.
Read [results and failures](results-and-failures.md) for the separate operational/scientific states.


## Editable checkout

Install `uv` and Git, then run from the selected checkout in an ordinary,
unactivated shell:

```sh
uv sync --locked --python 3.11.14 --group reactor-example
uv run --no-sync empirical-lawhood --help
uv run --no-sync empirical-lawhood doctor --route reactor --format json
uv run --no-sync empirical-lawhood example reactor-prefix --output-dir ./example-output/reactor-prefix
```

`uv sync` does not activate the environment in your shell. Use `uv run --no-sync`
after the selected sync. A subsequent default sync can remove optional groups.
For example, select both needed groups together:

```sh
uv sync --locked --python 3.11.14 --group reactor-example --group open-simulators
uv run --no-sync empirical-lawhood doctor --route open-simulators --format json
```

`doctor --route` accepts `reactor`, `prepared-response`, `open-simulators`, `reaction-response`, or
`grid2op`, `rc-challenges`, `finite-response-law`, or `preparation-applicability`. It reads distribution metadata and reports `installed_version`,
`required_version`, `available` and `version_matches`. Absence and version drift
are distinct reason codes. It does not import/initialize a simulator to perform
this check, authenticate source data, verify a fitted bank, or grant authority.
The report's successful diagnostic exit means inspection completed. Before you run a native command, examine its warnings and per-dependency matches.

The [Brian2 guide](../experiments/neuron-current-response/guide.md) installs its own locked
environment. Its interpreter can inspect versions without importing Brian2:

```sh
uv run --project experiments/neuron-current-response/native-env --no-sync python -c 'import importlib.metadata as m, platform; print(platform.python_version(), m.version("brian2"), m.version("numpy"))'
```

## Selected wheel

Dependency groups are checkout tooling, not wheel extras. To run the demo from
a selected wheel, create a fresh environment with the exact Python patch and
select the NumPy pin explicitly:

```sh
uv venv --python 3.11.14 /path/to/wheel-env
uv pip install --python /path/to/wheel-env/bin/python /path/to/empirical_lawhood-0.2.0-py3-none-any.whl 'numpy==2.4.6'
/path/to/wheel-env/bin/empirical-lawhood doctor --route reactor --format json
/path/to/wheel-env/bin/empirical-lawhood example reactor-prefix --output-dir /path/to/new-demo
```

For exact release reproduction, use the release packet's hash-pinned
`wheel-requirements.txt` with `uv pip install --require-hashes -r ...`, then
install the selected wheel with `--no-deps`. Examine the wheel's SHA-256 against
`release-manifest.json`. A `0.2.0` filename alone does not select its source.

A wheel can demonstrate, inspect and validate. Source-bound candidate, issue
and execution operations must import the tracked package from their selected
clean checkout. Naming that checkout with `--project-root` cannot make a wheel
its executing source. The [operator walkthrough](reactor-operator.md) uses the
editable route and explicit storage/trust inputs.

## Costs and checks

The observed minimal environment is about 0.5 GiB. The optional simulator
environment used for development is about 2.2 GiB. Allow additional space for
the uv cache and build/test checkouts. Downloads and native compilation can
take minutes.

The public prefix itself normally takes seconds and its three
output files occupy less than 1 MiB. These are planning estimates, not resource
guarantees. Issued tasks use their declared resource envelope and the storage
profile's configured free-space floor.

See [contributor checks](../CONTRIBUTING.md) for the clean release runner and
[test profiles](testing.md) for optional-native versus held-input jobs.

## References and research

The linked source modules, command metadata and existing checks own the implemented behavior described here.
The [scientific integrity guide](scientific-integrity.md) defines its separate evidence and authority boundaries.
The [program source register](../paper/SOURCES.md) identifies bounded historical results and unavailable primary records.

The [NumPy CPU options](https://numpy.org/doc/2.4/reference/simd/build-options.html) define the selected feature groups and runtime exclusions.
The [OpenBLAS runtime variables](https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/) define `OPENBLAS_CORETYPE`.

## Current numerical integrations

The current RC and matrix paper integrations reuse the main locked NumPy 2.4.6 / SciPy 1.17.1 environment.
This includes geometry, selected events, algebra, passive prediction, tangent, transient, baseline and preparation diagnostics.
No donor environment, old mount or additional simulator stack is required.
Static workflow/configuration discovery and dependency inspection initialize no native provider.
Explicit native source export and conformance use the current six-matrix marcher.

The listed native version checks retain their exact refusal contracts.
Native source entry authenticates the selected checkout and lock before source writes or numerical acquisition.
The shared matrix boundary enforces Python, NumPy, SciPy, host and single-thread pools.
Its provenance inventories every current package source and resource, excluding interpreter caches.
The observed HEAD and checkout cleanliness describe the working checkout; they do not claim that HEAD produced uncommitted bytes.

Descriptive analysis receipts retain actual interpreter, numerical-library, complete source and dependency-lock identities.
Retained readers authenticate that producing evidence without observing their own numerical environment or repeating calculations.
Versioned provenance records preserve earlier serialized records under their original identities.
A matching environment establishes software availability only.
Source, allocation, issue and outcome access remain separately authenticated.
See [the integration entry](integrations.md) for the selected operation's prerequisites and contact boundary.
