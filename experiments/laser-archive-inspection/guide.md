# Glenn 2026 retrospective archive inspection

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for preview and local import of a published laser-burst archive.
It gives a retrospective inspection procedure without live actuation or prospective qualification.
Use it to inspect the pinned release and understand its missing custody handoff.

The [strict target selector](config.json)
names the published [Zenodo 17163053 release](https://zenodo.org/records/17163053)
and declares its source role `RETROSPECTIVE_OUTCOME_VISIBLE`. This is a
logged physical archive import, not a live laser controller or prospective
intervention. The retained adapter pins the release's ZIP size/SHA-256 and
four safe selected members. The target config contributes a new local
inspection identity and accepted filename. It carries no old issued identity,
authority or physical result.

Preview the decoded upstream selection without network or outcome contact:

```sh
uv sync --locked --python 3.11.14
uv run --no-sync empirical-lawhood campaign glenn-import-check \
  --config experiments/laser-archive-inspection/config.json --preview \
  --output-dir /absolute/development/laser-archive-inspection/attempt-001
```

To supply the public bytes, create an external directory with enough space
for the 549,154,725-byte archive and download its exact record file. This is
an explicit researcher-managed acquisition, **not** a target custody receipt.
The CLI validates the archive size and SHA-256 before opening selected members.
Keep the release attribution and license record with your own source custody.

```sh
mkdir -p /absolute/held/glenn-2026
curl --fail --location \
  'https://zenodo.org/records/17163053/files/GDGlenn_PRR_2026.zip?download=1' \
  --output /absolute/held/glenn-2026/GDGlenn_PRR_2026.zip
uv run --no-sync empirical-lawhood campaign glenn-import-check \
  --config experiments/laser-archive-inspection/config.json \
  --source-root /absolute/held/glenn-2026
```

The source root must be absolute and the ZIP must be a regular non-symlink
file under it. The adapter validates the complete ZIP inventory.
It reads only the exact observation log, deposited model log, NumPy focal-radius array and figure notebook.
It does not execute the notebook. The successful local
check reports 101 independent logged **bursts**, 20 table columns and four
selected members. Ten shots inside a burst are nested observations, not ten
units.

Six command coordinates remain in their deposited
`source-native-command` units and on the burst clock. The deposited figure notebook scales source-native focal values by ×1000 into its micrometre display. The adapter preserves that exact recipe. Focal `r50` is a **post-command mediator**, not a prior prediction.

The source does not supply its generation method or uncertainty. The first burst has no
prior fitness. Each later `prior_fitness` is the previous burst's observed
fitness. The deposited model log is source validation input, not a pre-burst
forecast.

An action-stage delivery journal is unavailable, so no requested
versus applied versus realized physical intervention can be inferred.

From an unedited clean checkout with no external ZIP, the command refuses
before archive content contact. The [authentic scientific check](../../tests/test_glenn_authentic_source.py)
compares the causal one-burst lag and native NumPy focal conversion against
the external release. The [selector check](../../tests/test_glenn_quickstart.py)
validates the shipped config, absence/path/role refusals and the one-CLI import
surface. With your held file, run:

```sh
GLENN_PUBLIC_ARCHIVE=/absolute/held/glenn-2026/GDGlenn_PRR_2026.zip \
  uv run --no-sync pytest -q \
  tests/test_glenn_authentic_source.py tests/test_glenn_quickstart.py
```

This local inspection does **not** install a typed guarded custody receipt,
outcome-visible analysis authorization, selected production provider or
candidate. Its diagnostic audit uses an explicit unqualified-local-inspection
marker. For a later experiment, bind the exact release, physical storage and custody identity. Declare historical outcome access.

Supply the typed authorization required by the retained production manifests. The [experiment guide](../../docs/designing-and-running-an-experiment.md) describes issue, execution, recovery and the separate evaluator-reveal gate.
This selector is not an input to `campaign compile-candidate`. Historical measurement/order interpretation and retrospective response transformation are bounded here. Prospective local-law, transport, admission and controller-use claims do not follow.

See the [source assessment](../../docs/sources/glenn-2026-qualification.md)
for focal uncertainty and deposited action-label exceptions.

## Read results and failures

Preview reports `network_contacted` and `outcome_contacted` without obtaining the archive.
For an actual import, read `archive_sha256`, `independent_bursts`, `column_count` and `source_exceptions`.
Logged bursts are the independent units; nested shots are not. The output remains retrospective local inspection.

Read [the common result and failure guide](../../docs/results-and-failures.md) for retained attempts,
stdout/stderr and same-identity issued recovery. The route-specific outputs and stops below remain binding.
Use `workflow show laser-archive-inspection --format json` for static command effects and prerequisite roles.
Discovery does not inspect those prerequisites.

## Obtain the workflow inputs

Obtain this bundle from the selected repository checkout or source distribution. The guide and study inputs are repository workflow files, not wheel resources. Run each command from the checkout root in an unactivated shell. The relative paths use that root.

Keep generated data and receipts in the declared external storage. A wheel can perform its advertised inspection and native-input checks with absolute paths. Source-bound authoring and issue require a clean checkout that executes its own tracked package.

## References and research

G. D. Glenn, S. H. Glenzer and C. A. J. Palmer (2026), [*Dataset: Characterization and automated optimization of laser-driven proton beams from converging liquid sheet jet targets*](https://doi.org/10.5281/zenodo.17163053), version 1.
The source profile pins archive and member hashes. The [source assessment](../../docs/sources/glenn-2026-qualification.md) records attribution and transformation limits.

Gareth Seneque (2026), [*Empirical Lawhood*](../../paper/manuscript.md), edition 0.60, gives the scientific formulation.
The [program source register](../../paper/SOURCES.md) identifies historical results and unavailable primary records.
The linked scientific source modules and existing tests establish only their declared implementation and development boundaries.
