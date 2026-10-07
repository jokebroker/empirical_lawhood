# Paper reproduction on the first public release

SPDX-License-Identifier: CC-BY-4.0

This record documents reproduction on the local `paper_repro` branch.
It tracks scientific results and the failures encountered during reproduction.
Use the retained commands and reports to inspect each result.

The scope includes analytical and numerical RC experiments, forecast-mediated
matrix reuse, and preparation-mediated applicability in paper edition 0.60.
Run artifacts are retained in
`/media/chevre/SEMIOS/empirical-lawhood-paper-repro-20261008`; commands have separate
invocation, stdout, stderr and completion records. No commits are pushed.

The starting source is `b2f258d9731acf2ec7a573e5fb0ac81ee98f52cd`. The selected
environment is Linux x86-64, Python 3.11.14, NumPy 2.4.6, SciPy 1.17.1, installed
with `uv sync --locked --python 3.11.14 --group reactor-example --group dev`.
Subsequent commands use `uv run --no-sync` and single numerical threads.

Historical reproduction, current numerical repetition, software conformance
and fresh scientific qualification are recorded separately. Exposed roots and
original grants will not be presented as new scientific qualification.

## Initial attempts and blockers

1. The public [integration guide](integrations.md),
   [RC guide](../experiments/rc-challenges/guide.md),
   [forecast guide](../experiments/finite-response-law/guide.md) and
   [preparation guide](../experiments/preparation-applicability/guide.md) lead to
   current workflows. Historical primary records are explicitly absent from
   the public repository. The local original research checkout and its external
   disk retain additional evidence to inspect under its original identities.
2. `example rc-information --output-dir ...` and
   `campaign finite-response-native-check --output-dir ...` both refused before
   calculation on SEMIOS. The operational writer attempted `os.link`, which
   fails with `PermissionError` on the actual `vfat` mount. Its error message
   attributed the failure only to an unusable destination. Both failed command
   records and empty destinations are preserved. Development retention now uses
   the same Linux `renameat2(RENAME_NOREPLACE)` operation as scientific control
   publication, anchored to its open directory descriptor. Existing files and
   symlinks are still refused atomically. No scientific operands changed.
3. Storage diagnosis initially reported missing artifact storage. The profile
   constructor does not create its namespaces; creating the selected `artifacts`
   directory resolved that storage check. The actual volume is `/dev/sdb1`,
   `vfat`, identity `0C08-3C1A`. The profile retains a 10 GiB free-space floor and
   one campaign task at a time. Doctor confirms the numerical versions and
   storage, while reporting unavailable executor resource enforcement and absent
   local catalog separately. Those are not scientific results.
4. The original 48-root nomination export succeeded and reports no refitting.

## Verified calculations and development checks

- `rc-analytical-fat-fixed` completed on SEMIOS. The analytical separation is
  `0.1181175984276443591969259944323339078659610748157663731326851667645593446731837 V0`.
  The minimax bound is
  `0.05905879921382217959846299721616695393298053740788318656634258338227967233659185 V0`.
  The independent 110-digit equation check passed. This reproduces the
  analytical illustration without adding sampled evidence.
- `finite-native-fat-fixed` completed the excluded one-root canary on SEMIOS.
  Both numerical views and both futures completed. Pulse impulses were
  `0.5120000000000003` in dimensionless native-force time units. This checks
  native execution and retention. It does not reproduce the 64-root result.
- The focused retention and bounded descriptor suite passed all 137 tests.
  The first run directly on FAT passed 40 retention tests. Three symlink
  tests failed because FAT cannot create their deliberately unsafe inputs.
  The complete focused suite passed with temporary POSIX test fixtures.
  Durable test logs remain on SEMIOS. Lint and documentation checks passed.

Full RC and matrix scientific reproduction remains in progress.
