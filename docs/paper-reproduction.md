# Paper reproduction on the first public release

SPDX-License-Identifier: CC-BY-4.0

This record documents reproduction on the local `paper_repro` branch.
It tracks scientific results and the failures encountered during reproduction.
Use the retained commands and reports to inspect each result.

The scope includes analytical and numerical RC experiments, forecast-mediated
matrix reuse, and preparation-mediated applicability in paper edition 0.60.
Run artifacts are retained in
`/media/chevre/SEMIOS/empirical-lawhood-paper-repro-20261008`.
Commands have separate invocation, stdout, stderr and completion records.

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
   publication, anchored to its open directory descriptor.

   Existing files and symlinks are still refused atomically. No scientific operands changed.

3. Storage diagnosis initially reported missing artifact storage.
   The profile constructor does not create its namespaces.
   The storage check passed after creation of the selected `artifacts` directory.

   The actual volume is `/dev/sdb1`, `vfat`, identity `0C08-3C1A`. The profile retains a 10 GiB free-space floor and
   one campaign task at a time. Doctor confirms the numerical versions and
   storage, while reporting unavailable executor resource enforcement and absent
   local catalog separately. Those are not scientific results.

4. The original 48-root nomination export succeeded and reports no refitting.
   The original F export also succeeded. Its original payload remains
   `ab880c96a5e4d4e5070e4e2268331ea542f5cc0bedf18da43cded949d2462b1b`.
   The current wrapper binds those exact original bytes.

5. Exposed calibration packet preparation succeeded, but authoring refused its
   exposed units and streams. The public guide incorrectly described that sample
   as an authoring example. The guide now states this stop.

   A separate disjoint proposal passed full-size authoring and integration proof.
   It retained 32 units, 705 tasks and 2,051 outputs. No task ran and no authority
   was created. These proof allocations remain separate from the paper replay.
   The proof seed values are now exposed. Keep them in later exposure censuses.

6. A proposed packet required global `--project-root` and `--operator-profile`
   options. The packet prose omitted this requirement. The guide now states it.

7. The original forecast input helper refused the current donor lock hash.
   Its stored Python and NumPy versions agree with the selected environment.
   The original generic validator registrations also differ from current donor
   registrations. Historical records were read under their original identities.
   The audit checked original digests, receipts, closed member sets and publication
   commit bytes. It did not replace historical validator identities.

8. Original prefix custody includes successful acquisition and continuation
   forwarding receipts. Their scientific payload bytes agree, while their source
   and validator bindings differ. The replay keeps those receipts separate.
   It counts each prefix once. Early harness attempts refused this ambiguity.
   The corrected harness resumed saved current prefixes without reacquisition.

9. Other retained harness failures include an incorrect F option, old request
   purpose spellings, and an unsupported historical schema in a current wrapper.
   The corrections preserve the original records. Opaque transport retains
   original bytes and records the original schema separately.

10. The RC collector completed every native cell and saved recurrence and
    bootstrap records, then failed on a missing depth-14 summary key.
    A separate reducer reads the saved cells and retains all 36 history depths.
    No scientific acquisition is repeated to repair that summary.

11. The first complete portable software gate exhausted its POSIX temporary
    filesystem. It retained 1,707 passes, seven failures and 343 setup errors.
    The failures identify unavailable free space or `ENOSPC`.
    Its failed fixtures are preserved in a verified archive on SEMIOS.

    A memory-backed retry failed on the fixtures' original mount contracts.
    It was interrupted after focused diagnostics confirmed this mismatch.
    The first focused diagnostics also lacked their selected scratch parents.
    Those setup failures remain recorded.

12. The committed release runner retained passed temporary fixtures.
    The candidate runner already removed them under pytest's failed-only policy.
    The committed runner now uses the same policy.
    Failed fixtures remain available.
    The next complete test selection used system-root POSIX scratch and reclaimed space.
    Durable packets, archives and logs remain on SEMIOS.

13. A separate root-space audit identified inactive test fixtures and older release packets.
    The user approved three exact temporary-fixture directories after review of the expected recovery.
    Their SEMIOS archive passed a full comparison before removal.
    The cleanup recovered 4.22 GiB and left 12.15 GiB free at completion.
    Legacy worktrees and historical release packets remain available.
    `root-space-recovery` retains the inventory, approval, backup and completion records.

14. All 2,057 portable tests passed, but the complete release gate then refused stale generated inputs.
    Six current package-source bindings still selected the source before the FAT fix.
    The owner generator refreshed those digests.
    Scientific protocols, numerical allocations, seeds and original paper assets remain unchanged.
    The failed gate remains diagnostic evidence.

15. A separate verification decision avoids another complete local suite for these generated bindings.
    Focused workflow checks, release-runner checks and packaging checks retain their actual source identities.
    GitHub CI retains the complete final-source portable gate on push.
    The release runner now examines static errors, generators and document targets before expensive tests.
    A regression test requires generated-input drift to refuse before the portable suite.
    This change creates no scientific qualification.


16. The original GitHub CI packet retained one failed CLI assertion and twelve RC input errors.
    Its coloured diagnostic split a literal assertion, while exact RC development descriptor hashes differed.
    A local BLAS-dispatch diagnostic reproduced six descriptor mismatches under a different kernel.
    The original Haswell/NumPy `X86_V3` profile preserves all eighteen frozen descriptor hashes.
    CI now selects that profile and verifies the complete exposed descriptor census before expensive tests.
    The CLI test removes colour escapes before its message assertion.

    All 71 focused CI/release checks pass, including coloured-output cases.
    The mismatching numerical profile refuses in approximately half a second.

The CI diagnosis retains the original failed packet and its unchanged numerical references.
The precise dispatch used by that original hosted job was not recorded.
The dispatch explanation is supported by the local diagnostic, with final hosted verification still required.
No reference hash, scientific threshold or original paper result is changed.

## Historical input bindings

Original record names below identify historical provenance.
The original source checkout and original storage roots remained unchanged.

| Experiment | Original source or closure | Original custody |
|---|---|---|
| Numerical RC | `824ad09500c3a16a881477e046303ce4f21c2748` | `icf-yolo/runs/run.ipsmc-v3.evaluation.v9` on SEMIOS |
| Forecast reuse | Acquisition and continuation sources, with final closure `de96bbb6f70246ef0978024a1a76d3264115288c` | `icf-yolo/cc1-finite-lawhood-v1` on SEMIOS |
| Preparation applicability | `d4977137d15c595215b37927792323b05176906f` | `icf-yolo/experiments/cc1-boundary-production-v1` on SEMIOS |

The RC seed roster matches its original pre-outcome commitment.
Each regenerated descriptor matches the original descriptor bytes through the
explicit original numerical recipe. Current dense observers precede separate
sparse generator calls. All 36 blocks and 108 nested scale cells remain assigned.

The forecast closure selects the successful cohort repair receipt from attempt 002.
It retains the failed attempt. The terminal adjudication and 64 root control
seals remain bound by actual original receipts. The audit retains both delivered
and declined consumer slots.

The original 48-root development coefficients remain frozen.
The independent current calibration scorer checks all 32 root maxima, rank 30,
and all 8,192 request pairs for each boundary. It performs no fitting.

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

## Full scientific replays and independent arithmetic

The exposed numerical harness calls current scientific owners directly.
It writes new replay operands and compares them with authenticated originals.
It does not issue a current study or reuse original grants for a new study.
The operational records state `new_scientific_qualification=false`.

The full native cohort harnesses use up to four worker processes.
Each worker uses one numerical thread.
The issued-campaign profile is a separate, unused execution selection.

| Calculation | Reproduction result | Retained analysis directory |
|---|---|---|
| Analytical RC | Exact separation and minimax bound above | `rc-information-002` |
| Numerical RC, current targeter | Primary pair counts, recurrence and bootstrap reproduce | `rc-full-replay-002` |
| Numerical RC, original frozen choices | All 108 scientific adjudications reproduce | `rc-frozen-original-choices` |
| Calibration arithmetic | All four rank-30 q values and complete decision censuses reproduce | `historical-calibration-003` |
| Forecast arithmetic | 61/64 joint successes, zero false admissions, exact original inference | `historical-forecast-003` |
| Preparation native Q8/E32 | Every native numerical field matches the originals | `boundary-native-replay` |
| Calibration and forecast native C32/E64 | Every native numerical field matches the originals | `matrix-native-replay` |

The preparation comparison uses unchanged evolution H, selected negative preparation N and positive preparation P.

The preparation replay completed all 40 independent roots and 1,879,680 native
view intervals. Its 4,640 native phases match the original arrays, innovations,
effort measures and complete clocks exactly. Q8 reproduces 1,990/2,048 joint
successes. All eight roots pass the unchanged constructor crossing check before
E32 begins. E32 reproduces 7,965/8,192 joint successes for N, versus zero for H
and P. All false-admission counts remain zero.

All 32 E32 roots favour N, with exact one-sided `p=2.3283064365386963e-10`.
The both-view mask retains 7,719 successful pairs and 31 complete roots.
The conservative view envelope retains 6,720 successful pairs and 27 complete roots.
Excluded roots count as zero in these diagnostics.
All 8,192 pairs and 32 roots remain in their respective denominators.

The support-bypass diagnostic reproduces 7,916, 7,965 and 7,952 successful pairs for H, N and P.
The outcome-visible oracle reproduces 8,192 successful pairs for each preparation.
These diagnostics establish no prospective replacement policy or admission.

The preparation evidence concerns its fixed near-boundary source population.
The shared nominal waveform remains a source recipe, not an independent unit.
The evidence does not establish new physical capacity or an integrated controller.

The calibration and forecast native replay completed all 96 assigned roots.
Its 1,920 paired tasks retain both numerical views and 2,253,312 native view intervals.
All arrays, innovations, clocks, forces and work measures match authenticated originals exactly.
Exact numerical equality binds these new operands to the independent arithmetic below.
Original frozen coefficients and forecast control seals remain unchanged.

Forecast joint success is 61/64, with one-sided 95% lower bound `0.8832834120966406`.
For zero false-admission roots, the one-sided 95% upper bound is `0.045729702330762456`.
All 125 delivered task assays succeed.
Three tasks are declined before preparation, with no later cancellation.
All 64 roots remain in the success and false-admission denominators.

Response MSE improves by `27.550840477704286%` against cached reuse.
Added use remains unestablished: cached succeeds on 60 roots, composed on 61,
with one discordance and exact one-sided `p=0.5`.

The independent calibration q values are `0.7359961802516977` for lower,
`1.0174400111528847` for composed, `0.8689109739611955` for cached, and
`1.1953051889881836` for direct. These displayed binary-float values do not
replace the exact Decimal values in the original records.

Early RC comparisons treated Decimal spellings `1` and `1.0` as different.
Some adjudication scalars differ at floating-point rounding scale.
The comparison preserves canonical source bytes and checks numerical values
with the frozen `1e-10` agreement tolerance.

RC targeter repetition also changes some sink-boundary action selections.
The original and current selected scores differ by approximately `1e-16`.
The exact-score sort can change their order or select a different tied action.
Those changes alter some midpoint gates and false-safe/false-hold subcounts.
They are discrete differences, not accepted numerical agreement.
The diagnostic retains original nominations, current nominations and every gate difference.

The complete RC census contains all 36 seed blocks and 108 scale cells.
It reproduces 2,319 nominations, 1,524 response-adverse pairs and 899 decision-adverse pairs.
Every one of the eighteen endpoint comparisons retains twelve opposed seed blocks.
The simultaneous one-sided lower bound is `0.6123148491963758`.
The whole-seed-block bootstrap reproduces the original summaries.

All 108 algebraic rank predictions remain opposed.
All 72 larger-scale effective-rank endpoints remain right censored.

| History depth | Nominations | Response-adverse pairs | Decision-adverse pairs |
|---|---:|---:|---:|
| 0 | 864 | 697 | 432 |
| 1 | 567 | 488 | 279 |
| 2 | 432 | 293 | 144 |
| 4 | 188 | 46 | 44 |
| 8 | 144 | 0 | 0 |
| 9–13 | 124 | 0 | 0 |
| Total | 2,319 | 1,524 | 899 |

Declared depths 16, 31 and 35 remain untargetable under the fixed challenge limits.
Zero nominations at those depths do not establish closure.

The unchanged targeter also retains sixteen discrepant secondary adjudications.
All occur at 256 cells and concern sink-boundary choices at depth zero.
The new false-safe/false-hold action totals are 1,892/825, versus the original 1,891/826.
Three cells have future-metric differences above the cross-run `1e-10` comparison tolerance.
Among adjudication scalars, the largest absolute difference is `3.53679708e-9`.
All current dense-observer/sparse-generator agreement checks still pass for their own nominations.

The complete diagnostic retains 26 target-selection differences and 29 gate differences.
All gate differences concern `midpoint_sink_pass`.
The diagnostic binds each changed nomination and midpoint gate.
No threshold or original record is changed to remove a difference.

A second replay uses the original frozen choices for all 108 cells.
The current sparse generator consumes explicit numerical exports of the original sealed nominations.
All scientific adjudication fields reproduce within the original `1e-10` agreement tolerance.
The maximum absolute numerical difference is `1.69864e-14`.
The false-safe and false-hold totals reproduce the original 1,891 and 826.

This replay isolates targeter selection sensitivity.
It preserves the separate current-targeter differences and their complete diagnostic.
The three earlier frozen-choice checks are reused without further acquisition.

## Software verification

The FAT fix is committed as `82848f64207a621f02110e85e6cc525bf55a3054`.
The corrected portable release runner is committed as
`83ce55aeabc059b5a859ed034565e3240b46e585`.
Its package source bytes remain identical to the complete numerical replay source.

The failed packets are `analyses/release-82848f`, `analyses/release-82848f-002` and `analyses/release-83ce55a`.
All 45 focused release and mount-contract repair checks passed.
Their passed temporary fixtures were removed as configured.

The complete portable test selection at `83ce55a` passed all 2,057 tests.
Its 48 native or held tests were deselected, with no selected skip.
The test run took 9,451.53 seconds under branch coverage.
The overall release gate failed at the generated integration-input check.
This packet is not a passing release gate.

The declared coverage inventory verified executed statements in all fourteen required owners.
Its measured parent-process coverage is 66%.
That measurement does not establish repository-wide coverage or scientific qualification.

Commit `66a59d4c` refreshes the six generated current package-source digests through their owner.
Commit `ab3fc5ea` moves static, generator and document checks before the expensive test selection.
All 51 affected source/history/tangent workflow checks passed.
All 60 affected release-runner and coverage-contract checks passed.
The scientific package bytes remain unchanged since the complete numerical replay.

Fresh Python environments and temporary fixtures use the selected POSIX environment root in `/tmp`.
Durable artifacts and logs remain on SEMIOS.

The final scientific custody check verifies 28,339 unique retained files.
It verifies their complete comparison censuses and raw native payload bindings without new acquisition.
The final local software packet passed all seventeen selected checks at `ab3fc5ea`.
It builds the wheel through a new source archive and compares every package member with that source.
A fresh hash-pinned wheel environment passes all 149 CLI leaf-help checks outside the checkout.
Installed checks also cover the reactor example, partial retention, existing-output refusal and original operand delivery.
These results retain their software and exposed-development limits.

The final local software packet is `analyses/final-focused-software-verification`.
It records focused repairs and current-source packaging separately from the earlier complete test selection.
It does not claim a complete portable test run on the final tree.
GitHub CI retains that final-tree selection on the authorized push.
The final prose update has separate document-check and review records.
`control/final-verification-decision.json` binds this verification scope.

## Read the retained packet

Start with `closeout/reproduction-summary.json` under the selected artifact root.
Use `closeout/command-attempt-index.json` to inspect successful and failed attempts.
Each listed command retains its exact argument vector, producing commit and consumed environment.
Its completion record binds the observed exit status and log bytes.

The analysis directories above retain the complete assigned numerical cohorts.
Their input censuses bind the original publications, current operands and independent readouts.
`analyses/final-scientific-custody-check/verification.json` verifies those retained file bindings.
Replay harness sources and command-specific source snapshots remain in external custody.

## References and research

The [paper source index](../paper/SOURCES.md) binds the original result assets.
The [reproducibility guide](../REPRODUCIBILITY.md) states archive and qualification
limits. The [finite workflow guide](../experiments/finite-response-law/guide.md)
describes current study inputs. The [release procedure](release-artifacts.md)
defines the portable software gate.
