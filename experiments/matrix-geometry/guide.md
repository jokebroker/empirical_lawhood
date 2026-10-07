# Current six-matrix geometry scans

SPDX-License-Identifier: CC-BY-4.0

This development workflow runs the fixed six-member mathematical preset with an
explicit current allocation. It retains negative and invalid trajectories, cell
support, noncompensating member selection and numerical confirmation. Every
product is development-visible and `NON_PROMOTABLE`; numerical prerequisites,
software conformance and recurring-regime evidence are separate results.

## Select inputs and prove the complete census

Use the [locked base environment](../../docs/environments.md). Native entry
checks CPython 3.11.14, NumPy 2.4.6, SciPy 1.17.1, Linux x86-64 and one numerical
thread per worker. The supplied [input.json](input.json) is an exposed allocation.
Renaming its configuration or allocation leaves actual draws unchanged. Prepare
a new complete allocation with an explicit unsigned 256-bit master seed:

```bash
uv run --no-sync empirical-lawhood --project-root /selected/checkout \
  campaign matrix-geometry-prepare --config-id matrix-geometry.current-001 \
  --master-seed 87346 --workers 1 --output /work/current-scan.json
uv run --no-sync empirical-lawhood campaign matrix-geometry-prove \
  --input /work/current-scan.json
```

The example master seed above is public and exposed. Researchers choose their
own allocation and provide every prior exposed full seed through repeatable
`--prior-exposed-seed-sha256`. The input contains the complete explicit roster,
actual scientific source-file digest and selected dependency-lock digest. Those
digests are computed by the current owner; editing an ID cannot repair a stale
source or environment binding. Existing canonical configuration tools use
consumer `matrix-geometry` when manually editing permitted fields.

The proof instantiates the installed six-matrix factory and checks all 17,118
potential cells without executing a native task. Feasibility assigns 9,126 q2
trajectories: six members × 13×13 grid ×three histories ×three seeds. An eligible
selected member assigns 1,296 q3 primary confirmation roots and 36 nested secondary
views. Maximum actual integration work is 12,146,688 steps, within the preset's
15,000,000-step ceiling. Proof reports 4 GiB per worker, declared worker count,
16 KiB per retained rollout,64 MiB terminal bound and checkpoint cadence 256.

## Bounded source conformance and the current prerequisite

Select an explicit [operator storage profile](../../docs/configuration.md) for
every native or retained-result operation. A conformance coordinate has seven
integer entries: mathematical member ordinal, stage ordinal, X grid index,
Y grid index, history ordinal, seed slot and view ordinal. Stages are feasibility,
confirmation and numerical concordance in that order. The supplied first cell
is a declared 1,024-step trajectory; its negative geometry remains a result.

```bash
uv run --no-sync empirical-lawhood --project-root /selected/checkout \
  --operator-profile /work/operator-profile.json campaign matrix-geometry-conformance \
  --input /work/current-scan.json --coordinate 0,0,0,0,0,0,0
uv run --no-sync empirical-lawhood --project-root /selected/checkout \
  --operator-profile /work/operator-profile.json campaign matrix-geometry-qualification \
  --input /work/current-scan.json
uv run --no-sync empirical-lawhood --project-root /selected/checkout \
  --operator-profile /work/operator-profile.json campaign matrix-geometry-bind-qualification \
  --input /work/current-scan.json --qualification-input /work/current-scan.json \
  --output /work/qualified-scan.json
```

The qualification command performs the existing full current numerical checks;
it is separate from the one-cell conformance command. Binding requires its exact
completed receipt, all source-defined checks/canonical points/benchmarks, current
source and producing code. Missing or failed numerical qualification stops the
full scan before native contact. Historical C0 hashes do not qualify new science.

## Execute, read and recover

The following command acquires the full declared development scan. Do this only
when that acquisition is intended; the proof and one-cell command do not run it.

```bash
uv run --no-sync empirical-lawhood --project-root /selected/checkout \
  --operator-profile /work/operator-profile.json campaign matrix-geometry-run \
  --input /work/qualified-scan.json
uv run --no-sync empirical-lawhood --project-root /selected/checkout \
  --operator-profile /work/operator-profile.json campaign matrix-geometry-result \
  --input /work/qualified-scan.json
```

Outputs and exact completion receipts live under the input-bound run ID in the
selected guarded store. The CLI reports the observed producing Git commit;
dirty development code is not represented as a clean source closure. Current
scientific owner bytes, lock identity and actual native-runtime observation are
retained separately. Result reading authenticates the entire requested root/view
census and exact cell receipts, then replays the deterministic scientific reduction
without acquiring trajectories. No candidate is a valid complete negative terminal
and enters no confirmation.

Repeat the run command with the same exact input to reuse completed cells.
Receipt completeness is the recovery boundary. An interrupted native contact or
unreceipted output refuses automatic repetition because its effects are uncertain.
Serial execution retains each completed cell before entering the next one;
parallel interrupted batches can require inspection. No replacement seed,
favourable candidate substitution or retry permission is inferred from a missing
receipt. See [results and failures](../../docs/results-and-failures.md).

An authenticated valid trajectory may supply a separately identified
[selected-event investigation](../selected-events/guide.md), even if this scan's
recurring-regime gate fails. It does not become the paper's nominated instance.
