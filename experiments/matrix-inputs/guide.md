# Current matrix inputs and finite operands

SPDX-License-Identifier: CC-BY-4.0

Use this workflow to acquire explicitly allocated current Q2/CIR1 development arrays, then reuse those saved arrays for mechanical bridge fitting and entry readiness. Discover it with `empirical-lawhood workflow show matrix-inputs`. The base locked NumPy/SciPy environment supplies the native marcher; no extra simulator installation is required.

## Original numerical law

The original F is a fixed numerical operand. Its original payload, manifest, publication commit, calibration and qualification bytes travel together. The current wrapper preserves their identities; exporting it performs no fitting, new qualification or acquisition.

```bash
uv run --no-sync empirical-lawhood campaign original-f-export \
  --output-dir /explicit/new/original-f --payload-id study.original-f
uv run --no-sync empirical-lawhood campaign original-f-check \
  --operand /explicit/new/original-f/original-f.canonical.json \
  --source-directory /explicit/new/original-f
```

`config prepare` refuses this retained operand. Authenticate/import the original bytes instead.

## Select and acquire current sources

Copy `allocation.json` and either `preparation-source.json` or `matrix-source.json` into your working area. Allocation declares each root, native cohort, split role, every numeric purpose and passive-probe stream. The fixed preparation route requires exactly eight Q2 and sixteen CIR1 roots. The generic route accepts an explicitly declared bounded Q2/CIR1 roster; use separate batches for larger banks.

The supplied roots and numerical seeds are permanently exposed development examples. A role or root-name edit does not grant scientific freshness. `campaign matrix-source-config --config-id ID --allocation FILE --output NEWFILE` derives the actual source/code/protocol identities from your edited canonical allocation. Add `--original-f FILE --source-directory DIR` to select the fixed 24-root route. This avoids hand-editing identity hashes. The versioned source selector also binds the current dependency lock. Supply the selected global `--project-root` when preparing inputs for another checkout. Native drift refuses before source writes or acquisition. Validate/prepare copied inputs with consumers `matrix-allocation` and `preparation-source` or `matrix-source`.

```bash
uv run --no-sync empirical-lawhood config validate \
  --config /work/allocation.editable.json --consumer matrix-allocation --format json
uv run --no-sync empirical-lawhood config prepare \
  --config /work/allocation.editable.json --consumer matrix-allocation \
  --output-file /work/allocation.canonical.json --format json
uv run --no-sync empirical-lawhood --project-root /selected/clean/checkout \
  --operator-profile /work/operator-profile.json campaign preparation-source-export \
  --config /work/preparation-source.canonical.json \
  --allocation /work/allocation.canonical.json \
  --original-f /work/original-f/original-f.canonical.json \
  --source-directory /work/original-f --output-dir /external/new/source
```

The destination must be new and inside the selected guarded storage. Acquisition retains the canonical source/allocation before first native contact, then each completed cell. `arrays.canonical.json` is written only after complete acquisition. The independently published `analysis-completion.canonical.json` binds the full source/config/allocation/provenance and native-cell census. Only that authenticated completion makes the bank consumable. An interrupted directory is incomplete: preserve its diagnostics and complete cells; do not invent a manifest or reuse it as a completed bank. Restart explicitly with a new destination. This development export does not supply scheduler retry authority.

## Fit once and reuse

```bash
uv run --no-sync empirical-lawhood --project-root /selected/checkout \
  --operator-profile /work/operator-profile.json campaign preparation-operands-export \
  --manifest /external/new/source/arrays.canonical.json \
  --arrays /external/new/source/arrays.npz \
  --allocation /external/new/source/allocation.canonical.json \
  --original-f /work/original-f/original-f.canonical.json \
  --source-directory /work/original-f --output-dir /external/new/bridge \
  --operand-id study.preparation-operands
```

The import authenticates the complete NPZ member census, dimensions, physical hashes, source identities and purpose allocations before fitting. It produces the mechanical kernels, four outer fold operators and reusable readiness operands. The fit completion binds its actual interpreter, numerical libraries, lock and full producing package inventory. Retained consumers authenticate those bytes without fitting again. Failed/nonfinite cells retain their failure disposition. Fitting saved arrays does not repeat native acquisition, qualify F or mark follow-on work entered.

The [applicability workflow](../preparation-applicability/guide.md) consumes the derived bridge directory for current entry readiness.
The [transient and baseline workflow](../matrix-transient/guide.md) reuses its authenticated arrays and saved radial/all fits for full exploratory P08/P09 reports.
It performs no native reacquisition or new calibration of F.
Use `--attempt-dir` on export commands to retain bounded operational diagnostics.
The scientific/input outputs retain their own identities and development evidence limits.
