# Current transient preparation and baseline support

SPDX-License-Identifier: CC-BY-4.0

Discover `matrix-transient` for the full exploratory P08/P09 calculation. Use the base locked NumPy/SciPy environment with `uv run --no-sync`. This analysis reads explicitly supplied current saved operands; it does not acquire trajectories, refit F, or issue scientific qualification.

## Inputs and first stop

Follow the [current matrix input workflow](../matrix-inputs/guide.md) to export original F, acquire exactly eight Q2 and sixteen CIR1 roots, and run `preparation-operands-export` once. Retain both the native directory and derived bridge directory. Each directory carries its canonical manifest, exact NPZ bytes and allocation; the bridge also carries `bridge-input.canonical.json` and `bridge-spec.canonical.json`. Missing native observations or wrong clocks stop fitting. Historical files and public examples cannot stand in for newly acquired inputs.

Copy [input.json](input.json) and [baseline.json](baseline.json) into your work area. Validate and prepare them with consumers `matrix-transient-analysis` and `matrix-baseline-analysis`. `summary_population` selects the all-root, Q2 or CIR1 summary while preserving every root and fit. `model` selects the radial/all or nonrestoring/all signed decomposition. These edits change the report identity and output; no cutoff or fold changes.

```bash
uv run --no-sync empirical-lawhood config validate \
  --config /work/transient.editable.json --consumer matrix-transient-analysis --format json
uv run --no-sync empirical-lawhood config prepare \
  --config /work/transient.editable.json --consumer matrix-transient-analysis \
  --output-file /work/transient.canonical.json --format json
uv run --no-sync empirical-lawhood --project-root /selected/checkout \
  --operator-profile /work/operator-profile.json campaign matrix-transient-analyze \
  --input /work/transient.canonical.json --native-dir /external/current/native \
  --bridge-dir /external/current/bridge --original-f-dir /work/original-f \
  --output /external/new/transient --attempt-dir /work/transient-attempt
```

The fresh output must lie inside the selected guarded root. The complete calculation uses four whole-root outer folds, the fixed radial and nonrestoring kernels, all/early-time/duration/recovery regimes and three within-cohort inner folds. It re-evaluates and reuses all four saved radial/all outer fits, then performs 28 other outer fits and 24 inner fits. All 24 roots and all nested schedules, futures and views remain in the result.

## Interpret and evaluate the retained result

The transfer report compares interface increments, absolute handoffs and responses against current excluded-root endpoint U2, predicted HOLD without correction, matched nonrestoring dynamics and F at the actual handoff. A measured-future-HOLD anchor is an outcome-visible diagnostic only. It never enters a causal forecast or training feature.

The unchanged original F has its original q and source/calibration provenance. Its width at a predicted interface omits upper forecast error. The inner training-only maximum produces exploratory response boxes and feature enclosures; their qualification is `NONE`. Whole-root exclusion alone supplies no new confirmation or calibrated law. A report may show the endpoint map or nonrestoring comparator performing better.

```bash
uv run --no-sync empirical-lawhood config prepare \
  --config /work/baseline.editable.json --consumer matrix-baseline-analysis \
  --output-file /work/baseline.canonical.json --format json
uv run --no-sync empirical-lawhood --project-root /selected/checkout \
  --operator-profile /work/operator-profile.json campaign matrix-baseline-analyze \
  --input /work/baseline.canonical.json --transient-dir /external/new/transient \
  --original-f-dir /work/original-f --output /external/new/baseline \
  --attempt-dir /work/baseline-attempt
uv run --no-sync empirical-lawhood --project-root /selected/checkout \
  --operator-profile /work/operator-profile.json campaign matrix-analysis-result \
  --directory /external/new/baseline --attempt-dir /work/read-attempt
```

P09 authenticates the saved coefficients and complete disjoint masks without fitting. It retains baseline/transient feature errors, all signed cross terms, the lower-at-true-interface response term, both propagated terms and all three response cross terms. Support requires all 24 normalized features finite with maximum absolute value at most six; equality is inside. False-supported, false-unsupported, unsupported cells and actual failure indices remain complete. The historical count of ten is never a condition on new results.

## Results and interrupted output

`analysis-completion.canonical.json` is the independently guarded final completion publication. It binds exact report/configuration/input-manifest/allocation/array bytes without copying the numeric archive. `analysis-inputs.canonical.json` retains the complete selected roles and authenticated upstream identities. Version 2 reports bind actual producing runtime, package-source inventory, selected lock and observed Git state; the supported numerical profile is enforced before calculation. Report metrics count whole stochastic roots; nested cells do not become additional independent samples. `UNEVALUABLE` retains every requested root and missingness, and performs no dependent fits.

The result command authenticates the completion publication, required member census, closed metric roles and saved missingness without repeating source work or fitting. It validates the recorded producing environment rather than observing the reader as a new producer. A missing publication means incomplete output, even if a report exists. Preserve the partial directory and its retained attempt diagnostics, correct the cause, then select a fresh destination. Existing output, traversal, symbolic links and file ancestors refuse before input contact. This development analysis has no issued-effect resume or overwrite route.

The [current operative specification](../../docs/plans/empirical-lawhood.matrix-preparation-analysis.md) identifies the source-preserving baseline and its package resource. Guides and copied inputs belong to a selected checkout/source distribution; the fixed specification and original F resources ship with the wheel. Software tests use explicitly exposed arrays and supply no scientific qualification.
