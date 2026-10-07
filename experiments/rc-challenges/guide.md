# RC history-depth challenges

SPDX-License-Identifier: CC-BY-4.0

Discover `rc-challenges` for the independently authored canary → nomination → development → evaluation phases. This workflow reuses the current candidate compiler, executable bindings, guarded publications, resources, issue/package lifecycle and receipt recovery. It uses the base NumPy/SciPy environment. The separate [analytical RC example](../rc-information/guide.md) explains information loss without sampled confirmation.

## Copy, edit and publish numeric inputs

Copy the phase configs and `source-export.json`. Validate and prepare edited phase inputs with consumer `rc-challenge-phase`, and the source request with `rc-challenge-source`. Keep the actual phase config inside the source request identical to the config selected for authoring. The supplied source request and its fixed seed commitment are exposed examples, not fresh evaluation assignments.

```bash
uv run --no-sync empirical-lawhood config validate \
  --config /work/development.editable.json --consumer rc-challenge-phase --format json
uv run --no-sync empirical-lawhood config prepare \
  --config /work/development.editable.json --consumer rc-challenge-phase \
  --output-file /work/development.canonical.json --format json
uv run --no-sync empirical-lawhood --project-root /selected/clean/checkout \
  --operator-profile /work/operator-profile.json campaign rc-challenge-source-export \
  --config /work/source-export.canonical.json --output /work/new/numeric-input.json
```

A declared prior-source census requires every corresponding `--prior-source` payload in declared order. Publication checks source bytes, original/current descriptor correspondence, numeric units and actually consumed PCG64 operands; a label is not evidence of freshness. Known exposed original descriptors retain their authenticated opaque original fingerprints separately from current computed descriptors. An unavailable original descriptor envelope is never claimed to have been recreated.

## Author and prove one phase

```bash
uv run --no-sync empirical-lawhood --project-root /selected/clean/checkout \
  --operator-profile /work/operator-profile.json campaign rc-challenge-author \
  --config /work/development.canonical.json --numeric-input /work/new/numeric-input.json \
  --prerequisite /work/current-canary-result.json \
  --prerequisite /work/current-nomination-result.json \
  --evaluation-config /work/evaluation.canonical.json \
  --output-dir /external/new/development-authoring
uv run --no-sync empirical-lawhood --project-root /selected/clean/checkout \
  --operator-profile /work/operator-profile.json campaign integration-proof \
  --authoring-dir /external/new/development-authoring
```

Canary needs no numeric input or prerequisite. Nomination uses its current source. Development consumes the exact successful canary and nomination results and fixes the next evaluation design. Evaluation consumes that current design freeze and seed roster. Each phase has its own issued study. The proof reconstructs installed factories, the full graph, outputs, adjudication and derived resources while executing zero tasks. Missing inputs fail before scientific/native execution.

## Issue, execute, inspect and recover

Select this handoff with the global `--authoring-dir` for existing `campaign check-readiness`, issue/package operations, `campaign run`, `campaign status` and `campaign resume`. Follow the [shared lifecycle](../../docs/integrations.md) and generated CLI reference. A current issue and the applicable current execution/reveal authorities remain mandatory.

Export a selected result with `campaign rc-challenge-result`: supply exact `--run-id`, `--task-id`, `--receipt-id`, `--output-artifact-id`, a new `--result-id` and `--output`. For protected outcomes also supply the actual `--issued-study-id`, `--execution-authority-id` and `--reveal-authority-id`. The command loads those persisted objects, authenticates complete sibling publications and returns the scientific disposition without reacquisition. The output is the receipt-bound prerequisite selected by a dependent phase.

Operational success, complete output or passing software tests do not establish recurrence. Reports keep opposed/evaluable/requested counts, numerical stops and the full assigned-unit census. Recovery selects actual durable receipts; it does not change scientific retry permission or select an arbitrary latest attempt. Original paper evidence remains unchanged.
