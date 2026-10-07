# Analytical RC information loss

This paper v0.60 construction gives two equally weighted RC states the same initial mean voltage and predictor input. Their later receiver voltages differ, so every deterministic prediction from that common input incurs at least half the separation on one state. This is an analytical illustration, with an independent equation check; it has no experimental sample, confidence interval or qualification.

Use the base locked Python environment. No simulator, external scientific store, issue or reveal authority is required. Discover the route with `uv run --no-sync empirical-lawhood workflow show rc-information`.

Copy `experiments/rc-information/input.json` to your chosen work directory. `voltage_scale_volts` may be any positive value. The baseline fixes equal capacitances, time constants 1 and 4 seconds and initial states (1,1)V0 / (0.5,1.5)V0. Changes to states or time constants require `instance_kind: VARIANT`; unequal initial receiver means or predictor-input digests refuse the identical-input premise.

```bash
uv run --no-sync empirical-lawhood config validate --config /absolute/work/input.json --consumer rc-information
uv run --no-sync empirical-lawhood config prepare --config /absolute/work/input.json --consumer rc-information --output-file /absolute/work/prepared.json
uv run --no-sync empirical-lawhood example rc-information --config /absolute/work/prepared.json --output-dir /absolute/work/rc-attempt
```

Skip preparation when the copied input is already canonical. Preparation writes a new file and preserves your input. The operation prints the canonical report; optional `--output-dir` retains its report, input and bounded diagnostics in a fresh directory. It never overwrites a prior attempt.

The exact baseline separation is V0/4(exp(-t/4)-exp(-t)), maximized at 4log(4)/3 seconds. Its maximum is approximately 0.11812V0 and the unavoidable worst-case error is approximately 0.05906V0. The report carries exact expressions alongside numerical approximations, input/result identities and the independent check. Numerical approximations are not certified interval endpoints.

Inspect the retained report before interpreting a failed attempt. Invalid input or preparation requirements stop before calculation. Repair editable operands and choose a new attempt directory; no previous result is replaced and no scientific qualification is created.
