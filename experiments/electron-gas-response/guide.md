# Uniform electron-gas transverse-response development

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for the analytic uniform-electron-gas reference and development candidate.
It tests transverse-current calculations without establishing a material response.
Use it to run the reference or construct its candidate without native contact.

The [strict target config](config.json)
defines **one new synthetic analytic development acquisition**, with 20 nested
q/action conditions. It is a locally generable reference for the retained
transverse-response calculations. It is not a new gauge-closed material
measurement or a prospective evaluation roster.

Install the base environment from a clean target checkout, then use the one
installed CLI:

```sh
uv sync --locked --python 3.11.14
uv run --no-sync empirical-lawhood campaign electron-gas-reference-check \
  --config experiments/electron-gas-response/config.json \
  --output-dir /absolute/development/electron-gas-response/attempt-001
```

The config uses the owner's pre-response 3D neutralizing-jellium denominator:
free-electron parabolic dispersion at `r_s = 4` Bohr radii and 300 K. The
dimensionless action is `u = e v_F A_T / E_F`, with `u0 = 5e-7` and the signed
chart `−u0, −u0/2, 0, u0/2, u0`. The four finite wave numbers are
`q/k_F = 0.015625, 0.03125, 0.046875, 0.0625`. The wavevector points along x and the
transverse vector potential along y.

Requested, accepted, applied and receiver
clocks are respectively 0, 1, 2 and 3. The action is reported in `T·m`,
transverse current density in `A/m²` and the SI kernel in `A/(T·m³)`. The
receiver is the signed y-current at the static `ω = 0` acquisition, after
source convergence, followed by a finite-q fit before the q-to-zero limit. The predeclared field ceiling is `1e-3 T`.

The analytic positive-reference parameter is a penetration depth of `1e-7 m`. The slab is `5e-7 m` thick with 101 profile points. The local check constructs `J_T = −K(q) A_T` using `K(q) = [1/(μ₀ λ²)] [1 + 4(q/k_F)²]`. It evaluates the retained signed finite-difference estimator from **realized SI** vector potential and fits the q intercept.

It recovers λ from the independent slab center field ratio. The checked thresholds in the config preserve the owner's
action realization, locality, even remainder, normal cancellation, q
stability, shielding, Ward and numerical-view bounds. The original `base` and
`refined` numerical views are distinct planned checks. This analytic command
does not claim to execute those solvers or confer independent units.

A
nonlinear response, wrong-sign current, failed normal cancellation, Ward
failure, unstable q intercept or missing order preservation can falsify the
corresponding response–admission claim. Measurement source/gauge qualification and order-relation
opportunity remain prerequisites. Controller use is outside this static chart.

The [scientific tests](../../tests/test_uniform_electron_gas_native_science.py) compare the UEG density, Fermi scale and kernel to independent SI formulas.
They validate the finite-q intercept and slab inverse.
Changed **realized** A changes the estimate even when requested/applied A stays fixed.
A wrong-sign current cannot produce a positive penetration depth. They also
exercise nontransverse, clock and field-ceiling refusals. The command reports exactly
one independent unit and 20 nested conditions, with
`campaign_candidate_compiled: false` and `campaign_issued: false`.

For a different local analytic reference, copy the canonical JSON to a new file. Give it new `config_id` and `independent_unit_id` values. Edit the tagged decimals for the bounded denominator, action, q chart and slab. Pass that file with `--config`.

The CLI strictly decodes it and refuses
unsupported frames, clocks, resource bounds and noncanonical bytes before
emitting a result. This path needs no Python edits or guessed digest. No
research data or source authority is inferred from an analytic panel.

The retained truth-known transverse-screen authoring route remains closed over its
predeclared truth-case roster. The target-owned analytic route binds this config to one
selected native provider and an attached finite-q/slab evaluator. From a
clean target commit and an explicit
[operator storage profile](../../configs/operator-storage.example.json)
pointing to your guarded external artifact root, use:

```sh
uv run --no-sync empirical-lawhood --project-root /path/to/clean-target \
  --operator-profile /path/to/operator-profile.json \
  campaign electron-gas-author \
  --config /path/to/clean-target/experiments/electron-gas-response/config.json \
  --experiment-id my-uniform-electron-gas-development-001 \
  --output-dir /your/external-root/artifacts/authoring/my-uniform-electron-gas-development-001

uv run --no-sync empirical-lawhood --project-root /path/to/clean-target \
  --operator-profile /path/to/operator-profile.json \
  --electron-gas-authoring-dir /your/external-root/artifacts/authoring/my-uniform-electron-gas-development-001 \
  campaign check-readiness \
  --spec /your/external-root/artifacts/authoring/my-uniform-electron-gas-development-001/authoring.json \
  --extension-payload /your/external-root/artifacts/authoring/my-uniform-electron-gas-development-001/payload-0.json \
  --extension-payload /your/external-root/artifacts/authoring/my-uniform-electron-gas-development-001/payload-1.json \
  --decoder-registration /your/external-root/artifacts/authoring/my-uniform-electron-gas-development-001/decoder-0.json \
  --decoder-registration /your/external-root/artifacts/authoring/my-uniform-electron-gas-development-001/decoder-1.json \
  --expected-candidate /your/external-root/artifacts/authoring/my-uniform-electron-gas-development-001/candidate.json \
  --source-closure /your/external-root/artifacts/authoring/my-uniform-electron-gas-development-001/source-closure.json \
  --resource-envelope /your/external-root/artifacts/authoring/my-uniform-electron-gas-development-001/resources.json \
  --run my-uniform-electron-gas-development-001 --format json
```

`electron-gas-author` writes strict extension authoring, issued payloads/decoders,
candidate and pure run/execution projections. The generic preissue command
resolves the selected source and evaluator runners, exact config inputs,
output/adjudication locators, resource envelope and future authority gates
without native contact. The clean-checkout proof passed all six closure
sections with two tasks, three external inputs, three output locators and
four separate future authority gates. It reported no source contact, worker,
issue, outcome read or external byte write.

The [candidate test](../../tests/test_uniform_electron_gas_candidate.py)
validates the shipped config at the API seam and falsifies changed realized
action and wrong-sign current. This is an analytic development source with an
exposed public unit and planted positive kernel. It provides no independent
evidence of material order or external gauge closure. Measurement is limited to synthetic reference conformance.
Order relation, response, local law, admission and controller use require separate native evidence.

Before execution, predeclare a strict config with a distinct model/preparation, denominator, action chart and unit ID.
Use the same `--config` route.
A changed ID alone does not make the public deterministic source independent. The [researcher lifecycle guide](../../docs/designing-and-running-an-experiment.md) describes issue, package, plan, execution, recovery and separate reveal.
Each operation requires its exact typed proposer, custody, approval, execution and reveal records.
The shipped public unit cannot support a fresh-unit attestation.

The owner descoped the external gauge-closed material route from first-release on 29
September 2026. It has no target candidate/provider or public material claim.
The source truth-known conformance screen ended `TRANSVERSE_OPERAND_REQUIRED`: both nominated
papers were closure-only and supplied no eligible signed gauge-closed target
panel. Its zero target contact and analytic truth-known cases cannot become
external material evidence. To reopen this route, supply a legally usable source with these records:

- Requested, accepted, applied and realized `A_T` in `T·m`.
- Transverse current density and its error bound in `A/m²`.
- q/k_F and the four clocks.
- Normal cancellation and Ward controls.
- Order parameter and preservation operands.
- Numerical view IDs.
- One complete-acquisition independent-unit ID per prepared realization.
- A source custody receipt. The analytic provider rejects that external panel. It
does not pretend to qualify physical or independently simulated truth.

## Read results and failures

Read `density_m3`, `k_f_m1`, `finite_q_kernel_A_T_m3`, `intercept_stability_relative` and `fitted_penetration_depth_m`.
`independent_units=1` and `nested_q_u_conditions` distinguish the reference from its nested conditions.
`analytic_reference_only` limits the result. Authoring/readiness remains separate from native reference calculation.

Read [the common result and failure guide](../../docs/results-and-failures.md) for retained attempts,
stdout/stderr and same-identity issued recovery. The route-specific outputs and stops below remain binding.
Use `workflow show electron-gas-response --format json` for static command effects and prerequisite roles.
Discovery does not inspect those prerequisites.

## Obtain the workflow inputs

Obtain this bundle from the selected repository checkout or source distribution. The guide and study inputs are repository workflow files, not wheel resources. Run each command from the checkout root in an unactivated shell. The relative paths use that root.

Keep generated data and receipts in the declared external storage. A wheel can perform its advertised inspection and native-input checks with absolute paths. Source-bound authoring and issue require a clean checkout that executes its own tracked package.

## Recover an incomplete authoring export

A later authoring failure reports the incomplete output path and failed stage.
Keep that directory and its inputs for diagnosis. Use a fresh output path for the next attempt.
A partial export is not an issued campaign or a completed authoring handoff.
The command does not resume, replace or overwrite the incomplete directory.
An existing empty output directory remains accepted. An existing nonempty directory refuses.

## References and research

Gareth Seneque (2026), [*Empirical Lawhood*](../../paper/manuscript.md), edition 0.60, gives the scientific formulation.
The [program source register](../../paper/SOURCES.md) identifies historical results and unavailable primary records.
The linked scientific source modules and existing tests establish only their declared implementation and development boundaries.
