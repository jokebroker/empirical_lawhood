# Add a native adapter or method

SPDX-License-Identifier: CC-BY-4.0

This guide gives the procedure for a native adapter or scientific method extension.
It relates one declared scientific contract to records, providers and the maintained application route.
Use it to locate the owning seams and required existing checks.

## Prerequisites and first action

Use the [tested development environment](environments.md). Read the [interface policy](compatibility.md#four-supported-surfaces). Select one scientific contract and its existing adapter owner. Write the independent unit, preparation, available history, action chart, native units, receiver and horizon.

Keep the observations, views and arms nested within their actual independent units. Declare the outcome cutoff and the result that would refute the claim. Give missing and infinite outcomes an explicit rule.

A simulator adapter delivers and observes a native process.
A method consumes typed evidence and returns a bounded scientific product.
Registration and provider construction read no hidden outcomes and grant no authority.
They launch no native simulation.

The reactor supplies a complete implementation example:

| Part | Source owner |
|---|---|
| Native records and source bridge | [panel.py](../src/empirical_lawhood/adapters/simulators/reactor_prefix_response/panel.py), [assigned.py](../src/empirical_lawhood/adapters/simulators/reactor_prefix_response/assigned.py) |
| Native runner/provider | [provider.py](../src/empirical_lawhood/adapters/simulators/reactor_prefix_response/provider.py) |
| Static scientific descriptor | [Source extension bundle](../src/empirical_lawhood/adapters/simulators/reactor_prefix_response/extension_bundle.py) |
| Executable reconstruction | [Source binding](../src/empirical_lawhood/adapters/simulators/reactor_prefix_response/executable_binding.py) |
| Method custody and payload ports | [Method binding](../src/empirical_lawhood/adapters/methods/reactor_prefix_response/executable_binding.py) |
| Candidate composition | [authoring.py](../src/empirical_lawhood/adapters/composition/reactor_prefix_response/authoring.py) |

## Choose the smallest extension

| Researcher change | Existing path or new owner |
|---|---|
| Change a permitted parameter of the same source contract | Copy the existing input. Validate the copy. Prepare its canonical bytes. Preserve the scientific units, bounds and evidence role. |
| Calculate a new quantity from existing typed evidence | Add the method input and result. Add its evaluator or reducer. Reuse the source and authenticated custody. |
| Deliver or observe a different native process | Add source records and bounded implementation. Supply independently derived falsifiers. Define the complete provider contract. |
| Bind existing operations through a new scientific dependency | Add the family composition and strict input joins. Reuse installed providers, scheduler and custody. Validate parent admissibility. |

Complete the local numerical check before the installed route.
The method-only milestone below uses existing typed panels.
The installed trace identifies the additional seams.
Physical metrology, hidden-source access and genuine issue authority retain their separate prerequisites.

## Worked RC extension

Start with the existing numerical RC contract. Make the local change in a disposable copy outside the repository.
The [RC workflow](../experiments/rc-ladder-response/guide.md#edit-a-development-model) already supports copied-model configuration and retained output.
A new numerical model is not a new physical board or fresh confirmation.

### Milestone A: a typed local result

Trace the current route before adding a new one:

| Handoff | Owning file and requirement |
|---|---|
| Edited JSON → prepared canonical model | [Configuration](configuration.md), then [strict model](../src/empirical_lawhood/adapters/simulators/rc_ladder_response/contracts.py). Positive SI components, topology, action ordering and clock grid validate before contact. |
| Exact model → numerical diagnostic | [native_quickstart.py](../src/empirical_lawhood/adapters/simulators/rc_ladder_response/native_quickstart.py). Bounded 64 KiB read, strict decoding and two solver views. |
| Strict study → typed native panel | [campaign.py](../src/empirical_lawhood/adapters/simulators/rc_ladder_response/campaign.py), with [matrix exponential](../src/empirical_lawhood/adapters/simulators/rc_ladder_response/matrix_exponential.py) and [backward Euler](../src/empirical_lawhood/adapters/simulators/rc_ladder_response/refinement.py). |
| Typed panels → numerical falsifier | `check_native_views`, plus [method config](../src/empirical_lawhood/adapters/methods/rc_ladder_numerical_comparison/contracts.py). Keep one independent unit and two nested views. |
| CLI report → operational retention | [Results and failures](results-and-failures.md). Optional export changes custody of the attempt, not the scientific ceiling. |

A small worked change replaces the four-cell model with a one-cell numerical preparation.
Save this module as `local_rc_extension.py` in the disposable directory:

<!-- BEGIN WORKED RC PYTHON -->
```python
from decimal import Decimal

from empirical_lawhood.adapters.simulators.rc_ladder_response.contracts import (
    ResistorCapacitorLadderActionSegment,
    ResistorCapacitorLadderModelConfig,
    ResistorCapacitorLadderStudyConfig,
)
from empirical_lawhood.adapters.simulators.rc_ladder_response.campaign import (
    VIEWS, run_native_view,
)


def local_study():
    model = ResistorCapacitorLadderModelConfig(
        config_id="rc-worked-one-cell",
        scale_cells=1,
        capacitances_farads=(Decimal("1"),),
        interior_resistances_ohms=(),
        left_source_resistance_ohms=Decimal("1"),
        right_termination_resistance_ohms=Decimal("1"),
        initial_voltages_volts=(Decimal("0"),),
        action_segments=(ResistorCapacitorLadderActionSegment(
            "one-volt-step", Decimal("0"), Decimal("1"),
            Decimal("1"), Decimal("0"),
        ),),
        output_times_seconds=(Decimal("0"), Decimal("0.5"), Decimal("1")),
        component_metrology_frozen_before_response=True,
    )
    return ResistorCapacitorLadderStudyConfig(
        "rc-worked-study", "rc-worked-exposed-unit", model,
        Decimal("0.001"), Decimal("0.005"), Decimal("1"),
    )


def local_panels(study):
    return tuple(run_native_view(study, view) for view in VIEWS)
```
<!-- END WORKED RC PYTHON -->

All constants above are disclosed development choices. The coarse-grid charge-residual limit is deliberately explicit.
Declare voltage in V, capacitance in F, resistance in Ω and time in s.
The action is the finite-source one-volt step. The receiver is the capacitor voltage.
The independent preparation is the numerical model, not its two solvers or three clocks.

Use the independent circuit equation `dV/dt = 1 - 2V`, with `V(0)=0`.
It gives `V(t)=(1-exp(-2t))/2` and DC voltage 0.5 V.
The [worked check](../tests/test_worked_rc_extension.py) imports the copied module, reconstructs its canonical study/panels,
and compares the actual matrix-exponential output to this equation. It checks Euler error against the same independent relation.
Zero capacitance refuses before a solver. A tighter frozen voltage tolerance produces a negative numerical result.
Do not use an output from the solver as its own scientific oracle.

```sh
uv run --no-sync pytest -q --fail-on-skip tests/test_worked_rc_extension.py
```

This completes a local numerical boundary only. It supplies no real authority, sealed cohort or physical qualification.
The [existing DC/current tests](../tests/test_resistor_capacitor_study.py) and
[analytic/source-impedance checks](../tests/test_resistor_capacitor_native_science.py) remain independent verification owners.

### Method-only alternative

Reuse the typed exposed panels rather than inventing a source.
Construct `ResistorCapacitorLadderEvaluationConfig` with its exact study and `numerical_only=True`.
Use `check_native_views` to compare the two panels under the declared tolerances.
Retain its typed `ResistorCapacitorLadderNumericalCheck`, including `converged`, reasons and whole-unit count.
The worked check exercises this branch with native generation disabled during evaluation.
A lower tolerance is a negative result, not a request to rerun a source.
A physical-board claim or a substituted model/unit/grid refuses.
For issued evaluation, the existing [method runner](../src/empirical_lawhood/adapters/methods/rc_ladder_numerical_comparison/provider.py)
still requires sealed typed inputs, authentic dependency receipts and adjudication context.

### Milestone B: provider and issued integration

For a new installed adapter, enumerate the complete diff before claiming an issued route:

1. Scientific records, native implementation and independent falsifiers under the adapter owner.
2. Source `extension_bundle.py` and `executable_binding.py`, with exact provider factory, codec roster and complete ports.
3. Separate method descriptor/binding, input schemas, result contract and evaluator/reducer.
4. Family composition and strict authoring exports, candidate/source/resource/plan projections and bounded external input declarations.
5. API service/composition loaders and the family-specific CLI command/options/metadata, when an installed authoring route is intended.
6. Workflow/config registry and schema projections, guide/result fields and honest first missing prerequisite.
7. Provider selection, authority refusals, output semantics, receipt recovery and independent scientific checks.

### Trace the installed RC route

Use the numerical RC ladder as the installed template after milestone A.
The [family guide](../experiments/rc-ladder-response/guide.md) gives the actual `circuit-author` and `check-readiness` commands.
Its scientific unit is one declared model preparation.
The two solver views and receiver clocks are nested observations.

| Seam | Exact owner, exported contract and check |
|---|---|
| Source records and scientific clock | [contracts.py](../src/empirical_lawhood/adapters/simulators/rc_ladder_response/contracts.py) owns `ResistorCapacitorLadderModelConfig`, `ResistorCapacitorLadderStudyConfig` and `ResistorCapacitorLadderNativePanel`. Components use SI units. Preparation, ordered action segments and output times validate before calculation. |
| Bounded native implementation and falsifier | [campaign.py](../src/empirical_lawhood/adapters/simulators/rc_ladder_response/campaign.py) supplies `run_native_view` and `check_native_views`. These use the declared matrix-exponential and backward-Euler views. [Independent circuit checks](../tests/test_resistor_capacitor_native_science.py) test the analytic one-cell transient, charge balance and source impedance. |
| Static source contribution | [extension_bundle.py](../src/empirical_lawhood/adapters/simulators/rc_ladder_response/extension_bundle.py) exports `EXTENSION_BUNDLE_CONTRIBUTION`. It declares the capability, conformance obligations and evidence ceiling. Registration supplies no native result. |
| Executable source and codec roster | The [source binding](../src/empirical_lawhood/adapters/simulators/rc_ladder_response/executable_binding.py) exports `EXECUTABLE_BINDING_CONTRIBUTION` and `EXECUTABLE_RECORD_TYPES=(ResistorCapacitorLadderStudyConfig,)`. `EXECUTABLE_BINDING_FACTORIES` contains `ResistorCapacitorLadderNativeFactory`. Its complete platform-port set is empty. [runtime_provider.py](../src/empirical_lawhood/adapters/simulators/rc_ladder_response/runtime_provider.py) supplies two exact native runners and bounded study/view inputs. |
| Method input, result and evaluator | The [method contracts](../src/empirical_lawhood/adapters/methods/rc_ladder_numerical_comparison/contracts.py) own `ResistorCapacitorLadderEvaluationConfig`. The [provider](../src/empirical_lawhood/adapters/methods/rc_ladder_numerical_comparison/provider.py) consumes both sealed panels and that config. It returns `ResistorCapacitorLadderNumericalCheck` and `ScientificAdjudicationRecord`. The [binding](../src/empirical_lawhood/adapters/methods/rc_ladder_numerical_comparison/executable_binding.py) exports `ResistorCapacitorLadderNumericalFactory`, the evaluator codec roster and an empty platform-port set. |
| Scientific composition and candidate | [authoring.py](../src/empirical_lawhood/adapters/composition/rc_ladder_response/authoring.py) owns `build_rc_authoring`, `RCLadderResponseAuthoringBundle` and `ResistorCapacitorCandidateContextProvider`. Both source tasks supply the numerical evaluator. Authoring retains exact payloads, decoder registrations, source, resources and pure plan projections. |
| Application loader and public command | [fresh_authoring.py](../src/empirical_lawhood/adapters/composition/rc_ladder_response/fresh_authoring.py) owns `FreshRCProfile` and `load_fresh_rc_bundle`. [api/composition.py](../src/empirical_lawhood/api/composition.py) binds installed services. `campaign circuit-author` emits the closed directory selected by `--circuit-authoring-dir`. Readiness proves the exact three runners without execution. |
| Public configuration and discovery | [configuration_registry.py](../src/empirical_lawhood/api/configuration_registry.py) owns `rc-ladder-model` and `rc-ladder-study`. [workflows.py](../src/empirical_lawhood/api/workflows.py) owns `rc-ladder-response`. Existing schema, workflow and CLI generators project those facts. Guides and inputs belong to the selected source distribution. |
| Refusal, results and recovery | The [candidate integration check](../tests/test_resistor_capacitor_candidate.py) selects exact source/method providers and external inputs. The [family guide](../experiments/rc-ladder-response/guide.md#recover-an-incomplete-authoring-export) preserves partial authoring. A later attempt uses a fresh path. Issued evaluation requires actual dependency receipts, adjudication context and separate authority. [Receipt-based recovery](results-and-failures.md#issued-status-and-recovery) preserves completed effects. |

Inspect the native panels and their method consumer before copying integration code.
A completed numerical comparison can report `converged=false`.
Retain its reasons and one-unit denominator.
Scientific adjudication separates support from admission.
This numerical route establishes no physical-board result.

After a native report, the next implemented development task is strict study authoring.
No-contact readiness follows authoring.
Actual issue/run/reveal requires the genuine prerequisites in the family guide.

A held-source or physical adapter has a different complete port set.
Its external input contract binds exact schema, media, size, digest, access and custody before contact.
Platform ports supply the declared hardware/source or guarded-store services.
Obtain the applicable calibration and metrology.
Obtain the required grants.
Do not copy the RC zero-port factory into a different source contract.
Keep mounts and keys with the existing platform owners.

The existing RC source accepts exactly its issued study and no external platform ports.
The evaluator accepts exactly its evaluation config and no external platform ports.
[Source factory](../src/empirical_lawhood/adapters/simulators/rc_ladder_response/executable_binding.py) and
[method factory](../src/empirical_lawhood/adapters/methods/rc_ladder_numerical_comparison/executable_binding.py) own those exact sets.
Do not generalize that zero-port contract to physical or held-source adapters.

The [four selectors](architecture.md#four-authoring-directory-selectors) are manual:
reactor, circuit, electron-gas and synthetic-material authoring directories.
They are wired in CLI global inputs, API composition, family loaders and family authoring services.
A fifth descriptor does not create a fifth selector. Adding one requires those owners and matching refusal tests.
Keep existing selectors intact when a new route does not need an installed selector.

Regenerate extension aggregate → executable aggregate → CLI reference, then workflow/schema projections.
Inspect generated diffs. Verify provider construction without contact and existing synthetic authority/recovery fixtures.
Real source qualification, exposure census, physical metrology, issue/approval/execution/reveal authority and fresh cohorts remain separate missing acts.

## Provider and record contracts

Place the extension under its scientific owner in `adapters/simulators/`, `adapters/physical/` or `adapters/methods/`.
Define frozen dataclasses derived from `CanonicalRecord`.
Give each record an explicit `SCHEMA` and public format `VERSION`.
Constructor checks own units, finite values, sorted unique identities, shape and support.
Use `canonical_bytes()` and the registered strict decoder.
Keep pretty-printed JSON separate from canonical identity.

Bind external size, digest, containment and provenance before native contact.
Use the bounded descriptor readers and artifact-plane ports.
After a bounded read, keep its verified bytes for subsequent work.
Do not reopen that path with an unrestricted reader.
Model-bank plan identity, training roots and exposure role remain scientific operands.

Implement the existing [provider](../src/empirical_lawhood/runtime/providers.py) and [execution](../src/empirical_lawhood/runtime/execution.py) protocols.
Bind exact capability and implementation identities.
Declare external inputs, runners and output semantic contracts.
For canonical envelopes, declare the actual schema, `record_version` and required field keys.
Keep each output within its declared per-task ceiling.

Implement the factory's `build_provider(registry=..., records=..., platform_ports=...)` seam. Validate exact record types and revisions. Validate the complete port-key set. Use injected source, custody and payload ports.

Keep mounts, signing keys and external contact explicit. Use the existing scheduler. Compilation and construction remain operations without native effects.

## Register the two contributions

Add `extension_bundle.py` under the adapter's allowed family root.
Export `EXTENSION_BUNDLE_CONTRIBUTION` with capability, runtime, resource, evidence ceilings and conformance obligations.
Add `executable_binding.py` beside that owner.
Export `EXECUTABLE_BINDING_CONTRIBUTION`, `EXECUTABLE_BINDING_FACTORIES` and `EXECUTABLE_RECORD_TYPES`.
The record types define the closed canonical codec roster for this binding.

Regenerate the static aggregate before the executable aggregate:

```sh
uv run --no-sync python scripts/generate_extension_bundle_aggregate.py
uv run --no-sync python scripts/generate_executable_binding_aggregate.py
uv run --no-sync python scripts/generate_cli_reference.py
```

The expected outputs are the two generated composition modules and `docs/cli.md`. Examine their diff. The generators inspect the closed `adapters/{control,methods,physical,simulators}` descriptor roots. A module outside those roots does not become executable through registration.

Keep generated fingerprints under their generator owner. If a CLI verb is required, add its lifecycle and invocation/output/refusal metadata. Use the existing CLI behavior checks for that boundary.

## Compose the application route

Bind strict authoring records, candidate context, platform ports and resource envelope through [api/composition.py](../src/empirical_lawhood/api/composition.py).
Keep compilation, immutable issue, approval, execution and reveal as distinct acts.
Descriptor registration alone does not supply these family-specific bindings.
The [architecture guide](architecture.md#four-authoring-directory-selectors) documents the four manually wired authoring-directory selectors.
Select at most one of those directories for an operation.

Current integration families can use the closed generic `--authoring-dir` handoff instead.
[integration_handoffs.py](../src/empirical_lawhood/api/integration_handoffs.py) supplies existing composition inputs from exact typed records.
Its provider proof validates graph, resources, output census and bounded external-input bytes before native execution.
Use the existing port contracts and create fresh one-use streams for repeated proof or execution.
Do not add another scheduler, custody store or arbitrary module selector.

An exposed analytical operation can instead compose a typed API over existing source and array contracts.
Its adapter owns the mathematical object, cutoffs, complete denominators and independent falsifier.
Its API authenticates inputs and retains exact results under the declared evidence role.
The [history analysis](../experiments/matrix-history-analysis/guide.md) demonstrates this boundary without claiming an issued provider or scientific qualification.
Add its closed configuration and workflow facts through their existing owners.
Descriptor registration is required only when the route supplies the corresponding executable-provider contract.

Other retained families can have different, explicitly injected API/provider composition.
Their descriptor does not imply that a matching installed CLI authoring selector exists.
The [route register](substrate-route-register.md) owns each family's current input and execution boundary.
The [operator walkthrough](reactor-operator.md) gives the maintained checkout recipe for external custody and issued execution.
These recipes are checked for their documented release.
They do not promise stable arbitrary internal imports.

## Verification and honest stops

Use the existing decoder, factory, candidate, authority-substitution and recovery checks for the changed boundary. Choose a scientific falsifier independently of the implementation being checked. A golden output from the same algorithm cannot provide independent scientific evidence. For mechanics, compare mass, energy or actuator transitions with an independently derived relation.

For fitted methods, use a declared truth-known counterexample and whole-root split or permutation checks. Keep failed scientific results as valid results. They must not become retryable operational exceptions.

Use a bounded, nonpromotable integration when the changed route requires execution verification.
Exercise receipt replay and refusal before the relevant effect.
For recovery changes, verify that completed native effects do not repeat.
Reuse existing scientific checks instead of adding duplicate campaign runs.

Optional native distributions and authentic held inputs use separate [test profiles](testing.md).
A selected profile fails when its required prerequisite is absent.
Keep portable independent scientific checks in the portable profile.
Use the focused checks and all generator checks during development.
Before artifact selection, use the complete clean-source [release procedure](../CONTRIBUTING.md#development-and-release-checks).

Missing source qualification, records, ports, grants or authority remain explicit stops.
Compilation, registration and software conformance provide none of those acts.
For the report's meanings, read the [glossary](glossary.md#operational-and-scientific-outcomes).

## References and research

The linked reactor source and method modules provide the worked implementation example.
The [architecture map](architecture.md#application-composition-and-module-responsibilities) identifies affected service and verification owners.
The [scientific integrity guide](scientific-integrity.md) gives the preservation and authority requirements.
The [documentation policy](../CONTRIBUTING.md#documentation-policy) gives the authoring pattern and language rules.
