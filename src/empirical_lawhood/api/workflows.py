"""Closed public workflow facts. Discovery neither opens inputs nor builds providers.

SPDX-License-Identifier: MPL-2.0
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkflowInput:
    path: str
    role: str
    schema_id: str


@dataclass(frozen=True, slots=True)
class WorkflowOperation:
    command: str
    boundary: str
    inputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Workflow:
    workflow_id: str
    title: str
    kind: str
    substrates: tuple[str, ...]
    methods: tuple[str, ...]
    environment: str
    operations: tuple[WorkflowOperation, ...]
    inputs: tuple[WorkflowInput, ...]
    input_availability: str
    scientific_ceiling: str
    first_missing_prerequisite: str
    guide: str
    verification_owners: tuple[str, ...]
    output_retention: str = (
        "See each command's current effects and guide for explicit development export "
        "or closed authoring outputs. Discovery writes nothing."
    )
    resources: str = (
        "Guide and input paths belong to the selected source checkout/distribution, "
        "not the installed wheel. No current directory is assumed."
    )


WORKFLOWS = (
    Workflow(workflow_id='matrix-geometry', title='Six-matrix geometry scan', kind='fresh explicit native development scan',
        substrates=("six-matrix stochastic response",), methods=('Six-matrix geometry scan',), environment="base",
        operations=(
            WorkflowOperation("campaign matrix-geometry-prepare", "explicit complete numerical allocation", ()),
            WorkflowOperation("campaign matrix-geometry-prove", "explicit current input/result contract", ('experiments/matrix-geometry/input.json',)),
            WorkflowOperation("campaign matrix-geometry-qualification", "explicit current input/result contract", ('experiments/matrix-geometry/input.json',)),
            WorkflowOperation("campaign matrix-geometry-bind-qualification", "exact numerical prerequisite binding", ()),
            WorkflowOperation("campaign matrix-geometry-conformance", "explicit current input/result contract", ('experiments/matrix-geometry/input.json',)),
            WorkflowOperation("campaign matrix-geometry-run", "explicit current input/result contract", ('experiments/matrix-geometry/input.json',)),
            WorkflowOperation("campaign matrix-geometry-result", "explicit current input/result contract", ('experiments/matrix-geometry/input.json',)),
        ), inputs=(
            WorkflowInput('experiments/matrix-geometry/input.json', 'editable complete source/allocation/resource declaration', 'empirical-lawhood/matrix-geometry/scan-input'),
        ), input_availability='Complete source-preserving baseline and exposed explicit allocation supplied; actual cells are acquired separately.', scientific_ceiling='Development scan and numerical prerequisites; negative selection is a valid terminal; no issued grant.',
        first_missing_prerequisite='Select guarded storage, actual source/lock and sufficient declared resources before native cells.', guide='experiments/matrix-geometry/guide.md', verification_owners=('tests/test_matrix_geometry_integration.py',)),
    Workflow(workflow_id='selected-events', title='Selected conditional events', kind='authenticated nominated or alternate parent and new conditional branches',
        substrates=("six-matrix stochastic response",), methods=('Selected conditional events',), environment="base",
        operations=(
            WorkflowOperation("campaign selected-parent-reconstruct", "explicit current input/result contract", ('experiments/selected-events/parent.json',)),
            WorkflowOperation("campaign selected-parent-alternate", "explicit current input/result contract", ('experiments/selected-events/parent.json',)),
            WorkflowOperation("campaign selected-events-prepare", "explicit current input/result contract", ('experiments/selected-events/parent.json',)),
            WorkflowOperation("campaign selected-events-prove", "explicit current input/result contract", ('experiments/selected-events/parent.json',)),
            WorkflowOperation("campaign selected-events-conformance", "explicit current input/result contract", ('experiments/selected-events/parent.json',)),
            WorkflowOperation("campaign selected-events-run", "explicit current input/result contract", ('experiments/selected-events/parent.json',)),
            WorkflowOperation("campaign selected-events-result", "explicit current input/result contract", ('experiments/selected-events/parent.json',)),
        ), inputs=(
            WorkflowInput('experiments/selected-events/parent.json', 'editable fixed nominated-parent reconstruction request', 'empirical-lawhood/selected-events/parent-request'),
        ), input_availability='Nominated reconstruction request supplied; event input is prepared from its authenticated parent and a complete current null-reference scan.', scientific_ceiling='Development conditional experiment; checkpoint futures are nested under one parent, not independent parents.',
        first_missing_prerequisite='Reconstruct nominated parent, or select an explicit current alternate; acquire a complete P03 null reference with no recurrence-success gate.', guide='experiments/selected-events/guide.md', verification_owners=('tests/test_matrix_geometry_integration.py',)),
    Workflow(workflow_id='matrix-history-analysis', title='Algebra persistence and passive prediction', kind='current history acquisition or exposed saved-array analysis',
        substrates=("six-matrix stochastic response",), methods=('Algebra persistence and passive prediction',), environment="base",
        operations=(
            WorkflowOperation("campaign matrix-history-allocation", "explicit current input/result contract", ('experiments/matrix-history-analysis/allocation.json', 'experiments/matrix-history-analysis/source.json', 'experiments/matrix-history-analysis/algebra.json', 'experiments/matrix-history-analysis/passive.json')),
            WorkflowOperation("campaign matrix-history-source-config", "explicit current input/result contract", ('experiments/matrix-history-analysis/allocation.json', 'experiments/matrix-history-analysis/source.json', 'experiments/matrix-history-analysis/algebra.json', 'experiments/matrix-history-analysis/passive.json')),
            WorkflowOperation("campaign matrix-history-source-export", "explicit current input/result contract", ('experiments/matrix-history-analysis/allocation.json', 'experiments/matrix-history-analysis/source.json', 'experiments/matrix-history-analysis/algebra.json', 'experiments/matrix-history-analysis/passive.json')),
            WorkflowOperation("campaign matrix-history-import", "explicit current input/result contract", ('experiments/matrix-history-analysis/allocation.json', 'experiments/matrix-history-analysis/source.json', 'experiments/matrix-history-analysis/algebra.json', 'experiments/matrix-history-analysis/passive.json')),
            WorkflowOperation("campaign matrix-history-analyze", "explicit current input/result contract", ('experiments/matrix-history-analysis/allocation.json', 'experiments/matrix-history-analysis/source.json', 'experiments/matrix-history-analysis/algebra.json', 'experiments/matrix-history-analysis/passive.json')),
            WorkflowOperation("campaign matrix-history-result", "explicit current input/result contract", ('experiments/matrix-history-analysis/allocation.json', 'experiments/matrix-history-analysis/source.json', 'experiments/matrix-history-analysis/algebra.json', 'experiments/matrix-history-analysis/passive.json')),
        ), inputs=(
            WorkflowInput('experiments/matrix-history-analysis/allocation.json', 'editable exposed complete256-root allocation', 'empirical-lawhood/matrix-history-analysis/allocation'),
            WorkflowInput('experiments/matrix-history-analysis/source.json', 'editable current source selector', 'empirical-lawhood/matrix-history-analysis/source-config'),
            WorkflowInput('experiments/matrix-history-analysis/algebra.json', 'editable algebra analysis selector', 'empirical-lawhood/matrix-history-analysis/analysis-config'),
            WorkflowInput('experiments/matrix-history-analysis/passive.json', 'editable passive analysis selector', 'empirical-lawhood/matrix-history-analysis/analysis-config'),
        ), input_availability='Supplied source/allocation/analysis inputs; acquire each history or explicitly import exposed arrays.', scientific_ceiling='Whole stochastic histories are independent; incomplete roots remain UNENTERED/UNEVALUABLE; no inferred native or qualification attestation.',
        first_missing_prerequisite='Select guarded storage and current source/lock; missing history pairs retain the full requested denominator.', guide='experiments/matrix-history-analysis/guide.md', verification_owners=('tests/test_matrix_history_science.py', 'tests/test_matrix_history_public_api.py')),
    Workflow(workflow_id='matrix-tangent', title='Coupled tangent preparation', kind='current declared preparations and retained tangent analysis',
        substrates=("six-matrix stochastic response",), methods=('Coupled tangent preparation',), environment="base",
        operations=(
            WorkflowOperation("campaign matrix-tangent-config", "explicit current input/result contract", ('experiments/matrix-tangent/input.json',)),
            WorkflowOperation("campaign matrix-tangent-source-export", "explicit current input/result contract", ('experiments/matrix-tangent/input.json',)),
            WorkflowOperation("campaign matrix-tangent-analyze", "explicit current input/result contract", ('experiments/matrix-tangent/input.json',)),
            WorkflowOperation("campaign matrix-analysis-result", "explicit current input/result contract", ('experiments/matrix-tangent/input.json',)),
        ), inputs=(
            WorkflowInput('experiments/matrix-tangent/input.json', 'editable full tangent roster and pre-outcome ranks', 'empirical-lawhood/methods/matrix-preparation-analysis/tangent-preparation-config'),
        ), input_availability='Exposed full128-root source/rank recipe supplied; realised preparations are separately acquired.', scientific_ceiling='Outcome-visible exploratory comparison; future and numerical views remain nested whole-root measurements.',
        first_missing_prerequisite='Select guarded storage; analysis needs exact current saved source arrays and manifests.', guide='experiments/matrix-tangent/guide.md', verification_owners=('tests/test_matrix_preparation_analysis.py', 'tests/test_matrix_preparation_differential.py')),
    Workflow(workflow_id='matrix-transient', title='Radial increments and baseline support', kind='retained current preparation transfer and support analysis',
        substrates=("six-matrix stochastic response",), methods=('Radial increments and baseline support',), environment="base",
        operations=(
            WorkflowOperation("campaign preparation-operands-export", "explicit current input/result contract", ('experiments/matrix-transient/input.json', 'experiments/matrix-transient/baseline.json')),
            WorkflowOperation("campaign original-f-export", "explicit current input/result contract", ('experiments/matrix-transient/input.json', 'experiments/matrix-transient/baseline.json')),
            WorkflowOperation("campaign matrix-transient-analyze", "explicit current input/result contract", ('experiments/matrix-transient/input.json', 'experiments/matrix-transient/baseline.json')),
            WorkflowOperation("campaign matrix-baseline-analyze", "explicit current input/result contract", ('experiments/matrix-transient/input.json', 'experiments/matrix-transient/baseline.json')),
            WorkflowOperation("campaign matrix-analysis-result", "explicit current input/result contract", ('experiments/matrix-transient/input.json', 'experiments/matrix-transient/baseline.json')),
        ), inputs=(
            WorkflowInput('experiments/matrix-transient/input.json', 'editable transient population selector', 'empirical-lawhood/methods/matrix-preparation-analysis/transient-analysis-config'),
            WorkflowInput('experiments/matrix-transient/baseline.json', 'editable baseline comparison selector', 'empirical-lawhood/methods/matrix-preparation-analysis/baseline-support-config'),
        ), input_availability='Selectors supplied; use current authenticated native/bridge arrays and installed original numerical F.', scientific_ceiling='Descriptive exploratory transfer; width estimates confer no prospective qualification.',
        first_missing_prerequisite='Acquire current preparation source and export its saved kernels/fold fits; baseline reuses the retained transient result.', guide='experiments/matrix-transient/guide.md', verification_owners=('tests/test_matrix_preparation_analysis.py',)),
    Workflow(workflow_id='preparation-diagnostics', title='Applicability sensitivity diagnostics', kind='authenticated current Q8/E32 retained-result analysis',
        substrates=("six-matrix stochastic response",), methods=('Applicability sensitivity diagnostics',), environment="base",
        operations=(
            WorkflowOperation("campaign preparation-diagnostic-phase-prepare", "explicit current input/result contract", ()),
            WorkflowOperation("campaign preparation-diagnostics-prepare", "explicit current input/result contract", ()),
            WorkflowOperation("campaign preparation-diagnostics", "explicit current input/result contract", ()),
            WorkflowOperation("campaign preparation-diagnostics-result", "explicit current input/result contract", ()),
        ), inputs=(
        ), input_availability='Derive exact Q8/E32 phase inputs from current complete terminal indexes and explicitly selected grants; no fabricated example receipts.', scientific_ceiling='Outcome-visible post-hoc sensitivity; original F and original sealed choices remain unchanged; Q8 is not E32 confirmation.',
        first_missing_prerequisite='Complete current constructed Q/E runs and retain actual issue/execution/reveal authority before protected reads.', guide='experiments/preparation-diagnostics/guide.md', verification_owners=('tests/test_preparation_diagnostics.py',)),
    Workflow(
        workflow_id="rc-challenges", title="RC history-depth challenges", kind="current four-phase authoring and issued lifecycle",
        substrates=("multiscale disordered RC ladder",), methods=("effective history rank", "independent generator adjudication", "whole-block recurrence"),
        environment="base",
        operations=(WorkflowOperation("campaign rc-challenge-source-export", "numeric source/export custody", ("experiments/rc-challenges/source-export.json",)),
                    WorkflowOperation("campaign rc-challenge-author", "current phase candidate authoring", tuple("experiments/rc-challenges/"+phase+".json" for phase in ("canary", "nomination", "development", "evaluation"))),
                    WorkflowOperation("campaign integration-proof", "no-contact full graph/provider proof", ()),
                    WorkflowOperation("campaign check-readiness", "existing preissue integrity checks", ()),
                    WorkflowOperation("campaign run", "issued guarded execution", ()),
                    WorkflowOperation("campaign rc-challenge-result", "exact current scientific result and prerequisite selection", ()),
                    WorkflowOperation("campaign status", "receipt state inspection", ()),
                    WorkflowOperation("campaign resume", "receipt-based bounded recovery", ())),
        inputs=tuple(WorkflowInput("experiments/rc-challenges/"+phase+".json", "editable fixed phase design", "empirical-lawhood/simulator-morphism-challenges/simulator-morphism-challenge-config") for phase in ("canary", "nomination", "development", "evaluation")) + (WorkflowInput("experiments/rc-challenges/source-export.json", "exposed editable source-export example", "empirical-lawhood/simulator-morphism-challenges/source-export-config"),),
        input_availability="Supplied phase configs and exposed development source request; current dependent phases require explicitly selected upstream receipts.",
        scientific_ceiling="Each phase is separately authored and issued; software proof is not recurrence evidence or new qualification.",
        first_missing_prerequisite="Select current guarded storage and a clean source commit; dependent phases require exact current source/export and upstream custody.",
        guide="experiments/rc-challenges/guide.md",
        verification_owners=("tests/test_rc_challenge_integration.py",),
    ),
    Workflow(
        workflow_id="matrix-inputs", title="Current matrix sources and reusable finite operands", kind="explicit native development source / saved-operand analysis",
        substrates=("six-matrix Q2/CIR1 preparations",), methods=("bounded current source transport", "mechanical transient bridge", "fixed original numerical F"), environment="base",
        operations=(WorkflowOperation("campaign matrix-source-config", "derive source identities from edited allocations", ()),
                    WorkflowOperation("campaign original-f-export", "fixed operand delivery with provenance", ()),
                    WorkflowOperation("campaign original-f-check", "bounded authentication", ()),
                    WorkflowOperation("campaign matrix-source-export", "declared bounded native roster", ("experiments/matrix-inputs/matrix-source.json", "experiments/matrix-inputs/allocation.json")),
                    WorkflowOperation("campaign preparation-source-export", "fixed24 native development source", ("experiments/matrix-inputs/preparation-source.json", "experiments/matrix-inputs/allocation.json")),
                    WorkflowOperation("campaign preparation-operands-export", "guarded saved-fit operands and actual numerical provenance", ())),
        inputs=(WorkflowInput("experiments/matrix-inputs/allocation.json", "exposed editable root/purpose/split allocation", "empirical-lawhood/kernel/matrix-allocation"),
                WorkflowInput("experiments/matrix-inputs/matrix-source.json", "editable generic source selector", "empirical-lawhood/methods/finite-response-law/matrix-preparation-source-config"),
                WorkflowInput("experiments/matrix-inputs/preparation-source.json", "editable fixed24 source selector", "empirical-lawhood/methods/finite-response-law/current-preparation-source-config")),
        input_availability="Supplied exposed source/allocation examples and installed exact original-F provenance; realised arrays must be explicitly acquired or authenticated.",
        scientific_ceiling="Development/transfer operands, qualification NONE; declared split roles do not grant scientific eligibility.",
        first_missing_prerequisite="Select an explicit guarded storage profile before native contact; saved fits require complete authenticated native arrays.",
        guide="experiments/matrix-inputs/guide.md", verification_owners=("tests/test_matrix_operand_transport.py", "tests/test_current_preparation_source.py", "tests/test_transient_bridge.py"),
    ),
    Workflow(
        workflow_id="preparation-applicability", title="Ordinary screens and constructed preparation applicability", kind="current D/E and Q/E authoring / independent diagnostic reductions",
        substrates=("ordinary and fixed-nominal six-matrix preparations",), methods=("original-F seven-conjunct validity", "constructor gate", "whole-root service", "retained entry readiness"), environment="base",
        operations=(WorkflowOperation("campaign preparation-allocation-prepare", "expand explicit ordinary/constructed purpose allocations", ()),
                    WorkflowOperation("campaign preparation-conformance", "bounded current native software conformance", ("experiments/preparation-applicability/conformance-allocation.json",)),
                    WorkflowOperation("campaign preparation-exposure-inspect", "inspect actual prior numeric allocations", ()),
                    WorkflowOperation("campaign preparation-input-publish", "outcome-blind fixed input publication", ()),
                    WorkflowOperation("campaign preparation-selection-prepare", "prepare current stage/source from published inputs", ()),
                    WorkflowOperation("campaign preparation-result", "actual current ordinary/constructed scientific result", ()),
                    WorkflowOperation("campaign preparation-q-bind", "actual current qualified result selection", ()),
                    WorkflowOperation("campaign preparation-panel-bind", "exact current ordinary panel selection", ()),
                    WorkflowOperation("campaign preparation-panel-export", "export saved ordinary diagnostic operands", ()),
                    WorkflowOperation("campaign preparation-author", "exact ordinary or constructed current selection", ()),
                    WorkflowOperation("campaign integration-proof", "no-contact full graph/provider proof", ()),
                    WorkflowOperation("campaign run", "issued guarded acquisition and evaluation", ()),
                    WorkflowOperation("campaign preparation-screen", "ordinary96 headroom and retrospective prefix forecasts", ()),
                    WorkflowOperation("campaign preparation-readiness", "guarded saved current24 actual-failure entry screen", ()),
                    WorkflowOperation("campaign status", "receipt and scientific disposition inspection", ()),
                    WorkflowOperation("campaign resume", "receipt-based bounded recovery", ())),
        inputs=(WorkflowInput("experiments/preparation-applicability/conformance-allocation.json", "exposed editable three-source conformance allocation", "empirical-lawhood/kernel/matrix-allocation"),), input_availability="Select current OriginalF/exposure import publications and actual source conformance; E additionally requires exact completed currentQ custody. No synthetic qualification is supplied.",
        scientific_ceiling="Ordinary96/readiness are exposed screens; constructed E requires the fixed newQ gate before prefix contact and evaluates all32 assigned roots.",
        first_missing_prerequisite="Authenticate fixed OriginalF/current source conformance and exposure; missing or failedQ leaves constructedE unentered before native contact.",
        guide="experiments/preparation-applicability/guide.md", verification_owners=("tests/test_preparation_applicability_reducers.py", "tests/test_constructed_preparation_applicability_qualification.py", "tests/test_current_preparation_readiness.py"),
    ),
    Workflow(
        workflow_id="rc-information",
        title="Analytical RC information loss",
        kind="analytical calculation",
        substrates=("two-branch RC circuit",),
        methods=("identical-input minimax bound",),
        environment="base",
        operations=(WorkflowOperation(
            "example rc-information", "analytical calculation",
            ("experiments/rc-information/input.json",),
        ),),
        inputs=(WorkflowInput(
            "experiments/rc-information/input.json", "editable analytical operands",
            "empirical-lawhood/methods/rc-information/config",
        ),),
        input_availability="Supplied canonical baseline; positive voltage scaling is editable.",
        scientific_ceiling="Analytical construction and independent equation check; no sampled qualification.",
        first_missing_prerequisite="Prepare edited operands if their bytes are not canonical.",
        guide="experiments/rc-information/guide.md",
        verification_owners=("tests/test_rc_information.py", "tests/test_rc_information_workflow.py"),
        output_retention="Optional --output-dir retains the input, report and bounded attempt diagnostics.",
    ),
    Workflow(
        workflow_id="reactor-response",
        title="Reactor prefix and retained input routes",
        kind="development / authoring / input preflight / issued lifecycle",
        substrates=("chemical reactor",),
        methods=("paired response",),
        environment="reactor-example",
        operations=(
            WorkflowOperation("example reactor-prefix", "development", ()),
            WorkflowOperation(
                "campaign reactor-author",
                "candidate authoring",
                ("experiments/reactor-response/profile.json",),
            ),
            WorkflowOperation(
                "campaign reactor-preissue-proof", "no-contact proof", ()
            ),
            WorkflowOperation(
                "campaign reactor-batch-input-check",
                "held input preflight",
                ("experiments/reactor-response/batch-input.json",),
            ),
            WorkflowOperation(
                "campaign reactor-prepared-input-check",
                "held input preflight",
                (
                    "experiments/reactor-response/causal-response-study-input.json",
                    "experiments/reactor-response/feed-input.json",
                    "experiments/reactor-response/finite-control-frontier-input.json",
                    "experiments/reactor-response/local-input.json",
                    "experiments/reactor-response/matched-replay-history-input.json",
                    "experiments/reactor-response/regime-input.json",
                    "experiments/reactor-response/selected-action-response-input.json",
                    "experiments/reactor-response/staged-pulse-response-input.json",
                ),
            ),
        ),
        inputs=(
            WorkflowInput(
                "experiments/reactor-response/batch-input.json",
                "held source or parent selector",
                "empirical-lawhood/simulators/reactor-prefix-response/reactor-batch-input",
            ),
            WorkflowInput(
                "experiments/reactor-response/causal-response-study-input.json",
                "held source or parent selector",
                "empirical-lawhood/simulators/reactor-prefix-response/reactor-prepared-input",
            ),
            WorkflowInput(
                "experiments/reactor-response/feed-input.json",
                "held source or parent selector",
                "empirical-lawhood/simulators/reactor-prefix-response/reactor-prepared-input",
            ),
            WorkflowInput(
                "experiments/reactor-response/finite-control-frontier-input.json",
                "held source or parent selector",
                "empirical-lawhood/simulators/reactor-prefix-response/reactor-prepared-input",
            ),
            WorkflowInput(
                "experiments/reactor-response/local-input.json",
                "held source or parent selector",
                "empirical-lawhood/simulators/reactor-prefix-response/reactor-prepared-input",
            ),
            WorkflowInput(
                "experiments/reactor-response/matched-replay-history-input.json",
                "held source or parent selector",
                "empirical-lawhood/simulators/reactor-prefix-response/reactor-prepared-input",
            ),
            WorkflowInput(
                "experiments/reactor-response/profile.json",
                "authoring selector",
                "empirical-lawhood/composition/reactor-prefix-response/fresh-reactor-authoring-profile",
            ),
            WorkflowInput(
                "experiments/reactor-response/regime-input.json",
                "held source or parent selector",
                "empirical-lawhood/simulators/reactor-prefix-response/reactor-prepared-input",
            ),
            WorkflowInput(
                "experiments/reactor-response/selected-action-response-input.json",
                "held source or parent selector",
                "empirical-lawhood/simulators/reactor-prefix-response/reactor-prepared-input",
            ),
            WorkflowInput(
                "experiments/reactor-response/staged-pulse-response-input.json",
                "held source or parent selector",
                "empirical-lawhood/simulators/reactor-prefix-response/reactor-prepared-input",
            ),
            WorkflowInput(
                "experiments/reactor-response/exposed-inputs/exposed-inventory.json",
                "illustrative nonpromotable exposure metadata",
                "empirical-lawhood/examples/exposed-inventory",
            ),
            WorkflowInput(
                "experiments/reactor-response/exposed-inputs/finite-response-prior-exposure.json",
                "illustrative nonpromotable exposure metadata",
                "empirical-lawhood/methods/finite-response-law/native-exposure-metadata",
            ),
            WorkflowInput(
                "experiments/reactor-response/exposed-inputs/prepared-response-prior-exposure.json",
                "illustrative nonpromotable exposure metadata",
                "empirical-lawhood/composition/prepared-response/prepared-exposure-inspection",
            ),
            WorkflowInput(
                "experiments/reactor-response/exposed-inputs/reactor-assigned-exposed.json",
                "retained assigned exposed profile",
                "empirical-lawhood/composition/reactor-prefix-response/assigned-reactor-authoring-profile",
            ),
        ),
        input_availability="Public prefix inputs; authentic held sources and complete census for other routes.",
        scientific_ceiling="Five exposed demo units; held batch census is42 units, not a fresh result.",
        first_missing_prerequisite="Issued work needs clean source, complete exposure census, storage and distinct authorities.",
        guide="experiments/reactor-response/guide.md",
        verification_owners=(
            "tests/test_reactor_native_science.py",
            "tests/test_reactor_held_native_science.py",
            "tests/test_reactor_prepared_input.py",
            "tests/test_reactor_port_store.py",
            "tests/test_reactor_upstream_binding.py",
            "tests/test_reactor_classical_selected_action.py",
            "tests/test_reactor_empirical_synthetic_consumers.py",
            "tests/test_reactor_distinct_transform_boundaries.py",
        ),
    ),
    Workflow(
        workflow_id="prepared-response",
        title="Prepared matrix response",
        kind="development / authoring / parent preflight",
        substrates=("six-matrix medium",),
        methods=("paired prepared response",),
        environment="reactor-example",
        operations=(
            WorkflowOperation(
                "campaign prepared-response-native-check",
                "development",
                ("experiments/prepared-response/prepared-canary.json",),
            ),
            WorkflowOperation(
                "campaign matrix-response-author",
                "candidate authoring",
                (
                    "experiments/prepared-response/prepared-source-qualification-author.json",
                ),
            ),
            WorkflowOperation(
                "campaign dependent-response-input-check",
                "guarded parent preflight",
                (
                    "experiments/prepared-response/prepared-dependent-refinement-input.json",
                    "experiments/prepared-response/prepared-fresh-response-calibration-input.json",
                ),
            ),
        ),
        inputs=(
            WorkflowInput(
                "experiments/prepared-response/prepared-canary.json",
                "native canary",
                "empirical-lawhood/simulators/prepared-response/prepared-response-prepared-canary",
            ),
            WorkflowInput(
                "experiments/prepared-response/prepared-dependent-refinement-input.json",
                "held source or parent selector",
                "empirical-lawhood/composition/prepared-response/prepared-response-dependent-input",
            ),
            WorkflowInput(
                "experiments/prepared-response/prepared-fresh-response-calibration-input.json",
                "held source or parent selector",
                "empirical-lawhood/composition/prepared-response/prepared-response-dependent-input",
            ),
            WorkflowInput(
                "experiments/prepared-response/prepared-source-qualification-author.json",
                "authoring selector",
                "empirical-lawhood/composition/prepared-response/prepared-response-native-authoring-input",
            ),
        ),
        input_availability="Public canary; held plan/design/census and qualified parents are not supplied.",
        scientific_ceiling="One exposed canary; candidate is EXPOSED_DEVELOPMENT_NONPROMOTABLE.",
        first_missing_prerequisite="Dependent stages need authentic qualified parents and separate reveal/analysis authority.",
        guide="experiments/prepared-response/guide.md",
        verification_owners=(
            "tests/test_prepared_response_development_input.py",
            "tests/test_prepared_response_held_source.py",
            "tests/test_response_parent_custody.py",
            "tests/test_response_dependent_binding.py",
        ),
    ),
    Workflow(
        workflow_id="information-response",
        title="Information response library",
        kind="authoring / outcome-visible method",
        substrates=("six-matrix medium",),
        methods=("fitted information response",),
        environment="reactor-example",
        operations=(
            WorkflowOperation(
                "campaign matrix-response-author",
                "candidate authoring",
                ("experiments/information-response/information-author.json",),
            ),
        ),
        inputs=(
            WorkflowInput(
                "experiments/information-response/information-author.json",
                "authoring selector",
                "empirical-lawhood/composition/prepared-response/prepared-response-native-authoring-input",
            ),
        ),
        input_availability="Selector shipped; held source, plan, design, census and matching fitted bank are required.",
        scientific_ceiling="Provider construction and candidate compilation; no 64-root native result.",
        first_missing_prerequisite="No public fitting CLI or fresh fitted-bank custody is supplied.",
        guide="experiments/information-response/guide.md",
        verification_owners=("tests/test_information_prediction_held_source.py",),
    ),
    Workflow(
        workflow_id="causal-response",
        title="Causal response library",
        kind="authoring / outcome-visible method",
        substrates=("six-matrix medium",),
        methods=("fitted causal contrasts",),
        environment="reactor-example",
        operations=(
            WorkflowOperation(
                "campaign matrix-response-author",
                "candidate authoring",
                ("experiments/causal-response/causal-author.json",),
            ),
        ),
        inputs=(
            WorkflowInput(
                "experiments/causal-response/causal-author.json",
                "authoring selector",
                "empirical-lawhood/composition/prepared-response/prepared-response-native-authoring-input",
            ),
        ),
        input_availability="Selector shipped; held source, plan, design, census and matching causal bank are required.",
        scientific_ceiling="Candidate compilation without native tasks; no fresh qualification.",
        first_missing_prerequisite="Matching fitted bank and genuine exposure/custody remain required.",
        guide="experiments/causal-response/guide.md",
        verification_owners=("tests/test_causal_contrasts_held_source.py",),
    ),
    Workflow(
        workflow_id="finite-response-law",
        title="Finite response law current calibration and composition",
        kind="current authoring / issued lifecycle / exposed diagnostics",
        substrates=("six-matrix medium",),
        methods=("finite response law", "response-composition power"),
        environment="base",
        operations=(
            WorkflowOperation("campaign finite-response-allocation", "expand explicit current numeric allocation", ()),
            WorkflowOperation("campaign nomination-export", "fixed nominated coefficients with provenance", ()),
            WorkflowOperation("campaign nomination-check", "fixed nomination authentication", ()),
            WorkflowOperation("campaign finite-response-packet", "prepare current typed stage operands", ()),
            WorkflowOperation("campaign finite-response-parent", "join actual current native and qualification publications", ()),
            WorkflowOperation("campaign finite-response-result", "actual current protected scientific readout", ()),
            WorkflowOperation("campaign finite-response-author", "current calibration/qualification/evaluation candidate authoring", ("experiments/finite-response-law/rerun-calibration.json",)),
            WorkflowOperation("campaign integration-proof", "full-size no-contact graph/provider proof", ()),
            WorkflowOperation("campaign finite-response-bind-runtime", "actual current post-issue authority binding", ()),
            WorkflowOperation("campaign run", "issued guarded execution", ()),
            WorkflowOperation("campaign status", "current receipt/scientific disposition inspection", ()),
            WorkflowOperation("campaign resume", "receipt-based bounded recovery", ()),
            WorkflowOperation(
                "campaign finite-response-native-check",
                "development",
                ("experiments/finite-response-law/finite-canary.json",),
            ),
            WorkflowOperation(
                "campaign response-composition-power", "design diagnostic", ()
            ),
            WorkflowOperation(
                "campaign finite-response-input-check",
                "assignment preflight",
                ("experiments/finite-response-law/finite-calibration-input.json",),
            ),
            WorkflowOperation(
                "campaign finite-response-stage-input-check",
                "guarded stage preflight",
                (
                    "experiments/finite-response-law/calibration-input.json",
                    "experiments/finite-response-law/calibration-stage-input.json",
                    "experiments/finite-response-law/informative-composition-input.json",
                    "experiments/finite-response-law/preparation-screen-input.json",
                    "experiments/finite-response-law/prospective-continuation-input.json",
                    "experiments/finite-response-law/prospective-evaluation-input.json",
                    "experiments/finite-response-law/supplemental-development-input.json",
                ),
            ),
        ),
        inputs=(
            WorkflowInput("experiments/finite-response-law/rerun-calibration.json", "exposed editable current calibration packet", "empirical-lawhood/composition/finite-response-law/rerun-input"),
            WorkflowInput("experiments/finite-response-law/current-calibration-allocation.json", "exposed editable32-root current allocation", "empirical-lawhood/composition/finite-response-law/current-allocation"),
            WorkflowInput("experiments/finite-response-law/current-evaluation-allocation.json", "exposed editable64-root current allocation", "empirical-lawhood/composition/finite-response-law/current-allocation"),
            WorkflowInput("experiments/finite-response-law/current-exposure.json", "authenticated original/public allocation census", "empirical-lawhood/composition/finite-response-law/current-exposure"),
            WorkflowInput(
                "experiments/finite-response-law/calibration-input.json",
                "retained stage selector",
                "empirical-lawhood/composition/finite-response-law/finite-response-law-stage-input",
            ),
            WorkflowInput(
                "experiments/finite-response-law/calibration-stage-input.json",
                "retained stage selector",
                "empirical-lawhood/composition/finite-response-law/finite-response-law-stage-input",
            ),
            WorkflowInput(
                "experiments/finite-response-law/finite-calibration-input.json",
                "held source or parent selector",
                "empirical-lawhood/composition/finite-response-law/finite-response-law-calibration-input",
            ),
            WorkflowInput(
                "experiments/finite-response-law/finite-canary.json",
                "native canary",
                "empirical-lawhood/simulators/finite-response-law/finite-response-law-canary",
            ),
            WorkflowInput(
                "experiments/finite-response-law/informative-composition-input.json",
                "retained stage selector",
                "empirical-lawhood/composition/finite-response-law/finite-response-law-stage-input",
            ),
            WorkflowInput(
                "experiments/finite-response-law/preparation-screen-input.json",
                "retained stage selector",
                "empirical-lawhood/composition/finite-response-law/finite-response-law-stage-input",
            ),
            WorkflowInput(
                "experiments/finite-response-law/prospective-continuation-input.json",
                "retained stage selector",
                "empirical-lawhood/composition/finite-response-law/finite-response-law-stage-input",
            ),
            WorkflowInput(
                "experiments/finite-response-law/prospective-evaluation-input.json",
                "retained stage selector",
                "empirical-lawhood/composition/finite-response-law/finite-response-law-stage-input",
            ),
            WorkflowInput(
                "experiments/finite-response-law/supplemental-development-input.json",
                "retained stage selector",
                "empirical-lawhood/composition/finite-response-law/finite-response-law-stage-input",
            ),
        ),
        input_availability="Supplied exposed current32/64 allocation and fixed nomination examples; authentic current parent publications and grants are selected after execution.",
        scientific_ceiling="Full-size current calibration/qualification/evaluation authoring; public examples remain exposed and software proof is not scientific qualification.",
        first_missing_prerequisite="A clean source commit, guarded storage and current issue/execution/reveal authority; evaluation requires an actual qualified current parent.",
        guide="experiments/finite-response-law/guide.md",
        verification_owners=(
            "tests/test_finite_response_native_quickstart.py",
            "tests/test_finite_response_assignment.py",
            "tests/test_finite_response_assigned_native.py",
            "tests/test_finite_response_assigned_factory.py",
            "tests/test_finite_response_assigned_method_records.py",
            "tests/test_finite_response_assigned_evaluation_binding.py",
            "tests/test_finite_response_native_input.py",
            "tests/test_finite_response_consumer_binding.py",
            "tests/test_finite_response_retained_development_input.py",
            "tests/test_finite_response_stage_input.py",
            "tests/test_finite_response_method_counterexamples.py",
        ),
    ),
    Workflow(
        workflow_id="causal-transfer-audit",
        title="Outcome-visible response composition",
        kind="guarded input preflight",
        substrates=("qualified scalar parent outputs",),
        methods=("post-hoc response composition",),
        environment="locked base",
        operations=(
            WorkflowOperation(
                "campaign response-composition-input-check",
                "guarded parent preflight",
                ("experiments/causal-transfer-audit/analysis-input.json",),
            ),
        ),
        inputs=(
            WorkflowInput(
                "experiments/causal-transfer-audit/analysis-input.json",
                "guarded analysis selector",
                "empirical-lawhood/composition/response-composition/response-composition-input",
            ),
        ),
        input_availability="Analysis selector shipped; authentic qualified parents, custody and grants must be supplied.",
        scientific_ceiling="No independent source or new native tasks.",
        first_missing_prerequisite="Separate reveal and analysis authority before outcome reads.",
        guide="experiments/causal-transfer-audit/guide.md",
        verification_owners=(
            "tests/test_response_parent_input.py",
            "tests/test_causal_transfer_synthetic_development.py",
            "tests/test_causal_transfer_scalar_input.py",
        ),
    ),
    Workflow(
        workflow_id="rc-ladder-response",
        title="Numerical RC ladder",
        kind="development / candidate authoring",
        substrates=("numerical resistor-capacitor ladder",),
        methods=("numerical response conformance",),
        environment="reactor-example",
        operations=(
            WorkflowOperation(
                "campaign rc-ladder-native-check",
                "development",
                ("experiments/rc-ladder-response/model.json",),
            ),
            WorkflowOperation(
                "campaign circuit-author",
                "candidate authoring",
                ("experiments/rc-ladder-response/study.json",),
            ),
            WorkflowOperation("campaign check-readiness", "no-contact proof", ()),
        ),
        inputs=(
            WorkflowInput(
                "experiments/rc-ladder-response/model.json",
                "native model",
                "empirical-lawhood/adapters/simulators/rc-ladder-response/resistor-capacitor-ladder-model-config",
            ),
            WorkflowInput(
                "experiments/rc-ladder-response/study.json",
                "candidate study",
                "empirical-lawhood/simulators/rc-ladder-response/resistor-capacitor-ladder-study-config",
            ),
        ),
        input_availability="Public model and study; physical component metrology is not supplied.",
        scientific_ceiling="Two nested solver views of one synthetic model unit; no physical-board result.",
        first_missing_prerequisite="Fresh physical source/custody and broader circuit integration remain separate.",
        guide="experiments/rc-ladder-response/guide.md",
        verification_owners=(
            "tests/test_resistor_capacitor_candidate.py",
            "tests/test_resistor_capacitor_study.py",
            "tests/test_resistor_capacitor_native_science.py",
        ),
    ),
    Workflow(
        workflow_id="electron-gas-response",
        title="Uniform electron-gas reference",
        kind="reference / candidate authoring",
        substrates=("analytic uniform electron gas",),
        methods=("finite-q transverse response",),
        environment="locked base",
        operations=(
            WorkflowOperation(
                "campaign electron-gas-reference-check",
                "reference",
                ("experiments/electron-gas-response/config.json",),
            ),
            WorkflowOperation(
                "campaign electron-gas-author",
                "candidate authoring",
                ("experiments/electron-gas-response/config.json",),
            ),
            WorkflowOperation("campaign check-readiness", "no-contact proof", ()),
        ),
        inputs=(
            WorkflowInput(
                "experiments/electron-gas-response/config.json",
                "native development or import config",
                "empirical-lawhood/simulators/uniform-electron-gas-response/analytic-reference-config",
            ),
        ),
        input_availability="Public analytic reference config; no qualified physical material source.",
        scientific_ceiling="Synthetic analytic reference and candidate; no physical independent unit.",
        first_missing_prerequisite="Physical material inputs and qualification are separate.",
        guide="experiments/electron-gas-response/guide.md",
        verification_owners=(
            "tests/test_uniform_electron_gas_native_science.py",
            "tests/test_uniform_electron_gas_candidate.py",
        ),
    ),
    Workflow(
        workflow_id="lattice-pairing-method",
        title="Synthetic lattice pairing method",
        kind="method candidate authoring",
        substrates=("synthetic lattice inputs",),
        methods=("disclosed pairing evaluator",),
        environment="locked base",
        operations=(
            WorkflowOperation(
                "campaign synthetic-material-author",
                "candidate authoring",
                ("experiments/lattice-pairing-method/authoring.json",),
            ),
            WorkflowOperation("campaign check-readiness", "no-contact proof", ()),
        ),
        inputs=(
            WorkflowInput(
                "experiments/lattice-pairing-method/authoring.json",
                "authoring selector",
                "empirical-lawhood/simulators/ambient-pressure-superconductor/synthetic-material-response-method-config",
            ),
        ),
        input_availability="Synthetic method config supplied; physical Hamiltonian/position/pairing records are separate.",
        scientific_ceiling="Synthetic producer/evaluator conformance; no superconducting material result.",
        first_missing_prerequisite="Authentic material records and source qualification remain missing.",
        guide="experiments/lattice-pairing-method/guide.md",
        verification_owners=("tests/test_synthetic_material_response_method.py",),
    ),
    Workflow(
        workflow_id="battery-response-and-restart",
        title="Battery response, restart and reduction",
        kind="native development",
        substrates=("PyBaMM battery",),
        methods=("electrothermal response", "restart", "reduced observation"),
        environment="open-simulators",
        operations=(
            WorkflowOperation(
                "campaign battery-native-check",
                "development",
                ("experiments/battery-response-and-restart/config.json",),
            ),
        ),
        inputs=(
            WorkflowInput(
                "experiments/battery-response-and-restart/config.json",
                "native development or import config",
                "empirical-lawhood/simulators/py-ba-mm-development-input",
            ),
        ),
        input_availability="Public development config; optional numerical stack required.",
        scientific_ceiling="Five preparations across three distinct development contracts.",
        first_missing_prerequisite="No general issued battery campaign/provider route follows.",
        guide="experiments/battery-response-and-restart/guide.md",
        verification_owners=(
            "tests/test_battery_native_science.py",
            "tests/test_battery_development_input.py",
        ),
    ),
    Workflow(
        workflow_id="tokamak-heat-response",
        title="Tokamak heat response",
        kind="native development",
        substrates=("TORAX tokamak",),
        methods=("bounded heat response",),
        environment="open-simulators",
        operations=(
            WorkflowOperation(
                "campaign torax-native-check",
                "development",
                ("experiments/tokamak-heat-response/config.json",),
            ),
        ),
        inputs=(
            WorkflowInput(
                "experiments/tokamak-heat-response/config.json",
                "native development or import config",
                "empirical-lawhood/simulators/torax-native/native-torax-quickstart",
            ),
        ),
        input_availability="Public bounded heat config; optional TORAX stack required.",
        scientific_ceiling="One preparation and two nested 40 ms actions; no qualified source.",
        first_missing_prerequisite="Candidate/source qualification is not supplied.",
        guide="experiments/tokamak-heat-response/guide.md",
        verification_owners=("tests/test_torax_native_quickstart.py",),
    ),
    Workflow(
        workflow_id="tokamak-control-response",
        title="Tokamak control response",
        kind="native development",
        substrates=("Gym-TORAX tokamak",),
        methods=("paired selected-action control",),
        environment="open-simulators",
        operations=(
            WorkflowOperation(
                "campaign gym-torax-native-check",
                "development",
                ("experiments/tokamak-control-response/config.json",),
            ),
        ),
        inputs=(
            WorkflowInput(
                "experiments/tokamak-control-response/config.json",
                "native development or import config",
                "empirical-lawhood/simulators/gym-torax-native/gym-torax-native-quickstart",
            ),
        ),
        input_availability="Public quickstart; optional stack and selected source checkout required.",
        scientific_ceiling="Two nested 120-request episodes of one preparation; fixed canary limits.",
        first_missing_prerequisite="Controlled snapshot/provider integration remains unavailable.",
        guide="experiments/tokamak-control-response/guide.md",
        verification_owners=(
            "tests/test_gym_torax_native_quickstart.py",
            "tests/test_gym_torax_native_contract.py",
        ),
    ),
    Workflow(
        workflow_id="neuron-current-response",
        title="Neuron current response",
        kind="native development",
        substrates=("Brian2 LIF neuron",),
        methods=("current/spike response",),
        environment="separate Brian2 locked environment",
        operations=(
            WorkflowOperation(
                "campaign brian2-native-check",
                "development",
                ("experiments/neuron-current-response/config.json",),
            ),
        ),
        inputs=(
            WorkflowInput(
                "experiments/neuron-current-response/config.json",
                "native development or import config",
                "empirical-lawhood/simulators/brian2-neuron-current-response/brian2-lif-native-config",
            ),
        ),
        input_availability="Public config; explicit native Python from the separate locked environment.",
        scientific_ceiling="Two reset arms of one block; no population prevalence or issued candidate.",
        first_missing_prerequisite="Shared native binding/provider integration is absent.",
        guide="experiments/neuron-current-response/guide.md",
        verification_owners=("tests/test_brian2_native_science.py",),
    ),
    Workflow(
        workflow_id="reactor-flow-response",
        title="Flow-reactor response",
        kind="native development",
        substrates=("Cantera flow reactor",),
        methods=("flow/temperature response",),
        environment="reaction-response-simulators",
        operations=(
            WorkflowOperation(
                "campaign reaction-response-native-check",
                "development",
                ("experiments/reactor-flow-response/config.json",),
            ),
        ),
        inputs=(
            WorkflowInput(
                "experiments/reactor-flow-response/config.json",
                "native development or import config",
                "empirical-lawhood/reaction-diffusion-development/reaction-diffusion-development-input",
            ),
        ),
        input_availability="Public Cantera config; optional stack required.",
        scientific_ceiling="One complete development unit with 32 nested cells.",
        first_missing_prerequisite="No selected shared provider or fresh source qualification.",
        guide="experiments/reactor-flow-response/guide.md",
        verification_owners=("tests/test_independent_substrate_native_contracts.py",),
    ),
    Workflow(
        workflow_id="reaction-diffusion-response",
        title="Reaction-diffusion response",
        kind="native development",
        substrates=("FiPy reaction-diffusion field",),
        methods=("reaction-diffusion response",),
        environment="reaction-response-simulators",
        operations=(
            WorkflowOperation(
                "campaign reaction-response-native-check",
                "development",
                ("experiments/reaction-diffusion-response/config.json",),
            ),
        ),
        inputs=(
            WorkflowInput(
                "experiments/reaction-diffusion-response/config.json",
                "native development or import config",
                "empirical-lawhood/reaction-diffusion-development/reaction-diffusion-development-input",
            ),
        ),
        input_availability="Public FiPy config; optional stack required.",
        scientific_ceiling="One development unit with nested native cells/views.",
        first_missing_prerequisite="No selected shared provider or fresh source qualification.",
        guide="experiments/reaction-diffusion-response/guide.md",
        verification_owners=("tests/test_independent_substrate_native_contracts.py",),
    ),
    Workflow(
        workflow_id="grid-response-inputs",
        title="Offline grid response",
        kind="held native development",
        substrates=("Grid2Op electrical grid",),
        methods=("bounded grid action response",),
        environment="grid2op-held",
        operations=(
            WorkflowOperation(
                "campaign grid2op-native-check",
                "development with held input",
                ("experiments/grid-response-inputs/config.json",),
            ),
        ),
        inputs=(
            WorkflowInput(
                "experiments/grid-response-inputs/config.json",
                "native development or import config",
                "empirical-lawhood/simulators/grid2op-response/grid2-op-development-input",
            ),
        ),
        input_availability="Public selector; researcher must supply the declared offline chronic/source.",
        scientific_ceiling="One independent grid unit; no download, candidate or qualification.",
        first_missing_prerequisite="Held source/path bindings must pass before native contact.",
        guide="experiments/grid-response-inputs/guide.md",
        verification_owners=("tests/test_grid2op_native_quickstart.py",),
    ),
    Workflow(
        workflow_id="material-control-inputs",
        title="Material-control workflow inputs",
        kind="raw wrapping / input preflight",
        substrates=("five material control structures",),
        methods=("bounded solver-output inspection",),
        environment="locked base plus declared HDF5 tooling",
        operations=(
            WorkflowOperation(
                "campaign material-workflow-wrap-raw",
                "raw wrapping",
                ("experiments/material-control-inputs/control-inputs.json",),
            ),
            WorkflowOperation(
                "campaign material-workflow-input-check",
                "held input preflight",
                ("experiments/material-control-inputs/control-inputs.json",),
            ),
        ),
        inputs=(
            WorkflowInput(
                "experiments/material-control-inputs/control-inputs.json",
                "raw workflow manifest",
                "empirical-lawhood/simulators/ambient-pressure-superconductor/material-control-input-preflight",
            ),
        ),
        input_availability="Control manifest shipped; ten authentic raw solver-output envelopes must be supplied.",
        scientific_ceiling="Five structures and ten nested views; no solver execution or physical validity certification.",
        first_missing_prerequisite="All raw members, exact profiles and custody remain researcher supplied.",
        guide="experiments/material-control-inputs/guide.md",
        verification_owners=(
            "tests/test_material_epw_coupling.py",
            "tests/test_material_control_input_preflight.py",
            "tests/test_material_mgb2_authentic_reference.py",
        ),
    ),
    Workflow(
        workflow_id="laser-archive-inspection",
        title="Laser archive inspection",
        kind="retrospective import / preview",
        substrates=("published laser burst archive",),
        methods=("bounded logged-burst inspection",),
        environment="locked base",
        operations=(
            WorkflowOperation(
                "campaign glenn-import-check",
                "retrospective import",
                ("experiments/laser-archive-inspection/config.json",),
            ),
        ),
        inputs=(
            WorkflowInput(
                "experiments/laser-archive-inspection/config.json",
                "native development or import config",
                "empirical-lawhood/physical/glenn/glenn-input-quickstart",
            ),
        ),
        input_availability="Public pinned selection; full import requires the declared archive and burst table.",
        scientific_ceiling="Retrospective local inspection; no target guarded publication.",
        first_missing_prerequisite="Preview does not supply archive bytes or prospective qualification.",
        guide="experiments/laser-archive-inspection/guide.md",
        verification_owners=(
            "tests/test_glenn_authentic_source.py",
            "tests/test_glenn_quickstart.py",
        ),
    ),
    Workflow(
        workflow_id="response-method-reference",
        title="Physical scale-morphism reference",
        kind="truth-known method reference",
        substrates=("generated circuit reference worlds",),
        methods=("scale-morphism conformance",),
        environment="locked base",
        operations=(
            WorkflowOperation(
                "campaign scale-morphism-reference-check",
                "reference",
                ("experiments/response-method-reference/config.json",),
            ),
        ),
        inputs=(
            WorkflowInput(
                "experiments/response-method-reference/config.json",
                "native development or import config",
                "empirical-lawhood/reference-worlds/physical-scale-morphism/physical-scale-morphism-truth-suite-config",
            ),
        ),
        input_availability="Public generated reference suite; no physical records required.",
        scientific_ceiling="Thirty cases across fifteen fixtures with two nested variants.",
        first_missing_prerequisite="No physical source, independent physical evidence or issued campaign.",
        guide="experiments/response-method-reference/guide.md",
        verification_owners=("tests/test_circuit_truth_known_development_input.py",),
    ),
    Workflow(
        workflow_id="propulsion-reliability-reference",
        title="Propulsion reliability reference",
        kind="truth-known method reference",
        substrates=("generated propulsion worlds",),
        methods=("closure/reliability reference",),
        environment="locked base",
        operations=(
            WorkflowOperation(
                "campaign propulsion-reference-check",
                "reference",
                ("experiments/propulsion-reliability-reference/config.json",),
            ),
        ),
        inputs=(
            WorkflowInput(
                "experiments/propulsion-reliability-reference/config.json",
                "native development or import config",
                "empirical-lawhood/methods/synthetic-propulsion-development-input",
            ),
        ),
        input_availability="Public synthetic truth-known preparations.",
        scientific_ceiling="Reference worlds and nested decisions; no physical propulsion evidence.",
        first_missing_prerequisite="Physical preparation, custody and prospective qualification are not supplied.",
        guide="experiments/propulsion-reliability-reference/guide.md",
        verification_owners=("tests/test_synthetic_propulsion_development_input.py",),
    ),
)


def list_workflows() -> tuple[Workflow, ...]:
    """Return the closed task inventory; inspect no environment or source input."""
    return WORKFLOWS


def show_workflow(workflow_id: str) -> Workflow:
    """Look up an exact public identifier, never an import path or expression."""
    for workflow in WORKFLOWS:
        if workflow.workflow_id == workflow_id:
            return workflow
    raise ValueError(f"Unknown workflow: {workflow_id}. Use workflow list.")
