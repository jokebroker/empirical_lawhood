# Researcher glossary

SPDX-License-Identifier: CC-BY-4.0

This glossary gives the meanings of scientific and software terms used in these guides.
It keeps measurement, response, qualification, admission and permitted use distinct.
Start with the response operands below, then use the evidence stages to interpret a report.
The [reactor example](../experiments/reactor-response/guide.md#one-response-relation-in-ordinary-terms)
puts these terms in one concrete calculation.

## Scientific and software objects

| Term | Meaning here |
|---|---|
| Response relation | The bounded relation `L(D,H,A,R,τ)` between declared conditions, action and measured consequence |
| History | Observations causally available at the declared decision cutoff |
| Action | The specified native intervention and its delivery conditions |
| Horizon | The declared interval from the response origin to its measurement |
| Response-law package | A relation with its required observations/actions, supported domain, uncertainty, horizon, refusal rules and evidence |
| Denominator | The prepared systems, units and allowed conditions to which a response claim applies |
| Independent unit / root | A separately assigned physical or native realization. Its arms, views, request words and observations remain nested. |
| Numerical view | A declared discretization or solver setting for the same unit. It probes robustness and adds no independent unit. |
| Canonical record | Strict, versioned JSON emitted by `canonical_bytes()`. Applicable decoders reject unknown fields, duplicate keys and identity substitutions. |
| Object identity | ID, schema, version and canonical fingerprint together. Matching ID strings alone do not prove matching content. |
| Candidate | A design and its requirements compiled without evaluation outcomes. Compilation grants no custody, approval, execution or reveal authority. |
| Source closure | Exact clean Git commit and tree, with proof that the tracked package in that checkout executes |
| Issue | Publication of the reviewed candidate, source closure, proposer attestation and custody inputs as immutable records |
| Execution contracts | `ExperimentPackage`, `ProtocolRunPlan` and `ProtocolExecutionPlan` describe authorized delivery, scientific operations and executable tasks. Their format revisions are separate from scientific support. |
| Receipt | An immutable record of actual inputs, outputs, checks and implementation for a completed task. SQLite provides a rebuildable custody projection. |
| Exposed development | Inputs or outcomes already available for design or debugging. They cannot become prospective qualification input by changing a label. |
| No-contact proof | A software statement after provider, input, output, resource and control checks, before native tasks. It grants no research authority. |
| Software conformance | Implementation behavior established by tests under stated synthetic or development inputs. It grants no research authority. |
| Source qualification | Evidence that particular native bytes, preparation, clocks, units, delivery and receivers meet a declared source contract |
| Scientific qualification | Adjudication against preregistered response-law falsifiers and support limits, tied to exact source, inputs and receipts |
| Admission | A separate decision that permits a qualified claim to serve a specified consumer. Qualification alone does not grant controller use. |
| Reveal | Separately authorized evaluator access to sealed outcomes. Discovery and candidate compilation use no evaluation outcomes. |
| Cohort | The complete roster of independent units assigned to a declared role |
| Estimand | The quantity that the declared scientific analysis seeks to estimate |
| Receiver | The declared measured quantity, observation or downstream contract that receives a specified action or model output |
| Bootstrap | Resampling under the declared independent-unit and grouping rules to estimate uncertainty |
| Seed | A committed numerical input to a stated random generator. A view or nested observation does not create another independent unit. |
| Fingerprint | SHA-256 of the canonical record bytes. Changed canonical content requires a changed identity. |
| Preflight | Validation before the declared native or external effect. Success grants no execution authority |
| Projection | A derived computer or scientific view with its declared source identities and limits |
| Charter | The declared common response conditions that a prepared parent must qualify before dependent refinement |
| Model bank | Frozen fitted predictors with exact plan, training-root and development-result identities |
| Numerical chart | The finite declared conditions, views or solver settings used by a scientific contract |
| Grant | A separately published authority act with exact issuer, subject, scope, grantee and permissions |
| Post hoc | Analysis that has access to its declared existing outcomes. It cannot become prospective by relabeling them |
| Custody | The controlled retention, identity verification and authorized access of scientific inputs, outputs and immutable records |

## Evidence stages

These values match `EvidenceRung` and `EvidenceCeiling` in
[`kernel/evidence.py`](../src/empirical_lawhood/kernel/evidence.py).
Preparation and causal custody remain prerequisites at every applicable stage.
They do not replace measurement validity or order support.

| Stage | Canonical value | Question |
|---|---|---|
| Measurement | `MEASUREMENT` | Is the declared native measurement valid? |
| Order relation | `ORDER_RELATION` | Is the declared order relation supported? |
| Response | `RESPONSE` | Is the bounded action–response claim supported? |
| Local law | `LOCAL_LAW` | Does the local law survive its stated falsifiers and support limits? |
| Admission | `ADMISSION` | Is that result admitted to the specified consumer or transport use? |
| Controller use | `CONTROLLER_USE` | Does the separately authorized controller-use contract hold? |

`NON_PROMOTABLE` permits no evidence-stage claim.
A capability's maximum ceiling limits its possible claim.
That ceiling does not establish an achieved result.

## Response methods and study roles

| Term | Interpretation and boundary |
|---|---|
| Prepared response | Screening and evaluation over a declared finite word menu and whole-root cohorts. An authenticated negative evaluation cannot authorize dependent refinement. |
| Dependent refinement | Refinement that consumes an authenticated qualified screening parent. The parent does not manufacture a fresh independent source. |
| Fresh response calibration | Calibration of an authenticated refinement library nominated for this role, using a separately declared unexposed cohort and the complete parent/handoff contract |
| Information-based response prediction | A fitted model bank and its native handoff/receiver contract. Outcome-fitted banks remain exposed development input. |
| Causal response contrasts | A fitted model bank with paired native contrasts. Its scientific operands differ from information-based prediction. |
| Native cohort assignment | Reservation of physical roots for a stated role. Calibration and evaluation assignments have distinct roles and exposure rules. |
| Finite response-law specification | Frozen scientific operands and deterministic development requests. Specification binding does not establish universal engine readiness. |
| Opportunity screening | The finite response-law method's two-future opportunity gates |
| Informative composition | The retained composition method and its paired decision checks |
| Uncertainty calibration | Uncertainty and usability gates with complete whole-root rank accounting |
| Prospective consumer evaluation | Separately bound consumer and control evaluation. Continuation reuses the original root census. |
| Preparation screening | Screening with the unchanged qualified lower package and its declared root denominator |
| Preparation-policy development | The distinct supplied lower package, 24-root panel, nine-schedule chart and fixed development gates. Arithmetic agreement does not qualify a source. |
| Transverse response | Signed current response to the declared realized vector potential, with its gauge, finite-wavevector and unit conditions. Pairing alone does not establish it. |
| Post-hoc response composition | Analysis of authenticated, outcome-visible parent scalars. The analysis cannot relabel those inputs as prospective evidence. |

The [finite response-law specification](../src/empirical_lawhood/adapters/methods/finite_response_law/specification.md)
and [response input guide](response-input-formats.md) govern the detailed scientific operands.
Fresh calibration alone does not establish prospective use.

## Technical verbs

These verbs identify distinct operations. Their objects and evidence boundaries matter.

| Verb | Permitted technical meaning |
|---|---|
| Authenticate | Validate exact bytes, identity and authorization through the declared signature, receipt and custody replay. This does not establish physical validity. |
| Adjudicate | Apply preregistered scientific falsifiers to authorized evaluator inputs and record evaluability, support and limits |
| Calibrate | Estimate the declared uncertainty or response parameters from the specified independent calibration cohort |
| Fit | Estimate a declared model from its permitted training inputs. Do not use held evaluation outcomes for this estimate. |
| Persist | Write and replay a canonical record as an immutable publication under the declared computer-storage contract |
| Replay | Reconstruct and validate the exact stored record, receipt and lineage relationship. Do not repeat completed native effects. |
| Resolve | Select the registered provider, decoder, locator or grant from exact typed computer inputs |
| Support | Satisfy the declared scientific criterion within the specified denominator and evidence ceiling. This verb does not grant admission. |
| Compile | Construct a typed candidate and its obligations from a design. Compilation grants no authority. |
| Decode | Read a strict serialized input into its declared typed record, with the applicable validation |
| Serialize | Encode a typed record into its declared byte representation |
| Hash | Calculate SHA-256 from the exact stated bytes |
| Bind | Associate records through all required identity and contract fields. The association does not transfer authority. |
| Validate | Apply the declared computer predicates to an input or record. Success establishes only those predicates. |
| Execute | Perform the tasks of an issued execution contract under its stated controls |
| Freeze | Commit the specified scientific design or inputs as immutable records before the declared outcome-access cutoff |

## Operational and scientific outcomes

Operational completion and scientific adjudication answer different questions.
`SUCCEEDED` means that the operation completed.
A completed operation can have a legitimate negative scientific verdict.
`UNEVALUABLE` means that the declared evidence or evaluator contract could not decide the claim.
It provides no support for the claim.
`NOT_EVALUATED` admission authorizes no control.

Keep negative and unevaluable results.
A changed hypothesis, denominator or cohort requires a new predeclared design.
An execution retry cannot make that change.

## Notes for documentation authors

### Technical noun categories

These terms name specified scientific or computer objects under ASD-STE100 rule 1.5.
Mathematical and scientific terms use category 7.
Computer records and processes use category 19.
Exact software identifiers retain their declared spelling.

Use technical nouns as nouns in these contexts.
An -ing form can name a defined object or process, such as bootstrap resampling.
It cannot replace an instruction verb.

Use these technical verbs for the stated computer or scientific processes.
Use an approved dictionary verb when it preserves the exact meaning.
A technical word in generic prose does not give that prose an exception.
Computer verbs below use rule 1.12 category 2.
Scientific verbs use category 3a.
Use simple approved verb forms and retain the stated object and evidence boundary.

## References and research

The [scientific integrity guide](scientific-integrity.md) states the evidence boundaries used by these definitions.
The [documentation policy](../CONTRIBUTING.md#documentation-policy) uses ASD-STE100 Issue 9 and its technical noun/verb categories.
The linked public specification is the owner of the finite response-law operands.
