# Empirical Lawhood — first preprint, edition 0.60

SPDX-License-Identifier: CC-BY-4.0

The owner-selected first published preprint is edition **0.60**, dated 4 October 2026 in its supplied source. Read the [published preprint PDF](editions/v0.60-deposit/Empirical_Lawhood_Manuscript_v0.60.pdf). The supplied source-bundle inputs are the [Markdown](manuscript.md) and [PDF](manuscript.pdf), the [evidence and provenance companion](revision-and-evidence-notes.md) and [PDF](revision-and-evidence-notes.pdf), and the [revision summary](revision-summary.md).

## Choose a reading path

You can use the [reactor example](../experiments/reactor-response/guide.md) and
the [editable RC model](../experiments/rc-ladder-response/guide.md#edit-a-development-model)
before reading the theoretical background. The [workflow catalogue](../experiments/README.md)
selects current software operations; this index selects research reading.

Section numbers below refer to the supplied edition 0.60 manuscript. Follow the
linked Markdown sections or the same headings in the published PDF.

| Your question | Read next | What it supplies and assumes |
|---|---|---|
| What is a response law? | Manuscript §3, [Response laws and scientific inquiry](manuscript.md#response-laws-and-scientific-inquiry), then the reactor's [concrete response relation](../experiments/reactor-response/guide.md#one-response-relation-in-ordinary-terms) and [glossary](../docs/glossary.md) | A relation, its package and the task it serves; no philosophical prerequisite. The reactor maps the five operands to actual inputs and reports. |
| What does the empirical argument establish? | §4, [What descriptions preserve—and what they do not](manuscript.md#what-descriptions-preserveand-what-they-do-not); §5, [Supplying conditions for use](manuscript.md#supplying-conditions-for-use); §6, [Different systems, different requirements](manuscript.md#different-systems-different-requirements) | After §3, compare information loss, structural findings, forecast-mediated reuse, preparation and their distinct populations. Continue with the [evidence companion](revision-and-evidence-notes.md) for inferential and provenance limits. |
| Why ask about warrant and enabling conditions? | §2, [Warrant, participation and physical realisation](manuscript.md#warrant-participation-and-physical-realisation), then the foundations below | Optional conceptual background. It distinguishes entitlement, experimental participation and the physical conditions of usable regularity; it supplies no substitute for local qualification. |
| Can I inspect or reproduce a reported result? | [Source register and study crosswalk](SOURCES.md), then the [claim-to-evidence inventory](source/claim-evidence-inventory.json) | Check availability before choosing commands. Historical companions [18–22] are preserved externally and are not packaged here. Documentary overlap does not create independent cohorts. Current related operations do not reproduce the original observations. |
| How does the implementation carry these objects? | [Architecture](../docs/architecture.md), then the relevant family guide and [extension procedure](../docs/extending-the-engine.md) | Start with §3 and one software example. Follow typed scientific records, application composition and the separate scientific and executable graphs. |

## Read an implemented method

Start with its family guide for inputs, environment, effects and stopping point.
Read its scientific owner before changing an operand. These are method-specific
prerequisites; the conceptual works below are optional deeper reading.

| Method question | Procedure and scientific owner | Prior understanding and limit |
|---|---|---|
| Which information can a reduced circuit observation lose? | [Analytical RC guide](../experiments/rc-information/guide.md), [construction](../src/empirical_lawhood/adapters/methods/rc_information/analytic.py) and [independent equation check](../src/empirical_lawhood/adapters/methods/rc_information/checker.py) | Elementary RC equations and exponentials. An exact two-state illustration has no sampled confirmation or qualification. Continue to [RC history challenges](../experiments/rc-challenges/guide.md) for the distinct cohort-based procedure. |
| How can a forecast supply an input to a fixed response package? | [Finite response-law guide](../experiments/finite-response-law/guide.md), [frozen specification](../src/empirical_lawhood/adapters/methods/finite_response_law/specification.md) and [method provider](../src/empirical_lawhood/adapters/methods/finite_response_law/method_provider.py) | Whole-root calibration, support/refusal rules and separate evaluation cohorts. The fixed nomination and current parent joins preserve their exposure and authority boundaries; full scientific rerun remains unperformed. |
| Can preparation supply a condition for use without changing the package? | [Preparation applicability guide](../experiments/preparation-applicability/guide.md), [ordinary frozen specification](../src/empirical_lawhood/adapters/methods/preparation_applicability/frozen_specification.md) and [constructed frozen specification](../src/empirical_lawhood/adapters/methods/constructed_preparation_applicability/frozen_specification.md) | The fixed lower response package, support gates and phase roles. Read the guide's current operational specifications as well; preserved original specifications transfer no donor authority. |

For another method, select its entry in the [workflow catalogue](../experiments/README.md)
and use that guide's scientific references. Registration and passing software checks
do not establish an achieved evidence stage.

## Deeper background in the existing bibliography

Numbers identify entries in the unchanged [manuscript bibliography](manuscript.md#references-unnumbered).
Read §3 first for the framework's use of these ideas. The annotations describe
their roles in that argument, not additional measured results.

| Question | References and role | Useful prior understanding |
|---|---|---|
| When do observations suffice for a task? | Data informativity [4] separates identification from the information needed for control. Input–output computational mechanics [5] groups histories by their future consequences. | Dynamical systems, prediction and conditional distributions. |
| What must connected descriptions preserve? | Assume-guarantee contracts [6] address composition requirements; causal abstraction [7] addresses error across connected causal descriptions. | Input/output contracts or causal models, after the composition distinction in §3. |
| What does a prediction permit a decision to do? | Conformal prediction [8] concerns calibrated prediction; task-based learning [9] connects model fitting to decision objectives. Conditional predictive limits [14] constrain distribution-free guarantees; binomial intervals [15] concern finite-sample proportions. | Probability, calibration/evaluation separation and the declared estimand. These references do not make every implemented interval conformal. |
| Why can reduction require memory? | Mori–Zwanzig and optimal prediction [10] explain memory in reduced descriptions. | Differential equations and projection methods; read the circuit information example first. |
| What motivates the matrix structural investigations? | Emergent geometry [12] supplies matrix-model context; noiseless subsystems [13] supplies context for retained algebraic structure. | Matrix dynamics and linear algebra. Neither reference qualifies the selected matrix events or supplies new cohorts. |
| Why distinguish warrant, participation and physical possibility? | Kant [1] concerns entitlement to knowledge; Wheeler [25,26] concerns apparatus-elicited questions and participation; Gell-Mann/Hartle [2,3,17,24] concern histories, quasiclassical regularity and knowing systems. Cartwright [11] concerns arrangements that sustain regularities; Heidegger [16] concerns equipment and practical dependence. | Begin with §2's distinctions. The local scientific contracts do not depend on adopting every philosophical or cosmological claim. |
| Why preserve people's ability to redirect inquiry? | Engelbart [27] and meaningful human control [28] inform the scientific-agency motivation in [§7](manuscript.md#from-dependencies-to-cumulative-capability). | No method prerequisite. Research direction is a design aim, distinct from the measured study outcomes. |
| Where is the project's detailed research history? | Historical companions [18–22] supply the studies and overlapping analyses identified in the [source register](SOURCES.md); current companion [23] supplies evidence and provenance. | Read §§4–6 first. Only [23](revision-and-evidence-notes.md) is packaged as a companion here; use the inventory to check external documentary and native-record availability. |

## Publication selection and preservation

The supplied source-bundle documents and all six figure assets retain the exact supplied ZIP-member bytes. No PDF regeneration, metadata change, navigation correction or numerical plot edit was applied. The [curated v0.60 source export](editions/Empirical_Lawhood_v0.60_Selected_Source.zip), [member map](source/canonical-inputs.json) and [preservation receipt](source/source-preservation.json) identify 17 original members and 32 public paths. This is a current-edition selection, not the complete original bundle.

The complete original 40-member ZIP is preserved byte-exact in the external maintainer backup. Earlier paper editions, source papers, editorial drafts, diffs and the historical validator are excluded from this public selection. Their original names and hashes remain provenance metadata in the preservation receipt. The retained publication text and supplied validation report still describe their original evidence and editorial history; those references do not mean the predecessor payloads are included here.

The [source register](SOURCES.md) and [evidence inventory](source/claim-evidence-inventory.json) distinguish included documents and figures from available external research records, and record the limits of each evidence binding. Included historical documents retain their original identities and reported negative, mixed and unevaluable results. Their source hashes authenticate their bytes; they do not establish a new qualification or an independent reproduction.

The original [figure provenance](assets/inherited_figure_provenance.json) is retained as supplied. Its earlier-edition source/output hashes remain earlier-edition claims; current figure hashes are bound independently by the member map and [selected visual review](source/pdf-preflight.json).

The owner selected [Zenodo record 23083988](https://zenodo.org/records/23083988) as the canonical first-preprint deposit. Verified deposit metadata identifies the manuscript DOI 10.5281/zenodo.23083988, preprint edition 0.60 and publication date 1 October 2026. The original source-bundle README's earlier software-DOI label and absent-manuscript-DOI statement are retained in external preservation and recorded in [metadata](metadata.json). This preprint is not peer reviewed.

Use the [paper guide](../docs/paper-build.md) to validate this selection or build an optional derivative externally. The supplied [build recipe](editions/v0.60/build.sh) and [validation report](editions/v0.60/validation_report.json) are included unchanged; the latter reports the original full bundle, not a new check of this curated export. Its historical validator needs excluded predecessor drafts and is preserved externally. The maintained [selection validator](source/validate_bundle.py) checks the current public payload. Complete producing-tool versions, commands and TeX logs were not supplied. Validation and visual review of these preserved bytes require no scientific rerun. The later complete experiment programme and fresh prospective experiments have separate protocols and authority.

The [deposited manuscript PDF](editions/v0.60-deposit/Empirical_Lawhood_Manuscript_v0.60.pdf) is retained separately with [verified deposit metadata](editions/v0.60-deposit/zenodo-record.json). Its hash differs from the supplied source-bundle PDF. Only page18 differs in extracted text and rendered appearance: the publication-identifier paragraph identifies the manuscript DOI. All other pages, PDF metadata and outlines agree in the recorded comparison. The source-bundle Markdown and PDFs remain unchanged. The deposit date1October and printed manuscript date4October are distinct supplied metadata.
