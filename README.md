![Empirical Lawhood: A civilisational response-law engine. Light passes through two stone rings.](assets/empirical-lawhood-hero.jpg)

# Empirical Lawhood

We inhabit a universe whose regularities make knowledge and action possible. The observers, instruments and records through which we investigate it belong to that same universe. Both the conditions of knowing and the grounds for our claims to knowledge are open to inquiry.

**Empirical Lawhood (EL) investigates how knowledge can help produce the conditions for further knowledge and action.** It brings together Kant's question of warrant, Wheeler's account of participation, and Gell-Mann and Hartle's physical account of regularity. Each poses a distinct demand: justify reliance on a description; account for how an investigation's question and apparatus shape what it can reveal; explain the physical conditions that sustain observers, apparatus and dependable records.

A response relation earns **empirical lawhood** when evidence warrants its repeatability and prospective use under stated conditions. EL asks whether another operation can supply a condition that use requires. Producing the condition and establishing warrant for the resulting use are separate achievements.

A missing measurement, unsuitable preparation or unresolved distinction can become another scientific target. Supplying what is needed can make further questions answerable and actions possible. Each new capability has requirements of its own, giving inquiry a recursive direction: its means can themselves be investigated, maintained and changed.

In a [numerical study](paper/editions/v0.60-deposit/Empirical_Lawhood_Manuscript_v0.60.pdf#page=12) of a population constructed near a fixed support boundary, we changed preparation while keeping the response model and every rule governing its use fixed. The unchanged package supported successful use on **7,965 of 8,192 request pairs, nested within 32 fresh episodes**, versus **zero under unchanged preparation**. A conservative numerical check retained 6,720 pairs. The intervention supplied a physical condition under which existing evidence justified using the package.

The operational ambition is a **civilisational response-law engine**: scientific contributions supplying one another's requirements through tested connections. Dependable fusion propulsion to Neptune motivated the inquiry. The broader aim is to expand what people can discover and build, and their ability to choose and redirect that work. This repository is the first public software release.

**Start here:** [run an example](#install-and-run), [change a model](#make-it-your-own), [inspect the research evidence](#what-the-research-has-established), or [extend the engine](docs/extending-the-engine.md)\.

## What you can do here

- **Run a supported study** from a model or saved observations, using an explicit design and independent experimental units\.
- **Change the RC development model** and inspect how its measured response changes\.
- **Inspect the records behind a result or refusal**, then extend an existing route or add a substrate or method\.

The engine checks inputs and plans and retains episode records, reports and provenance\. The [release inventory](docs/release-inventory.md) records which operations are available in this extraction of the research codebase\.

## Install and run

The tested host is Linux x86\-64 with CPython 3\.11\.14\. Read the [environment support table](docs/environments.md) before selecting a dependency group\. From this checkout in an unactivated shell:

```sh
uv sync --locked --python 3.11.14 --group reactor-example
uv run --no-sync empirical-lawhood --help
uv run --no-sync empirical-lawhood doctor --route reactor --format json
uv run --no-sync empirical-lawhood capability list --limit 5 --format json
uv run --no-sync empirical-lawhood example reactor-prefix --output-dir ./example-output/reactor-prefix
```

Open `example-output/reactor-prefix/report.md` first\. The example uses packaged, pinned public reactor inputs, performs 20 short native branches with 40 observed decisions, and writes:

|Output       |What to inspect                 |
|-------------|--------------------------------|
|`report.md`  |The readable report: start here.|
|`report.json`|Exact clocks and summaries.     |
|`panel.json` |The canonical episode record.   |

The example illustrates the same pattern in a small, inspectable setting: an input is prepared, a fixed response route is applied, and the records show whether the declared conditions were met\. It is not a claim that the packaged demonstration independently establishes the scientific result described above\.

**This is a demonstration / unqualified local measurement\.** Its fixed, previously exposed inputs add no fresh independent support and do not reproduce the paper’s historical observations\. The five independent units contain nested arms and numerical views; those nested records do not increase the independent sample size\. No operator profile or scientific authority is needed for this example\. See the [reactor guide](experiments/reactor-response/guide.md) and [results and failures](docs/results-and-failures.md)\.

## Make it your own

**To start hacking, follow the [RC development loop](experiments/rc-ladder-response/guide.md)\.** It is the entry point for editing a model and checking its response\. Use the guide’s editable inputs and operations, then inspect the resulting records before changing the study\. The [configuration guide](docs/configuration.md) explains the configuration surface; the [source](src/empirical_lawhood) and [architecture](docs/architecture.md) show where the engine and adapters live\.

### Choose your next task

|What you want to do                                     |Start here                                                                                                                              |
|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
|Edit a model and check its response                     |[RC development loop](experiments/rc-ladder-response/guide.md), then [configuration](docs/configuration.md)                             |
|Choose a system for a scientific question               |[System selector](docs/substrates.md), then its family guide                                                                            |
|Run an implemented operation or find a paper-study rerun|[Workflow catalogue](experiments/README.md); [paper-study integrations](docs/integrations.md) for RC and matrix reruns                  |
|Design a study or prepare an issued handoff             |[Experiment design](docs/designing-and-running-an-experiment.md); [assigned reactor authoring](docs/fresh-experiment.md) for that family|
|Interpret a result, refusal or interrupted attempt      |[Results and failures](docs/results-and-failures.md)                                                                                    |
|Add a substrate or method                               |[Extension recipe](docs/extending-the-engine.md), [architecture](docs/architecture.md) and [supported Python use](docs/compatibility.md)|

For changes to the project itself, read [Contributing](CONTRIBUTING.md) and the [testing guide](docs/testing.md)\. Generated runs stay outside the repository\.

The [complete system inventory](docs/release-inventory.md#systems-and-current-states) and [route register](docs/substrate-route-register.md) distinguish implemented operations, deferred integrations and historical evidence\. Their state labels describe software availability\. Development checks and software tests grant no scientific qualification\.

An installed wheel supports static discovery, document validation, the example and advertised family checks with their declared dependencies\. Guide and editable input paths belong to a selected checkout or source archive\. Guarded operations require explicit source/storage selections and their separate scientific authorities\. See [environments](docs/environments.md) and the [CLI reference](docs/cli.md)\.

## What the research has established

A **response law** relates a specified intervention to a measured consequence under declared conditions\. A usable package also records the observations and interventions it requires, its supported domain, uncertainty, time horizon, refusal rules and evidence\. EL asks what a proposed use requires, which requirements have been established, and whether another operation can supply what is missing\.

The preparation study gives the central concrete case\. The response package was held fixed; only the preparation changed\. Under the original preparation, none of the tested request pairs met the package’s conditions for successful use\. After the intervention, the same package supported 7,965 of 8,192 pairs\. The result therefore concerns the production of applicability conditions, not a revised response model\.

The inquiry has produced scientific findings as well as tested connections between capabilities:

|Question                                                                |Finding                                                                                                                                                                                                                                                                                                                            |Evidence                                                                                        |
|------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
|**What must an observation preserve?**                                  |An analytical circuit example establishes an unavoidable prediction error when different futures share exactly the same permitted input. A separate numerical challenge found 1,524 response-adverse pairs, including 899 that also changed the tested decision. These targeted witnesses are nested in 36 independent seed blocks.|[Preprint §4.1](paper/editions/v0.60-deposit/Empirical_Lawhood_Manuscript_v0.60.pdf#page=7)     |
|**What does an apparent success actually establish?**                   |Matrix studies separated geometric organisation from dynamical commitment, an approximate algebra from the scalar detector that rejected it, and accurately predicted intervention-induced changes from an uncertain absolute future. Each distinction changed what the next investigation needed to resolve.                      |[Preprint §§4.2–4.4](paper/editions/v0.60-deposit/Empirical_Lawhood_Manuscript_v0.60.pdf#page=8)|
|**Can a forecast supply information needed by another operation?**      |Predictions and choices were committed before preparation. A forecast fed fixed response coefficients, and both tasks succeeded in **61/64 fresh episodes**. The composition received separate uncertainty calibration.                                                                                                            |[Preprint §5.1](paper/editions/v0.60-deposit/Empirical_Lawhood_Manuscript_v0.60.pdf#page=10)    |
|**Can an intervention supply a condition for using existing knowledge?**|Preparation made an entirely unchanged package support **7,965/8,192 successful request pairs across 32 fresh episodes**, versus zero under unchanged preparation. The conservative numerical check retained **6,720/8,192**.                                                                                                      |[Preprint §5.2](paper/editions/v0.60-deposit/Empirical_Lawhood_Manuscript_v0.60.pdf#page=12)    |

The preparation result concerns **qualified applicability**: satisfying the existing package’s evidence requirements\. It was tested in a local population deliberately constructed near a fixed support boundary\. A post hoc diagnostic that bypassed only that boundary found suitable responses already widely available without preparation\. The intervention chiefly changed when the existing knowledge justified successful use; the large gain does not measure newly attainable physical response\. Request pairs are nested within episodes, and the paired assays do not establish simultaneous control on one trajectory\.

These are separate studies\. The forecast and preparation constructions have not been joined into one qualified controller\. Further simulations and analyses of physical records examine batteries, plasma systems, cortical activity and an optical mirror, with distinct findings and limits for each\. Read the [preprint](paper/editions/v0.60-deposit/Empirical_Lawhood_Manuscript_v0.60.pdf), the [evidence and provenance companion](paper/revision-and-evidence-notes.md), and the [paper source register](paper/SOURCES.md) for the claims and available evidence\.

## Why the question reaches further

The consequence is a way to organise scientific work around what it can make possible next\. A materials result could enable an instrument; that instrument could resolve a distinction needed to investigate another material or control a reactor\. In the concrete preparation example, an intervention does not replace the response law: it supplies the conditions under which the existing law can be used\. Each connection would need its own evidence, uncertainty and operating conditions\. The engine’s ambition is to accumulate tested means of asking and answering further questions, with people able to choose and revise the direction of inquiry\.

The [preprint's philosophical foundations](paper/editions/v0.60-deposit/Empirical_Lawhood_Manuscript_v0.60.pdf#page=3) and [discussion of cumulative capability](paper/editions/v0.60-deposit/Empirical_Lawhood_Manuscript_v0.60.pdf#page=15) develop this connection.

## References and research

The [paper reading guide](paper/README.md) leads to the published preprint, its definitions and bibliography, and the evidence and provenance companion\. The [paper source register](paper/SOURCES.md) identifies historical evidence and its public availability\. The [source guides](docs/sources.md) describe the retained datasets and their acquisition limits\.

## Release status

|Component                |Version|
|-------------------------|-------|
|Software                 |0.2.0  |
|Public schema baseline   |1.0.0  |
|Selected preprint edition|0.60   |

**Scope:** first public source release\. Reactor and prepared/finite response have checked input and consuming contracts with the limits recorded in the [release inventory](docs/release-inventory.md)\. The [release status](docs/release-status.md) identifies the exact source/artifact packet and distinguishes software checks, scientific qualification and historical paper evidence\. See [compatibility and support](docs/compatibility.md), [changelog](CHANGELOG.md) and [operator input handoff](docs/operator-handoff.md)\.

The preprint DOI is [10\.5281/zenodo\.23083988](https://doi.org/10.5281/zenodo.23083988)\. Use the [citation guide](CITATION.md) and [reproducibility statement](REPRODUCIBILITY.md) when referring to this work or its evidence\.

<details>
<summary>Scientific qualification and publication provenance</summary>

The assigned reactor route supports five fresh native noise seeds bound to a complete prior\-exposure census\. The fixed public example remains exposed development input\. Qualification receipts bind the exact source commit and are retained outside this repository\.

The verified [Zenodo record](https://zenodo.org/records/23083988) identifies preprint edition 0\.60 with publication date 1 October 2026\. The supplied source bundle retains its earlier project/software DOI label as historical provenance\. It does not identify a verified release of this software candidate\. The [paper evidence inventory](paper/source/claim-evidence-inventory.json) states included and external historical evidence limits\.

</details>

## Repository map

|Location                                                                                                                                                                                                   |Purpose                                                     |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
|[Source](src/empirical_lawhood)                                                                                                                                                                            |Scientific engine, adapters, API, CLI and packaged demo.    |
|[Architecture](docs/architecture.md), [CLI](docs/cli.md) and [integrity](docs/scientific-integrity.md)                                                                                                     |General reference.                                          |
|[Experiments](experiments)                                                                                                                                                                                 |Task guides and study inputs. Generated runs stay external. |
|[Shared configuration](configs)                                                                                                                                                                            |Deployment/source settings and provenance templates.        |
|[Published preprint](paper/editions/v0.60-deposit/Empirical_Lawhood_Manuscript_v0.60.pdf), [evidence and provenance companion](paper/revision-and-evidence-notes.md) and [paper build](docs/paper-build.md)|Manuscript, evidence context and build instructions.        |
|[Contributing](CONTRIBUTING.md) and [testing](docs/testing.md)                                                                                                                                             |Contribution guidance, generators and verification profiles.|

## Use of AI

<details>
<summary>Model token usage</summary>

|Model      |Tokens            |
|-----------|-----------------:|
|gpt-5.6-sol|34,879,384,556    |
|gpt-6-astra|4,261,390,753     |
|gpt-5.5    |3,501,138,732     |
|gpt-6.1-sol|2,418,476,539     |
|gpt-6-sol  |1,622,755,196     |
|**Total**  |**46,683,145,776**|

</details>

## Licence

The code is MPL\-2\.0; project\-authored documentation, manuscript and figures are CC\-BY\-4\.0; generated evidence and plotted\-value data are CC0\-1\.0; third\-party material keeps its upstream terms\. See [licensing](docs/licensing.md)\.

SPDX\-License\-Identifier: CC\-BY\-4\.0