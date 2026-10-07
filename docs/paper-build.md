# Preserve, validate and optionally build preprint edition 0.60

SPDX-License-Identifier: CC-BY-4.0

The primary reader asset is the verified [published preprint PDF](../paper/editions/v0.60-deposit/Empirical_Lawhood_Manuscript_v0.60.pdf), edition **0.60**. The separately supplied source-bundle [Markdown](../paper/manuscript.md), [PDF](../paper/manuscript.pdf), [evidence and provenance](../paper/revision-and-evidence-notes.md), [companion PDF](../paper/revision-and-evidence-notes.pdf), and six figure assets retain the exact supplied bytes. The [selected-source ZIP](../paper/editions/Empirical_Lawhood_v0.60_Selected_Source.zip), [paper inventory](../paper/README.md), [member map](../paper/source/canonical-inputs.json) and [preservation receipt](../paper/source/source-preservation.json) bind this selection. The curated ZIP contains 17 unchanged current-edition source members. The complete 40-member original source ZIP remains byte-exact in external provenance custody; it is not the public ZIP. Only edition 0.60 publication assets are selected here. No navigation edits were made.

## Validate the selected original

Use the selected checkout and its [paper environment](environments.md), with a new external receipt directory under an actual mounted storage root:

```sh
uv run --no-sync python paper/source/validate_bundle.py \
  --output-dir /absolute/external/paper-validation \
  --storage-root /absolute/external \
  --storage-mount /absolute/active-mount
```

Validation checks the selected local inventory, the curated ZIP and mapped members, unchanged selected documents, local links, the exact 28-reference roster, reported-summary arithmetic and hash-bound visual review. The selected [review](../paper/source/pdf-preflight.json) covers all 19 manuscript pages, six companion pages and all six figure formats. It retains the supplied navigation and metadata.

The [source register](../paper/SOURCES.md) and [claim evidence inventory](../paper/source/claim-evidence-inventory.json) state availability and interpretation limits. A passing publication check does not reproduce native trajectories, fitting or bootstrap results, authenticate missing primary bindings, or qualify the current software scientifically.

## Original build evidence and optional derivatives

The original [build recipe](../paper/editions/v0.60/build.sh) and [supplied validation report](../paper/editions/v0.60/validation_report.json) remain byte-exact. The supplied report describes its original source bundle; the maintained validator checks the current public selection. Complete original producing-tool versions, command transcripts and TeX logs were not supplied. The preservation receipt records that limitation; selecting original published bytes requires no new build.

For an optional new derivative, install the lock's paper dependency group and supply Pandoc, XeLaTeX, Linux Libertine O, DejaVu Sans/Mono and Latin Modern Math. The maintained builder checks declared fonts and records actual inputs/tools. It copies canonical inputs to external scratch, uses the supplied header/filter, and makes no navigation patch. Run it only when a derivative is wanted:

```sh
uv run --no-sync python paper/source/build_pdfs.py \
  --pandoc /absolute/path/pandoc \
  --engine /absolute/path/xelatex \
  --output-dir /absolute/external/paper-build \
  --scratch-dir /absolute/external/paper-scratch \
  --storage-root /absolute/external \
  --storage-mount /absolute/active-mount
```

Keep the generated PDFs, TeX, logs, recorder files and build manifest. Render and inspect every page and figure of each new derivative, including actual navigation, captions, tables and glyphs. Bind the review to its exact hashes. A derivative does not replace the selected original PDFs automatically. After an authorised selection change, [write the inventory](../paper/source/write_manifest.py), validate it, and follow the [selection contract](release-status.md#selection-record-contract).

## Identifiers and scientific boundary

Gareth Seneque, [ORCID 0009-0003-3046-6899](https://orcid.org/0009-0003-3046-6899), authored the non-peer-reviewed preprint, dated 4 October 2026 as supplied. The original bundle calls DOI 10.5281/zenodo.23083988 project/software and assigns no manuscript DOI. The verified [Zenodo record](https://zenodo.org/records/23083988) identifies the preprint DOI, edition 0.60 and publication date 1 October 2026. Keep those provenance statements distinct.

The later complete preprint experiment programme follows the final source fixes. Archived-cohort reproduction and fresh prospective repetition remain distinct; neither publication preservation nor SOURCE_READY starts them. [New experiments](designing-and-running-an-experiment.md) require their own inputs, independent units, exposure boundaries, authority and receipts.

Project-authored text/PDFs/figures use CC-BY-4.0; executable builders use MPL-2.0. Third-party materials retain their own terms; see [licensing](licensing.md).

The separately retained [deposited PDF](../paper/editions/v0.60-deposit/Empirical_Lawhood_Manuscript_v0.60.pdf) has different bytes from the supplied source-bundle PDF. Its publication-identifier paragraph on page18 names the manuscript DOI; all other pages render identically in the hash-bound comparison. Review covers the changed page directly and the remaining pages through proven pixel equality. The producing transformation and its original tool records remain unavailable. Both originals are retained without rewriting either.
