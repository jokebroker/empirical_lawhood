# Licensing policy

SPDX-License-Identifier: CC-BY-4.0

This policy defines the repository licenses and third-party reuse requirements.
It separates software, documentation, generated data and upstream material.
Use it to identify the terms that apply to a selected file.

Empirical Lawhood is intentionally a mixed-license repository.

## Default scopes

### Software — MPL-2.0

The following are MPL-2.0 unless an individual file says otherwise:

- source code under `src/`.
- executable experiment machinery under `experiments/`.
- executable examples under `src/empirical_lawhood/examples/`.
- executable build and figure scripts under `paper/`.
- internal maintainer scripts under `scripts/`.
- other source-code-form software in the repository that is not explicitly assigned another license.

For new source files, use the applicable comment syntax with:

```
SPDX-License-Identifier: MPL-2.0
```

The complete MPL-2.0 text is available at the repository root as `LICENSE` and at `LICENSES/MPL-2.0.txt`.

### Documentation and figures — CC-BY-4.0

Project-authored prose, PDFs and figures under `docs/` and `paper/` use Creative Commons Attribution 4.0 International.
Repository documentation with a CC-BY-4.0 notice uses the same license. Executable scripts in `paper/` retain MPL-2.0.

Before canonical citation metadata is published, attribution may identify the creator as **Empirical Lawhood project**, include the repository URL, and identify CC-BY-4.0. Code reproduced verbatim inside documentation retains the license applying to that code.

### Generated evidence and data — CC0-1.0

The former `evidence/` directory policy dedicated project-generated evidence, results, receipts, measurement tables and datasets placed there under CC0 1.0 Universal unless a file said otherwise. That directory contained only the provenance template, now at `configs/templates/evidence-provenance.yaml`, whose CC0 notice is unchanged. Relocation does not extend the dedication to private external custody records.
The project-generated `paper/source/reported-values.json` published summary operands use CC0-1.0.

CC0 is used to minimize friction for validation, recombination, meta-analysis, and machine-readable scientific reuse.

This does not waive or alter rights held by third parties in any upstream input material.

### Third-party material

Third-party material is not covered by the project's MPL, CC-BY, or CC0 grants merely because it appears in this repository. Its original terms control.

Before you add third-party code, data, models, figures or assets, write their provenance and licensing.
Before you add that material, put this information beside the actual vendored payload or in the relevant source reference.

## Precedence

1. A file-specific license notice or SPDX identifier controls that file.
2. Otherwise, the directory policy above applies.
3. For software not otherwise marked, the root MPL-2.0 license is the default.
4. Third-party rights remain subject to their upstream terms.

## Redistribution and provenance

Third-party material retains its upstream license and is **not** relicensed under MPL-2.0, CC-BY-4.0, or CC0-1.0 merely because it appears in this repository.

For each third-party item, record:

- name.
- upstream source.
- version, release, commit, DOI, or other identifier.
- license.
- local path or usage.
- modifications, if any.
- any redistribution or attribution requirements.

When redistribution is not permitted or is unnecessary, prefer recording how to obtain the dependency rather than committing a copy.

The three packaged reactor inputs retain their upstream Apache-2.0 license,
unmodified source bytes and exact pins. Their specific account remains beside
the payload in [PROVENANCE.md](../src/empirical_lawhood/_vendor/terminal_bench_science/PROVENANCE.md).

The former template-only evidence directory is not a primary-results deposit.
The template and plotted-value CC0 notices retain their original scope.
Private custody records and third-party inputs are
not relicensed by relocating the provenance template.

## Neptune photograph

The repository retains an unmodified NASA/JPL photograph of Neptune.
This third-party asset illustrates the project's long-term motivation.
Use the linked source and terms when you reuse the photograph.

| Provenance field | Selected value |
|---|---|
| Asset | `assets/neptune-voyager-2.jpg` |
| Image ID | PIA01492 |
| Title | Neptune Full Disk From Voyager 2 |
| Capture | Voyager 2, summer 1989 |
| Source | [NASA Science image page](https://science.nasa.gov/image-detail/pia01492-neptune-full-disk-16x9/) |
| Download | [Original 1920 × 1080 JPEG](https://science.nasa.gov/wp-content/uploads/2024/03/pia01492-neptune-full-disk-16x9-1.jpg) |
| Credit | Neptune, Voyager 2 (1989). Courtesy NASA/JPL-Caltech. |
| Download date (UTC) | 2026-10-02 |
| SHA-256 | `f113eae8ce9db322495f3dacaf6cbee14f72ebcfbbf0da2a408bd3af0f4abb11` |
| Modifications | None. The retained asset preserves the original 16:9 composition. |
| Reuse terms | [JPL Image Use Policy](https://www.jpl.nasa.gov/jpl-image-use-policy/) and [NASA Images and Media Usage Guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/) |

Retain the credit when you reuse this image.
Its upstream terms control. The project's CC-BY-4.0 documentation license does not cover the photograph.
NASA, JPL and Caltech have not endorsed this project.

### References and research

- NASA/JPL-Caltech, *Neptune Full Disk From Voyager 2*, PIA01492; source page updated 18 April 2024. The page identifies the image and credit.
- Jet Propulsion Laboratory, *JPL Image Use Policy*, consulted 2 October 2026. The policy supplies the image-use conditions and required credit.
- NASA, *Images and Media Usage Guidelines*, consulted 2 October 2026. The guidelines explain informational use, third-party rights and endorsement limits.
