# Public sources and external custody

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for public source selection and guarded external custody.
It separates source acquisition from native scientific qualification.
Use it to preview the bounded plasma selection and identify acquisition prerequisites.

Source declarations live in [`configs/sources/`](../configs/sources). They select public endpoints, finite members, schemas and transfer bounds. They contain neither experimental payloads nor a publisher checksum. The storage profile supplies the external root. The exact downloaded bytes, SHA-256 values, time and source selection are recorded in a guarded receipt.

The [NREL inverter qualification assessment](sources/nrel-inverter-qualification.md)
records a newer authentic public reference and the physical action journal it
does not contain. The prospective NREL physical route is outside first-release scope. The
assessment is a source decision, not an acquisition declaration or candidate
route.

The [Glenn 2026 archive assessment](sources/glenn-2026-qualification.md)
records a distinct official public retrospective source. Its [strict local
quick start](../experiments/laser-archive-inspection/guide.md) now previews the pinned release and
validates a researcher-held ZIP through the one CLI. It is an outcome-visible
dataset transform without a guarded custody/authorization handoff or
prospective physical intervention claim.

The [Grid2Op held-source assessment](sources/grid2op-qualification.md)
records an authentic excluded chronic and the installed native action/receiver
check. The installed CLI accepts an explicit local development selection.
A prospective route still needs a newly qualified chronic roster and a selected campaign provider.

## FAIR-MAST Level 2

[`fair-mast-level2.json`](../configs/sources/fair-mast-level2.json) selects public shot 30421, campaign M9, `summary/time`, its Zarr v3 array metadata and first compressed chunk. This is a bounded source sample, not the complete shot or a scientific observation panel. The declaration follows the [MAST Data Catalog's Level 2 access guide](https://mastapp.site/level2-data.html) and [REST API guide](https://mastapp.site/rest_api.html). [UKAEA](https://www.ukaea.org/service/fair-mast/) identifies the public JSON and S3 services and the archive data license, CC BY-SA 4.0. Software and data have separate licenses.

From any install, preview without network contact:

```sh
empirical-lawhood source preview --source-config configs/sources/fair-mast-level2.json
```

For live acquisition, use the selected checkout's tracked package and tested Python environment.
Supply a strict `OperatorStorageProfile` for the guarded external mount.
Install separate typed `SOURCE_ACQUISITION` and `CUSTODY_PUBLICATION` records in its authority store. The two records must bind the exact selection, grantee, storage root and relative custody scope. They are researcher supplied. `--yes` confirms the already authorized effect. No owner key, mount or grant is installed by this package.

```sh
empirical-lawhood --project-root /path/to/clean-target \
  --operator-profile /path/to/operator-profile.json \
  source acquire --source-config /path/to/clean-target/configs/sources/fair-mast-level2.json \
  --acquisition-authority-id SOURCE_AUTHORITY_ID \
  --custody-authority-id CUSTODY_AUTHORITY_ID --yes
```

`source recover` takes the same arguments and `--yes`. It verifies an existing receipt or completes the **same** durable attempt. It never silently starts a second selection. The gateway accepts only declared HTTPS hosts and exact URLs.

It refuses redirects and enforces per-member and aggregate byte limits. It validates shot/campaign metadata, Zarr metadata and the compressed chunk header. It stores raw bytes under the external custody root and hashes those bytes in the receipt. A changed response, missing member or changed local byte refuses verification.

It does not use the generic public-source service, which requires publisher checksums and has no Zarr member type.

To turn a verified receipt into a prospective design input, supply a strict `InformationCutoff` document and a UTC authoring time after custody:

```sh
empirical-lawhood --project-root /path/to/clean-target \
  --operator-profile /path/to/operator-profile.json \
  source design-input --source-config /path/to/clean-target/configs/sources/fair-mast-level2.json \
  --cutoff /path/to/cutoff.json --input-id MY_INPUT_ID \
  --operator-id MY_OPERATOR_ID --authoring-at-utc 2026-09-28T12:00:00Z
```

The output is a strict `DesignInputRecord` binding the receipt identity and its hash. Put it in the new `StudyDraft.design_inputs` before candidate compilation. It is an outcome-blind motivation input. A later execution must declare and qualify its native observation input separately.
Receipt binding establishes no scientific fitness, issue authority or plasma result. The example acquisition receipt is external and is never shipped in the wheel or repository.

## References and research

The linked source modules, command metadata and existing checks own the implemented behavior described here.
The [scientific integrity guide](scientific-integrity.md) defines its separate evidence and authority boundaries.
The [program source register](../paper/SOURCES.md) identifies bounded historical results and unavailable primary records.
