# Select exact release artifacts

SPDX-License-Identifier: CC-BY-4.0

This guide gives the clean-source procedure for software artifacts and their evidence packet.
It provides the software evidence bound by the external release selection.
Use it to build, examine and select exact artifact bytes before owner review.

## Prerequisites and first action

Use the [tested environment](environments.md) and a clean committed checkout.
Complete any public-history collapse before selecting that commit for verification.
The packet must bind the exact root or other commit intended for delivery; a later squash, reword, amend or merge creates a different source identity.
Select a new absolute output directory outside the checkout.
Keep sufficient space for the fresh environments and full worked example.
From that checkout, run:

```sh
uv run --no-sync python scripts/release_check.py --profile portable --output-dir /path/to/new-release-packet
```

The output directory must not already exist.
If its filesystem does not support Python environment symlinks, add
`--environment-root /path/on-posix-filesystem/new-release-environments`.
This directory must be new and outside the source checkout and packet.
It holds the fresh environments and temporary files. The packet retains their paths.
Keep sufficient free space on both filesystems.
Use `--offline` only with a complete verified dependency and interpreter cache.
The [test profile guide](testing.md) gives native and held prerequisites.
Select native profiles for each execution claim that requires them.
An absent selected prerequisite fails its profile.

## What the runner produces

For a once-only precommit portable gate, settle and stage the intended source, then run `uv run --no-sync python scripts/release_candidate_check.py --output-dir /absolute/new-candidate-packet`. It records actual loaded bytes and executable modes, the staged tree, exact environment, one complete portable selection with required-skip checks, sealed logs and coverage. It refuses unstaged public changes and excluded working plans. Passed disposable pytest fixtures are removed during the run to bound scratch use; failed evidence remains available.

Commit that exact unchanged candidate. Then run the release runner with `--portable-evidence /absolute/new-candidate-packet/precommit-portable-evidence.json` in addition to its normal portable packet options. This option authenticates the original logs/inventory/coverage against the final clean commit, preserves their precommit attribution and runs artifact/static/installed checks without invoking pytest again. A changed candidate, filtered/collect-only selector, failed check or altered evidence refuses reuse. Preserve failed packets and use focused repair checks; another full suite requires a new explicit decision.

The candidate checker saves its progress before each subprocess and after each sealed log.
A catchable interruption records `INTERRUPTED` with an unobserved exit code.
An abrupt termination can leave `RUNNING` evidence.
Neither state permits artifact-gate reuse.

The [release runner](../scripts/release_check.py) clones the exact committed source into the new packet.
It installs the locked environment and retains the actual Python, distribution and backend versions.
Before testing or building, it verifies the executing package belongs to that
checkout and that source, module and distribution versions agree.
The portable profile executes the complete portable tests and required critical coverage.
Coverage checks configured paths separately from the actual measured critical
owners, including the active catalog migration.
It also examines static errors, all registered generators and shipped-document targets.
Selected skips fail.

The runner builds the wheel through its newly produced sdist. It compares every package source/resource member with the selected Git source. It installs those exact artifact bytes in a separate wheel environment. Installed checks cover leaf help, metadata, packaged notices, native refusals and the full public reactor example.

They also cover partial failure, existing-output refusal and console bootstrap before NumPy import. These are software and exposed-development checks. They grant no scientific authority or admission.

## Examine the packet

Read `release-manifest.json`.
Require `status=PASSED` and the exact selected source identity.
Compare its tree-inventory and lock hashes with that source.
Read its actual tags, host and uv version.
Compare every required check's command, exit status and log hash.
Keep the environment log with the packet.

Each bound log is a new snapshot containing at most the bytes observed after
the direct child returned or its wait was interrupted. `unbound_live_log` names
the original diagnostic log; descendants may continue writing it, so its later
bytes are not part of the bound evidence. Successful and failed checks use
sealed snapshots. Catchable gate failures record `FAILED` and retain the
attempted command, exception type and an unobserved/null exit code where needed,
then propagate the original failure.

Read `package-inventory.json`. Compare each selected package member's byte count and SHA-256. Select the wheel and sdist from the manifest's `artifacts` entries. Those entries bind path, byte count and hash.

Use the packet's `wheel-requirements.txt` for a separate exact wheel installation. The packet also retains critical-coverage evidence and its subprocess-measurement limit. The [testing guide](testing.md) defines that coverage scope.

`release-status.json` binds the original manifest hash and the same source/artifact identities.
Its scientific qualification status is `NOT_PERFORMED`.
Keep that snapshot unchanged.
The later [external selection record](release-status.md#selection-record-contract) binds the software snapshot with separate paper, qualification and review evidence.
It owns the release stage and selected evidence.

The paper binding is separate from software artifact construction.
For the selected published preprint, retain the supplied archive, PDFs, figures and historical producing identities unchanged.
The selection contract accepts an original publication build manifest or the [as-is preservation receipt](../paper/source/source-preservation.json), together with exact asset, numerical/document-check and visual-review bindings.
That receipt states which original recipe, validation and tool records exist and which remain unavailable; it does not invent a build by the selected software.
Neither a new PDF build nor a preprint experiment rerun is required to select `SOURCE_READY`.
New derivatives and later experimental results retain separate identities and reviews.

## Selection and stopping conditions

A failed packet is retained diagnostic evidence.
It cannot select a passing candidate.
An incomplete packet without terminal manifest/status records is also
nonpassing. Abrupt termination, power loss or failed final storage writes may
prevent finalization; do not interpret a missing terminal record as success.
For a source correction, commit the reviewed change and use a new packet directory.
Keep earlier failure evidence unchanged.
Never select an artifact by its familiar filename or from an old ignored build directory.
A rebuilt artifact needs verification of its own bytes.

The source migration does not move existing campaign elapsed journals.
The default store refuses an existing original journal root before it can initialize a new ledger.
Continue an existing campaign only through a separately verified custody-preserving migration
that retains every snapshot, transition, envelope binding and accumulated time.
Selecting an explicit root does not make incompatible records valid or grant execution authority.

Historical tags and artifacts retain their original identities and custody.
They do not qualify a later source.
The runner creates no release tag and performs no publication.
Fresh qualification uses the separately authorized [operator lifecycle](reactor-operator.md).
The external selection binds those later acts without changing the qualified source.

## References and research

The [release runner](../scripts/release_check.py) owns the packet and artifact construction.
The [environment guide](environments.md) and [test profiles](testing.md) own their actual prerequisites.
The [release selection contract](release-status.md) owns the final evidence join and stage conditions.
The [paper build guide](paper-build.md) owns publication-asset consistency and visual review.
The [scientific integrity guide](scientific-integrity.md) states the separate scientific and authority boundaries.
