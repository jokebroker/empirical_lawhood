# CI and selected profiles

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for hosted verification jobs and their input requirements.
It gives the same exact-source software gate used for local release checks.
Use it to select a job and retain its complete evidence packet.

[Release checks](../.github/workflows/release-check.yml) runs
`portable-release-check` on pushes and pull requests using Ubuntu 24.04 x86-64,
CPython 3.11.14 and uv 0.11.28. It calls the same clean-source release runner used locally. The runner validates the locked environment, portable tests, scoped coverage, static errors, generators and documentation. It builds through the source archive and examines the installed wheel outside the checkout.

The portable job selects the [frozen RC numerical profile](environments.md#frozen-rc-numerical-input-profile).
It retains exact original descriptor and fibre hashes.
The runner verifies those exposed inputs, generators and document targets before expensive tests.
A profile mismatch or source drift therefore stops early.
The complete portable test selection remains required in CI.
These software checks confer no scientific qualification.

A 30-day artifact retains manifest, status, logs,
coverage and artifact hashes. Download and preserve the complete selected packet
before CI retention expires. Failed checks upload their available diagnostic
packet. Artifact existence alone is not success.

`workflow_dispatch` always runs portable and separately selects either
`native-open`, `native-brian2`, `held-reactor` or `held-response`. Unselected jobs are
not evidence. Native profiles install their declared stacks and must fail if a
selected prerequisite is absent. Held profiles run only on a maintainer-managed
Linux x86-64 runner labeled `empirical-lawhood-held` through the `held-inputs`
environment.

Before you enable the job, configure environment reviewers and permitted read-only sources.
Before you enable the job, configure the actual guarded mounts, native software and explicit input variables documented in [testing](testing.md). Never run an untrusted
pull request on this runner. Held packets remain on the permitted runner for
review. No raw sources, outcomes, keys or private logs are uploaded publicly.

A queued or absent held runner is pending, not a green skip or qualification.

An owner-approved upstream push and a passing run on the exact candidate are prerequisites.
A maintainer can then select `portable-release-check` as a required branch-protection or ruleset check. Require it on the protected release branch.
Where appropriate, disallow bypasses.
Make sure that the configured protection blocks a deliberately failing proposed change. Workflow YAML does not configure protection. No hosted run or service
change is claimed by committing it. Action references are pinned commits.
Review updates using the official [checkout](https://github.com/actions/checkout),
[setup-uv](https://github.com/astral-sh/setup-uv) and
[artifact](https://github.com/actions/upload-artifact) documentation.

Budget hours for the complete portable job. The pre-finish local run described
in [testing](testing.md) took 2 h 35 min before artifact verification. It ended
with two stale CLI-fixture failures; its large-case passes retain their original
source identity, and it is not a passing final-source gate. The portable CI job
permits 240 minutes for the full suite, cold downloads and artifact checks.
Hosted duration remains unmeasured; this budget does not guarantee a hosted pass.
Allow several GiB for cached packages, source, test and wheel environments.
Optional native stacks need substantially more.

The cache keys use `uv.lock`.
A warm cache reduces downloads but never relaxes `--locked` or checks. CI uses
network installation. `--offline` is only for a separately verified complete
local cache. Held science has no universal duration/storage promise: inspect
the exact native resource envelope before authorization.

## References and research

The linked source modules, command metadata and existing checks own the implemented behavior described here.
The [scientific integrity guide](scientific-integrity.md) defines its separate evidence and authority boundaries.
The [program source register](../paper/SOURCES.md) identifies bounded historical results and unavailable primary records.
