# Five-material held-input preflight

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for envelope validation of five held material controls.
It counts two numerical views per structure without creating ten independent materials.
Use it to wrap permitted solver outputs and identify missing material operands.

The [strict target input](control-inputs.json)
selects five independent control material structures under two SSSP numerical views.
It names **five independent controls and ten nested workflow outputs**, not
ten independent materials. The fixed profile IDs, structures, classes and
views come from the retained five-material control design, frozen before that control run produced outcomes. The `empirical-lawhood-` config identity and `controls/<profile>/`
staging layout are new target-owned input conventions. No old run ID, result,
authority or source hash is used as a prospective default.

This is a read-only source-input boundary. It does not run QE/PH/EPW, qualify
a material, calculate transverse response, freeze material-control science, compile a candidate or issue a
campaign. The [synthetic lattice-pairing method guide](../lattice-pairing-method/guide.md) has a different evidence
ceiling. Its nine fixtures do not qualify material controls.

## Read results and failures

Read `independent_control_structures`, `nested_numerical_views`, `input_envelopes_checked` and `profiles`.
`material_operands_qualified` remains false: member presence is not physical validity or custody qualification.
Raw wrapping requires a fresh output; the preflight executes no material solver.

Read [the common result and failure guide](../../docs/results-and-failures.md) for retained attempts,
stdout/stderr and same-identity issued recovery. The route-specific outputs and stops below remain binding.
Use `workflow show material-control-inputs --format json` for static command effects and prerequisite roles.
Discovery does not inspect those prerequisites.

## Supply an accepted raw output

For each `workflow_profile_id`, supply the raw **solver-output** `tar.gz` for that exact control structure and SSSP view.
Supply its own source custody, solver version and numerical history. The
retained workflow profile uses QE 7.6/EPW 6.1, eight CPU cores, at least 24 GiB
of memory and a network-disabled local scratch environment. The archived
source executor caps a raw archive at half its 8–64 GiB output budget. This
target input path bounds one raw tar at 32 GiB, one member at 2 GiB and total
expanded members at 64 GiB.

Output member names must be safe and unique. For positive Pb/MgB2 controls, the tar requires one same-gauge `<prefix>_hr.dat` Hamiltonian and separate `<prefix>_r.dat` position matrix.
It also requires the material reducer's pairing, shell and phonon outputs. An EPW gap or a Wannier center in a log cannot replace the position matrix. The other three controls require their native SCF or phonon-stability output
for later scientific reduction.

This preflight validates only the envelope and member presence. It does not certify the output's physical validity or custody.

Place the ten independently obtained raw tar archives in a held source
directory outside the repository. Wrap each with the **one installed CLI**,
selecting its exact profile from the config. For example, for the efficiency
view's MgB2 control:

```sh
uv sync --locked --python 3.11.14
mkdir -p /absolute/held/material-controls/controls/workflow.material-control-pbe-efficiency-base-calibration-mgb2-alb2
uv run --no-sync empirical-lawhood campaign material-workflow-wrap-raw \
  --archive /absolute/held/material-controls/mgb2-efficiency/raw-workflow.tar.gz \
  --profile-id workflow.material-control-pbe-efficiency-base-calibration-mgb2-alb2 \
  --output /absolute/held/material-controls/controls/workflow.material-control-pbe-efficiency-base-calibration-mgb2-alb2/raw-workflow.h5
```

The output path must be fresh. Repeat for all ten profile IDs, using the
relative `archive_relative_paths` in the checked-in config or a strictly
decoded copy with the same one-to-one profile roster. Then run:

```sh
uv run --no-sync empirical-lawhood campaign material-workflow-input-check \
  --config experiments/material-control-inputs/control-inputs.json \
  --source-root /absolute/held/material-controls \
  --output-dir /absolute/development/material-control-inputs/attempt-001
```

The CLI validates the exact HDF5 schema, profile, one-byte-stream dataset and
logical tar SHA-256. It reports five structures and ten views only when every
held file is present and its envelope matches. From a clean target checkout,
the shipped config refuses at the first missing held file before solver or
material contact. The public repository distributes no QE output, SSSP
pseudopotential, material archive, source authority or exposure-free control roster. Obtain and license those inputs separately. Do not reuse R10/R11
outcome-visible products as a prospective control set.

The [scientific checks](../../tests/test_material_epw_coupling.py) independently
integrate an EPW `alpha2F` example and reject a spectral-peak substitution.
The [held-input check](../../tests/test_material_control_input_preflight.py) loads the shipped config and validates the five-versus-ten denominator.
It rejects absent outputs, wrong profiles and changed logical bytes.
It requires a distinct `mgb2_r.dat` before positive-tar wrapping. The optional
[authenticated tutorial check](../../tests/test_material_mgb2_authentic_reference.py)
exercises native MgB2 units and the missing-position/projection refusals when
the external R11 reference is supplied. It is a historical numerical check,
not a prospective control.

Pb SCF qualification and staged material design need their own source and authoring contracts.
Material control further needs all five independently qualified control
panels, same-gauge material operands, a checked response reduction and a
fresh selected candidate/provider path. Material preparation, measurement
conformance, order-relation and response would be relevant after those inputs
exist. Local law, admission and controller-use claims remain outside
this input preflight. Missing position matrices, failed local projection, ambiguous EPW coupling and incomplete control intersection are unevaluable stops.
They establish no zero stiffness or 300 K claim. The [historical review](../../docs/sources/ambient-pressure-material-historical-review.md)
and [operand assessment](../../docs/sources/ambient-pressure-material-operands.md) trace the
source limits.

## Obtain the workflow inputs

Obtain this bundle from the selected repository checkout or source distribution. The guide and study inputs are repository workflow files, not wheel resources. Run each command from the checkout root in an unactivated shell. The relative paths use that root.

Keep generated data and receipts in the declared external storage. A wheel can perform its advertised inspection and native-input checks with absolute paths. Source-bound authoring and issue require a clean checkout that executes its own tracked package.

## References and research

Wannier90 Developers (undated), [*File formats*](https://wannier90.readthedocs.io/en/stable/user_guide/wannier90/files/).
The documented Hamiltonian and position-matrix formats are separate operands. The [material assessment](../../docs/sources/ambient-pressure-material-operands.md) records source-specific limits.

Gareth Seneque (2026), [*Empirical Lawhood*](../../paper/manuscript.md), edition 0.60, gives the scientific formulation.
The [program source register](../../paper/SOURCES.md) identifies historical results and unavailable primary records.
The linked scientific source modules and existing tests establish only their declared implementation and development boundaries.
