# Ambient-pressure material operand assessment

SPDX-License-Identifier: CC-BY-4.0

This assessment gives the requirements for material operands of the retained transverse-response reducer.
The available tutorials do not qualify the material-control intersection.
Use it to inspect a permitted source and identify the first missing operand.

This is a source qualification assessment. The separate
[five-material held-input guide](../../experiments/material-control-inputs/guide.md) now supplies a
strict one-CLI wrap/preflight path, but no runnable material-control result.
The separate [historical design and data review](ambient-pressure-material-historical-review.md)
maps Pb solver inspection, staged material design, the method amendment and material controls to their original scientific roles.
The retained staged material graph includes a conditional transverse-response provider.
The material-control route includes a Wannier/EPW material-response reducer. The separate
[fresh synthetic method suite](../../experiments/lattice-pairing-method/guide.md) now has a selected
generated binding and strict candidate single-CLI route, but it carries no qualified
material operand. Neither staged material design nor material control is selected
by a fresh target material candidate or no-effect production route yet.

## Public formalism specifications

The [formalism declarations](../../src/empirical_lawhood/adapters/simulators/ambient_pressure_superconductor/material_source_design_design.py) use the source label `source.plan.uniform-electron-gas-transverse-receiver-screen`.
This label identifies those retained declarations.
It is not a material archive, source qualification or operator grant.

| Declared formalism | Public implementation | Evidence boundary |
|---|---|---|
| `formalism.gauge-closed-transverse-kubo` | [Signed-current SI reduction](../../src/empirical_lawhood/adapters/simulators/uniform_electron_gas_response/physics.py) and its [analytic panel check](../../src/empirical_lawhood/adapters/simulators/uniform_electron_gas_response/analytic_science.py). The [multiband material method](../../src/empirical_lawhood/adapters/simulators/ambient_pressure_superconductor/multiband_strong_coupling_response.py) gives the normal-referenced, gauge-closed finite-q calculation. | The analytic fixture supplies no qualified material operands. The same-gauge Hamiltonian, self-energy and position/projection checks stated below are necessary for material use. |
| `formalism.finite-slab-receiver-admittance` | The [slab profile and inverse](../../src/empirical_lawhood/adapters/simulators/uniform_electron_gas_response/physics.py) and [analytic panel check](../../src/empirical_lawhood/adapters/simulators/uniform_electron_gas_response/analytic_science.py). | The analytic receiver map supplies no independent material measurement or control qualification. |

These links give the project specifications and implementations.
They do not publish an earlier private plan or supply primary literature support.
The material operands, fresh control cohort, selected binding and separate authority are prerequisites.

## What the available sources establish

- The [official EPW superconductivity tutorial](https://docs.epw-code.org/tutorials/tutorial_04/index.html)
  supplies Pb and MgB2 input recipes for phonons, Wannierisation and
  isotropic/anisotropic pairing. Its input bundle is a starting point for a
  numerical calculation, not a gauge-closed material transverse-current
  panel or a 300 K pairing state.
- [Wannier90 documents](https://wannier90.readthedocs.io/en/stable/user_guide/wannier90/files/)
  `seedname_hr.dat` and `seedname_r.dat` as distinct Hamiltonian and position
  matrices. `write_rmn = true` is required to emit the latter. A Wannier center
  printed in a log cannot replace the off-diagonal position matrix or its
  omission bound.
- The archived later tutorial runs authenticate Pb/MgB2 Hamiltonians,
  phonon output and low-temperature pairing products. Their complete raw
  member inventories have no `pb_r.dat` or `mgb2_r.dat`. The MgB2 archive has
  an EPW `egnv` shell and Wannier checkpoint/overlap files, but lacks the
  `.eig` eigenvalue file needed for a documented Wannier90 interpolation
  restart. The existing orbital-local projection does not pass its fixed
  validity limit on that archived run. These are historical,
  outcome-visible diagnostics and cannot be reused as a new control cohort.
- The authors' [Materials Cloud EPW v4 archive](https://archive.materialscloud.org/records/a6p4k-eh221)
  was checked at its published MD5. Its 81-member, 1.8 MiB tar supplies
  MgB2/Pb calculation inputs and plotting scripts, but no `*_hr.dat`,
  `*_r.dat` or anisotropic gap output. It is a reproducibility recipe, not a
  compatible material-response operand bundle. A separate
  [Materials Cloud superconductivity archive](https://archive.materialscloud.org/records/0wv1s-yqg46)
  advertises EPW calculations and anisotropic superconductivity outputs. The published 2.3 MiB manual anisotropic ZIP matched its published MD5.
Its complete 408-member listing includes gap outputs and Wannier inputs, with no `*_hr.dat` or `*_r.dat`. The newer
  [AiiDA export](https://archive.materialscloud.org/records/0kqen-w8h90)
  indexes numerical repository objects by hashes. The 7.8 GiB EPW archive has no qualification against the same-gauge position, pairing and five-control contract. Even a compatible operand inside it would need a new
  material/preparation and control qualification. The catalogue is not a material-control science
  freeze or a substitute for the owner's fixed control intersection.
- The earlier partial five-material control run recovered only its Cu and diamond
  negative controls. Its positive and unstable controls were unevaluable
  after source-workspace failures. The later audit stopped after two tutorial workflows.
  Neither issued a complete material-control science freeze or contacted a target material.
- The later outcome-visible audit also exposed a reducer error: the old EPW parser
  selected an `alpha2F` spectral peak as MgB2's coupling. The target reducer
  now requires EPW's paired spectral/cumulative table and matching footer.
  It returns the first integrated endpoint (`0.5729519` on the authenticated
  MgB2 tutorial archive) or refuses an ambiguous table. This repairs a
  scientific observation rule for a *future* target experiment. It does not
  reclassify either historical run or supply the missing transverse operands.

The [method-amendment source trace](../substrate-route-register.md) distinguishes two historical development states.
The initial design inspection lacked a producer.
The later amendment supplied a conditional method and provider. **Qualified material operands**
and a complete control intersection remain absent. Analytic method fixtures,
low-temperature tutorial gaps and the excluded Pb SCF solver check do not measure a
material-specific 300 K transverse response.

An operator who already has lawful access to the authenticated historical MgB2
`raw-workflow.h5` may run the optional
[native reference check](../../tests/test_material_mgb2_authentic_reference.py) with
`EMPIRICAL_LAWHOOD_MATERIAL_MGB2_REFERENCE_H5=/absolute/path/to/raw-workflow.h5 uv run --no-sync pytest -q tests/test_material_mgb2_authentic_reference.py`.
It verifies the EPW shell/Wannier alignment, the two-gap scale in native eV
and the missing-position/projection refusal. Without that external input it
skips. It is a historical diagnostic, not a clean-checkout quick-start pass.

## Required researcher input and present stop

Supply a legally usable source bundle under separate custody with these records:

- Exact material/preparation identity.
- QE/EPW and Wannier versions.
- Native lattice, reciprocal and SI maps.
- Same-gauge `*_hr.dat` and `*_r.dat`.
- Finite-temperature Eliashberg self-energy, or the EPW gap and fine-shell operands needed to derive it.
- Phonon spectra.
- Requested, accepted, applied and realized numerical preparation.
- Receiver clock and finite-q transverse-current conventions.
- Resource bounds.
 The
positive Pb/MgB2 controls need their declared temperatures, and a claimed
300 K material requires its own 300 K pairing state. Two SSSP numerical views
remain nested within each of five independent control material structures.

Before promotion, the adapter must validate these conditions:

- Energy and gauge alignment.
- Orbital-local projection and position-operator omission.
- Matsubara convergence and the Migdal proxy.
- Finite-q/current sign and normal-state subtraction.
- The control intersection.
 A failure or missing member is an
unevaluable validity stop, not zero stiffness. The current material-control reducer accepts
a bounded `tar.gz` raw-workflow archive and requires exact member names. The
R11 tutorials stop at the missing position matrix. The provider also needs a
fresh target-owned authoring/selected-binding path and no-effect production
check before this route can support a prospective campaign. That integration
is deferred for the first release. The material-control input CLI can wrap a
researcher-supplied bounded raw tar and validate ten HDF5 envelopes. It does not
qualify their source, execute a solver, reduce material response or select a
candidate. Solver qualification and staged material design have no corresponding fresh material authoring path yet.

Running the official tutorial with `write_rmn` and a POSIX scratch filesystem
is a plausible way to *generate* some missing control operands. That is a
proposed new calculation, not a recovered or already qualified source. It
would still require independent material validity checks, all five controls,
fresh custody and the target candidate/provider integration above. No
superconducting material, ambient-pressure 300 K target or material admission is
established by this assessment.

## References and research

EPW Developers (2026), [*Superconductivity tutorial*](https://docs.epw-code.org/tutorials/tutorial_04/index.html), provides the Pb/MgB2 input recipes.
Wannier90 Developers (undated), [*Files*](https://wannier90.readthedocs.io/en/stable/user_guide/wannier90/files/), defines the separate Hamiltonian and position formats.
Samuel Poncé, Elena Roxana Margine, Carla Verdi and Feliciano Giustino (2020), [*EPW: Electron-phonon coupling, transport and superconducting properties using maximally localized Wannier functions*](https://doi.org/10.24435/materialscloud:tf-kf), version 1.
Marnik Bercx et al. (2025), [*Charting the landscape of Bardeen-Cooper-Schrieffer superconductors in experimentally known compounds*](https://doi.org/10.24435/materialscloud:c8-gs), version 1.
The same authors' [version 4 archive](https://doi.org/10.24435/materialscloud:x2-ap) contains the separately listed AiiDA export.
Source versions, observed byte inventories and scientific limits are stated beside each claim above.
The [program source register](../../paper/SOURCES.md) identifies project results whose primary records are not deposited.
