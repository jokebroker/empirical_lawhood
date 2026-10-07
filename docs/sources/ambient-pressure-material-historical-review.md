# Historical ambient-pressure material design and data review

SPDX-License-Identifier: CC-BY-4.0

This review gives information about the historical ambient-pressure material program and its separate stages.
It distinguishes solver qualification, method development and incomplete material controls.
Use it to locate the original evidence limits before proposing new material work.

The ambient-pressure 300 K superconductor program has four distinct scientific roles.
They are Pb solver inspection, staged material design, synthetic transverse-response method development and five-material controls.
Current material tasks use the scientific names in the linked experiment guides.
Historical record identities and the detailed source crosswalk remain in their original custody.
This review is a research starting map.
It does not supply an executable material candidate or establish reproduction.

## Scientific design that preceded material outcomes

The program tests three independent questions:

- Does a finite supported transformation path exist in the entered material graph (`H_PATH`)?
- Does a response-law exploration policy beat equal-budget comparators (`H_SEARCH`)?
- Does a conventional phonon-mediated candidate pass every declared ambient-pressure 300 K material and receiver gate (`H_TARGET`)?

The target
operating point is 300 K and 101,325 Pa. The staged material design specifies a 10⁻⁴ T weak magnetic probe and a 1 mm slab.
It requires an 86,400 s ambient hold.
The process-history ceiling is 1 GPa, 2,000 K and 604,800 s. A pressure
assisted process does not change the ambient **operating** pressure.

The physical independent unit is one realized material preparation: composition,
prototype family, ordering or defect state, process history and ambient-pressure
relaxed structure. Atoms, bands, k/q points, temperatures, solver jobs and
numerical views are nested coordinates. Material families are grouped before
the control, development and sealed prospective material split. The finite
action chart includes substitution, stoichiometry, vacancy/interstitial,
ordering, polytype, strain, process-history and observation-refinement moves.

Requested, accepted and realized structures remain distinct. The receiver
intersects stability, metallicity, pairing, signed transverse response,
finite-body shielding, preservation and synthesis reachability. Scalar `Tc`
alone is not an admission rule. A missing or mismatched operand yields `HOLD`
or `UNEVALUABLE` before material promotion.

The initial design freezes ten initial development actions.
It assigns two waves of two actions per policy, totaling four actions and eight compute units per policy.
Each wave permits one family restart and bridge action.
At most two high-fidelity survivors remain. The three
policies are response-guided, scalar-predicted-`Tc` and stratified random. Those
are pre-material-outcome design choices.
The initial design-readiness failure was repaired in the later synthetic method amendment.
Those method choices followed the initial inspection and preceded all development and prospective material target contact.
Later control workflows and observation choices were revised after earlier control outcomes.
A fresh experiment must identify them as later choices.

## Historical stages and actual data

| Stage | Source implementation and operator role | Data and observed ceiling |
|---|---|---|
| Excluded Pb solver inspection | Pb QE/EPW source inspector, fixed `pw.x` executor, QE output extractor and solver authoring/registry/runtime. Its historical operator bound frozen config, local source assets, authority and a standard candidate. | The first attempt chose provisional SCF accuracy and stopped. The next attempt corrected that rule. One excluded Pb SCF component then passed: 14 electrons, eight iterations and final 6.7×10⁻¹³ Ry accuracy below 10⁻¹² Ry. It did not calculate phonons, pairing, `Tc` or transverse response. |
| Initial staged material design | Source qualification, material roster, search/science design and three-task audit. Its historical operator supplied frozen source and issue identities. | A 14-asset, 39-pseudopotential roster and finite design were inspected with zero target contact. Design freeze refused because the development template, providers and gauge-covariant material-response producer were then missing. |
| Synthetic transverse-response method amendment | Strict gauge-covariant method fixtures, compatibility maps, eleven-capability staged graph and conditional material-response provider. Its historical operator bound the amendment and source extension. | Nine synthetic method fixtures passed, including wrong-sign, normal-cancellation, finite-q, Ward and view-disagreement refusals. The provider required same-preparation Wannier/pairing/unit/validity operands. Material control returned `SOURCE_OPERAND_REQUIRED`. Material result count was zero. |
| Five-material controls and later tutorial audit | Five-material control producer, QE/PH/EPW workflow profiles, raw HDF5/tar envelope and Wannier/EPW material-response reducers. Its historical operator bound frozen control archives, local environment, authority and its source-bound candidate route. | The partial control run recovered Cu and diamond negative classes. Pb, MgB₂ and unstable compressed Pb were unevaluable after workflow failures. The later audit paused after authenticated Pb/MgB₂ tutorials and before the third control workflow. It issued no terminal material-control science freeze or fresh material experiment. It contacted no development or prospective material target. The two SSSP views per material are nested, not ten independent controls. |

The authenticated later tutorial products are useful **outcome-visible reference
data**, not fresh controls. They contain Pb/MgB₂ phonon, Wannier Hamiltonian and
low-temperature pairing output, but neither `pb_r.dat` nor `mgb2_r.dat`. The
MgB₂ orbital-local projection exceeds its fixed validity limit. The archived
pairing solutions cannot supply a 300 K material state or a same-gauge
position-operator panel.

The later tutorial audit also found an extractor error: a peak in
MgB₂'s `alpha2F` spectral column was reported as `lambda ≈ 0.8923162`. The
first integrated `2 alpha2F / omega` endpoint is **0.5729519**. The retained
[material-control reducer](../../src/empirical_lawhood/adapters/simulators/ambient_pressure_superconductor/material_control_solver.py)
now selects and verifies the integrated endpoint and rejects ambiguous tables. The [focused check](../../tests/test_material_epw_coupling.py) uses an independent
integral and malformed layouts.

The optional
[authenticated-reference check](../../tests/test_material_mgb2_authentic_reference.py)
also verifies the historical MgB₂ bytes when an authorized holder supplies
them. This target correction does not revise any issued source result.

## Researcher continuation boundary

The target retains the four scientific adapter roles, but the historical
scripts were source-bound issue wrappers. The separate [synthetic lattice-pairing method guide](../../experiments/lattice-pairing-method/guide.md) now binds one disclosed synthetic suite to strict extension authoring.
The same suite has a selected fixture producer and attached evaluator through the single CLI. That method-only path does not qualify the
distinct Pb SCF solver, staged material design or material-control route. A
researcher cannot compile a fresh material candidate by substituting a file
into the old config.

Pb SCF solver qualification needs an independently qualified QE/EPW environment and Pb input.
Staged material design needs fresh source and roster identities.
Material control needs all five independent structures, exact source/control archives and a complete compatible same-gauge material operand set. The accepted format,
native-to-SI maps, validity gates and missing-member refusal are detailed in
the [operand assessment](ambient-pressure-material-operands.md). A new material-control act
must rerun all five material units. Neither historical run can be pooled into it.

The material program is deferred for the first release.
Prospective use requires fresh authoring, a selected provider, no-effect production proof and the complete control-source contract. Its method fixtures and historical tutorial diagnostics
make no 300 K superconductivity, admission, controller-use or manufacturing claim.

## References and research


Source versions, observed byte inventories and scientific limits are stated beside each claim above.
The [program source register](../../paper/SOURCES.md) identifies project results whose primary records are not deposited.
