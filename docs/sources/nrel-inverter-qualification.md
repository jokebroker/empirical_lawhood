# NREL inverter source qualification boundary

SPDX-License-Identifier: CC-BY-4.0

This assessment gives information about the missing intervention metadata in published inverter archives.
It records the deferred physical route and its precontact decoder requirements.
Use it to distinguish measured traces from a qualified action-stage journal.

Prospective NREL physical-inverter work is outside first-release scope.
The retained historical independent-recurrence physical-inverter adapter requires
a logged intervention, not only electrical measurements. Its
`NRELSafeDecodeProfile` requires 23
distinct, named roles before opening a source member. Those roles include a
preparation and run identity, requested/accepted/applied/realized action and
their clocks, AC/DC power with uncertainty, and trip/saturation/validity
flags.

One complete preparation is the independent unit. Samples, phases and
mode epochs within it are nested observations. The archive must be held
offline, bound to a terms and acquisition receipt, and have an explicit
outcome-visibility disposition. The current adapter deliberately refuses a
profile with missing roles before source contact.

The [official 2024 NREL/NLR mode-transition submission](https://data.nlr.gov/submissions/253)
(DOI 10.7799/2483511) is a fresh public reference compared with the older
OpenEI 8255 archive. It includes one 91.9 MB CSV and experiment documentation. The documentation describes a 150 kW three-phase fuel-cell inverter, a DC
source, grid-connected and islanded operation, and eight measured electrical
signals. A bounded read of the official CSV found ten unlabeled numeric
columns.

The documentation gives approximate mode epochs, but no
time-aligned request, acceptance, application and realization journal or
independent reset/preparation ledger. The [2023 PV inverter version 2
submission](https://data.nlr.gov/submissions/217) and [2024 fuel-cell
submission](https://data.nlr.gov/submissions/251) likewise did not supply
those typed roles in the checked resources. A mode label, inferred edge or
measurement column cannot be relabeled as an action stage.

These public releases support exposed observational analysis only. They do
not supply a qualified native independent-recurrence physical action source, a prospective
evaluation unit, or the selected provider and strict candidate context. No
target command advertises them as such. The [catalogue license](https://data.nlr.gov/node/253/license)
permits use or copying subject to its full notice and credit terms. It does
not add the missing experimental metadata.

For a future physical route, obtain a UTF-8 CSV/TSV logged-intervention table from the source operator.
Obtain its separate source dossier. The table needs the exact 23 roles above, headers,
units and a preparation selector. The dossier needs these records:

- Apparatus/site and reset ledger.
- Timebase and calibration.
- Action-code meanings.
- Receiver derivation and uncertainty.
- Safety/trip/saturation rules.
- Independent-unit grouping.
- File digest and size.
- License/access terms.
- Custody/visibility receipt.
 A new adapter-owned selector, candidate context and executable
provider would then bind that dossier through the single CLI. This public
checkout has no such command. The present decoder stops before source contact.
The descoped route and its attached consistency method contribute no first-release
starter, provider or intervention result.

## References and research

Nils Nemsow et al. (2024), [*Fuel Cell Inverter Transition Between Modes of Operation (Grid-Forming and Grid-Following)*](https://doi.org/10.7799/2483511).
Kumaraguru Prabakar et al. (2023), [*PV Inverter Experimental Dataset Version 2 with 100 Percent Power*](https://doi.org/10.7799/2205350).
Kumaraguru Prabakar et al. (2024), [*Fuel Cell Inverter Dataset*](https://doi.org/10.7799/2472828).
The official catalogue metadata identifies the versions and creators.
The [catalogue license](https://data.nlr.gov/node/253/license) defines its credit and use terms.
The checked releases' missing intervention roles are explained above.
The [program source register](../../paper/SOURCES.md) identifies project results whose primary records are not deposited.
