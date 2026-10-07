# MAST-U public voltage source assessment

SPDX-License-Identifier: CC-BY-4.0

This assessment gives information about the missing physical semantics of released tokamak voltage arrays.
It keeps supply association, diagnostic current and realized actuator voltage distinct.
Use it to identify the source metadata required before a new consuming route.

The held 26M5-EY02 trace and JMPP-4C57 machine configuration are separate
public source objects. Their archive inventories establish selected object
bytes and path provenance. They do not establish a calibrated realized-voltage
coordinate for the proposed voltage-law donor. Its historical source gate
stopped `SOURCE_SEMANTICS_UNEVALUABLE` before issue or protected shot contact.
The target has no typed trusted conversion, qualified post-hoc import, selected
provider or one-CLI no-effect route for this source.

The [UKAEA PF-active mapping overview](https://ukaea.github.io/IMAS_MASTU_mappings/mappings/pf_active/overview/)
associates the release's `/XCM/.../VOUT` and `/XCM/.../VOLTS` path family with
named power supplies. The [detailed mapping](https://ukaea.github.io/IMAS_MASTU_mappings/mappings/pf_active/mappings/)
uses independently timed current data. It maps no supply-voltage data field. It specifies no calibration, positive direction, validity flags or exact sensing point for a released voltage array.

The [UKAEA controller assessment](https://scientific-publications.ukaea.uk/wp-content/uploads/UKAEA-CCFE-PR25362.PDF) compares simulated and measured power-supply output voltages after voltage requests in vacuum discharges.
Section 3.3 gives this comparison.
Figure 5(b,c) gives D1 and P5 coil power-supply outputs in volts.
The report does not authenticate those values as the same shot-specific arrays in 26M5 or provide their missing per-channel metadata.
A supply association and a figure caption cannot give that byte-specific qualification.

The FreeGSNKE authors' public [MAST-U loader](https://github.com/FusionComputingLab/freegsnke/blob/678ff6b49246b8d36003750237d2fa6757b7f6bf/freegsnke/mastu_tools.py)
separately reads the XCM `VOUT`/`VOLTS` family as coil supply output and the
XDC `PF/F` family as requested voltage. Its XCM loader preserves the UDA
`units` field, which the held release did not serialize. The loader's explicit
two-channel sign correction is applied in its XDC requested-voltage
transformation. Copying that correction onto the released XCM arrays would
confuse action stages. The code narrows the likely source role but does not
recover the lost per-shot metadata, calibration or synchronized physical
coordinate from the selected release bytes.

To reopen this route, provide these inputs:

- Exact release and shot custody.
- Permission for trusted local conversion of the selected pickle members.
- A version- and path-specific authoritative signal dictionary or signed maintainer record.
- A circuit map compatible with the FreeGSNKE receiver.

The signal record must specify these meanings:

- Physical unit, polarity and request-versus-sensed role.
- Measurement point, calibration and uncertainty.
- Quality/status rules, timebase and synchronization.
 Conversion must be bounded, typed and outside the public Git/wheel
archives. The target must validate converted scientific values on an authentic permitted reference.
It must bind a fresh source/consumer candidate and prove the exact no-effect authorization boundary. No existing CLI command supplies
that substitution today.

The retained processed EFIT `currents_input` contract is a different source
role: a 23-channel diagnostic constraint target in amperes, excluding `pc`.
It has no requested, accepted, applied or realized actuator-current journal.
It cannot repair the voltage donor by changing the action label, and it needs
its own qualified observational consuming route.

## References and research

- UKAEA / Fusion Computing Lab (undated where no year is stated), [UKAEA PF-active mapping overview](https://ukaea.github.io/IMAS_MASTU_mappings/mappings/pf_active/overview/). The source role and limits are stated above.
- UKAEA / Fusion Computing Lab (undated where no year is stated), [detailed mapping](https://ukaea.github.io/IMAS_MASTU_mappings/mappings/pf_active/mappings/). The source role and limits are stated above.
- Lvovskiy, A. et al. (2025), [*Framework for Assessment of Magnetic Equilibrium Controller Performance on the MAST Upgrade Spherical Tokamak*](https://scientific-publications.ukaea.uk/wp-content/uploads/UKAEA-CCFE-PR25362.PDF), UKAEA-CCFE-PR(25)362. Section 3.3 and Figure 5 give the comparison stated above.
- UKAEA / Fusion Computing Lab (undated where no year is stated), [MAST-U loader](https://github.com/FusionComputingLab/freegsnke/blob/678ff6b49246b8d36003750237d2fa6757b7f6bf/freegsnke/mastu_tools.py). The source role and limits are stated above.

Source versions, observed byte inventories and scientific limits are stated beside each claim above.
The [program source register](../../paper/SOURCES.md) identifies project results whose primary records are not deposited.
