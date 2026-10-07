# Scientific integrity boundaries

SPDX-License-Identifier: CC-BY-4.0

This reference gives the evidence and authority boundaries of the scientific engine.
It keeps measurement, response-law qualification, admission and controller use separate.
Use it to interpret a result within its declared scientific scope.

A response law is qualified for a declared denominator, action chart,
receiver, horizon, support and use. An operationally successful command does
not make a scientific result positive or transferable.

## Evidence and independent units

`kernel/worlds.py` distinguishes analytic reference, numerical simulator,
hardware in loop and physical experiment worlds. A transition between worlds
requires a declared rule and independent validation before it can promote
evidence. `EvidenceUnitScope` separates a physical independent unit from a
nested numerical view. Repeated views or solver resolutions do not silently
increase the independent sample count.

Evidence ceilings and outcome access
follow records through planning and runtime. Exploratory outcome-visible work
is nonpromotable. A fresh confirmatory target needs its own prospective plan.

## Causal timing and action delivery

Planning records information cutoffs and prospective obligations before
results are available. Native adapters distinguish requested, accepted,
applied and observed delivery, and retain what actually happened. The kernel
models action and receiver semantics. A simulator or instrument adapter may
report a refusal, failed source qualification, missing measurement or
incomplete delivery. Those are recorded states, not replacement successes.

## Authority, custody and admission

Outcome-blind candidate compilation grants no authority.
Issue verifies the candidate, source closure, proposer attestation and custody authority.
The issued manifest and publication receipt supply the scientific approval proposal.
Package assembly requires scientific approval and separate execution authority.
Execution and recovery validate the issued package, source identity, execution resource envelope, guarded storage and receipts. Reveal
and adjudication have separate authority. `adapters/control/`,
`runtime/controller_*` and the prospective services keep reachability,
controller admission and controller use distinct from law measurement. An admission method
or controller-use result cannot be inferred from an adapter import or a
successful demonstration command.

Negative, mixed, held, unsupported and unevaluable results remain first class.
`doctor` and the capability commands report environment and static binding
state only. `EXECUTABLE_INPUTS_REQUIRED` means source data, native software or
other declared inputs remain to be supplied. No descriptor is promoted to
ready by discovery. For the exact prerequisites and exit codes, see the
[CLI reference](cli.md).

Fresh approval and response-algebra authority entry points, and persisted response-algebra conformance authorizations, use the
[strict kernel UTC parser](../src/empirical_lawhood/kernel/time.py): valid calendar dates, seconds, optional one-to-six digit fractions and a final `Z`.
Direct construction and canonical decoding now refuse conformance timestamps outside that grammar.
Persisted approval records retain their calendar-valid legacy parser for identity-bearing signed strings,
including omitted seconds, compact spellings and comma fractions. Replay preserves their original bytes and signatures.

## Current verification ceiling

The portable gate tests synthetic issue, execution, recovery, bounded six-matrix custody and joined consumers. It tests known-exposure refusal, canonical records, SQL, CLI effects and source-identity refusal. It also examines generated bindings, the installed wheel and the native reactor example. Its scoped coverage report records missing decisions.

Subprocess-only native
paths are not counted as covered by the parent process. Optional-native and held
profiles are separately selected. A passing gate does not certify every retained
adapter or scientifically qualify the final source. The authoritative
[release status](release-status.md) binds the selected commit, profiles and
artifacts and distinguishes historical qualification from final-source work.

The paper's primary native archives and receipts are not deposited.
[its source register](../paper/SOURCES.md) identifies that limit. New qualification
needs its own reviewed unexposed roster, permitted custody and exact-source
receipts. A human attestation cannot promote a known exposed reactor assignment:
prospective issue refuses with `OUTCOME_SEPARATION_REQUIRED`.

## Exact Decimal storage

The shared [canonical encoder](../src/empirical_lawhood/kernel/serialization.py) preserves each finite Decimal's numerical value.
It uses exact fixed-point formatting, without arithmetic normalization.
Precision, rounding mode, exponent limits and traps cannot change the bytes or fingerprint.
The encoder does not change the caller's context or flags.
Signed zero and trailing fractional zeros have one canonical spelling.
Their representational detail is not a separate scientific value.

This rule applies to all canonical records, including nested records, authoring exports, archives and provider outputs.
The two history-bundle handoff checks provide additional corruption detection.
They are no longer the only loss-prevention boundary.
The tagged Decimal grammar and record field schemas have not changed.
Values that previously lost digits now produce different bytes and fingerprints.
Bind fresh records to their actual source and bytes.

## Rigorous threshold rank

Receiver-history, split-cohort and phase-diagram use one
[rank implementation](../src/empirical_lawhood/adapters/methods/certified_rank.py).
Its algorithm identity is `arb-inertia-exact-dyadic-v2`.
Current configurations require this identity.
All three families bind its source and the
[endpoint converter](../src/empirical_lawhood/adapters/methods/_arb.py) in their scientific and observer source identities.
Those identities also include the canonical encoder.

For the entered real binary64 matrix A, the mathematical target is:

```text
r_tau(A) = #{i : sigma_i(A) > tau}
tau = max(n_rows, n_columns) * 2^-160 * max(1, ||A||_2)
```

This is threshold rank of the numerical matrix.
It does not establish the symbolic rank or discretization accuracy of a continuous model.
The rank implementation rejects empty, nonfinite and complex inputs.
It supports finite binary64 values, including subnormals and the largest finite value, subject to available computation resources.
Other real numeric inputs first enter binary64 explicitly.

Each family captures a private binary64 matrix before certification.
The certificate and `matrix_sha256` use the same captured values, including when the caller later changes its array.
The standalone certificate entry point also detaches its input from caller-owned memory.
Capture does not promise an atomic observation of concurrent writes during the copy;
callers requiring a coherent external state must keep the input stable while it is captured.

The certificate follows these rules:

1. Convert each entered binary64 value to its exact dyadic rational.
   Arb encloses the smaller row or column Gram matrix G.
2. Enclose the spectral norm between the largest column norm and the Frobenius norm.
   Clamp both norm bounds below by one.
   Directed Arb operations give exact dyadic endpoints `tau_lower` and `tau_upper`.
3. Count positive inertia of `G - t^2 I` with interval arithmetic.
   Each symmetric pivot excludes zero before division.
   A positive pivot contributes one direction under Sylvester's inertia law.
   At `tau_upper`, the count is a lower rank bound.
   At `tau_lower`, it is an upper rank bound.
4. Cap the upper count by exact rational matrix rank.
   This cap never promotes a direction below the threshold.
5. Enclose the residual and compare its upper endpoint against `tau_lower / 4`.
   All sums, products, roots, elimination steps and comparisons remain in Arb.
   No mpmath approximation participates in the certificate.

Let M contain the Gram ball midpoints, and let R contain their nonnegative radii.
For the exact Gram matrix:

```text
|G_ij - M_ij| <= R_ij
||G - M||_2 <= ||G - M||_F <= sqrt(sum_ij R_ij^2)
residual_norm >= (sum_ij R_ij^2)^(1/4)
```

Thus `residual_norm` bounds the square root of the Gram error in operator norm.
It is not a QR reconstruction residual.
Interval elimination separately encloses all errors in the inertia calculation.
The [endpoint converter](../src/empirical_lawhood/adapters/methods/_arb.py) stores the directed dyadic bound as an exact Decimal.
It does not truncate the bound to a fixed number of decimal digits.
`separation_ratio` remains absent because this method makes no separation claim.

The precision request must be an integer of at least 256 bits.
Work starts at 384 bits, or at least 512 bits when the smaller dimension is 128 or more.
The implementation retries once at 256 additional bits.
An unresolved pivot or insufficient residual budget then returns the conservative bracket `[0, exact_rank]` and the actual residual bound.
Equality at the threshold can remain unresolved.
Family records retain `RANK_UNRESOLVED` before any comparison with structural rank.
Increased precision can improve resolution, but is not an assumption in the enclosure proof.
Calls serialize changes to the shared Arb precision context and restore it on failure as well as success.

Phase-diagram now uses this certificate in place of its approximate QR combination.
The new algorithm identity records this scientific-method change.
It does not claim an independent QR certificate.
The family record types and threshold definition remain the same.
The [implementation ownership guide](scientific-implementation-forks.md) identifies the remaining distinct family methods.

## Numerical acceptance and evidence limits

The [rank acceptance tests](../tests/test_rank_enclosure_characterization.py) use exact rational references and analytic singular values.
They test the inward-endpoint counterexamples, exact Gram errors, threshold equality and adjacent values, dependent matrices and extreme binary64 magnitudes.
They also test unresolved arithmetic, source binding and context restoration.
The standalone [enclosure probe](../tests/probes/rank_enclosure_probe.py) must return success.
The [Decimal tests](../tests/kernel/test_decimal_custody_characterization.py) require exact numerical values and identical bytes across arithmetic contexts.
The [writer checks](../tests/test_decimal_output_custody.py) cover buffered and streaming history output, authoring exports and archives.

These checks justify the stated numerical and storage contracts for fresh use.
They do not qualify an experiment, validate matrix construction from a physical model or establish controller admission.
New scientific claims still need their declared inputs, prospective design, independent evidence and exact-source custody.
Earlier software gates and stored certificates do not acquire these guarantees through this source change.

## References and research

The linked source modules, command metadata and existing checks own the implemented behavior described here.
The [scientific integrity guide](scientific-integrity.md) defines its separate evidence and authority boundaries.
The [program source register](../paper/SOURCES.md) identifies bounded historical results and unavailable primary records.
