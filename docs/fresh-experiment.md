# Author one assigned reactor campaign

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for reactor candidate authoring and the proof before native contact.
It is the bounded entry to the separately authorized operator lifecycle.
Use it to construct an explicit assignment and identify the first genuine owner handoff.

## Prerequisites and first action

Use the [tested Linux environment](environments.md) and a clean committed checkout executing its tracked package. Create an explicit [operator storage profile](../configs/operator-storage.example.json) for an active guarded external mount. The profile grants no scientific authority. Obtain the complete owner-reviewed source inventory and prior unit/seed census before assigning new units.

Missing history is a scientific stop. The public [input examples](input-contracts.md#exposed-input-construction-examples) explicitly declare incomplete development history. They cannot replace the actual census.

For the exposed development check, use the fixed [profile](../experiments/reactor-response/profile.json):

```sh
uv run --no-sync empirical-lawhood --project-root /path/to/clean-target \
  --operator-profile /path/to/operator-profile.json campaign reactor-author \
  --profile /path/to/clean-target/experiments/reactor-response/profile.json \
  --output-dir /path/to/external-root/artifacts/authoring/reactor-development
uv run --no-sync empirical-lawhood --project-root /path/to/clean-target \
  --operator-profile /path/to/operator-profile.json \
  --reactor-authoring-dir /path/to/external-root/artifacts/authoring/reactor-development \
  campaign reactor-preissue-proof
```

Choose a new output directory beneath the profile's artifact namespace.
The public profile reports `PUBLIC_EVALUATION_UNITS_AND_SEEDS_EXPOSED` as its prospective issue stop.
Both commands execute zero native tasks.

## Assigned scientific inputs

For new qualification, supply a canonical `AssignedReactorAuthoringProfile` to the same `--profile` option.
Its `assignment` is a `ReactorPrefixAssignment`.
It contains five unique unit IDs and five distinct positive 63-bit native noise seeds.
The declared public scenario conditions remain in `SCENARIOS` order.
Its `prior_census` is a `ReactorPrefixPriorCensus` with complete sorted prior units, seeds and source-inventory identity.
The assignment binds that census through its exact fingerprint.

Draw the five seeds once from an operating-system cryptographic random source.
Reject each unit, numeric seed, stream and alias collision against the complete census.
Freeze the assignment before native contact.
Paired arms and two numerical views share each unit's seed.
The native bridge installs that seed before simulation.
A new ID alone cannot make an exposed preparation fresh.

The assignment ID is `<experiment_id>.cohort`. Each unit ID adds the scenario name with underscores replaced by hyphens. Its independent identity is `unit.<unit_id>`. Constructors reject duplicate, aliased or previously excluded identities.

The complete census remains immutable. The issue projection omits only numeric native-reactor aliases outside the permitted positive 63-bit seed domain. All valid seed aliases and other source-purpose identities remain. Authoring and proof retain the 64 MiB authoring-member limit.

A large complete census must not be reduced to pass that bound.

The [operator handoff](operator-handoff.md#canonical-constructors) gives the exact helper fields and conflict refusal.
The [operator walkthrough](reactor-operator.md#1-select-storage-and-freeze-the-census) gives the source inventory and assignment procedure.
A typed constructor cannot establish that the researcher's exposure history is complete.

## Expected records and interpretation

Authoring writes the strict authoring package, payloads, decoders, candidate, source closure, resource envelope and projected plans. The proof resolves the actual CLI/API provider and every runner. It validates native inputs, output/adjudication locators, resource ceilings and separate authority gates. It writes and replays the declared control records under guarded custody.

It acquires no native observations. Read all six closure sections and the reported issue stop. A successful projection grants no issue, execution or reveal authority.

The planned study has 21 tasks: twenty native branches and one finite-chain adjudication.
Five independent scenario/noise units each have two arms and two nested numerical views.
The delayed receiver is reactor temperature under the fixed prefix contract.
A supported result would be a finite conditional contrast.
It would establish no population confirmation, physical safety certificate or historical manuscript reproduction.
The [reactor guide](../experiments/reactor-response/guide.md) gives the exact actions, clocks, receivers and falsifiers.

## Separate issue, execution and reveal

Continue through the [operator walkthrough](reactor-operator.md) only with its actual reviewed inputs. An accountable proposer attests the outcome-blind design and complete exposure lineage. Separate custody authority permits immutable base and extension issue. Independent scientific approval and execution authority bind the issued package.

Compilation produces the scientific run plan and executable graph without native effects. Native tasks produce guarded receipts. Separately authorized reveal permits evaluator access and adjudication. Recovery reuses the same issued identities and immutable receipts.

Operational completion, evaluability, support, admission and permitted use remain separate.
A negative scientific result is not a retryable execution error.
The [CLI reference](cli.md) gives exact names and effects.
Other scientific families need their own selected source, candidate context and provider contracts.
This reactor binding supplies no default family authority.

## References and research

The [reactor experiment guide](../experiments/reactor-response/guide.md) owns the worked numerical contract and upstream attribution.
The [scientific integrity guide](scientific-integrity.md) states the evidence boundaries.
The [public specification](../src/empirical_lawhood/adapters/methods/finite_response_law/specification.md) owns its separate finite-response scientific operands.
