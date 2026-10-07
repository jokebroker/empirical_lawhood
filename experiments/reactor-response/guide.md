# Reactor response: demo and development authoring

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for the public reactor example and development authoring.
It is the first native software check after installation.
Use it to create a report and understand the separate route-input stops.

**Demonstration and exposed development only.** No fresh qualification or
issued scientific result follows from these starts.

## Read results and failures

Read `report.md` first, then `report.json` for clocks and paired summaries.
`panel.json` preserves the five independent units and nested arms/views.
A failed acquisition retains `partial.json`; it does not supply a successful full panel.
Held batch/prepared preflight reports describe census, fingerprints and missing ports without native tasks.

Read [the common result and failure guide](../../docs/results-and-failures.md) for retained attempts,
stdout/stderr and same-identity issued recovery. The route-specific outputs and stops below remain binding.
Use `workflow show reactor-response --format json` for static command effects and prerequisite roles.
Discovery does not inspect those prerequisites.

## Run the public demo

From a fresh Linux CPU checkout, install the exact numerical runtime and run:

```sh
uv sync --locked --python 3.11.14 --group reactor-example
uv run --no-sync empirical-lawhood example reactor-prefix --output-dir /absolute/path/to/new-reactor-demo
```

The readable implementation is
[`reactor_prefix.py`](../../src/empirical_lawhood/examples/reactor_prefix.py).
It reads three public, pinned inputs packaged under the installed code, as
recorded in [licensing policy](../../docs/licensing.md). It runs 20 short
native branches: five scenario/seed units, two numerical views and two arms,
with two observed decisions per branch. The output directory receives a typed
`panel.json` and short `report.json` and `report.md`. Allow space for a few
megabytes. A failed branch receives `partial.json` and a nonzero exit, with no
invented full panel.

**Demonstration only / unqualified local measurement.** The roster and actions
are already known, so this run adds no fresh independent support. It does not
issue or traverse a production campaign and does not reproduce paper evidence.

## One response relation, in ordinary terms

The engine asks about a relation `L(D,H,A,R,τ)`, not an unconstrained universal
law. In the public reactor example:

| Operand | Concrete meaning |
|---|---|
| Denominator `D` | A zero-feed prefix in five fixed public reactor scenarios, with declared initial conditions and noise seeds |
| Available history `H` | Only measurements available at each decision clock. A later receiver cannot choose an earlier action |
| Action `A` | Jacket requests at 0 and 10 seconds: pulse arm 315 then 316 K, comparator 316 then 316 K. Native clipping/rate limits remain visible |
| Receiver `R` | Reactor temperature in kelvin returned at the 20-second callback, carrying the delayed 10-second state and native measurement noise |
| Horizon `τ` | This two-decision prefix and its declared delayed receiver, with 1-second and 0.5-second numerical views |

There are **five independent units**. Two arms and two numerical views per unit
produce twenty branches, and two deliveries per branch produce forty decisions.
Those nested observations do not increase the independent denominator to twenty
or forty. The exposed public seeds cannot be made fresh by renaming a file.

## Read the report

Open `report.md` for the paired contrasts, then `report.json` for exact clocks,
units, source digests and all four delivery stages. `requested` is what the
controller asked for. `accepted`, `applied` and the native realized exposures
show what the actuator actually did. Each `pulse_minus_comparator_k` is paired
within one physical unit and one numerical view. The displayed mean is a
descriptive mean of five paired contrasts. It is not a population estimate or
a law qualification.

`panel.json` is the canonical scientific record. If acquisition fails,
`partial.json` retains the completed episodes and failure, the command exits
nonzero, and no successful full panel/report is invented. Choose a new demo
directory for a separate invocation. The example refuses to overwrite retained
output.


## Author a development candidate

This route uses the three packaged, pinned public Terminal Bench Science input
files. It needs CPython 3.11.14 and NumPy 2.4.6 (`reactor-example`). The source
bytes are included with the package, so no source hash needs to be invented.
The [strict profile](profile.json) selects a
new target experiment identity. Its public scenario IDs and seeds are exposed.
This is a **development candidate**.
The current proposer attestation cannot truthfully authorize prospective issue of these units.

## Frozen scientific question

The denominator is a zero-feed reactor prefix with prescribed initial state.
Its five public scenario/seed units are `cooling_loss`, `fouling`, `nominal` for
calibration and `feed_temp`, `kinetic_hot` for held-out checks. Every unit has
both jacket arms and both nested numerical views. The twenty branches are five
independent units, not twenty replicates. At episode times 0 and 10 s, feed is
requested at 0 kg/s.

The first jacket command is 315 K in the pulse arm and
316 K in the comparator. Both request 316 K at 10 s. The native actuator may
clip and rate-limit commands. Its accepted, applied and realized values remain
distinct from requests.

The receiver is reactor temperature in K at the 20 s
callback, which carries the delayed 10 s state with native measurement noise. The two RK4 views use 1 and 0.5 s plant steps on the same scenario units,
with a 10 s sampling clock. More than 0.01 K held-out absolute contrast error or 0.001 K view disagreement falsifies the bounded claim.
Neither threshold is a safety limit. This route supplies preparation and measurement
operands for order-relation, response and a bounded local-law question.

It does not establish admission or controller use. The packaged prefix is an unqualified
native demonstration. The candidate and preissue proof below execute no
native task.

## Inputs and commands

Use a clean target checkout executing its own `src/` package. Copy the
[storage profile example](../../configs/operator-storage.example.json)
to a location outside either repository. Replace its two `/path/to/...`
locators with the absolute path of an **active external mount** and a dedicated
subdirectory on that mount. Create that subdirectory and its distinct
`artifacts` and `scratch` children.

The mount guard validates that the root is
active, writable and outside the checkout, home and `/tmp`. The profile grants
no authority. Set `expected_mount_source` and `allowed_filesystem_types` if
your storage policy requires an exact device and filesystem.

```sh
uv sync --locked --python 3.11.14 --group reactor-example
uv run --no-sync empirical-lawhood --project-root /path/to/clean-target \
  --operator-profile /path/to/operator-profile.json \
  campaign reactor-author \
  --profile /path/to/clean-target/experiments/reactor-response/profile.json \
  --output-dir /path/to/external-root/artifacts/authoring/reactor-prefix-development
uv run --no-sync empirical-lawhood --project-root /path/to/clean-target \
  --operator-profile /path/to/operator-profile.json \
  --reactor-authoring-dir /path/to/external-root/artifacts/authoring/reactor-prefix-development \
  campaign reactor-preissue-proof
```

The author command writes the strict `ExecutableStudyDefinition`, extension
payloads and decoders, compiled extension candidate, source closure, resource
envelope and projected plans. The proof resolves the installed provider and
runner, every native input, output and adjudication locator at its resource
ceiling, and the typed authority gates. It persists and replays control
records in guarded storage. Both commands report `native_tasks_executed: 0`.
The proof reports `PUBLIC_EVALUATION_UNITS_AND_SEEDS_EXPOSED` as the issue stop.
The [scientific check](../../tests/test_reactor_native_science.py) separately
contacts the public native simulator and validates delivery and delayed-callback
behavior against an independent actuator and thermal reference.

For a new prospective experiment, use the assigned profile and provider in the [fresh campaign guide](../../docs/fresh-experiment.md).
The fixed public roster cannot support an unexposed-unit attestation. The [CLI reference](../../docs/cli.md) describes issue, package, plan, execution, recovery and separate reveal/adjudication.
Each needs its exact candidate, source closure, custody, approval, execution and reveal records.
No issue or native campaign run is claimed by this guide.

## Full-batch source gate and distinct response contracts

The retained full-batch route is a different 2,880-callback/28,800 s forecast
contract. The target-owned [batch input](batch-input.json)
fixes a 10 s callback clock, 1 and 0.5 s nested plant views and one assigned
scenario/seed as the independent unit. Its source port accepts an **external**
Terminal Bench Science tree containing these four required relative members (additional files are allowed):
`tests/plant.py`, `environment/spec/plant_params.json`,
`environment/spec/scenarios_public.json`, and `solution/controller.py`. The
preflight bounds sizes and authenticates the retained upstream hashes without
executing either Python member:

```sh
uv run --no-sync empirical-lawhood campaign reactor-batch-input-check \
  --config experiments/reactor-response/batch-input.json \
  --source-root /absolute/path/to/held-terminal-bench-science \
  --output-dir /absolute/development/reactor-response/attempt-001
```

The first no-input run refuses `SOURCE_ROOT_REQUIRED`. Selecting the three
packaged prefix resources as a source tree refuses
`SOURCE_MEMBER_REQUIRED: solution/controller.py`. Those resources are not a
full-batch source. The held-source preflight checks all four member hashes,
the 28,800 s/10 s clock and the 1/0.5 s views before selecting a provider.

The target canonical
source-record digest is `2fc284e231c6e8ce1cdca06190e9df6516f928951a045a8553a5af653886eef4`.
This binds the current public schema to the unchanged authentic member bytes.
The earlier public digest was stale after the namespace migration; historical
records retain their original identities and do not acquire new qualification.
The source-namespace record had a different digest because its schema name
was different. The command selected the registered
`ReactorBatchFactory` and constructed its `ReactorBatchRunner` provider
from the authenticated outcome-blind source port. It executed no Python
member, candidate, native task or issue. The current batch provider and composition still require the old fixed 42-unit
roster and exact 84 native tasks.

Those unit IDs and seeds are exposed. A
new scenario generator, task graph and provider binding are required before a
fresh batch candidate can be represented. The 10 s forecast receivers are reactor temperature, dose and conversion in K, kg and dimensionless units.
Requested feed/jacket actions use kg/s and K.
Retain accepted, applied and realized delivery separately.

With the authentic held source mounted, the [bounded native check](../../tests/test_reactor_held_native_science.py)
exercises one **exposed development** batch scenario/seed in both 1 and 0.5 s
views. From the clean checkout after installing `reactor-example`, run:

```sh
REACTOR_HELD_SOURCE_ROOT="$HELD_REACTOR_TREE" \
  REACTOR_LOCAL_DISCOVERY="$HELD_LOCAL_DISCOVERY_JSON" \
  uv run --no-sync pytest -q tests/test_reactor_held_native_science.py
```

The batch check observed 2,880 ordered callbacks and action-stage rows in each view.
It observed 28,801/57,601 native grid rows and an exact callback-to-grid applied-action journal.
Terminal dose was 287.3 kg within numerical integration tolerance.
The two views agreed within 0.001 K and 0.00001 dimensionless conversion at
the terminal receiver. This is an adapter-level diagnostic on an old assigned
unit, separate from the one-CLI no-effect provider preflight. It is no issued
batch campaign or new independent evidence.

Eight response routes share the bounded four-member source authentication. Six
of them also consume the raw local-discovery publication: local, feed, regime,
finite control frontier, selected action and staged pulse. Each has a distinct typed design
and native config. Select its [route input](../../experiments/reactor-response)
(`matched-replay-history-input.json`, `local-input.json`, `feed-input.json`,
`regime-input.json`, `finite-control-frontier-input.json`, `selected-action-response-input.json`,
`staged-pulse-response-input.json` or `causal-response-study-input.json`) and run:

```sh
uv run --no-sync empirical-lawhood campaign reactor-prepared-input-check \
  --config experiments/reactor-response/feed-input.json \
  --source-root "$HELD_REACTOR_TREE" \
  --discovery-file "$HELD_LOCAL_DISCOVERY_JSON"
```

The source root must contain the four batch members above. For the six
discovery consumers, the discovery file must be an absolute regular JSON file
at most 4 MiB. Omit `--discovery-file` for matched replay and forecast and causal response. The
command refuses an irrelevant discovery argument.

Causal response additionally
requires the pinned published REF, EKF and fixed-schedule comparator Python
files under `solution/` and `authoring/evidence/` in the held tree. Their
bytes are bounded and authenticated without executing them. The `freeze_discovery` decoder bounds the supplied bytes, records their actual SHA-256, and validates the exact retained domain specification before using its parameters. This is a new target nomination contract, not a reconstruction of an earlier publication identity.

The command constructs the selected typed native config and reports these retained independent-root counts:

| Route | Roots |
|---|---|
| Matched replay/history | 42 |
| Local atlas | 64 |
| Feed contrast | 64 |
| Regime preparation | 208 |
| Finite frontier | 192 |
| Selected action | 96 |
| Staged pulse | 480 |
| Causal controller | 96 |
 Each root has two
nested plant views. Local keeps its nine-action atlas. Feed, regime preparation and selected action use the prepared `d11010` action indices 1/4/7.

Frontier instead has a
zero word and nine finite rate/duration pulses. The staged-pulse route has separate local and induced pair requests.
The causal-controller route has nine named arms and eight confirmation roots among its 96 staged roots. The discovery
is an **outcome-visible development input** where consumed, and all retained
unit/seed rosters are exposed. The command selects each route's registered
factory.

Matched replay and forecast constructs its `ReactorBatchRunner` provider from the
authenticated batch source. The other seven report
`UPSTREAM_PORTS_REQUIRED`, `provider_built=false` and their exact missing
platform port keys. Their authenticated batch source alone does not satisfy
the distinct native source and custody port contracts. No native assay,
candidate compilation or issue occurs.

Missing source stops at `REACTOR_SOURCE_ROOT_REQUIRED`.
Missing required discovery stops at `REACTOR_DISCOVERY_REQUIRED`. Altered
domain parameters refuse at the domain-specification check.
Missing or changed empirical comparators refuse at `REACTOR_COMPARATOR_REQUIRED`
or `REACTOR_COMPARATOR_MISMATCH`. The
[prepared-input check](../../tests/test_reactor_prepared_input.py) loads each shipped
selection and, when the held source is mounted, verifies all eight typed
constructions and these refusals.

For any of the seven routes with extra ports, an operator can add
`--upstream-binding /absolute/path/to/binding.json` to the same command. The bounded canonical `EnvironmentBoundReactorSourceBinding` names the route and exact config/source hashes printed by the input check.
It declares `EXPOSED_DEVELOPMENT_NONPROMOTABLE`.
It supplies one sorted `ObjectIdentity` for every required **non-source** port key. The decoder refuses extra,
missing or reordered keys, a changed config/source digest, or another role. The seven factory rosters determine those keys.

Finite control frontier and staged pulse
include their `*-inputs` tuple among them. The four-member tree is also checked
against the comparator pins and rebuilt into the retained
`EmpiricalStudySource` payload for a provider construction. The binding also contains `EmpiricalNativeEnvironment`: an explicit image SHA-256, public source commit, exact file census and the separate verifier package versions. The entire supplied census is checked.

Selectors without this operand refuse `REACTOR_NATIVE_ENVIRONMENT_REQUIRED`. This preflight does not run the verifier.

With an explicit top-level `--operator-profile`, the installed CLI now resolves
these identities through `ReactorExternalPortStore`. The operator publication
procedure is in the [reactor walkthrough](../../docs/reactor-operator.md#other-retained-reactor-port-contexts). Explicit operator tooling publishes
strict `ReactorPortContext` records and their immutable manifests at
`operator/reactor-ports/<context_id>.json` under that profile's artifact root. Each record binds its route/key, config, run, parent, output census, resources,
issued program and separate execution/reveal authority, plus applicable
phase, stage, inputs or retained continuation.

Fixed installed recipes create
the actual custody and resource ports. JSON cannot choose executable code. Missing store/publication or changed authority/identity refuses before provider
construction. The [store check](../../tests/test_reactor_port_store.py) checks
publication replay and tampering. The [upstream checks](../../tests/test_reactor_upstream_binding.py)
construct all seven real factories using explicitly synthetic guarded ports and
the authenticated public source.

No native task or publication occurs in those
checks. Omitting a binding still reports `UPSTREAM_PORTS_REQUIRED`.

For matched replay and forecast, use the same command with `--config
experiments/reactor-response/matched-replay-history-input.json --source-root
"$HELD_REACTOR_TREE"` and no discovery file. For causal response, replace the
config with `causal-response-study-input.json` and supply the full comparator tree.
This input command does not run the source's pinned native verifier image
(Python 3.12.13, NumPy 2.1.3, SciPy 1.14.1, pytest 8.4.1). The input pass
therefore cannot certify its benchmark or feedback result.

The matched replay and forecast route adds a prescribed −1 K jacket pulse at 7,200–7,800 s.
Recovery extends to 9,000 s.
Matched-input replay uses the same independent scenario. The same bounded native check completes one old history root with
two feedback views and three matched/pulsed replays. Its requested jacket
change is −1 K only at callbacks 720–779. The accepted/applied journal records
the actuator rate limit at onset and common input again at recovery. The
five episodes and two forecasts remain nested under one root.

The same bounded test acquires one exposed feed root in both views, using the
authenticated local discovery. It finds the causal anchor at callback 326
and completes eight native episodes: two exploration views and six matched
action views. Feed actions 1/4/7 request 0/.016/.032 kg/s. The actuator
applies 0/.016/.020 kg/s, realizing 0/.16/.20 kg during the ten-second assay.

The jacket stays fixed and all action views retain the donor's pre-action prefix.
Peak-temperature and zero-feed cooling receivers agree across views within their declared numerical bounds. An independent affine
calculation validates the frozen model's zero-action contrast. This is one old
development root. It does not publish the required feed port context or qualify the 64-root roster.

The bounded test runs the first exposed local root through three exploration donors.
It finds `d11010` at callback 290 before any measured branch response. Poisoning all later observations and native grid values
leaves the nine selected requests and causal features unchanged. All nine
actions then complete in both native views with matched donor prefixes,
separate requested/accepted/applied stages, integrated feed dose and a
peak-temperature receiver in K. Their receiver values differ across actions
and agree across views within 0.01 K. This validates one atlas domain and one
old root. The 64-root local panel and its public upstream ports remain open.

Feed and local routes select causal anchors from prior callbacks and assay
feed words under the actuator. They cannot read later grid truth at their
cutoff. Regime adds committed preparation and phase-local windows.
Its C and P request tapes share the pre-action prefix.
Each requests 4.8 kg over the first 300 s, with a different P pulse shape. The bounded
held-source check now completes six C/P preparation episodes on one exposed
fit root in both views, with 4.8 kg realized under each path.

It selects
the common causal `t0` at callback 304 and verifies the `c_q`/`p_q` prepared
prefixes through the retained safety reducer. At both 330 s and 930 s after
`t0`, zero and nonzero feed branches measure positive phase-local cooling
in K with a between-view contrast difference below 1e-6 K. These are
adapter-level development branches before the separate sealed-prediction
barrier. The full 48-slot panel and public regime provider remain open.

The held local
discovery publication authenticated its raw byte pin, and its typed feed
domain/design were rebuilt under target schemas. Their target fingerprints
replace the two source-namespace pins consumed by regime, frontier and
selected-action/staged-pulse preparation. Typed instantiation checks accept regime,
finite control frontier and selected-action/staged-pulse native configs with that rebuilt domain.
None was issued. Finite control frontier adds finite pulse/budget words and separates
short response from guard safety.

The bounded held-source test compares its
zero word with 0.016 kg/s for 30 s on one exposed early root in both views. The native 120 s guard integrates 0.48 kg, remains below 356.2 K, and shows a
**higher**, not lower, peak K than the zero word by about 0.0038 K. It also
validates the exact 96 independent-root binomial denominator. The two views do
not double it.

This early development response is a measured warning for any
cooling claim, not a frontier qualification. Selected action and staged pulse use distinct
selected-action and induced history receiver charts. The same bounded
held-source test runs selected action's zero/0.016 kg/s ten-second pair on one
old exposed regime root in both views. The selected word applies 0.16 kg and
its zero-minus-action peak is about 0.002519 K in each view, inside the frozen
0.0012–0.0044 K band.

The [64-root gate test](../../tests/test_reactor_classical_selected_action.py)
still uses a synthetic receiver fixture to show that 62/64 qualifies and
61/64 does not. The native root here supplies neither that panel nor an
issued selected action decision. The bounded staged pulse test uses one declared
historical development root for its FIRST and induced pair in both views. FIRST requests 0.032 kg/s but applies 0.020 kg/s, realizing 0.20 kg.

The
second 0.016 kg/s word realizes another 0.16 kg. Its 120 s first window and
240 s pair window preserve the full requested/accepted/applied/realized stage
journal. The first ten-second local and pre-second induced receivers show
about 0.001714 and 0.001411 K cooling respectively. The full 240 s peak is
higher under added feed.

These are distinct receiver horizons. The typed
induced context binds the nominal causal prefix before the second response,
and the refined prefix matches it without fabricating a second typed seal. The stage's upstream custody, provider, full root panel and issued decision
remain open. Causal response has a fitted controller and eight-root
delivery/feedback design, not the reference-policy singleton.

Its bounded
[held-source diagnostic](../../tests/test_reactor_held_native_science.py)
authenticates the REF/EKF/fixed-schedule comparator file pins, then checks
one old exposed root at the 600 s callback in both numerical views. The nine
declared action IDs produce six distinct requests there because the upper
jacket clip aliases words. Two matched feed branches preserve the causal
prefix and agree with the actuator's requested, accepted, applied and realized
stages. Their post-run K, conversion and kg labels match native grid slices.

The feed branch has positive realized dose and a lower ten-second peak K than
zero feed. A separate [synthetic stage fixture](../../tests/test_reactor_empirical_synthetic_consumers.py)
validates nine affine fit/nomination calls, the 16/8/32/32/8 root denominator,
32-root calibration and qualification refusal, and eight-root confirmation
nonentry. Its incomplete callback rows are rejected by the full census gate. This fixture is not native qualification.

A benchmark pass additionally needs
the separately pinned native verifier. Their retained
authors/providers remain closed on historical fixed rosters or source-era
selections, and no target-owned selected candidate has been checked for them. These contracts require separate authoring ports and focused scientific
checks. The batch input gate and prefix candidate do not certify them.

The [bounded transform checks](../../tests/test_reactor_distinct_transform_boundaries.py) independently test these operands:

- The history pulse boundary.
- Empirical actuator rate/mass relation and causal features.
- Feed's zero-action contrast and future-blind candidate features.
- Regime C/P request integral and realized-dose prefix in both views.
- Frontier exact-binomial inversion.
- Staged-pulse adverse scale.
 Frontier's 0.016 kg/s × 30 s word realizes 0.48 kg over
its 120 s guard. Staged pulse's FIRST word requests 0.032 kg/s for 10 s but
the actuator applies 0.020 kg/s and realizes 0.20 kg over its distinct 240 s
guard. Local's nine-word candidate context is unchanged when later native
truth is poisoned.

The [selected action check](../../tests/test_reactor_classical_selected_action.py)
independently integrates selected feed mass, validates two-view cooling and
action-stage refusal, and tests the exact 64-root one-sided qualification
gate. These checks do not execute the remaining full native contracts.
## Independent retained arithmetic

Use the [full-batch trace diagnostics](../../src/empirical_lawhood/adapters/methods/reactor_prefix_response/posthoc_metrics.py)
for the original 28,800 s trace, exact interval peaks and assigned 32-calibration/
10-held-out sensitivity census. Use the [empirical diagnostics](../../src/empirical_lawhood/adapters/methods/reactor_causal_response/posthoc.py) for the distinct 24-root discovery chart.
The chart has 16 fit roots and eight nomination roots.
It has five episodes and two nested views. Nominal RMSE and maximum error over both
views have different denominators. Intervention audits retain the original
shared-prefix, exposure-contact and native K/conversion checks.

The regime [fit verifier](../../src/empirical_lawhood/adapters/methods/reactor_regime_response/fit_verification.py)
independently reconstructs saved root-weighted affine, radial-basis and local
coefficients. The [qualification verifier](../../src/empirical_lawhood/adapters/methods/reactor_regime_response/qualification_verification.py)
validates supplied precision, matched-root contrasts and declared saved-input edges.
It retains both native views, all 64 assigned roots, the 48-contact minimum and
fixed 99 percent one-sided bootstrap limits. It emits no scientific adjudication.

Authenticate the retained operands and obtain separate analysis authority first.
These installed Python library functions perform no storage or native effects.
They provide independent arithmetic and outcome-visible diagnostics. They do not
select authentic inputs or qualify a new experiment.

`verify_saved_development` also validates the frozen nomination and every original
200-by-32 ordinary-root bootstrap draw before reporting independent fit checks.
`verify_saved_qualification` reconstructs all 32 calibration and 64 qualification
roots, their causal prediction barriers, five contrasts, local-law operands and terminal
receipt ancestry. Its complete evidence denominator is 292 tasks, including the
160 authenticated retained tasks and 132 new tasks when using a continuation.
Supply authenticated records keyed by `(task_id, schema)`, the successful
receipts and the frozen terminal configuration publication operands.

The [saved service verifier](../../src/empirical_lawhood/adapters/methods/reactor_regime_response/independent_prospective_check.py)
provides `verify_saved_service` for the distinct 64-root prospective contract.
It preserves nonentry, both native views, exact word delivery, the installed
owner's task and probe gates, and all-assigned A/J/C/F counts and bounds. Its
typed plan has the original 387-task denominator. These readouts require the
caller to authenticate canonical payloads, publication markers, successful
receipts, frozen source and exposure, and separate reveal/analysis authority.
Supplying typed records or task names alone does not establish those proofs.

For retained feasibility work, the [chart and fold reductions](../../src/empirical_lawhood/adapters/methods/reactor_regime_response/retained_feasibility.py)
keep delivered-mass and peak/cooling measurements, shared-prefix checks and
whole-root current/history comparisons. The [root sensitivity](../../src/empirical_lawhood/adapters/methods/reactor_regime_response/retained_sensitivity.py)
keeps the original 5,000 descriptive resamples. These exposed-data calculations
cannot supply missing delayed-probe labels, fresh power or admission.

`analyze_retained_inputs` accepts the separately authenticated 24-root global,
64-root feed and 64-root local panels, including their native arrays and derived
reports. It returns the original root census, chart selection, whole-root fits
and measured/provisional opportunity report. `analyze_retained_margins` consumes
that report plus the authenticated 64-root native feature charts and old support
domain. It preserves support faces, interval widths, temperature headroom and
whole-root sensitivity. Both functions keep the old law explicitly unqualified
and return data to the caller without publishing it.

## Obtain the workflow inputs

Obtain this bundle from the selected repository checkout or source distribution. The guide and study inputs are repository workflow files, not wheel resources. Run each command from the checkout root in an unactivated shell. The relative paths use that root.

Keep generated data and receipts in the declared external storage. A wheel can perform its advertised inspection and native-input checks with absolute paths. Source-bound authoring and issue require a clean checkout that executes its own tracked package.

## References and research

Gareth Seneque (2026), [*Empirical Lawhood*](../../paper/manuscript.md), edition 0.60, gives the scientific formulation.
The [program source register](../../paper/SOURCES.md) identifies historical results and unavailable primary records.
The linked scientific source modules and existing tests establish only their declared implementation and development boundaries.
