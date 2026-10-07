# Grid2Op held-chronic native development check

SPDX-License-Identifier: CC-BY-4.0

This guide gives instructions for a held power-grid chronic development check.
It measures two reset branches nested within one complete numerical episode.
Use it to select permitted offline inputs and interpret the bounded native report.

The [strict target selection](config.json)
uses the retained independent-recurrence Grid2Op adapter. It selects one externally held
`l2rpn_case14_sandbox`-format chronic, two reset action branches (hold and
disconnect line 000), and receiver horizons at native steps 1 and 3. A
complete chronic is **one independent unit**. Branches and horizons are nested
views. The native receiver records UTC step time, maximum line thermal loading
`rho` (dimensionless), disconnected-line count, connected-substation-component
count and a topology-vector digest. Reward is ignored.

## Read results and failures

Read `source_binding_sha256`, `independent_units`, `nested_action_branches`, `native_observation_complete` and `traces`.
Inspect requested, accepted, applied and realized actions with each receiver clock/unit.
A missing offline source is a pre-contact input stop, not a successful empty grid calculation.

Read [the common result and failure guide](../../docs/results-and-failures.md) for retained attempts,
stdout/stderr and same-identity issued recovery. The route-specific outputs and stops below remain binding.
Use `workflow show grid-response-inputs --format json` for static command effects and prerequisite roles.
Discovery does not inspect those prerequisites.

## Supply a held source

Install the pinned Linux x86-64 `grid2op-held` profile (Grid2Op 1.12.5 and
LightSim2Grid 0.13.1). Obtain the two matching wheel files and a lawfully held
offline sandbox-format grid from their upstream distributors. Examine the
dataset's own license and access terms. The repository does not contain the
grid, chronics or wheel bytes. Arrange a source root with the selection's relative paths:

- The Grid2Op and LightSim2Grid wheel files.
- `datasets/l2rpn_case14_sandbox/grid.json` and `config.py`.
- `chronics/<native_chronic_id>/time_interval.info` and `start_datetime.info`.
- Compressed `load_p`, `load_q`, `prod_p` and `prod_v` CSV files beneath that chronic.
 Optional hazard, maintenance and forecast members are included in
the inventory when present. The four required series, timestamp and positive
timestep must describe an executable episode. The [held-source assessment](../../docs/sources/grid2op-qualification.md)
explains its native provenance and historical ceiling.

Edit only the relative source paths and `native_chronic_id` in a copy of the
strict selection to match your held files. The shipped
`researcher-chronic-001` is a **placeholder**, not an acquired or frozen
evaluation chronic. Keep `fresh_target_identity_disjoint: false` for exposed or
unqualified development data. Set it true only after separately establishing
a genuinely disjoint prospective chronic roster.

That declaration alone does
not supply source qualification, custody or execution authority. The command
still runs a development diagnostic and never compiles or issues a candidate. The wheel, grid, rules and chronic hashes and the installed observation
operator identity enter the reported source binding. The command reads the
held files and makes no network request.

From a clean target checkout, with your source root outside the repository:

```sh
uv sync --locked --python 3.11.14 --group grid2op-held
uv run --no-sync empirical-lawhood campaign grid2op-native-check \
  --config experiments/grid-response-inputs/config.json \
  --source-root /absolute/path/to/your/held-grid2op-source \
  --output-dir /absolute/development/grid-response-inputs/attempt-001
```

The unedited selection should refuse before simulator contact because its
placeholder chronic and external root have not been supplied. For a selected source, the command reports requested, accepted, applied and realized action codes.
It reports two native horizons, receiver values, source hash, one independent unit and any invalid observation reasons. The [focused checks](../../tests/test_grid2op_native_quickstart.py) load the shipped config and reject missing source/path escape before contact.
They inventory an explicitly synthetic file fixture.
They prevent a nondisjoint source from entering prospective design or sealed execution. A read-only held reference
was also exercised through this target CLI in the pinned runtime: both
branches had the same reset hash.

Line 000 was disconnected and realized at
steps 1 and 3, while hold kept zero disconnected lines. Maximum `rho` was
0.844417 and 0.861682 on disconnect versus 0.814236 and 0.827904 on hold. These are exposed development observations, not fresh evidence.

The retained recurrence method and its historical negative evaluation cannot
become a new candidate through this command. A prospective route still needs
a source-qualified, disjoint chronic roster, typed custody and authority,
selected provider and shared candidate/production binding. The command
reports `campaign_candidate_compiled=false` and `campaign_issued=false`.

## Obtain the workflow inputs

Obtain this bundle from the selected repository checkout or source distribution. The guide and study inputs are repository workflow files, not wheel resources. Run each command from the checkout root in an unactivated shell. The relative paths use that root.

Keep generated data and receipts in the declared external storage. A wheel can perform its advertised inspection and native-input checks with absolute paths. Source-bound authoring and issue require a clean checkout that executes its own tracked package.

## References and research

Grid2Op Developers (undated), [*Available environments*](https://grid2op.readthedocs.io/en/stable/available_envs.html).
This is upstream dataset/interface documentation. The selected software is Grid2Op 1.12.5 with LightSim2Grid 0.13.1.

Gareth Seneque (2026), [*Empirical Lawhood*](../../paper/manuscript.md), edition 0.60, gives the scientific formulation.
The [program source register](../../paper/SOURCES.md) identifies historical results and unavailable primary records.
The linked scientific source modules and existing tests establish only their declared implementation and development boundaries.
