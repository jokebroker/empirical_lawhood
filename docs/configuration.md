# Configuration

Copy a supplied input, edit its values, and validate it before running its owning
command. Configuration validation reads one bounded input and invokes its
declared parser or typed constructor. It does not run a simulator, fetch a source,
inspect an external census, construct a provider, open a catalog, or grant authority.

The [workflow index](../experiments/README.md) identifies the relevant guide and
input role. The closed [configuration inventory](../src/empirical_lawhood/resources/config_schemas/index.json)
covers the supplied experiment JSON inputs and their closed root schema IDs, the operator
storage and FAIR-MAST source configs, and the human provenance YAML template.
These are consumer-specific contracts; an object with a `schema` field is not
automatically an editable experiment input.

## Validate and prepare a copied input

For the portable [RC ladder](../experiments/rc-ladder-response/guide.md), use a
fresh working directory outside the checkout. Run from the selected checkout
and locked main environment described in [environments](environments.md).

```bash
WORK=$(mktemp -d /tmp/empirical-lawhood-config.XXXXXX)
cp experiments/rc-ladder-response/model.json "$WORK/model.editable.json"
# Edit the copy; whitespace and property order may change.
uv run --no-sync empirical-lawhood config validate \
  --config "$WORK/model.editable.json" --consumer rc-ladder-model --format json
uv run --no-sync empirical-lawhood config prepare \
  --config "$WORK/model.editable.json" --consumer rc-ladder-model \
  --output-file "$WORK/model.canonical.json" --format json
uv run --no-sync empirical-lawhood campaign rc-ladder-native-check \
  --config "$WORK/model.canonical.json"
```

`validate` reports the input's SHA256, selected consumer, disposition, typed
validity, canonical SHA256/size when applicable, and consumer byte readiness.
Valid values can require preparation because an exact-byte consumer rejects
formatting changes. A consumer using the authoring codec can already read a
pretty document, provided it fits that consumer's own byte ceiling.

`prepare` writes only an explicitly selected **new file** under an existing real
directory. It refuses overwrite, in-place conversion, symlinks in input or output
paths, and parent-directory traversal. It preserves the input file and reports
the original and prepared byte hashes. It does not update source hashes,
approvals, assignments, seed labels or exposure claims. An edited scientific
value changes the relevant configuration identity.

Unchanged pretty RC and Brian2 JSON copies prepare to the exact supplied
canonical bytes. Brian2 execution still requires its separate environment;
validation/preparation use the main environment without Brian2 execution.
Do not add `$schema` to an input envelope or convert tagged Decimal values to
binary numbers. These are closed records:

```json
{"decimal": "0.001"}
```

The exact Decimal string carries the value. In canonical records, duplicate keys,
bare fractional JSON numbers, nonfinite values, extra fields, wrong nested
schemas/revisions, and semantic constructor violations refuse. JSON Schema
does not replace these lexical and semantic checks.

Successful reports go to stdout. Validation/preparation refusals exit 2 and
write the chosen text or JSON error to stderr, leaving stdout empty. Click
option/usage errors also exit 2 with their ordinary stderr diagnostic.

## Consumer limits and dispositions

The `--consumer` selector names a closed input role. It must match the declared
schema; it is required whenever a schema has more than one registered role.
Schema text never selects a Python import or a native factory. Shared schema
IDs can serve multiple consumers with different limits or preparation rules.
Unknown roots, issued/authority records and opaque
plan/design files have no fallback parser in this interface.

Canonical output must fit the existing consumer's limit. Editable input has a
separate ceiling of four times that limit, capped at the existing authoring
codec's 128 MiB maximum; retained imports still fit their original byte ceiling.
With an explicit consumer, the file is bounded at that role's authoring ceiling
before parsing. Automatic root discovery has a 128 MiB read ceiling and then
enforces the selected role's tighter ceiling. Existing nesting/node bounds also
apply. Decimal fixed-point expansion and total canonical output are bounded
before serialization. Extra whitespace does not relax the eventual consumer
limit.

| Consumer | Owning input/command | Consumer maximum | Disposition |
|---|---|---:|---|
| `rc-ladder-model` | `campaign rc-ladder-native-check` | 64 KiB | Editable; exact JSON consumer |
| `rc-ladder-study` | `campaign circuit-author` study | 128 KiB | Editable; exact JSON consumer |
| `brian2-current` | `campaign brian2-native-check` | 16 KiB | Editable; exact JSON consumer |
| `electron-gas-reference` | `campaign electron-gas-reference-check` | 32 KiB | Editable; exact JSON consumer |
| `lattice-method-authoring` | `campaign synthetic-material-author` | 128 KiB | Editable; exact JSON consumer |
| `reactor-fresh-profile` | `campaign reactor-author` profile | 64 MiB | Editable; exact JSON consumer |
| `reactor-assigned-profile` | Assigned exposed profile, same authoring owner | 64 MiB | Retained exact bytes; preparation forbidden |
| `matrix-response-authoring` | `campaign matrix-response-author`, route in input | 1 MiB | Editable authoring record |
| `reaction-development` | `campaign reaction-response-native-check`, backend in input | 32 KiB | Editable authoring record |
| `tokamak-heat` | `campaign torax-native-check` | 32 KiB | Editable authoring record |
| `material-control-input` | `campaign material-workflow-input-check` | 32 KiB | Editable authoring record |
| `battery-development`, `grid-development`, `tokamak-control` | Their native-check guides | 16 KiB | Editable authoring records |
| `propulsion-reference`, `scale-morphism-reference`, `glenn-import-selection` | Their reference/import-check guides | 16 KiB | Editable authoring selections; external inputs uninspected |
| `reactor-batch-input`, `reactor-prepared-input` | Reactor input-check commands | 16 KiB | Editable development selectors |
| `prepared-canary`, `prepared-dependent-input` | Prepared native/dependent input checks | 16 KiB | Editable development selectors; parent/custody uninspected |
| `finite-canary`, `finite-calibration-input`, `finite-stage-input` | Finite native/input/stage checks | 16 KiB | Editable development selectors; stage/parent checks remain with owner |
| `response-composition-input` | `campaign response-composition-input-check` | 16 KiB | Editable analysis selector; operands uninspected |
| `operator-storage` | Operator profile, used by doctor/guarded commands | 128 MiB | Editable deployment record; no mount access or authority |
| `fair-mast-selection` | `source acquire --source-config` and source preview | 64 KiB | Editable selection; exact JSON consumer; no acquisition |
| `prepared-prior-census` | `matrix-response-author --prior-inspection` | 64 MiB | Retained import; owning `_prior_census` parser only |
| `finite-prior-census` | Finite input check `--prior-exposure` | 128 MiB | Retained flat import; owning `_census` parser only |
| `exposed-inventory` | Supplied example documentary inventory | New 64 KiB shape-check ceiling | Helper shape only; member hashes/custody not authenticated |
| `evidence-provenance-template` | `configs/templates/evidence-provenance.yaml` | No consuming limit | Human template; runner validation/preparation unsupported |

YAML is accepted only for the editable roles whose consuming command already
uses the restricted authoring codec. Exact JSON consumers and retained imports
stay JSON-only. The existing safe YAML grammar refuses anchors, aliases,
explicit tags and duplicate mappings; quote Decimal text. Preparation returns
canonical JSON and leaves the YAML input unchanged.

The prepared census example shares the larger inspection record's schema ID but
contains only the fields consumed by the historical import reader. It is **not**
decoded as that larger canonical inspection record. The flat finite census has
its own import envelope. Import schemas retain the owner's extensible metadata
fields, and the tool never sorts, rewrites or normalizes imported evidence.
Successful import validation only establishes the owning census parser's
acceptance; it does not establish complete history, authentic custody or fresh
units. Held, assigned, issued and authority inputs cannot be repaired through
general preparation.

## Offline editor schemas

Export the installed resource for your copied RC input:

```bash
uv run --no-sync empirical-lawhood config schema \
  --schema-id empirical-lawhood/adapters/simulators/rc-ladder-response/resistor-capacitor-ladder-model-config \
  > "$WORK/rc-ladder-model.schema.json"
```

Associate the schema outside the scientific payload. For example, VS Code
workspace settings can associate the copied filename with the local export:

```json
{
  "json.schemas": [
    {"fileMatch": ["model.editable.json"], "url": "./rc-ladder-model.schema.json"}
  ]
}
```

Every projection declares Draft 2020-12 and uses local `$defs` references. No
network resolver is needed. Projections include required fields, nested exact
envelopes/revisions, enums, nullable values, integer/boolean distinction, tuple
shapes, closed tagged Decimal objects and explicit unit metadata for declared
unit suffixes. Constructor defaults do not make serialized fields optional.
Schema structural acceptance does not imply canonical byte spelling, duplicate
key safety, valid dimensions/clocks, scientific qualification or authority.
JSON Schema can accept an integer-valued numeric token such as `4.0`; the
runtime's canonical JSON parser still rejects that lexical spelling.

Library users can read/export resources without a checkout:

```python
import json
from empirical_lawhood.api.configuration_schemas import configuration_schema

schema = configuration_schema(
    "empirical-lawhood/adapters/simulators/rc-ladder-response/resistor-capacitor-ladder-model-config"
)
print(json.dumps(schema, indent=2))
```

The installed wheel contains the schema resources and inventory, not the
checkout's experiment inputs. Supply your own copied input. The API entrypoints
are `validate_configuration(Path(...), consumer=...)` and
`prepare_configuration(Path(...), Path(...), consumer=...)` in
[configuration.py](../src/empirical_lawhood/api/configuration.py).

Maintainers change the closed registry/type owners, regenerate with
`python scripts/generate_config_schemas.py`, and check drift with
`python scripts/generate_config_schemas.py --check`. Unsupported annotations fail
generation; they never become unconstrained schemas. The schema acceptance
tests use an offline Draft 2020-12 validator from the development environment.

## Current paper integration inputs

[Current integration guides](integrations.md) add explicit purpose/seed/split allocations, source requests and phase selections through the same inventory.
The inventory owns each consumer's exact limit, parser and preparation rule.
Use its consumer identifier when one schema has several roles.
The table above describes the established development selectors rather than every integration consumer.

Scientific protocol, editable selection, environment identity and retained evidence remain separate records.
Array operands carry exact byte identities and bounded member manifests.
Source and stage helpers derive object identities from authenticated selections.
Researchers need not fabricate manifests or grant hashes.
Configuration preparation does not acquire arrays, replace fixed protocols or refresh source bindings.

Current matrix source configurations bind the exact selected dependency lock and complete package bytes.
Prepare a new configuration when either binding changes.
Native APIs require the explicit selected project root before numerical contact.
Producing provenance is immutable completion evidence, separate from editable allocation or configuration.
Its observed environment and working-source inventory do not create scientific authority.
Changed serialized source and result contracts use explicit record versions; original records remain unchanged.

A current wrapper around frozen OriginalF or the 48-root nomination preserves its original numerical bytes and provenance.
It does not upgrade their evidence status.
Retained result, current parent, import-publication and fixed-operand records use their owning authentication APIs.
The public [history allocation](../experiments/matrix-history-analysis/guide.md) and [tangent configuration](../experiments/matrix-tangent/guide.md) expose actual numeric master inputs.
Changing a namespace leaves those numerical draws unchanged.

Public seed examples remain exposed.
Independent exposure checks use actually consumed numeric operands, including truncated generator seeds where applicable.
A role label, canonical bytes or passing schema establishes no freshness, scientific qualification or operational authority.
