#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Write the selected local v0.60 inventory. This is not a native-evidence check."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]


def main() -> None:
    if any(p.is_symlink() for p in BASE.rglob('*')):
        raise ValueError('Publication files must be regular local files without compatibility links.')
    records = []
    for path in sorted(BASE.rglob('*')):
        if (not path.is_file() or path == BASE/'bundle-manifest.json'
                or '__pycache__' in path.parts or path.name == '.DS_Store'):
            continue
        data = path.read_bytes()
        records.append({'path':path.relative_to(BASE).as_posix(), 'bytes':len(data),
                        'sha256':hashlib.sha256(data).hexdigest()})
    result = {'edition':'v0.60','date':'2026-10-07','hash_algorithm':'sha256',
              'scope':'Curated current-edition-only public files except this manifest. Complete original archive and predecessor paper payloads are externally preserved, not packaged. Local byte identity only; not external native evidence.',
              'files':records}
    (BASE/'bundle-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Wrote {len(records)} selected local file identities.')


if __name__ == '__main__':
    main()
