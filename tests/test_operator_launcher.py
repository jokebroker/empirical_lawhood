# SPDX-License-Identifier: MPL-2.0
"""The documented external recipe works from outside an unactivated checkout."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_literal_external_operator_launch(tmp_path):
    script = tmp_path / "operator.py"
    script.write_text('''import json
import os
import sys
from pathlib import Path
assert "numpy" not in sys.modules
from empirical_lawhood.cli.entry import require_single_thread_bootstrap
assert require_single_thread_bootstrap()["OPENBLAS_NUM_THREADS"] == "1"
from scripts.operator_records import read_record
from empirical_lawhood.api.composition import create_cli_api
from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile
profile = read_record(Path(os.environ["PROFILE"]), OperatorStorageProfile)
assert profile.authority_granted is False
try:
    read_record(Path("unprovided-authority.json"), OperatorStorageProfile)
except FileNotFoundError:
    print(json.dumps({"boundary": "operator input required", "args": sys.argv[1:]}))
else:
    raise AssertionError("launcher supplied an operator act")
''')
    env = {key: value for key, value in os.environ.items()
           if key not in {"PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}}
    env.update(UV_PROJECT_ENVIRONMENT=sys.prefix,
               PROFILE=str(ROOT / "configs/operator-storage.example.json"),
               OPENBLAS_NUM_THREADS="7", OMP_NUM_THREADS="9")
    result = subprocess.run(
        ["uv", "run", "--project", str(ROOT), "--no-sync", "python",
         str(ROOT / "scripts/run_operator.py"), str(script), "retained-identity"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=40,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "boundary": "operator input required", "args": ["retained-identity"],
    }
    assert list(tmp_path.iterdir()) == [script]
