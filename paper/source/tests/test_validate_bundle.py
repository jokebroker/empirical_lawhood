# SPDX-License-Identifier: MPL-2.0
"""Paper-profile regressions for the original publication preservation boundary."""
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "source"))
spec = importlib.util.spec_from_file_location("paper_preservation_validator", PAPER / "source/validate_bundle.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class OriginalPublicationPreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="paper-preservation-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name) / "paper"
        shutil.copytree(PAPER, self.base)
        self.owner = patch.object(validator, "BASE", self.base)
        self.owner.start()
        self.addCleanup(self.owner.stop)

    def test_exact_original_member_map_and_both_pdf_reviews_accept(self):
        self.assertEqual(validator.check_canonical_inputs(), 32)
        self.assertEqual(len(validator.check_author_qa()), 3)

    def test_changed_supplied_alias_refuses(self):
        path = self.base / "manuscript.md"
        path.write_bytes(path.read_bytes() + b"changed\n")
        with self.assertRaisesRegex(ValueError, "Canonical member differs"):
            validator.check_canonical_inputs()

    def test_changed_curated_zip_refuses(self):
        path = self.base / validator.SELECTED_ARCHIVE
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Curated archive bytes differ"):
            validator.check_canonical_inputs()

    def test_coherent_map_omission_refuses(self):
        for name,key in (("canonical-inputs.json","files"), ("source-preservation.json","selected_inputs")):
            path = self.base / "source" / name
            record = json.loads(path.read_text())
            record[key] = [r for r in record[key] if r["path"] != "manuscript.md"]
            path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "Selected public member map differs"):
            validator.check_canonical_inputs()

    def test_old_edition_payload_refuses_even_with_coherent_manifest(self):
        path = self.base / "editions/v0.10/older.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"excluded predecessor payload")
        manifest_path = self.base / "bundle-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"].append({"path":"editions/v0.10/older.pdf", "bytes":path.stat().st_size,
                                  "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Predecessor paper payload must remain external"):
            validator.check_manifest()

    def test_unselected_pdf_with_generic_name_refuses(self):
        (self.base / "source/legacy.pdf").write_bytes(b"excluded predecessor payload")
        with self.assertRaisesRegex(ValueError, "Unselected public paper payload"):
            validator.check_canonical_inputs()

    def test_old_edition_zip_member_with_coherent_archive_claim_refuses(self):
        path = self.base / validator.SELECTED_ARCHIVE
        with zipfile.ZipFile(path, "a") as bundle:
            bundle.writestr("Empirical_Lawhood_v0.60/sources/Manuscript_v0.10.pdf", b"excluded predecessor payload")
        for name in ("canonical-inputs.json", "source-preservation.json"):
            claim_path = self.base / "source" / name
            record = json.loads(claim_path.read_text())
            record["archive"].update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), members=18)
            record["archive_sha256"] = record["archive"]["sha256"]
            claim_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "Curated archive binding differs"):
            validator.check_canonical_inputs()

    def test_coherent_excluded_provenance_tamper_refuses(self):
        for name in ("canonical-inputs.json", "source-preservation.json"):
            path = self.base / "source" / name
            record = json.loads(path.read_text())
            record["excluded_original_members"][0]["sha256"] = "0" * 64
            path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "Excluded original member inventory differs"):
            validator.check_canonical_inputs()

    def test_navigation_edit_claim_refuses(self):
        path = self.base / "source/source-preservation.json"
        record = json.loads(path.read_text())
        record["public_translation"]["pdf_navigation_value_edits"] = 1
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "Original publication bytes must remain"):
            validator.check_canonical_inputs()


if __name__ == "__main__":
    unittest.main()
