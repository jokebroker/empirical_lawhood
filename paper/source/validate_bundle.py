#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Validate the curated original v0.60 publication and preservation receipts.

No canonical regeneration, navigation correction or native scientific execution.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import zipfile
from urllib.parse import unquote, urlsplit
sys.dont_write_bytecode = True
import pymupdf
from storage import add_storage_arguments, external_directory
BASE = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = "69bf607bb2a0c7f1c3980872c89ca432d6df2a864988b6d336d72777835ef30b"
SELECTED_ARCHIVE_SHA256 = "9e3ea52ae38cf9113f21dc4f479ac6f51f64e0a0d58a1f70fb977674c15df1b5"
SELECTED_ARCHIVE = "editions/Empirical_Lawhood_v0.60_Selected_Source.zip"
EXCLUDED_MEMBER_INVENTORY_SHA256 = "6138357f8723841ed3e512c40e840ba0bcbbfc562514ba5500a2018c2c56d0bd"
SELECTED_MEMBERS = (
    "Empirical_Lawhood_Evidence_and_Provenance_v0.60.md",
    "Empirical_Lawhood_Evidence_and_Provenance_v0.60.pdf",
    "Empirical_Lawhood_Manuscript_v0.60.md",
    "Empirical_Lawhood_Manuscript_v0.60.pdf", "Revision_Summary_v0.60.md",
    *(f"assets/figure{i}.{kind}" for i in (1, 2, 3) for kind in ("pdf", "png")),
    "assets/inherited_figure_provenance.json", "build.sh", "evidence_metadata.tex",
    "header.tex", "layout.lua", "validation_report.json",
)
ALIASES = {
    "Empirical_Lawhood_Evidence_and_Provenance_v0.60.md": "revision-and-evidence-notes.md",
    "Empirical_Lawhood_Evidence_and_Provenance_v0.60.pdf": "revision-and-evidence-notes.pdf",
    "Empirical_Lawhood_Manuscript_v0.60.md": "manuscript.md",
    "Empirical_Lawhood_Manuscript_v0.60.pdf": "manuscript.pdf",
    "Revision_Summary_v0.60.md": "revision-summary.md",
    **{name: name for name in SELECTED_MEMBERS if name.startswith("assets/")},
    "evidence_metadata.tex": "source/evidence_metadata.tex",
    "header.tex": "source/preamble.tex", "layout.lua": "source/pdf_filter.lua",
}
MAINTAINED_FILES = {
    "README.md", "SOURCES.md", "metadata.json", "source_inventory.json",
    "source/bibliography.json", "source/build_pdfs.py", "source/canonical-inputs.json",
    "source/claim-evidence-inventory.json", "source/evidence-manifest.json",
    "source/pdf-compatibility.tex", "source/pdf-preflight.json", "source/reported-values.json",
    "source/source-preservation.json", "source/storage.py", "source/tests/test_validate_bundle.py",
    "source/validate_bundle.py", "source/write_manifest.py",
    "editions/v0.60-deposit/Empirical_Lawhood_Manuscript_v0.60.pdf",
    "editions/v0.60-deposit/zenodo-record.json", SELECTED_ARCHIVE,
}
DOCS = ("manuscript.md", "revision-and-evidence-notes.md", "README.md", "SOURCES.md")
PDFS = ("manuscript.pdf", "revision-and-evidence-notes.pdf")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_json(relative: str) -> dict:
    return json.loads((BASE/relative).read_text())

def packaged_files() -> set[str]:
    require(not any(p.is_symlink() for p in BASE.rglob('*')),
            'Publication files must be regular local files without compatibility links.')
    return {p.relative_to(BASE).as_posix() for p in BASE.rglob('*') if p.is_file()
            and p != BASE/'bundle-manifest.json' and '__pycache__' not in p.parts
            and p.name != '.DS_Store'}

def anchors(text: str) -> set[str]:
    result = set(re.findall(r'<a\s+id=[\"\']([^\"\']+)[\"\']', text))
    result.update(re.findall(r'\{#([^\s}]+)', text))
    for heading in re.findall(r'^#{1,6}\s+(.+)$', text, re.M):
        heading = re.sub(r'\s*\{[^}]*\}\s*$', '', heading)
        heading = re.sub(r'[^\w\s-]', '', heading.lower())
        result.add(re.sub(r'\s+', '-', heading.strip()))
    return result

def lower_binomial(successes: int, trials: int, tail: float = .05) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo+hi)/2
        probability = sum(math.comb(trials,n)*mid**n*(1-mid)**(trials-n)
                          for n in range(successes,trials+1))
        if probability < tail:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2

def check_manifest() -> int:
    manifest = read_json("bundle-manifest.json")
    require(manifest["edition"] == "v0.60", "Unexpected selected paper edition.")
    expected = {r["path"]: r for r in manifest["files"]}
    require(len(expected) == len(manifest["files"]), "Duplicate manifest path.")
    require(packaged_files() == set(expected), "Publication inventory differs.")
    check_current_edition_only()
    for rel, r in expected.items():
        require((BASE/rel).stat().st_size == r["bytes"] and digest(BASE/rel) == r["sha256"],
                f"Manifest identity mismatch: {rel}")
    return len(expected)


def check_current_edition_only() -> None:
    allowed = MAINTAINED_FILES | set(ALIASES.values()) | {"editions/v0.60/"+name for name in SELECTED_MEMBERS}
    for rel in packaged_files():
        require(not rel.startswith("sources/")
                and not re.search(r"v0\.(?!60(?:[^0-9]|$))[0-9]|editorial_history|(?:^|/)changes/", rel),
                f"Predecessor paper payload must remain external: {rel}")
        require(rel in allowed, f"Unselected public paper payload: {rel}")
        if rel.startswith("editions/v0.60/"):
            require(rel.removeprefix("editions/v0.60/") in SELECTED_MEMBERS,
                    f"Unselected supplied payload: {rel}")
    require(not (BASE/"editions/Empirical_Lawhood_v0.60_Source_Bundle.zip").exists(),
            "Complete original ZIP must remain external.")


def check_canonical_inputs() -> int:
    inputs = read_json("source/canonical-inputs.json")
    preservation = read_json("source/source-preservation.json")
    require(inputs["edition"] == preservation["edition"] == "0.60", "Preservation edition differs.")
    require(inputs["files"] == preservation["selected_inputs"], "Canonical maps differ.")
    archive = inputs["archive"]
    require(archive == preservation["archive"]
            and archive == {"path":SELECTED_ARCHIVE, "bytes":2128309,
                "sha256":SELECTED_ARCHIVE_SHA256, "members":17,
                "kind":"CURATED_CURRENT_EDITION_ONLY_EXPORT"}
            and inputs["archive_sha256"] == preservation["archive_sha256"] == SELECTED_ARCHIVE_SHA256,
            "Curated archive binding differs.")
    require(inputs["original_archive"] == preservation["original_archive"] == {
        "filename":"Empirical_Lawhood_v0.60_Source_Bundle.zip", "bytes":3456552,
        "sha256":ARCHIVE_SHA256, "members":40, "status":"EXTERNALLY_PRESERVED_NOT_PACKAGED"},
        "External original archive provenance differs.")
    excluded = inputs["excluded_original_members"]
    require(excluded == preservation["excluded_original_members"] and len(excluded) == 23
            and len({r["member"] for r in excluded}) == 23
            and all(r["publicly_packaged"] is False for r in excluded)
            and hashlib.sha256(json.dumps(excluded,sort_keys=True,separators=(",",":")).encode()).hexdigest()
                == EXCLUDED_MEMBER_INVENTORY_SHA256,
            "Excluded original member inventory differs.")
    check_current_edition_only()
    archive_path = BASE/archive["path"]
    require(archive_path.stat().st_size == archive["bytes"]
            and digest(archive_path) == SELECTED_ARCHIVE_SHA256, "Curated archive bytes differ.")
    records = inputs["files"]
    require(len(records) == len({r["path"] for r in records}), "Duplicate selected path.")
    prefix = "Empirical_Lawhood_v0.60/"
    expected_map = {"editions/v0.60/"+name:prefix+name for name in SELECTED_MEMBERS}
    expected_map.update({path:prefix+member for member,path in ALIASES.items()})
    require({r["path"]:r["member"] for r in records} == expected_map, "Selected public member map differs.")
    with zipfile.ZipFile(archive_path) as z:
        require(z.testzip() is None, "Curated ZIP CRC check failed.")
        members = {i.filename for i in z.infolist() if not i.is_dir()}
        require(len(z.infolist()) == 17 and members == {prefix+name for name in SELECTED_MEMBERS},
                "Curated17-member map differs.")
        require(not members & {r["member"] for r in excluded}, "Selected and excluded members overlap.")
        for r in records:
            raw = z.read(r["member"])
            require(len(raw) == r["bytes"] and hashlib.sha256(raw).hexdigest() == r["sha256"]
                    and (BASE/r["path"]).read_bytes() == raw, f"Canonical member differs: {r['path']}")
    translation = preservation["public_translation"]
    require(all(translation[k] == 0 for k in ("scientific_text_edits", "bibliographic_edits",
            "pdf_page_content_edits", "pdf_metadata_edits", "pdf_navigation_value_edits", "figure_edits")),
            "Original publication bytes must remain as supplied.")
    for rel, r in preservation["documents"].items():
        require(r["supplied_bytes_equal"] and r["supplied_sha256"] == r["public_sha256"] == digest(BASE/rel),
                f"Supplied document identity differs: {rel}")
    return len(records)


def check_links_and_references() -> int:
    checked = 0
    for rel in DOCS:
        p = BASE/rel
        plain = re.sub(r"```.*?```", "", p.read_text(), flags=re.S)
        for target in re.findall(r'!?\[[^\]\n]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)', plain):
            split = urlsplit(target.strip("<>"))
            if split.scheme or split.netloc:
                continue
            dest = (p.parent/unquote(split.path)).resolve() if split.path else p
            require(BASE.parent in dest.parents and dest.exists(), f"Missing local link: {rel}: {target}")
            if split.fragment and dest.suffix == ".md":
                require(unquote(split.fragment) in anchors(dest.read_text()), f"Missing heading: {target}")
            checked += 1
    main = (BASE/"manuscript.md").read_text()
    actual = [(int(n), t.strip()) for n,t in re.findall(r"^\[(\d+)\] (.+)$", main, re.M)]
    registry = read_json("source/bibliography.json")
    require(actual == [(r["number"],r["entry"]) for r in registry["external_references"]]
            and [n for n,_ in actual] == list(range(1,29)), "Selected28-reference roster differs.")
    return checked


def check_summary_arithmetic() -> dict:
    values = read_json('source/reported-values.json')
    require(values['edition'] == '0.60', 'Summary edition differs.')
    main = (BASE/'manuscript.md').read_text()
    rc, g, f, t, m, b = (values[k] for k in ('rc','geometry','factor','transient','fixed_coefficient_composition','boundary_production'))
    require((rc['seed_blocks'],rc['nominations'],rc['nonadverse'],rc['response_adverse_only'],rc['response_and_decision_adverse'])
            == (36,2319,795,625,899), 'RC unit/partition operands differ.')
    require(rc['nonadverse']+rc['response_adverse_only']+rc['response_and_decision_adverse'] == rc['nominations'], 'RC partition differs.')
    rc_time = (4/3)*math.log(4)
    separation = (math.exp(-rc_time/4)-math.exp(-rc_time))/4
    require(round(separation,5) == rc['max_separation_over_V0'] and round(separation/2,5) == rc['minimax_bound_over_V0'], 'RC rounded analytical bound differs.')
    require(g['label_00']+g['label_01'] == g['valid_trajectories'] == 9126, 'Geometry feasibility partition differs.')
    require(g['shooting_cohorts']*g['branches_per_cohort'] == g['resolved_branches'] == 448, 'Conditional shooting roster differs.')
    require(g['source_admitted_checkpoints']*g['branches_per_cohort'] == g['source_admitted_branches'] == 384
            and g['rolling_target_first_successes'] == 0 and not g['dependent_constitutive_stages_executed'], 'Geometry event/negative stage role differs.')
    require(f['histories'] == 64 and f['horizon_ticks'] == 32 and f['histories_worst_loss_above_one'] == 63,
            'Factor/directional analysis population differs.')
    require(not t['fresh_confirmation'] and not t['future_hold_baseline_available_to_selector']
            and t['support_failures_forecast_inside'] == 10, 'Retained transient causal/evidence boundary differs.')
    ratio = t['baseline_squared_contribution']/t['increment_squared_contribution']
    require(round(ratio) == t['rounded_ratio'] == 781, 'Rounded baseline/increment ratio differs.')
    require(m['episodes'] == 64 and m['joint_successes'] == 61 and m['false_admissions'] == 0
            and m['delivered_assays'] == 125 and m['nonattempts'] == 3 and m['cancellations'] == 0, 'Fixed coefficient composition independent-unit/accounting operands differ.')
    require(abs(lower_binomial(61,64)-m['one_sided_95_success_lower']) < .00005
            and abs(1-.05**(1/64)-m['one_sided_95_false_admission_upper']) < .000005, 'Fixed coefficient composition rounded exact bounds differ.')
    require(m['calibration_order_statistic'] == 30 and m['calibration_episodes'] == 32
            and m['marginal_target'] == .9 and m['final_first_example_admission_interval'] is None
            and m['paired_p'] == .5, 'Fixed coefficient composition calibration/absence/comparison role differs.')
    mse_cached = 100*(1-m['composed_mse']/m['cached_mse'])
    mse_direct = 100*(1-m['composed_mse']/m['direct_mse'])
    require(abs(mse_cached-m['loss_reduction_vs_cached_percent']) < .005
            and abs(mse_direct-m['loss_reduction_vs_direct_percent']) < .005, 'Fixed coefficient composition descriptive rounded MSE ratios differ.')
    require(b['qualification_episodes_excluded'] == 8 and b['confirmation_episodes'] == 32
            and b['pairs_per_episode'] == 256 and b['pairs'] == 8192
            and b['confirmation_episodes']*b['pairs_per_episode'] == b['pairs'], 'Qualification/confirmation/nested pair roles differ.')
    require(b['primary_successes'] == {'H':0,'N':7965,'P':0}
            and b['primary_validity_episodes'] == {'H':0,'N':32,'P':0}, 'Primary capstone outcomes differ.')
    require(b['paired_p'] == 2.0**(-32) and b['paired_favourable'] == 32 and b['paired_reverse'] == 0,
            'Episode paired comparison differs.')
    gain = 100*7965/8192
    bypass = 100*(7965-7916)/8192
    conservative = 100*6720/8192
    require(abs(gain-b['primary_gain_percentage_points']) < .005
            and abs(bypass-b['support_bypass_gain_percentage_points']) < 1e-6
            and abs(conservative-b['conservative_gain_percentage_points']) < .005, 'Primary/posthoc gain arithmetic differs.')
    require(b['support_bypass_counts'] == {'H':7916,'N':7965,'P':7952}
            and b['support_bypass_difference_pairs'] == 49
            and b['both_view_episodes'] == 31 and b['both_view_pairs'] == 7719
            and b['conservative_episodes'] == 27 and b['conservative_pairs'] == 6720, 'Sensitivity populations/counts differ.')
    require(abs(100*(1-.05**(1/32))-b['zero_failure_one_sided_95_upper_percent']) < .005
            and b['false_admission_episodes'] == 0 and b['uncertainty_unit'] == 'episode, not request pair', 'Episode risk statement differs.')
    require(b['bootstrap_confidence_level'] is None and b['formal_admission'] == 'NOT_EVALUATED', 'Absent bootstrap specification or formal status was invented.')
    require(b['source_perturbation_scale']**2 == b['source_perturbation_variance']
            and (b['preparation_selection_tick'],b['handoff_tick'],b['response_end_tick']) == (4096,4496,4688), 'Source perturbation/causal timing differs.')
    require(all(token in main for token in ('9,126', '0/384', '0.740', '93.0%', '90.1%', '781', '7,965', '6,720')),
            'Selected published numerical anchors differ.')
    return {'scope':'Recoverable published-summary arithmetic and role checks only. No native replay, model fitting or bootstrap.',
            'RC_max_separation_over_V0':separation,'RC_minimax_bound_over_V0':separation/2,
            'fixed_coefficient_composition_success_one_sided_95_lower':lower_binomial(61,64),
            'fixed_coefficient_composition_zero_failure_one_sided_95_upper':1-.05**(1/64),
            'fixed_coefficient_composition_MSE_reduction_vs_cached_percent':mse_cached,'fixed_coefficient_composition_MSE_reduction_vs_direct_percent':mse_direct,
            'baseline_to_increment_squared_ratio':ratio,'capstone_paired_p':2.0**(-32),
            'primary_gain_percentage_points':gain,'bypass_gain_percentage_points':bypass,
            'conservative_gain_percentage_points':conservative,'capstone_episode_failure_one_sided_95_upper':1-.05**(1/32)}

def check_author_qa() -> dict:
    metadata = read_json("metadata.json")
    qa = read_json("source/pdf-preflight.json")
    require(metadata["edition"] == qa["edition"] == "0.60", "Current QA edition differs.")
    require(qa["result"] == "PASS_ORIGINAL_PUBLICATION_VISUAL_QA" and not qa["critical_visual_findings"], "Current visual review incomplete.")
    require(not qa["native_scientific_reproduction"] and qa["navigation_edits"] == 0, "Review ceiling differs.")
    observed = {}
    require(set(qa["documents"]) == set(PDFS), "Selected PDF review roster differs.")
    for rel in PDFS:
        r = qa["documents"][rel]
        require(digest(BASE/rel) == r["sha256"] and (BASE/rel).stat().st_size == r["bytes"], f"PDF review bytes differ: {rel}")
        with pymupdf.open(BASE/rel) as doc:
            require(len(doc) == r["pages"] and r["visually_inspected_pages"] == list(range(1,len(doc)+1)), "Complete original-page review differs.")
            require(doc.metadata == r["metadata"] and all(p.get_text().strip() for p in doc), "Original PDF metadata/text differs.")
            require(all(1 <= t[2] <= len(doc) for t in doc.get_toc()), "Original PDF has unresolved bookmark destination.")
            observed[rel] = {"sha256":r["sha256"],"pages":len(doc),"navigation":"Original supplied bytes unchanged"}
    require(len(qa["figures"]) == 6, "Original six-format figure review differs.")
    for r in qa["figures"]:
        require(r["visually_inspected"] and digest(BASE/r["path"]) == r["sha256"], "Figure review binding differs.")
    deposited = qa["deposited_manuscript"]
    path = BASE/deposited["path"]
    require(digest(path) == deposited["sha256"] == "542c7450f5504316ffab6a28f4f626a963225ff6536753577781acafbd7760db"
            and path.stat().st_size == deposited["bytes"] == 875723
            and hashlib.md5(path.read_bytes()).hexdigest() == deposited["md5"], "Deposited publication bytes differ.")
    require(deposited["visually_inspected_pages"] == list(range(1,20)), "Deposited publication review incomplete.")
    require(digest(BASE/deposited["record_path"]) == deposited["record_sha256"], "Verified deposit record differs.")
    record = read_json(deposited["record_path"])
    require(record["metadata"]["doi"] == "10.5281/zenodo.23083988"
            and record["metadata"]["version"] == "0.60"
            and record["metadata"]["publication_date"] == "2026-10-01"
            and record["metadata"]["resource_type"]["subtype"] == "preprint", "Verified deposit metadata differs.")
    require(read_json("source/source-preservation.json")["deposited_publication"] == deposited,
            "Deposit preservation and review bindings differ.")
    observed[deposited["path"]] = {"sha256":deposited["sha256"],"pages":19,"role":"Verified deposited original, distinct supplied PDF"}
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_storage_arguments(parser)
    args = parser.parse_args()
    output = external_directory(args.output_dir,args,parser)
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new empty validation directory. Keep earlier receipts.")
    file_count = check_manifest()
    canonical_count = check_canonical_inputs()
    links = check_links_and_references()
    arithmetic = check_summary_arithmetic()
    qa = check_author_qa()
    output.mkdir(parents=True,exist_ok=True)
    result = {"edition":"0.60","result":"PASS_LOCAL_CANONICAL_SELECTION_AND_AUTHOR_QA",
              "scope":"Original publication preservation, references, reported-summary arithmetic and exact selected visual review. No paper regeneration, native reproduction or new scientific qualification.",
              "packaged_files":file_count,"canonical_inputs":canonical_count,"local_links":links,
              "pdfs":qa,"arithmetic":arithmetic,"navigation_corrections":[],
              "owner_sha256":digest(Path(__file__)),"bundle_manifest_sha256":digest(BASE/"bundle-manifest.json"),
              "source_preservation_sha256":digest(BASE/"source/source-preservation.json"),"qa_sha256":digest(BASE/"source/pdf-preflight.json")}
    (output/"numerical-validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("result","packaged_files","canonical_inputs","local_links")},indent=2))


if __name__ == "__main__":
    main()
