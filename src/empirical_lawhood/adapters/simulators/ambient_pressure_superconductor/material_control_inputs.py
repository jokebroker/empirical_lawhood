'Deterministic material control control-input construction from the frozen material source design roster.\n\nThe generated archive is a source operand, not an experiment result.  It\ncontains exact QE/PH/EPW inputs and only the selected SSSP pseudopotentials for\nthe five calibration controls.  Materialization is non-overwriting and confined to the\nguarded external scientific volume.\n'

from __future__ import annotations

from empirical_lawhood._required_inputs import required_external_path

from dataclasses import dataclass
from gzip import GzipFile
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import tarfile
from typing import Final

from .material_source_design_contracts import MaterialStructureSpec, SolverViewFreeze
from .material_source_design_design import build_material_roster, build_science_design


INPUT_BUNDLE_LOCATOR: Final = (
    'sources/ambient-pressure-superconductor/material-control-control-inputs/material-control-control-inputs.tar.gz'
)

_SSSP_LOCATORS: Final = {
    "view.pbe-efficiency-base": 'sources/ambient-pressure-superconductor/sssp-1.3.0-pbe-efficiency',
    "view.pbe-precision-refined": 'sources/ambient-pressure-superconductor/sssp-1.3.0-pbe-precision',
}

_MASS: Final = {
    "B": "10.811",
    "C": "12.011",
    "Cu": "63.546",
    "Mg": "24.305",
    "Pb": "207.2",
}


@dataclass(frozen=True, slots=True)
class GeneratedInputBundle:
    payload: bytes
    sha256: str
    size_bytes: int
    member_sha256s: tuple[tuple[str, int, str], ...]


def _view_pseudos(
    view: SolverViewFreeze,
) -> tuple[dict[str, bytes], dict[str, dict[str, object]]]:
    root = required_external_path("EMPIRICAL_LAWHOOD_AP_EXTERNAL_ROOT") / _SSSP_LOCATORS[view.view_id]
    archive_path = next(root.glob("*.tar.gz"))
    metadata_path = next(root.glob("*.json"))
    decoded = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) and isinstance(value, dict) for key, value in decoded.items()
    ):
        raise ValueError("SSSP metadata is not an element-object mapping")
    metadata: dict[str, dict[str, object]] = decoded
    wanted = {
        element: str(metadata[element]["filename"]) for element in ("Pb", "Mg", "B", "Cu", "C")
    }
    payloads: dict[str, bytes] = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        by_name = {Path(member.name).name: member for member in archive if member.isfile()}
        for element, filename in wanted.items():
            member = by_name[filename]
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f'SSSP pseudopotential cannot be read: {filename}')
            payload = stream.read(member.size + 1)
            if len(payload) != member.size:
                raise ValueError(f'SSSP pseudopotential size differs: {filename}')
            payloads[element] = payload
    return payloads, metadata


def _namelist(name: str, rows: tuple[tuple[str, str], ...]) -> str:
    body = "\n".join(f'  {key:<18} = {value}' for key, value in rows)
    return f'&{name}\n{body}\n/\n'


def _metadata_float(metadata: dict[str, object], key: str) -> float:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'SSSP metadata lacks numeric {key}')
    return float(value)


def _structure_cards(
    structure: MaterialStructureSpec,
    *,
    pseudo_names: dict[str, str],
) -> str:
    elements = tuple(element for element, _count in structure.formula)
    species = "\n".join(
        f'  {element:<3} {_MASS[element]:>8} {pseudo_names[element]}' for element in elements
    )
    positions = "\n".join(
        f'  {site.element:<3} '
        + " ".join(f'{str(value):>18}' for value in site.fractional_coordinates)
        for site in structure.sites
    )
    cell = "\n".join(
        "  " + " ".join(f'{str(value):>18}' for value in vector)
        for vector in structure.lattice_vectors_A
    )
    return (
        f'ATOMIC_SPECIES\n{species}\n'
        f'ATOMIC_POSITIONS crystal\n{positions}\n'
        f'CELL_PARAMETERS angstrom\n{cell}\n'
    )


def _scf_input(
    structure: MaterialStructureSpec,
    view: SolverViewFreeze,
    *,
    pseudo_names: dict[str, str],
    prefix: str,
    ecutwfc_Ry: float,
    ecutrho_Ry: float,
    fixed_occupations: bool = False,
) -> bytes:
    system_rows: list[tuple[str, str]] = [
        ("ibrav", "0"),
        ("nat", str(len(structure.sites))),
        ("ntyp", str(len(structure.formula))),
        ("ecutwfc", str(ecutwfc_Ry)),
        ("ecutrho", str(ecutrho_Ry)),
    ]
    if fixed_occupations:
        system_rows.extend((("occupations", "'fixed'"), ("nbnd", "8")))
    else:
        system_rows.extend(
            (
                ("occupations", "'smearing'"),
                ("smearing", "'m-p'"),
                ("degauss", str(view.smearing_Ry)),
            )
        )
    text = _namelist(
        "CONTROL",
        (
            ("calculation", "'scf'"),
            ("prefix", f"'{prefix}'"),
            ("restart_mode", "'from_scratch'"),
            ("pseudo_dir", "'../pseudo'"),
            ("outdir", "'./out'"),
            ("verbosity", "'high'"),
            ("tprnfor", ".true."),
            ("tstress", ".true."),
        ),
    )
    text += _namelist("SYSTEM", tuple(system_rows))
    text += _namelist(
        "ELECTRONS",
        (
            ("diagonalization", "'david'"),
            ("mixing_mode", "'local-TF'"),
            ("mixing_beta", "0.3"),
            ("electron_maxstep", "200"),
            ("conv_thr", "1.0d-10"),
        ),
    )
    text += _structure_cards(structure, pseudo_names=pseudo_names)
    text += "K_POINTS automatic\n" + "  " + " ".join(map(str, view.k_mesh)) + " 0 0 0\n"
    return text.encode("ascii")


def _explicit_k_points(mesh: tuple[int, int, int]) -> str:
    count = mesh[0] * mesh[1] * mesh[2]
    weight = 1.0 / count
    rows = ["K_POINTS crystal", str(count)]
    for i in range(mesh[0]):
        for j in range(mesh[1]):
            for k in range(mesh[2]):
                rows.append(
                    f'  {i / mesh[0]:.12f} {j / mesh[1]:.12f} {k / mesh[2]:.12f} {weight:.16e}'
                )
    return "\n".join(rows) + "\n"


def _nscf_input(
    structure: MaterialStructureSpec,
    view: SolverViewFreeze,
    *,
    pseudo_names: dict[str, str],
    prefix: str,
    nbnd: int,
    ecutwfc_Ry: float,
    ecutrho_Ry: float,
) -> bytes:
    text = _namelist(
        "CONTROL",
        (
            ("calculation", "'nscf'"),
            ("prefix", f"'{prefix}'"),
            ("restart_mode", "'from_scratch'"),
            ("pseudo_dir", "'../pseudo'"),
            ("outdir", "'./out'"),
            ("verbosity", "'high'"),
        ),
    )
    text += _namelist(
        "SYSTEM",
        (
            ("ibrav", "0"),
            ("nat", str(len(structure.sites))),
            ("ntyp", str(len(structure.formula))),
            ("nbnd", str(nbnd)),
            ("ecutwfc", str(ecutwfc_Ry)),
            ("ecutrho", str(ecutrho_Ry)),
            ("occupations", "'smearing'"),
            ("smearing", "'m-p'"),
            ("degauss", str(view.smearing_Ry)),
            ("nosym", ".true."),
            ("noinv", ".true."),
        ),
    )
    text += _namelist(
        "ELECTRONS",
        (
            ("diagonalization", "'david'"),
            ("mixing_mode", "'local-TF'"),
            ("mixing_beta", "0.3"),
            ("electron_maxstep", "200"),
            ("conv_thr", "1.0d-10"),
        ),
    )
    text += _structure_cards(structure, pseudo_names=pseudo_names)
    text += _explicit_k_points(view.k_mesh)
    return text.encode("ascii")


def _ph_input(prefix: str, view: SolverViewFreeze) -> bytes:
    q1, q2, q3 = view.q_mesh
    return (
        f'{prefix} material control control\n'
        + _namelist(
            "INPUTPH",
            (
                ("prefix", f"'{prefix}'"),
                ("outdir", "'./out'"),
                ("trans", ".true."),
                ("reduce_io", ".true."),
                ("fildyn", f"'{prefix}.dyn'"),
                ("fildvscf", "'dvscf'"),
                ("ldisp", ".true."),
                ("nq1", str(q1)),
                ("nq2", str(q2)),
                ("nq3", str(q3)),
                ("tr2_ph", "1.0d-14"),
                ("alpha_mix(1)", "0.1"),
                ("alpha_mix(10)", "0.3"),
            ),
        )
    ).encode("ascii")


def _q2r_input(prefix: str) -> bytes:
    return _namelist(
        "INPUT",
        (
            ("fildyn", f"'{prefix}.dyn'"),
            ("flfrc", f"'{prefix}.fc.xml'"),
            ("zasr", "'crystal'"),
        ),
    ).encode("ascii")


def _matdyn_input(prefix: str, view: SolverViewFreeze) -> bytes:
    n1, n2, n3 = view.fine_q_mesh
    return _namelist(
        "INPUT",
        (
            ("flfrc", f"'{prefix}.fc.xml'"),
            ("asr", "'crystal'"),
            ("dos", ".true."),
            ("fldos", f"'{prefix}.phdos'"),
            ("nk1", str(n1)),
            ("nk2", str(n2)),
            ("nk3", str(n3)),
        ),
    ).encode("ascii")


def _epw_common(prefix: str, view: SolverViewFreeze) -> tuple[tuple[str, str], ...]:
    nk1, nk2, nk3 = view.k_mesh
    nq1, nq2, nq3 = view.q_mesh
    return (
        ("prefix", f"'{prefix}'"),
        ("outdir", "'./out'"),
        ("dvscf_dir", "'../phonon/save'"),
        ("etf_mem", "0"),
        ("epw_memdist", ".true."),
        ("lifc", ".true."),
        ("asr_typ", "'crystal'"),
        ("nk1", str(nk1)),
        ("nk2", str(nk2)),
        ("nk3", str(nk3)),
        ("nq1", str(nq1)),
        ("nq2", str(nq2)),
        ("nq3", str(nq3)),
    )


def _pb_epw_inputs(view: SolverViewFreeze) -> tuple[bytes, bytes]:
    common = _epw_common("pb", view)
    epw1 = common + (
        ("amass(1)", "207.2"),
        ("elph", ".true."),
        ("epbwrite", ".true."),
        ("epbread", ".false."),
        ("epwwrite", ".true."),
        ("epwread", ".false."),
        ("wannierize", ".true."),
        ("nbndsub", "4"),
        ("bands_skipped", "'exclude_bands = 1:5'"),
        ("num_iter", "3000"),
        ("iprint", "2"),
        ("iverbosity", "2"),
        ("dis_froz_max", "16.5"),
        ("dis_froz_min", "-1.0"),
        ("proj(1)", "'Pb:sp3'"),
        ("wdata(1)", "'write_hr = .true.'"),
        ("wdata(2)", "'guiding_centres = .true.'"),
        ("wdata(3)", "'use_ws_distance = .true.'"),
        ("wdata(4)", "'write_rmn = .true.'"),
        ("wdata(5)", "'write_u_matrices = .true.'"),
        ("nkf1", "1"),
        ("nkf2", "1"),
        ("nkf3", "1"),
        ("nqf1", "1"),
        ("nqf2", "1"),
        ("nqf3", "1"),
    )
    fk1, fk2, fk3 = view.fine_k_mesh
    fq1, fq2, fq3 = view.fine_q_mesh
    epw2 = common + (
        ("amass(1)", "207.2"),
        ("elph", ".true."),
        ("epwread", ".true."),
        ("epwwrite", ".false."),
        ("wannierize", ".false."),
        ("nbndsub", "4"),
        ("bands_skipped", "'exclude_bands = 1:5'"),
        ("fsthick", "0.2"),
        ("degaussw", "0.05"),
        ("degaussq", "0.15"),
        ("a2f_iso", ".true."),
        ("eliashberg", ".true."),
        ("liso", ".true."),
        ("limag", ".true."),
        ("lpade", ".true."),
        ("lacon", ".true."),
        ("nsiter", "500"),
        ("npade", "12"),
        ("conv_thr_iaxis", "1.0d-4"),
        ("conv_thr_racon", "1.0d-4"),
        ("wscut", "0.1"),
        ("muc", "0.1"),
        (
            "temps",
            "0.3 0.9 1.5 2.1 2.7 3.3 4.0 4.5 4.6 4.7 4.8 4.9 5.0 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9 6.0",
        ),
        ("mp_mesh_k", ".true."),
        ("nkf1", str(fk1)),
        ("nkf2", str(fk2)),
        ("nkf3", str(fk3)),
        ("nqf1", str(fq1)),
        ("nqf2", str(fq2)),
        ("nqf3", str(fq3)),
    )
    return _namelist("INPUTEPW", epw1).encode("ascii"), _namelist("INPUTEPW", epw2).encode("ascii")


def _mgb2_epw_inputs(view: SolverViewFreeze) -> tuple[bytes, bytes]:
    common = _epw_common("mgb2", view)
    fk1, fk2, fk3 = view.fine_k_mesh
    fq1, fq2, fq3 = view.fine_q_mesh
    epw1 = common + (
        ("ep_coupling", ".true."),
        ("elph", ".true."),
        ("epwwrite", ".true."),
        ("epwread", ".false."),
        ("wannierize", ".true."),
        ("nbndsub", "5"),
        ("num_iter", "500"),
        ("dis_froz_max", "10.5"),
        ("proj(1)", "'B:pz'"),
        ("proj(2)", "'f=0.5,1.0,0.5:s'"),
        ("proj(3)", "'f=0.0,0.5,0.5:s'"),
        ("proj(4)", "'f=0.5,0.5,0.5:s'"),
        ("wdata(1)", "'write_hr = .true.'"),
        ("wdata(2)", "'guiding_centres = .true.'"),
        ("wdata(3)", "'use_ws_distance = .true.'"),
        ("wdata(4)", "'write_rmn = .true.'"),
        ("wdata(5)", "'write_u_matrices = .true.'"),
        ("iverbosity", "2"),
        ("fsthick", "0.2"),
        ("degaussw", "0.05"),
        ("degaussq", "0.5"),
        ("ephwrite", ".true."),
        ("eliashberg", ".true."),
        ("laniso", ".true."),
        ("limag", ".true."),
        ("lpade", ".true."),
        ("lacon", ".false."),
        ("nsiter", "500"),
        ("conv_thr_iaxis", "1.0d-4"),
        ("wscut", "0.5"),
        ("muc", "0.1"),
        ("nstemp", "9"),
        ("temps", "5 45"),
        ("mp_mesh_k", ".true."),
        ("nkf1", str(fk1)),
        ("nkf2", str(fk2)),
        ("nkf3", str(fk3)),
        ("nqf1", str(fq1)),
        ("nqf2", str(fq2)),
        ("nqf3", str(fq3)),
    )
    epw2 = common + (
        ("ep_coupling", ".false."),
        ("elph", ".false."),
        ("epwwrite", ".false."),
        ("epwread", ".true."),
        ("wannierize", ".false."),
        ("nbndsub", "5"),
        ("fsthick", "0.2"),
        ("degaussw", "0.05"),
        ("degaussq", "0.5"),
        ("ephwrite", ".false."),
        ("eliashberg", ".true."),
        ("laniso", ".true."),
        ("fbw", ".true."),
        ("limag", ".true."),
        ("lpade", ".true."),
        ("lacon", ".false."),
        ("nsiter", "500"),
        ("conv_thr_iaxis", "1.0d-4"),
        ("wscut", "0.5"),
        ("muc", "0.1"),
        # The exact EPW 6.1 environment was stable through 45 K but its
        # full-bandwidth 50 K solve hit a Broyden factorization failure.  Keep
        # the same predeclared temperature support as the Fermi-surface run.
        ("nstemp", "9"),
        ("temps", "5 45"),
        ("mp_mesh_k", ".true."),
        ("nkf1", str(fk1)),
        ("nkf2", str(fk2)),
        ("nkf3", str(fk3)),
        ("nqf1", str(fq1)),
        ("nqf2", str(fq2)),
        ("nqf3", str(fq3)),
    )
    return _namelist("INPUTEPW", epw1).encode("ascii"), _namelist("INPUTEPW", epw2).encode("ascii")


def build_material_control_input_files() -> dict[str, bytes]:
    roster = build_material_roster()
    structures = {value.structure_id: value for value in roster.structures}
    views = {
        value.view_id: value
        for value in build_science_design().solver_views
        if value.view_id in _SSSP_LOCATORS
    }
    control_ids = (
        'structure.calibration-c-diamond',
        'structure.calibration-cu-fcc',
        'structure.calibration-mgb2-alb2',
        'structure.calibration-pb-fcc',
        'structure.calibration-pb-sc-compressed',
    )
    result: dict[str, bytes] = {}
    for view_id in sorted(views):
        view = views[view_id]
        pseudo_payloads, metadata = _view_pseudos(view)
        pseudo_names = {element: str(metadata[element]["filename"]) for element in pseudo_payloads}
        view_name = view_id.removeprefix("view.")
        for structure_id in control_ids:
            structure = structures[structure_id]
            case = structure_id.removeprefix("structure.")
            root = f'material-control-control-inputs/{view_name}/{case}'
            for element, _count in structure.formula:
                result[f'{root}/pseudo/{pseudo_names[element]}'] = pseudo_payloads[element]
            prefix = {
                'structure.calibration-pb-fcc': "pb",
                'structure.calibration-mgb2-alb2': "mgb2",
                'structure.calibration-cu-fcc': "cu",
                'structure.calibration-c-diamond': "diamond",
                'structure.calibration-pb-sc-compressed': "pbsc",
            }[structure_id]
            ecutwfc = max(
                _metadata_float(metadata[element], "cutoff_wfc")
                for element, _count in structure.formula
            )
            ecutrho = max(
                _metadata_float(metadata[element], "cutoff_rho")
                for element, _count in structure.formula
            )
            fixed = structure_id == 'structure.calibration-c-diamond'
            result[f'{root}/scf/scf.in'] = _scf_input(
                structure,
                view,
                pseudo_names=pseudo_names,
                prefix=prefix,
                ecutwfc_Ry=ecutwfc,
                ecutrho_Ry=ecutrho,
                fixed_occupations=fixed,
            )
            if structure_id in {
                'structure.calibration-pb-fcc',
                'structure.calibration-mgb2-alb2',
                'structure.calibration-pb-sc-compressed',
            }:
                result[f'{root}/phonon/scf.in'] = _scf_input(
                    structure,
                    view,
                    pseudo_names=pseudo_names,
                    prefix=prefix,
                    ecutwfc_Ry=ecutwfc,
                    ecutrho_Ry=ecutrho,
                )
                result[f'{root}/phonon/ph.in'] = _ph_input(prefix, view)
                result[f'{root}/phonon/q2r.in'] = _q2r_input(prefix)
                result[f'{root}/phonon/matdyn.in'] = _matdyn_input(prefix, view)
            if structure_id in {
                'structure.calibration-pb-fcc',
                'structure.calibration-mgb2-alb2',
            }:
                nbnd = 16
                result[f'{root}/epw/scf.in'] = _scf_input(
                    structure,
                    view,
                    pseudo_names=pseudo_names,
                    prefix=prefix,
                    ecutwfc_Ry=ecutwfc,
                    ecutrho_Ry=ecutrho,
                )
                result[f'{root}/epw/nscf.in'] = _nscf_input(
                    structure,
                    view,
                    pseudo_names=pseudo_names,
                    prefix=prefix,
                    nbnd=nbnd,
                    ecutwfc_Ry=ecutwfc,
                    ecutrho_Ry=ecutrho,
                )
                epw1, epw2 = (
                    _pb_epw_inputs(view)
                    if structure_id == 'structure.calibration-pb-fcc'
                    else _mgb2_epw_inputs(view)
                )
                result[f'{root}/epw/epw1.in'] = epw1
                result[f'{root}/epw/epw2.in'] = epw2
    manifest_rows = [
        {
            "path": path,
            "sha256": sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for path, payload in sorted(result.items())
    ]
    manifest = {
        "base_roster_sha256": build_material_roster().fingerprint(),
        "base_science_sha256": build_science_design().fingerprint(),
        "member_count_excluding_manifest": len(result),
        "members": manifest_rows,
        "schema": 'empirical-lawhood/material/public-control-readiness/input-manifest',
        "target_contact_count": 0,
        "version": '1.0.0',
    }
    result["material-control-control-inputs/manifest.json"] = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return result


def build_material_control_input_bundle() -> GeneratedInputBundle:
    files = build_material_control_input_files()
    tar_buffer = BytesIO()
    with GzipFile(fileobj=tar_buffer, mode="wb", mtime=0, filename="") as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            directories = {
                str(parent) for path in files for parent in Path(path).parents if str(parent) != "."
            }
            for directory in sorted(directories):
                info = tarfile.TarInfo(directory)
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info)
            for path, payload in sorted(files.items()):
                info = tarfile.TarInfo(path)
                info.size = len(payload)
                info.mode = 0o644
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, BytesIO(payload))
    payload = tar_buffer.getvalue()
    return GeneratedInputBundle(
        payload=payload,
        sha256=sha256(payload).hexdigest(),
        size_bytes=len(payload),
        member_sha256s=tuple(
            (path, len(value), sha256(value).hexdigest()) for path, value in sorted(files.items())
        ),
    )


def materialize_material_control_input_bundle(path: Path | None = None) -> GeneratedInputBundle:
    external_root = required_external_path("EMPIRICAL_LAWHOOD_AP_EXTERNAL_ROOT")
    if path is None:
        path = external_root / INPUT_BUNDLE_LOCATOR
    external = external_root.resolve()
    mount = required_external_path("EMPIRICAL_LAWHOOD_AP_MOUNT")
    if not os.path.ismount(mount) or not os.access(mount, os.W_OK):
        raise ValueError("external scientific storage is not an active writable mount")
    lexical_parent = path.parent.absolute()
    if external != lexical_parent and external not in lexical_parent.parents:
        raise ValueError('material control input bundle resolves outside external scientific storage')
    parent = path.parent
    while parent != external:
        if parent.exists() and parent.is_symlink():
            raise ValueError('material control input parent chain contains a symlink')
        parent = parent.parent
    if path.exists() or path.is_symlink():
        raise ValueError('material control input materialization is non-overwriting')
    if shutil.disk_usage(external).free < 10 * 1024**3:
        raise ValueError('material control input materialization lacks its free-space floor')
    bundle = build_material_control_input_bundle()
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = path.parent.resolve()
    if external != resolved_parent and external not in resolved_parent.parents:
        raise ValueError('created material control input parent escaped external storage')
    partial = path.with_name(f'.{path.name}.partial')
    if partial.exists() or partial.is_symlink():
        raise ValueError('material control input partial path is not fresh')
    try:
        with partial.open("xb") as stream:
            stream.write(bundle.payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()
    return bundle


__all__ = [
    "GeneratedInputBundle",
    "INPUT_BUNDLE_LOCATOR",
    'build_material_control_input_bundle',
    'build_material_control_input_files',
    'materialize_material_control_input_bundle',
]
