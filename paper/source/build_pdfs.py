#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Build optional v0.60 publication derivatives in external storage.

Use the supplied canonical Markdown, header and Lua filter without changing
selected canonical PDFs. Requires Pandoc, XeLaTeX, the declared fonts and PyMuPDF.
This command builds publication assets. It executes no scientific experiments.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
import pymupdf
from storage import add_storage_arguments, external_directory

BASE = Path(__file__).resolve().parents[1]
STEMS = ('manuscript', 'revision-and-evidence-notes')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path, root: Path | None = None) -> dict:
    return {'path': path.relative_to(root).as_posix() if root else str(path.resolve()),
            'bytes': path.stat().st_size, 'sha256': digest(path)}


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, **kwargs)
    if result.returncode:
        raise RuntimeError('Command failed: ' + ' '.join(command) + '\n'
                           + result.stdout[-12000:] + '\n' + result.stderr[-4000:])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scratch-dir', type=Path, required=True)
    parser.add_argument('--pandoc', default=shutil.which('pandoc'))
    parser.add_argument('--engine', default=shutil.which('xelatex'))
    add_storage_arguments(parser)
    args = parser.parse_args()
    if not args.pandoc:
        import pypandoc
        args.pandoc = pypandoc.get_pandoc_path()
    if not args.engine:
        parser.error('Supply the installed XeLaTeX executable with --engine.')
    for label, executable in (('Pandoc', args.pandoc), ('XeLaTeX', args.engine)):
        if not Path(executable).is_file():
            parser.error(f'{label} executable is absent: {executable}')
    scratch = external_directory(args.scratch_dir, args, parser)
    output = external_directory(args.output_dir, args, parser)
    if output == scratch or output in scratch.parents or scratch in output.parents:
        parser.error('Use separate output and scratch directories.')
    if output.exists() and any(output.iterdir()):
        parser.error('Use a new empty output directory. Keep earlier evidence.')
    selected = json.loads((BASE/'source/canonical-inputs.json').read_text())
    input_hashes = {r['path']: r['sha256'] for r in selected['files']}
    preservation = json.loads((BASE/'source/source-preservation.json').read_text())
    for stem in STEMS:
        input_hashes[stem+'.pdf'] = preservation['documents'][stem+'.pdf']['public_sha256']
    for rel, expected in input_hashes.items():
        if digest(BASE/rel) != expected:
            parser.error(f'Canonical input identity differs: {rel}')
    scratch.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    tex_root = output/'source/tex'
    tex_root.mkdir(parents=True)
    reports = {}
    commands = []
    actual_dependencies = {}
    font_patterns = ('Linux Libertine O', 'DejaVu Sans', 'DejaVu Sans Mono')
    font_records = []
    fontconfig = shutil.which('fc-match')
    if not fontconfig:
        parser.error('Fontconfig fc-match is required to check the declared fonts.')
    for pattern in font_patterns:
        answer = run([fontconfig, '-f', '%{family}\n%{file}\n', pattern]).stdout.splitlines()
        if len(answer) < 2 or pattern not in [name.strip() for name in answer[0].split(',')]:
            parser.error(f'Declared font is absent or substituted: {pattern}')
        font_records.append({'requested_family':pattern, 'matched_family':answer[0],
                             **identity(Path(answer[1]))})
    for path in (BASE/'source/preamble.tex', BASE/'source/pdf_filter.lua', BASE/'source/pdf-compatibility.tex'):
        actual_dependencies[str(path.resolve())] = identity(path)
    with tempfile.TemporaryDirectory(prefix='el_publication_', dir=scratch) as directory:
        work = Path(directory)
        # Compile from copied inputs so supplied Lua image paths resolve externally.
        for record in selected['files']:
            source = BASE/record['path']
            dest = work/record['path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
        shutil.copyfile(BASE/'source/pdf-compatibility.tex', work/'source/pdf-compatibility.tex')
        for stem in STEMS:
            tex = tex_root/(stem+'.tex')
            command = [args.pandoc, str(work/(stem+'.md')), '-s', '--from=markdown',
                       '--to=latex', '--number-sections',
                       '--lua-filter='+str(work/'source/pdf_filter.lua'),
                       '-V', 'documentclass=article', '-V', 'fontsize=11pt',
                       '-V', 'papersize=a4',
                       '-V', 'geometry=left=22mm,right=22mm,top=22mm,bottom=22mm,headsep=6mm',
                       '-V', 'mainfont=Linux Libertine O',
                       '-V', 'sansfont=DejaVu Sans', '-V', 'monofont=DejaVu Sans Mono',
                       '-V', 'monofontoptions=Scale=0.78',
                       '-V', 'mathfont=latinmodern-math.otf', '-V', 'linestretch=1.015',
                       '-V', 'colorlinks=false',
                       '--include-in-header='+str(work/'source/preamble.tex'),
                       '--include-in-header='+str(work/'source/pdf-compatibility.tex'), '-o', str(tex)]
            if stem == 'revision-and-evidence-notes':
                command.insert(-2, '--include-in-header='+str(work/'source/evidence_metadata.tex'))
            commands.append(command)
            run(command, cwd=work)
            generated = tex.read_text()
            # Preserve the supplied edition's generated navigation; no old
            # v0.10 bookmark correction is transferred to a new derivative.
            count = 0
            console = []
            for _ in range(3):
                command = [args.engine, '-interaction=nonstopmode', '-halt-on-error',
                           '-recorder', '-output-directory='+str(work), str(tex)]
                commands.append(command)
                result = run(command, cwd=work)
                console.append(result.stdout+'\n'+result.stderr)
            (output/(stem+'.console.txt')).write_text('\n'.join(console))
            for suffix in ('.log', '.fls'):
                shutil.copyfile(work/(stem+suffix), output/(stem+suffix))
            shutil.copyfile(work/(stem+'.pdf'), output/(stem+'.pdf'))
            for line in (work/(stem+'.fls')).read_text(errors='replace').splitlines():
                if line.startswith('INPUT '):
                    dep = Path(line[6:])
                    dep = dep if dep.is_absolute() else work/dep
                    if dep.is_file():
                        actual_dependencies[str(dep.resolve())] = identity(dep)
            log = (output/(stem+'.log')).read_text(errors='replace')
            with pymupdf.open(output/(stem+'.pdf')) as doc:
                reports[stem] = {'pages':len(doc), 'metadata':doc.metadata,
                    'warnings':[line for line in log.splitlines() if any(t in line for t in
                        ('Overfull', 'Missing character', 'LaTeX Warning', 'Package hyperref Warning'))],
                    'unnumbered_heading_destinations_added':count,
                    'scientific_source_sha256':digest(BASE/(stem+'.md')),
                    'generated_tex_before_navigation_patch_sha256':hashlib.sha256(generated.encode()).hexdigest(),
                    'generated_tex_after_navigation_patch_sha256':digest(tex),
                    'navigation_patch_scope':'No navigation edits; current original supplied recipe inputs. New derivative needs its own review.',
                    'selected_canonical_pdf_sha256':digest(BASE/(stem+'.pdf')),
                    'rebuilt_derivative_sha256':digest(output/(stem+'.pdf')),
                    'visual_review':'REQUIRED_FOR_THIS_NEW_DERIVATIVE'}
            print(f'Built optional {stem}.pdf: {reports[stem]["pages"]} pages. '
                  f'{len(reports[stem]["warnings"])} warnings', flush=True)
    for rel, expected in input_hashes.items():
        if digest(BASE/rel) != expected:
            raise RuntimeError(f'Canonical input changed during external build: {rel}')
    (output/'pdf-build-report.json').write_text(json.dumps(reports,indent=2)+'\n')
    repository = BASE.parent
    tools = {}
    for label, executable in (('pandoc',args.pandoc), ('xelatex',args.engine), ('fc-match',fontconfig)):
        path = Path(executable).resolve()
        version = run([str(path),'--version'])
        tools[label] = {**identity(path), 'version':(version.stdout or version.stderr).splitlines()[0]}
    source = {'git_head':run(['git','rev-parse','HEAD'],cwd=repository).stdout.strip(),
              'working_tree_clean':not bool(run(['git','status','--porcelain'],cwd=repository).stdout.strip()),
              'scope':'Observed owner inputs. Separate final source selection required.'}
    environment = {key:os.environ.get(key) for key in ('TEXMFHOME','TEXMFVAR','TEXMFCACHE','TEXINPUTS','FONTCONFIG_FILE','FONTCONFIG_PATH')}
    manifest = {'edition':'0.60', 'scope':'Optional publication derivatives. No native acquisition, fit, bootstrap, adjudication or selected canonical PDF replacement.',
                'source':source, 'python':{'executable':sys.executable,'version':platform.python_version()},
                'packages':{'PyMuPDF':importlib.metadata.version('PyMuPDF')},
                'tools':tools, 'fontconfig_matches':font_records, 'environment':environment,
                'commands':commands, 'inputs':[identity(BASE/r,BASE) for r in sorted(input_hashes)],
                'owner_inputs':[identity(BASE/'source'/n,BASE) for n in ('build_pdfs.py','storage.py','canonical-inputs.json','source-preservation.json','pdf-compatibility.tex')],
                'project_inputs':[identity(repository/n) for n in ('pyproject.toml','uv.lock')],
                'actual_tex_inputs':list(actual_dependencies.values()),
                'outputs':[identity(p,output) for p in sorted(output.rglob('*')) if p.is_file()]}
    (output/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
