#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for tool in pandoc xelatex; do
  command -v "$tool" >/dev/null || { printf 'Required build tool not found: %s\n' "$tool" >&2; exit 1; }
done
mkdir -p build
for stem in Empirical_Lawhood_Manuscript_v0.60 Empirical_Lawhood_Evidence_and_Provenance_v0.60; do
  extra_header=()
  if [[ "$stem" == *Evidence_and_Provenance* ]]; then
    extra_header=(--include-in-header=evidence_metadata.tex)
  fi
  pandoc "$stem.md" -s --from=markdown --to=latex --number-sections \
    --lua-filter=layout.lua \
    -V documentclass=article -V fontsize=11pt -V papersize=a4 \
    -V 'geometry=left=22mm,right=22mm,top=22mm,bottom=22mm,headsep=6mm' \
    -V mainfont='Linux Libertine O' -V sansfont='DejaVu Sans' \
    -V monofont='DejaVu Sans Mono' -V 'monofontoptions=Scale=0.78' \
    -V mathfont=latinmodern-math.otf \
    -V linestretch=1.015 -V colorlinks=false \
    --include-in-header=header.tex "${extra_header[@]}" -o "build/$stem.tex"
  for run in 1 2; do
    xelatex -interaction=nonstopmode -halt-on-error -output-directory=build "build/$stem.tex" > "build/${stem}_compile_${run}.txt"
  done
  cp "build/$stem.pdf" "$stem.pdf"
done
printf '\nBuilt manuscript and evidence/provenance PDFs.\n'
