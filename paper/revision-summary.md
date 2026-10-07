# Empirical Lawhood — revision summary

**Edition:** 0.60  
**Previous edition:** 0.5.0  
**Date:** 4 October 2026

## Single authorised text change

The opening four sentences of the abstract are replaced with:

> How can reliable scientific capabilities accumulate when the conditions needed to acquire and use knowledge must themselves be discovered or constructed? Motivated by dependable fusion propulsion for a journey to Neptune, Empirical Lawhood investigates how knowledge can help establish the means of further discovery and action. Kant’s inquiry into warrant, Wheeler’s observer-participancy and Gell-Mann and Hartle’s physical account of regularity ground an investigation in which observers, instruments and the conditions of inquiry belong to the world being investigated. The framework is developed and applied through human-directed scientific discovery and prospective tests of particular constructions.

The rest of the abstract and manuscript is unchanged. The companion's text is unchanged. Source and PDF version metadata, output filenames, build targets and package inventories use 0.60; the original preprint date remains unchanged.

## Verification

`validate.py` checks that reversing this opening replacement and the version-metadata change exactly reproduces the previous manuscript Markdown. It also checks that reversing the version-metadata change exactly reproduces the previous companion Markdown. The prior Markdown files are retained in `sources/editorial_history/`.

The exact manuscript diff is `changes/v0.5.0_to_v0.60.diff`; the companion metadata diff is `changes/evidence_v0.5.0_to_v0.60.diff`. Inherited numerical figure assets and documentary sources remain byte-identical.

The checks concern document integrity and presentation, not new validation of the experiments.
