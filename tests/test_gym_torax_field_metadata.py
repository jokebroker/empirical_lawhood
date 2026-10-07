from __future__ import annotations

from dataclasses import replace

import pytest

from empirical_lawhood.adapters.simulators.gym_torax_native.field_metadata import build_gym_torax_field_metadata_manifest, verify_installed_gym_torax_metadata_sources


def test_field_metadata_is_the_exact_retained_closure() -> None:
    manifest = build_gym_torax_field_metadata_manifest()

    assert len(manifest.fields) == 17
    assert set(manifest.by_key()) == {
        ("profiles", "T_e"),
        ("profiles", "T_i"),
        ("profiles", "n_e"),
        ("profiles", "psi"),
        ("profiles", "q"),
        ("scalars", "H98"),
        ("scalars", "P_heat_total"),
        ("scalars", "P_radiation_e"),
        ("scalars", "Q_fusion"),
        ("scalars", "beta_N"),
        ("scalars", "fgw_n_e_volume_avg"),
        ("scalars", "q95"),
        ("scalars", "q_min"),
        ("numerics", "inner_solver_iterations"),
        ("numerics", "outer_solver_iterations"),
        ("numerics", "sawtooth_crash"),
        ("numerics", "solver_error_state"),
    }
    assert all(
        value.native_unit != "source-unit-unspecified" for value in manifest.fields
    )
    assert all(value.native_frame_id for value in manifest.fields)
    assert all(value.native_dimension_ids for value in manifest.fields)


def test_unspecified_unit_is_a_construction_error() -> None:
    field = build_gym_torax_field_metadata_manifest().fields[0]

    with pytest.raises(ValueError, match="unspecified units are prohibited"):
        replace(field, native_unit="source-unit-unspecified")


@pytest.mark.native("gymtorax", "torax")
def test_installed_defining_source_bytes_match_manifest() -> None:
    verify_installed_gym_torax_metadata_sources(build_gym_torax_field_metadata_manifest())
