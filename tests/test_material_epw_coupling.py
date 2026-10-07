"""Independent integral and refusal checks for EPW's native alpha2F table."""

from decimal import Decimal

import pytest

from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.material_control_solver import _epw_operands, parse_epw_a2f_integrated_lambda


def _two_smearing_table() -> bytes:
    # alpha2F(w) = w/4 in the first smearing, so 2*alpha2F/w = 1/2.
    # Integrating from zero to each displayed frequency gives 0.5, 1, 1.5.
    rows = []
    for frequency in (1, 2, 3):
        w = Decimal(frequency)
        rows.append(f"{w} {w / 4} {w * 3 / 10} {w / 2} {w * 3 / 5}")
    return (
        "w[meV] a2f and integrated 2*a2f/w for 2 smearing values\n"
        + "\n".join(rows)
        + "\nIntegrated el-ph coupling\n# 1.5 1.8\n"
        + "Phonon smearing (meV)\n# 0.5 0.6\n"
    ).encode("ascii")


def test_first_integrated_coupling_not_spectral_peak_or_second_view(tmp_path) -> None:
    payload = _two_smearing_table()
    # Compute the physical integral from the spectral column independently
    # of the cumulative columns and parser selection.
    rows = [tuple(Decimal(part) for part in line.split()) for line in payload.decode().splitlines()[1:4]]
    integral = sum((2 * row[1] / row[0]) for row in rows)
    assert parse_epw_a2f_integrated_lambda(payload) == integral

    (tmp_path / "mgb2.a2f").write_bytes(payload)
    (tmp_path / "mgb2.imag_aniso_gap0_020.00").write_text("0 1.2 20 0 0\n1 6.5 20 0 0\n")
    coupling, temperature, gap_low, gap_high = _epw_operands(tmp_path, "mgb2")
    assert coupling == integral
    assert temperature == Decimal(20)
    assert (gap_low, gap_high) == (Decimal("1.2"), Decimal("6.5"))


def test_missing_coupling_is_unevaluable_even_with_gap(tmp_path) -> None:
    (tmp_path / "mgb2.imag_aniso_gap0_020.00").write_text("0 1.2 20 0 0\n")
    coupling, _, _, gap_high = _epw_operands(tmp_path, "mgb2")
    assert coupling == 0 and gap_high > 0


@pytest.mark.parametrize(
    "corruption",
    (
        lambda data: data.replace(b"# 1.5 1.8", b"# 1.8 1.5"),
        lambda data: data.replace(b"3 0.75 0.9 1.5 1.8", b"3 0.75 0.9 1.5"),
        lambda data: data.replace(b"2 0.5 0.6 1 1.2", b"2 0.5 0.6 0.1 1.2"),
        lambda data: data.replace(b"2 0.5 0.6 1 1.2", b"2 0.5 BAD 1 1.2"),
        lambda data: data.replace(b"Integrated el-ph coupling", b"unlabelled values"),
    ),
)
def test_ambiguous_or_inconsistent_epw_integral_refuses(corruption) -> None:
    with pytest.raises(ValueError, match="EPW alpha2F"):
        parse_epw_a2f_integrated_lambda(corruption(_two_smearing_table()))
