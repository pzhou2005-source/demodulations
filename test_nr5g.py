import numpy as np
import pytest

from nr5g_demod import demodulate_nr5g
from nr5g_waveform import NR5GConfig, generate_dmrs, generate_nr5g_waveform
from vsa_89600 import VSA89600, simulate_vsa_result


@pytest.mark.parametrize("slot", [0, 1])
@pytest.mark.parametrize("cell_id", [0, 42])
def test_dmrs_has_unit_power(slot, cell_id):
    dmrs = generate_dmrs(612, slot, 2, cell_id)

    np.testing.assert_allclose(np.abs(dmrs), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.abs(dmrs.real), 1 / np.sqrt(2))
    np.testing.assert_allclose(np.abs(dmrs.imag), 1 / np.sqrt(2))


@pytest.mark.parametrize("snr_db", [None, 30.0])
def test_64qam_demodulation(snr_db):
    cfg = NR5GConfig(n_slots=2, snr_db=snr_db, seed=42)
    waveform = generate_nr5g_waveform(cfg)
    result = demodulate_nr5g(
        waveform["time_signal"], cfg,
        tx_bits=waveform["tx_bits"],
        tx_symbols=waveform["tx_symbols"],
        data_positions=waveform["data_positions"],
    )

    assert result.ber == 0.0
    assert result.evm_rms < (5.0 if snr_db is not None else 1e-10)
    assert np.all(np.isfinite(result.evm_per_symbol))
    expected_peak = np.max(np.abs(result.rx_symbols - waveform["tx_symbols"]))
    expected_peak *= 100 / np.sqrt(np.mean(np.abs(waveform["tx_symbols"]) ** 2))
    assert result.evm_peak == pytest.approx(expected_peak)


def test_simulated_vsa_is_marked():
    cfg = NR5GConfig(n_slots=1, snr_db=30.0)
    waveform = generate_nr5g_waveform(cfg)

    result = simulate_vsa_result(waveform["time_signal"], cfg)

    assert result.is_simulated
    assert result.evm_peak > result.evm_rms


@pytest.mark.parametrize("invalid_value", [None, np.nan, np.inf])
def test_missing_vsa_metrics_are_not_reported_as_zero(monkeypatch, invalid_value):
    vsa = object.__new__(VSA89600)
    monkeypatch.setattr(vsa, "_find_trace", lambda name: None)
    monkeypatch.setattr(vsa, "_get_scalar", lambda name: invalid_value)

    with pytest.raises(RuntimeError, match="Missing or invalid VSA measurements"):
        vsa.get_results()


def test_missing_vsa_constellation_is_rejected(monkeypatch):
    vsa = object.__new__(VSA89600)
    monkeypatch.setattr(vsa, "_find_trace", lambda name: None)
    monkeypatch.setattr(vsa, "_get_scalar", lambda name: 0.0)

    with pytest.raises(RuntimeError, match="no valid constellation"):
        vsa.get_results()


def test_simulated_comparison_reports_provenance(tmp_path):
    from compare import run_full_comparison

    python_result, vsa_result, report = run_full_comparison(output_dir=str(tmp_path))

    assert python_result.ber == 0.0
    assert vsa_result.is_simulated
    assert "SIMULATED" in report
    assert "Not an independent VSA measurement" in report
    assert f"{python_result.evm_peak:.3f}" in report
    assert (tmp_path / "comparison_report.txt").read_text(encoding="utf-8") == report
    assert "SIMULATED" in (tmp_path / "correlation_report.txt").read_text(encoding="utf-8")


def test_real_vsa_failure_does_not_fall_back_to_simulation(monkeypatch, tmp_path):
    import compare
    import vsa_89600

    def unavailable(*args, **kwargs):
        raise RuntimeError("VSA unavailable")

    def forbidden_simulation(*args, **kwargs):
        pytest.fail("A real VSA request must not use simulated measurements")

    monkeypatch.setattr(vsa_89600, "run_vsa_demod", unavailable)
    monkeypatch.setattr(compare, "simulate_vsa_result", forbidden_simulation)

    with pytest.raises(RuntimeError, match="VSA unavailable"):
        compare.run_full_comparison(use_real_vsa=True, output_dir=str(tmp_path))

    assert not (tmp_path / "comparison_report.txt").exists()