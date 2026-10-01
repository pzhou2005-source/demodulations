import numpy as np
import pytest

from nr5g_demod import demodulate_nr5g
from nr5g_waveform import NR5GConfig, generate_dmrs, generate_nr5g_waveform
from nr5g_waveform import _cp_lengths, ofdm_modulate, ofdm_demodulate
from vsa_89600 import VSA89600, simulate_vsa_result


def test_vsa_grid_excludes_only_dmrs_positions():
    from vsa_89600 import _pdsch_data_symbols

    cfg = NR5GConfig(n_slots=2)
    waveform = generate_nr5g_waveform(cfg)
    actual = _pdsch_data_symbols(waveform["resource_grid"].ravel(), cfg.n_sc)
    np.testing.assert_array_equal(actual, waveform["tx_symbols"])


def test_correlation_locates_subframe_inside_long_recording():
    from compare import correlate_symbol_streams

    rng = np.random.default_rng(42)
    reference = rng.normal(size=500) + 1j * rng.normal(size=500)
    measured = reference[213:313] * 1.7 * np.exp(0.4j)
    offset, correlation = correlate_symbol_streams(
        reference, measured.real, measured.imag, max_lag=None)
    assert offset == -213
    assert correlation == pytest.approx(1.0)


@pytest.mark.parametrize("slot", [0, 1])
@pytest.mark.parametrize("cell_id", [0, 42])
def test_dmrs_has_unit_power(slot, cell_id):
    dmrs = generate_dmrs(612, slot, 2, cell_id)

    np.testing.assert_allclose(np.abs(dmrs), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.abs(dmrs.real), 1 / np.sqrt(2))
    np.testing.assert_allclose(np.abs(dmrs.imag), 1 / np.sqrt(2))


@pytest.mark.parametrize("mu", [0, 1, 2, 3])
def test_normal_cp_subframe_duration(mu):
    n_fft = 2048
    lengths = [length for slot in range(2 ** mu)
               for length in _cp_lengths(n_fft, mu, slot)]

    assert sum(n_fft + length for length in lengths) == n_fft * 15 * (2 ** mu)
    assert lengths.count(144 + 16 * (2 ** mu)) == 2


def test_30khz_normal_cp_pattern():
    assert _cp_lengths(2048, 1) == [176] + [144] * 13
    assert len(generate_nr5g_waveform(NR5GConfig(n_slots=2))["time_signal"]) == 61440


def test_nr_resource_grid_is_contiguous_across_dc():
    grid = np.arange(1, 13, dtype=complex).reshape(1, 12)
    signal = ofdm_modulate(grid, 128, [10])
    spectrum = np.fft.fftshift(np.fft.fft(signal[10:])) / np.sqrt(128)

    np.testing.assert_allclose(spectrum[58:70], grid[0], atol=1e-14)
    np.testing.assert_allclose(ofdm_demodulate(signal, 128, 12, [10], 1), grid)


@pytest.mark.parametrize("offset", [0, -1, -5, -10])
def test_ofdm_fft_window_within_prefix_recovers_same_grid(offset):
    rng = np.random.default_rng(52)
    grid = rng.normal(size=(4, 12)) + 1j * rng.normal(size=(4, 12))
    signal = ofdm_modulate(grid, 128, [12, 10, 10, 10])

    recovered = ofdm_demodulate(signal, 128, 12, [12, 10, 10, 10], 4,
                                fft_window_offset=offset)

    np.testing.assert_allclose(recovered, grid, atol=1e-13)


@pytest.mark.parametrize("offset", [1, -11, 0.5])
def test_ofdm_rejects_fft_windows_outside_prefix(offset):
    with pytest.raises(ValueError, match="within the cyclic prefix"):
        ofdm_demodulate(np.ones(138, dtype=complex), 128, 12, [10], 1,
                        fft_window_offset=offset)


@pytest.mark.parametrize("snr_db", [30.0, 80.0, None])
def test_nr_early_fft_window_end_to_end(snr_db):
    cfg = NR5GConfig(n_slots=2, snr_db=snr_db, fft_window_offset=-66)
    waveform = generate_nr5g_waveform(cfg)
    result = demodulate_nr5g(waveform["time_signal"], cfg,
                              tx_bits=waveform["tx_bits"], tx_symbols=waveform["tx_symbols"],
                              data_positions=waveform["data_positions"])

    assert result.ber == 0.0
    assert np.all(np.isfinite(result.rx_symbols))
    assert result.evm_rms < (5.0 if snr_db is not None else 1e-10)


def test_waveform_matches_independent_nr_reference():
    reference = pytest.importorskip("py3gpp")
    cfg = NR5GConfig(n_slots=2)
    waveform = generate_nr5g_waveform(cfg)
    expected, info = reference.nrOFDMModulate(
        grid=waveform["resource_grid"].T, scs=30, initialNSlot=0,
        Nfft=cfg.n_fft, SampleRate=cfg.sample_rate_mhz * 1e6,
        Windowing=0, CarrierFrequency=0,
    )
    np.testing.assert_array_equal(info["CyclicPrefixLengths"], waveform["cp_lengths"])
    np.testing.assert_allclose(waveform["time_signal"], expected * np.sqrt(cfg.n_fft), atol=1e-14)

    carrier = reference.nrCarrierConfig(NSizeGrid=51, SubcarrierSpacing=30)
    pdsch = reference.nrPDSCHConfig()
    pdsch.NSizeBWP = 51
    pdsch.PRBSet = np.arange(51)
    pdsch.DMRS.NIDNSCID = 0
    pdsch.DMRS.DMRSTypeAPosition = 2
    pdsch.DMRS.DMRSAdditionalPosition = 0
    np.testing.assert_allclose(generate_dmrs(612, 0, 2, 0), reference.nrPDSCHDMRS(pdsch, carrier))


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
    with np.load(tmp_path / "comparison_data.npz") as saved:
        assert saved["is_simulated"]
        assert "vsa_measurement_status" in saved
        np.testing.assert_array_equal(saved["vsa_symbols"],
                                      vsa_result.symbols_i + 1j * vsa_result.symbols_q)


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


@pytest.mark.parametrize("offset", [-3, 0, 3])
def test_correlation_handles_short_delayed_streams(offset):
    from compare import correlate_symbol_streams

    rng = np.random.default_rng(12)
    symbols = rng.standard_normal(32) + 1j * rng.standard_normal(32)
    prefix = rng.standard_normal(abs(offset)) + 1j * rng.standard_normal(abs(offset))
    py_symbols = np.concatenate([prefix, symbols]) if offset < 0 else symbols
    vsa_symbols = np.concatenate([prefix, symbols]) if offset > 0 else symbols
    vsa_symbols = vsa_symbols * 1.7 * np.exp(0.4j)

    lag, correlation = correlate_symbol_streams(py_symbols, vsa_symbols.real, vsa_symbols.imag)

    assert lag == offset
    assert correlation == pytest.approx(1.0)


def test_identical_streams_have_unit_correlation_on_every_subcarrier():
    from compare import correlate_results

    cfg = NR5GConfig(n_rb=2, n_slots=2)
    waveform = generate_nr5g_waveform(cfg)
    python_result = demodulate_nr5g(waveform["time_signal"], cfg)
    vsa_result = simulate_vsa_result(waveform["time_signal"], cfg)

    result = correlate_results(python_result, vsa_result, cfg)

    assert result.symbol_correlation == pytest.approx(1.0)
    assert result.evm_of_difference < 1e-10
    np.testing.assert_allclose(result.per_symbol_corr, 1.0, atol=1e-14)
    np.testing.assert_allclose(result.per_subcarrier_corr, 1.0, atol=1e-14)


@pytest.mark.parametrize("seed", [7, 42, 99])
def test_flat_channel_estimator_reduces_awgn_evm(seed):
    from dataclasses import replace

    cfg = NR5GConfig(n_slots=2, snr_db=30.0, seed=seed)
    waveform = generate_nr5g_waveform(cfg)
    references = dict(tx_bits=waveform["tx_bits"], tx_symbols=waveform["tx_symbols"],
                      data_positions=waveform["data_positions"])
    baseline = demodulate_nr5g(waveform["time_signal"], cfg, **references)
    improved = demodulate_nr5g(waveform["time_signal"],
                              replace(cfg, channel_estimation="flat"), **references)

    assert improved.evm_rms < baseline.evm_rms * 0.85
    assert improved.ber == 0.0
    np.testing.assert_allclose(improved.channel_est,
                               np.broadcast_to(improved.channel_est[:, :1],
                                               improved.channel_est.shape))


def test_comparison_import_preserves_plotting_backend(monkeypatch):
    import importlib
    import matplotlib
    import compare

    def forbid_backend_change(*args, **kwargs):
        pytest.fail("Importing comparison must preserve the editor plotting backend")

    monkeypatch.setattr(matplotlib, "use", forbid_backend_change)
    importlib.reload(compare)


@pytest.mark.parametrize("modulation", ["QPSK", "16QAM", "64QAM", "256QAM"])
@pytest.mark.parametrize("channel_estimation", ["linear", "flat"])
@pytest.mark.parametrize("seed", [7, 42, 99])
def test_high_snr_demodulation_consistency(modulation, channel_estimation, seed):
    from compare import correlate_symbol_streams

    previous_snr = None
    previous_evm = None
    previous_correlation = 0.0
    previous_ber = 1.0
    reference_bits = None
    for snr_db in (30.0, 40.0, 50.0, 60.0, 80.0, 100.0, None):
        cfg = NR5GConfig(modulation=modulation, channel_estimation=channel_estimation,
                         n_slots=2, snr_db=snr_db, seed=seed)
        waveform = generate_nr5g_waveform(cfg)
        result = demodulate_nr5g(
            waveform["time_signal"], cfg,
            tx_bits=waveform["tx_bits"], tx_symbols=waveform["tx_symbols"],
            data_positions=waveform["data_positions"],
        )
        context = f"{modulation}, {channel_estimation}, seed={seed}, SNR={snr_db}"

        if reference_bits is None:
            reference_bits = waveform["tx_bits"]
        np.testing.assert_array_equal(waveform["tx_bits"], reference_bits, err_msg=context)
        measured_ber = np.mean(result.rx_bits != reference_bits)
        assert result.ber == pytest.approx(measured_ber), context
        assert result.ber <= previous_ber, context
        if snr_db is None or snr_db >= 40.0:
            np.testing.assert_array_equal(result.rx_bits, reference_bits, err_msg=context)
            assert result.ber == 0.0, context
        else:
            assert result.ber < 1e-4, context
        assert np.all(np.isfinite(result.channel_est)), context
        assert np.all(np.isfinite(result.rx_symbols)), context
        assert np.all(np.isfinite(result.evm_per_symbol)), context
        assert np.isfinite(result.evm_rms), context
        assert np.isfinite(result.evm_peak), context
        assert result.evm_peak >= result.evm_rms, context

        offset, correlation = correlate_symbol_streams(
            waveform["tx_symbols"], result.rx_symbols.real, result.rx_symbols.imag,
            max_lag=0,
        )
        assert offset == 0, context
        assert 0.999 < correlation <= 1.0, context
        assert correlation >= previous_correlation - 1e-14, context

        if snr_db is None:
            assert result.evm_rms < 1e-10, context
            assert result.evm_peak < 1e-10, context
            np.testing.assert_allclose(result.evm_per_symbol, 0.0, atol=1e-10,
                                       err_msg=context)
            np.testing.assert_allclose(result.channel_est, 1.0, atol=1e-13,
                                       err_msg=context)
            assert correlation == pytest.approx(1.0, abs=1e-14), context
        else:
            noise = waveform["time_signal"] - waveform["time_signal_clean"]
            measured_snr = 10 * np.log10(
                np.mean(np.abs(waveform["time_signal_clean"]) ** 2)
                / np.mean(np.abs(noise) ** 2)
            )
            assert measured_snr == pytest.approx(snr_db, abs=0.15), context
            assert result.evm_rms > 0.0, context
            if previous_evm is not None:
                expected_evm = previous_evm * 10 ** (-(snr_db - previous_snr) / 20)
                assert result.evm_rms == pytest.approx(expected_evm, rel=0.01), context

        previous_snr = snr_db
        previous_evm = result.evm_rms
        previous_correlation = correlation
        previous_ber = result.ber