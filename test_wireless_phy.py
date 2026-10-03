import numpy as np
import pytest

from wireless_phy import WiFi7Config, generate_wifi7_waveform, demodulate_wifi7
from wireless_phy import BluetoothConfig, generate_bluetooth_waveform, demodulate_bluetooth
from wireless_phy import BluetoothEDRConfig, generate_bluetooth_edr_waveform, demodulate_bluetooth_edr
from wireless_phy import bluetooth_edr_bits_from_state_indices, bluetooth_edr_symbols_from_state_indices
from wireless_phy import UWBConfig, generate_uwb_waveform, demodulate_uwb
from wireless_phy import plot_wireless_result


@pytest.mark.parametrize("bandwidth_mhz", [20, 40, 80, 160, 320])
@pytest.mark.parametrize("modulation", ["QPSK", "256QAM", "4096QAM"])
def test_wifi7_noiseless_roundtrip(bandwidth_mhz, modulation):
    cfg = WiFi7Config(bandwidth_mhz=bandwidth_mhz, modulation=modulation, n_symbols=2)
    waveform = generate_wifi7_waveform(cfg)
    result = demodulate_wifi7(waveform["time_signal"], cfg, waveform["tx_bits"],
                              waveform["tx_symbols"])

    assert cfg.sample_rate_hz / cfg.n_fft == 78125
    assert len(waveform["time_signal"]) == (cfg.n_symbols + 1) * (cfg.n_fft + cfg.cp_length)
    assert np.all(waveform["resource_grid"][:, 0] == 0)
    np.testing.assert_array_equal(result.rx_bits, waveform["tx_bits"])
    assert result.ber == 0.0
    assert result.evm_rms < 1e-10


@pytest.mark.parametrize("guard_interval_us", [0.8, 1.6, 3.2])
@pytest.mark.parametrize("channel_estimation", ["flat", "per_subcarrier"])
def test_wifi7_high_snr_and_complex_gain(guard_interval_us, channel_estimation):
    cfg = WiFi7Config(guard_interval_us=guard_interval_us,
                      channel_estimation=channel_estimation, snr_db=60.0)
    waveform = generate_wifi7_waveform(cfg)
    received = waveform["time_signal"] * 0.7 * np.exp(0.8j)
    result = demodulate_wifi7(received, cfg, waveform["tx_bits"], waveform["tx_symbols"])

    assert result.ber == 0.0
    assert 0 < result.evm_rms < 0.3
    assert np.all(np.isfinite(result.rx_symbols))


def test_wifi7_does_not_require_known_payload():
    cfg = WiFi7Config(modulation="64QAM")
    waveform = generate_wifi7_waveform(cfg)
    result = demodulate_wifi7(waveform["time_signal"], cfg)

    np.testing.assert_array_equal(result.rx_bits, waveform["tx_bits"])
    assert result.ber is None
    assert result.evm_rms is None


def test_wifi7_equalizes_multipath_within_guard_interval():
    cfg = WiFi7Config(channel_estimation="per_subcarrier")
    waveform = generate_wifi7_waveform(cfg)
    channel = np.array([1.0, 0, 0, 0, 0, 0.25 + 0.15j])
    received = np.convolve(waveform["time_signal"], channel)[:len(waveform["time_signal"])]
    result = demodulate_wifi7(received, cfg, waveform["tx_bits"], waveform["tx_symbols"])

    assert result.ber == 0.0
    assert result.evm_rms < 1e-10
    expected_channel = np.fft.fft(channel, cfg.n_fft)[waveform["active_bins"]]
    np.testing.assert_allclose(result.diagnostics["channel_est"], expected_channel, atol=1e-13)


@pytest.mark.parametrize("kwargs", [dict(bandwidth_mhz=30), dict(modulation="8192QAM"),
                                  dict(guard_interval_us=0.4), dict(n_symbols=0),
                                  dict(snr_db=np.inf), dict(channel_estimation="magic")])
def test_wifi7_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        WiFi7Config(**kwargs)


def test_wifi7_rejects_truncated_capture():
    cfg = WiFi7Config()
    waveform = generate_wifi7_waveform(cfg)

    with pytest.raises(ValueError, match="aligned IQ samples"):
        demodulate_wifi7(waveform["time_signal"][:-1], cfg)


@pytest.mark.parametrize("phy", ["LE1M", "LE2M"])
@pytest.mark.parametrize("pattern", ["zeros", "ones", "alternating"])
def test_bluetooth_known_bit_patterns(phy, pattern):
    cfg = BluetoothConfig(phy=phy, n_bits=128)
    bits = {"zeros": np.zeros(128, dtype=np.uint8),
            "ones": np.ones(128, dtype=np.uint8),
            "alternating": np.arange(128, dtype=np.uint8) % 2}[pattern]
    waveform = generate_bluetooth_waveform(cfg, bits)
    result = demodulate_bluetooth(waveform["time_signal"], cfg, bits)

    np.testing.assert_allclose(np.abs(waveform["time_signal_clean"]), 1.0)
    np.testing.assert_array_equal(result.rx_bits, bits)
    assert result.ber == 0.0
    assert result.evm_rms is None
    assert np.max(np.abs(waveform["frequency_hz"])) <= cfg.frequency_deviation_hz * (1 + 1e-14)
    if pattern != "alternating":
        expected = cfg.frequency_deviation_hz * (1 if pattern == "ones" else -1)
        np.testing.assert_allclose(result.diagnostics["frequency_hz"], expected)


@pytest.mark.parametrize("phy", ["LE1M", "LE2M"])
@pytest.mark.parametrize("snr_db", [20.0, 40.0, 80.0, None])
def test_bluetooth_noise_and_phase_invariance(phy, snr_db):
    cfg = BluetoothConfig(phy=phy, snr_db=snr_db)
    waveform = generate_bluetooth_waveform(cfg)
    result = demodulate_bluetooth(waveform["time_signal"] * 1.5 * np.exp(1.2j),
                                  cfg, waveform["tx_bits"])

    assert result.ber == 0.0
    assert np.all(np.isfinite(result.rx_symbols))
    assert result.evm_rms is None


def test_bluetooth_demodulates_independent_constant_tone():
    cfg = BluetoothConfig(n_bits=16)
    sample_index = np.arange(cfg.n_bits * cfg.samples_per_symbol + 1)
    samples = np.exp(-2j * np.pi * 250e3 * sample_index / cfg.sample_rate_hz)
    result = demodulate_bluetooth(samples, cfg)

    np.testing.assert_array_equal(result.rx_bits, np.zeros(cfg.n_bits, dtype=np.uint8))
    assert result.ber is None


@pytest.mark.parametrize("kwargs", [dict(phy="EDR"), dict(n_bits=0),
                                  dict(samples_per_symbol=2), dict(seed=-1)])
def test_bluetooth_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        BluetoothConfig(**kwargs)


def test_bluetooth_rejects_nonbinary_payload():
    with pytest.raises(ValueError, match="binary bits"):
        generate_bluetooth_waveform(BluetoothConfig(n_bits=3), [0, 1, 2])


@pytest.mark.parametrize("phy", ["EDR2M", "EDR3M"])
def test_bluetooth_edr_noiseless_roundtrip(phy):
    cfg = BluetoothEDRConfig(phy=phy, n_symbols=64)
    waveform = generate_bluetooth_edr_waveform(cfg)
    result = demodulate_bluetooth_edr(
        waveform["time_signal"], cfg, waveform["tx_bits"], waveform["tx_symbols"])

    np.testing.assert_array_equal(result.rx_bits, waveform["tx_bits"])
    np.testing.assert_array_equal(result.diagnostics["symbol_indices"],
                                  waveform["tx_symbol_indices"])
    assert result.ber == 0.0
    assert result.evm_rms < 1e-10


@pytest.mark.parametrize("phy", ["EDR2M", "EDR3M"])
def test_bluetooth_edr_tolerates_constant_complex_gain(phy):
    cfg = BluetoothEDRConfig(phy=phy, n_symbols=64, snr_db=None)
    waveform = generate_bluetooth_edr_waveform(cfg)
    received = waveform["time_signal"] * 0.6 * np.exp(1.1j)
    result = demodulate_bluetooth_edr(received, cfg, waveform["tx_bits"],
                                      waveform["tx_symbols"])

    np.testing.assert_array_equal(result.rx_bits, waveform["tx_bits"])
    assert result.ber == 0.0
    assert result.evm_rms < 1e-10


@pytest.mark.parametrize("phy", ["EDR2M", "EDR3M"])
def test_bluetooth_edr_vsa_state_conversion_roundtrip(phy):
    cfg = BluetoothEDRConfig(phy=phy, n_symbols=4 if phy == "EDR2M" else 8)
    state_indices = np.arange(cfg.phase_order)
    bits = bluetooth_edr_bits_from_state_indices(state_indices, cfg)
    symbols = bluetooth_edr_symbols_from_state_indices(state_indices, cfg)
    waveform = generate_bluetooth_edr_waveform(cfg, bits)

    np.testing.assert_array_equal(waveform["tx_symbol_indices"], state_indices)
    np.testing.assert_allclose(waveform["tx_symbols"], symbols, atol=1e-14)


def test_uwb_all_position_and_polarity_combinations():
    cfg = UWBConfig(n_symbols=4)
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
    waveform = generate_uwb_waveform(cfg, bits)
    blocks = waveform["time_signal_clean"].reshape(5, cfg.samples_per_symbol)
    burst = waveform["burst_template"]
    half_symbol = cfg.samples_per_symbol // 2

    np.testing.assert_allclose(blocks[1, :burst.size], burst)
    np.testing.assert_allclose(blocks[2, :burst.size], -burst)
    np.testing.assert_allclose(blocks[3, half_symbol:half_symbol + burst.size], burst)
    np.testing.assert_allclose(blocks[4, half_symbol:half_symbol + burst.size], -burst)
    np.testing.assert_array_equal(blocks[1:3, half_symbol:], 0)
    np.testing.assert_array_equal(blocks[3:5, :half_symbol], 0)

    result = demodulate_uwb(waveform["time_signal"], cfg, bits, waveform["tx_symbols"])

    np.testing.assert_array_equal(result.rx_bits, bits)
    assert result.evm_rms < 1e-10
    assert result.ber == 0.0


@pytest.mark.parametrize("snr_db", [10.0, 30.0, 60.0, 100.0, None])
@pytest.mark.parametrize("samples_per_chip", [2, 4, 8])
def test_uwb_noise_and_complex_gain(snr_db, samples_per_chip):
    cfg = UWBConfig(snr_db=snr_db, samples_per_chip=samples_per_chip)
    waveform = generate_uwb_waveform(cfg)
    result = demodulate_uwb(waveform["time_signal"] * 0.6 * np.exp(1.1j), cfg,
                            waveform["tx_bits"], waveform["tx_symbols"])

    assert result.ber == 0.0
    assert np.isfinite(result.evm_rms)
    assert np.all(np.isfinite(result.rx_symbols))
    if snr_db is None:
        assert result.evm_rms < 1e-10
    else:
        assert result.evm_rms > 0


def test_uwb_does_not_require_known_payload():
    cfg = UWBConfig()
    waveform = generate_uwb_waveform(cfg)
    result = demodulate_uwb(waveform["time_signal"], cfg)

    np.testing.assert_array_equal(result.rx_bits, waveform["tx_bits"])
    assert result.ber is None
    assert result.evm_rms is None


@pytest.mark.parametrize("kwargs", [dict(chips_per_symbol=63), dict(chips_per_burst=33),
                                  dict(chip_rate_hz=0), dict(n_symbols=0),
                                  dict(samples_per_chip=1)])
def test_uwb_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        UWBConfig(**kwargs)


def test_uwb_rejects_missing_training_burst():
    cfg = UWBConfig()
    waveform = generate_uwb_waveform(cfg)
    waveform["time_signal"][:cfg.samples_per_symbol] = 0

    with pytest.raises(ValueError, match="Training burst"):
        demodulate_uwb(waveform["time_signal"], cfg)


@pytest.mark.parametrize("protocol", ["wifi7", "bluetooth", "uwb"])
@pytest.mark.parametrize("seed", [7, 42, 99])
def test_wireless_high_snr_consistency(protocol, seed):
    previous_error = None
    for snr_db in (60.0, 80.0, 100.0):
        if protocol == "wifi7":
            cfg = WiFi7Config(snr_db=snr_db, seed=seed)
            waveform = generate_wifi7_waveform(cfg)
            result = demodulate_wifi7(waveform["time_signal"], cfg,
                                      waveform["tx_bits"], waveform["tx_symbols"])
            error = result.evm_rms
        elif protocol == "bluetooth":
            cfg = BluetoothConfig(snr_db=snr_db, seed=seed)
            waveform = generate_bluetooth_waveform(cfg)
            result = demodulate_bluetooth(waveform["time_signal"], cfg, waveform["tx_bits"])
            error = np.sqrt(np.mean((result.diagnostics["frequency_hz"] - waveform["frequency_hz"]) ** 2))
        else:
            cfg = UWBConfig(snr_db=snr_db, seed=seed)
            waveform = generate_uwb_waveform(cfg)
            result = demodulate_uwb(waveform["time_signal"], cfg,
                                    waveform["tx_bits"], waveform["tx_symbols"])
            error = result.evm_rms
        assert result.ber == 0.0
        assert np.isfinite(error) and error > 0
        if previous_error is not None:
            assert error == pytest.approx(previous_error / 10, rel=0.01)
        previous_error = error


@pytest.mark.parametrize("protocol", ["wifi7", "bluetooth", "uwb"])
def test_wireless_plots_render_without_changing_backend(protocol):
    import io
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if protocol == "wifi7":
        cfg = WiFi7Config(snr_db=50.0)
        waveform = generate_wifi7_waveform(cfg)
        result = demodulate_wifi7(waveform["time_signal"], cfg,
                                  waveform["tx_bits"], waveform["tx_symbols"])
    elif protocol == "bluetooth":
        cfg = BluetoothConfig(snr_db=20.0)
        waveform = generate_bluetooth_waveform(cfg)
        result = demodulate_bluetooth(waveform["time_signal"], cfg, waveform["tx_bits"])
    else:
        cfg = UWBConfig(snr_db=20.0)
        waveform = generate_uwb_waveform(cfg)
        result = demodulate_uwb(waveform["time_signal"], cfg,
                                waveform["tx_bits"], waveform["tx_symbols"])
    backend = matplotlib.get_backend()
    figure = plot_wireless_result(waveform, result)
    try:
        output = io.BytesIO()
        figure.savefig(output, format="png")
        assert output.getbuffer().nbytes > 10000
        assert len(figure.axes) == 4
        assert all(axis.lines or axis.collections for axis in figure.axes)
        assert matplotlib.get_backend() == backend
    finally:
        plt.close(figure)


@pytest.mark.parametrize("protocol", ["wifi7", "bluetooth", "uwb"])
@pytest.mark.parametrize("seed", [7, 42, 99])
def test_wireless_noiseless_soft_signal_correlation(protocol, seed):
    from compare import correlate_symbol_streams

    if protocol == "wifi7":
        cfg = WiFi7Config(seed=seed)
        waveform = generate_wifi7_waveform(cfg)
        result = demodulate_wifi7(waveform["time_signal"], cfg)
        reference = waveform["tx_symbols"]
        measured = result.rx_symbols
    elif protocol == "bluetooth":
        cfg = BluetoothConfig(seed=seed)
        waveform = generate_bluetooth_waveform(cfg)
        result = demodulate_bluetooth(waveform["time_signal"], cfg)
        reference = waveform["frequency_hz"]
        measured = result.diagnostics["frequency_hz"].astype(complex)
    else:
        cfg = UWBConfig(seed=seed)
        waveform = generate_uwb_waveform(cfg)
        result = demodulate_uwb(waveform["time_signal"], cfg)
        reference = waveform["tx_symbols"]
        measured = result.rx_symbols

    offset, correlation = correlate_symbol_streams(reference, measured.real, measured.imag,
                                                   max_lag=0)

    assert offset == 0
    assert correlation == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(measured, reference, atol=1e-8, rtol=1e-10)