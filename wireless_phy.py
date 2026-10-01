"""Uncoded, sample-aligned wireless PHY teaching models, not conformance tools.

Wi-Fi 7 model: EHT symbol timing, 20-320 MHz bandwidth and square QAM through
4096-QAM. Uses a generic DC-null allocation, comb pilots and one training
symbol, not IEEE EHT PPDU framing, RU maps, bit labeling, FEC, MIMO or MLO.
Bluetooth model: uncoded LE 1M/2M GFSK (BT=0.5, modulation index=0.5).
No preamble/access-address acquisition, whitening, CRC, hopping, LE Coded
or BR/EDR. Frequency-discriminator outputs are not QAM constellation EVM.
UWB model: HRP-inspired BPM-BPSK with Gaussian pulses and a training burst.
No IEEE 802.15.4z framing, spreading codes, time hopping, FEC, STS or ranging.
UWB EVM refers to the two matched-filter outputs, not a conformance metric.
All SNR values refer to average time-domain sample power, not Eb/N0.
Receiver timing is supplied by the caller; acquisition and CFO recovery are
not implemented. Known payloads are used only for optional error metrics.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch
from scipy.signal.windows import gaussian
from scipy.spatial import cKDTree

from nr5g_waveform import _qam_constellation, qam_modulate


@dataclass
class WirelessResult:
    """Recovered soft symbols and bits, with optional reference-based metrics."""

    rx_bits: np.ndarray
    rx_symbols: np.ndarray
    sample_rate_hz: float
    evm_rms: Optional[float] = None
    ber: Optional[float] = None
    diagnostics: dict = field(default_factory=dict)


def _positive_integer(name, value, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _validate_noise(snr_db, seed):
    if snr_db is not None and not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite or None")
    _positive_integer("seed", seed, minimum=0)


def _bits(values, size):
    values = np.asarray(values)
    if values.shape != (size,) or not np.all((values == 0) | (values == 1)):
        raise ValueError(f"Expected exactly {size} binary bits")
    return values.astype(np.uint8)


def _samples(values, size):
    values = np.asarray(values, dtype=complex)
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(f"Expected exactly {size} finite, aligned IQ samples")
    return values


def _add_awgn(clean, snr_db, rng):
    if snr_db is None:
        return clean.copy()
    noise_rms = np.sqrt(np.mean(np.abs(clean) ** 2)) * 10 ** (-snr_db / 20)
    noise = rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)
    return clean + noise_rms * noise / np.sqrt(2)


def _measure_result(result, tx_bits, tx_symbols=None):
    if tx_bits is not None:
        reference_bits = _bits(tx_bits, result.rx_bits.size)
        result.ber = float(np.mean(result.rx_bits != reference_bits))
    if tx_symbols is not None:
        reference = _samples(tx_symbols, result.rx_symbols.size)
        power = np.mean(np.abs(reference) ** 2)
        if power == 0:
            raise ValueError("Reference symbols must have nonzero power")
        result.evm_rms = float(100 * np.sqrt(
            np.mean(np.abs(result.rx_symbols - reference) ** 2) / power))
    return result


@dataclass(frozen=True)
class WiFi7Config:
    """EHT-numerology OFDM model with a generic, nonstandard pilot allocation."""

    bandwidth_mhz: int = 20
    modulation: str = "4096QAM"
    guard_interval_us: float = 0.8
    n_symbols: int = 4
    channel_estimation: str = "flat"
    snr_db: Optional[float] = None
    seed: int = 42

    def __post_init__(self):
        if self.bandwidth_mhz not in (20, 40, 80, 160, 320):
            raise ValueError("bandwidth_mhz must be 20, 40, 80, 160 or 320")
        if self.modulation not in ("QPSK", "16QAM", "64QAM", "256QAM", "1024QAM", "4096QAM"):
            raise ValueError("Unsupported Wi-Fi QAM modulation")
        if self.guard_interval_us not in (0.8, 1.6, 3.2):
            raise ValueError("guard_interval_us must be 0.8, 1.6 or 3.2")
        if self.channel_estimation not in ("flat", "per_subcarrier"):
            raise ValueError("channel_estimation must be 'flat' or 'per_subcarrier'")
        _positive_integer("n_symbols", self.n_symbols)
        _validate_noise(self.snr_db, self.seed)

    @property
    def sample_rate_hz(self):
        return self.bandwidth_mhz * 1e6

    @property
    def n_fft(self):
        return int(self.bandwidth_mhz * 12.8)

    @property
    def cp_length(self):
        return int(round(self.sample_rate_hz * self.guard_interval_us * 1e-6))

    @property
    def bits_per_symbol(self):
        return {"QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8,
                "1024QAM": 10, "4096QAM": 12}[self.modulation]


def _wifi_allocation(cfg):
    tone_count = {20: 242, 40: 484, 80: 996, 160: 1992, 320: 3984}[cfg.bandwidth_mhz]
    offsets = np.concatenate([np.arange(-tone_count // 2, 0),
                              np.arange(1, tone_count // 2 + 1)])
    active = offsets % cfg.n_fft
    pilot_mask = np.arange(tone_count) % 16 == 0
    return active, pilot_mask


def generate_wifi7_waveform(cfg: WiFi7Config, tx_bits=None):
    """Generate a training symbol followed by uncoded OFDM data and AWGN."""
    rng = np.random.default_rng(cfg.seed)
    active, pilot_mask = _wifi_allocation(cfg)
    bit_count = cfg.n_symbols * np.count_nonzero(~pilot_mask) * cfg.bits_per_symbol
    bits = (rng.integers(0, 2, bit_count, dtype=np.uint8) if tx_bits is None
            else _bits(tx_bits, bit_count))
    symbols = qam_modulate(bits, cfg.bits_per_symbol)
    grid = np.zeros((cfg.n_symbols + 1, cfg.n_fft), dtype=complex)
    grid[0, active] = 1.0
    grid[1:, active[pilot_mask]] = 1.0
    grid[1:, active[~pilot_mask]] = symbols.reshape(cfg.n_symbols, -1)
    useful = np.fft.ifft(grid, axis=1) * np.sqrt(cfg.n_fft)
    clean = np.concatenate([useful[:, -cfg.cp_length:], useful], axis=1).ravel()
    return {"config": cfg, "tx_bits": bits, "tx_symbols": symbols,
            "resource_grid": grid, "active_bins": active,
            "data_bins": active[~pilot_mask], "pilot_bins": active[pilot_mask],
            "time_signal_clean": clean, "time_signal": _add_awgn(clean, cfg.snr_db, rng)}


def demodulate_wifi7(rx_signal, cfg: WiFi7Config, tx_bits=None, tx_symbols=None):
    """Recover aligned OFDM with training-based equalization and pilot tracking.

    Flat mode assumes a frequency-flat channel. Per-subcarrier mode estimates
    each active tone independently. Neither mode uses known payload bits.
    """
    length = cfg.n_fft + cfg.cp_length
    samples = _samples(rx_signal, (cfg.n_symbols + 1) * length)
    useful = samples.reshape(cfg.n_symbols + 1, length)[:, cfg.cp_length:]
    grid = np.fft.fft(useful, axis=1) / np.sqrt(cfg.n_fft)
    active, pilot_mask = _wifi_allocation(cfg)
    channel = grid[0, active]
    if cfg.channel_estimation == "flat":
        channel = np.full(channel.shape, np.mean(channel), dtype=complex)
    if np.any(np.abs(channel) < 1e-12):
        raise ValueError("Training symbol has zero or unusable channel gain")
    equalized = grid[1:, active] / channel
    pilot_gain = np.mean(equalized[:, pilot_mask], axis=1)
    if np.any(np.abs(pilot_gain) < 1e-12):
        raise ValueError("Pilots have zero or unusable channel gain")
    equalized /= pilot_gain[:, None]
    symbols = equalized[:, ~pilot_mask].ravel()
    constellation = _qam_constellation(cfg.bits_per_symbol)
    tree = cKDTree(np.column_stack([constellation.real, constellation.imag]))
    _, indices = tree.query(np.column_stack([symbols.real, symbols.imag]))
    bits = ((indices[:, None] >> np.arange(cfg.bits_per_symbol)) & 1).astype(np.uint8).ravel()
    result = WirelessResult(bits, symbols, cfg.sample_rate_hz,
                            diagnostics={"channel_est": channel, "pilot_gain": pilot_gain,
                                         "equalized_grid": equalized})
    return _measure_result(result, tx_bits, tx_symbols)


@dataclass(frozen=True)
class BluetoothConfig:
    """Uncoded Bluetooth LE GFSK with known bit timing and zero residual CFO."""

    phy: str = "LE1M"
    n_bits: int = 256
    samples_per_symbol: int = 8
    snr_db: Optional[float] = None
    seed: int = 42

    def __post_init__(self):
        if self.phy not in ("LE1M", "LE2M"):
            raise ValueError("phy must be 'LE1M' or 'LE2M'")
        _positive_integer("n_bits", self.n_bits)
        _positive_integer("samples_per_symbol", self.samples_per_symbol, minimum=4)
        _validate_noise(self.snr_db, self.seed)

    @property
    def symbol_rate_hz(self):
        return 1e6 if self.phy == "LE1M" else 2e6

    @property
    def sample_rate_hz(self):
        return self.symbol_rate_hz * self.samples_per_symbol

    @property
    def frequency_deviation_hz(self):
        return self.symbol_rate_hz * 0.25


def generate_bluetooth_waveform(cfg: BluetoothConfig, tx_bits=None):
    """Generate Gaussian-filtered, continuous-phase binary FSK payload samples.

    The leading sample has zero phase, so each following phase increment
    represents one frequency sample. Constant edge extension removes FIR
    startup transients; it is not an over-the-air packet preamble.
    """
    rng = np.random.default_rng(cfg.seed)
    bits = (rng.integers(0, 2, cfg.n_bits, dtype=np.uint8) if tx_bits is None
            else _bits(tx_bits, cfg.n_bits))
    levels = 2 * bits.astype(float) - 1
    sigma = np.sqrt(np.log(2)) * cfg.samples_per_symbol / (2 * np.pi * 0.5)
    shaped = gaussian_filter1d(np.repeat(levels, cfg.samples_per_symbol), sigma,
                               mode="nearest", truncate=4.0)
    frequency = cfg.frequency_deviation_hz * shaped
    phase = np.concatenate([[0.0], 2 * np.pi * np.cumsum(frequency) / cfg.sample_rate_hz])
    clean = np.exp(1j * phase)
    return {"config": cfg, "tx_bits": bits, "tx_symbols": levels.astype(complex),
            "frequency_hz": frequency, "time_signal_clean": clean,
            "time_signal": _add_awgn(clean, cfg.snr_db, rng)}


def demodulate_bluetooth(rx_signal, cfg: BluetoothConfig, tx_bits=None):
    """Noncoherent GFSK frequency discrimination and per-bit integration.

    rx_symbols contains normalized frequency decision metrics, not QAM
    symbols. EVM is intentionally None; BER needs an optional bit reference.
    """
    samples = _samples(rx_signal, cfg.n_bits * cfg.samples_per_symbol + 1)
    if np.any(np.abs(samples) == 0):
        raise ValueError("GFSK discriminator requires nonzero IQ samples")
    frequency = np.angle(samples[1:] * np.conj(samples[:-1])) * cfg.sample_rate_hz / (2 * np.pi)
    decisions = frequency.reshape(cfg.n_bits, cfg.samples_per_symbol).mean(axis=1)
    decisions /= cfg.frequency_deviation_hz
    result = WirelessResult((decisions >= 0).astype(np.uint8), decisions.astype(complex),
                            cfg.sample_rate_hz,
                            diagnostics={"frequency_hz": frequency, "decision_metric": decisions})
    return _measure_result(result, tx_bits)


@dataclass(frozen=True)
class UWBConfig:
    """Aligned BPM-BPSK burst model, not an IEEE 802.15.4z packet generator."""

    n_symbols: int = 128
    chips_per_symbol: int = 64
    chips_per_burst: int = 8
    samples_per_chip: int = 4
    chip_rate_hz: float = 499.2e6
    snr_db: Optional[float] = None
    seed: int = 42

    def __post_init__(self):
        _positive_integer("n_symbols", self.n_symbols)
        _positive_integer("chips_per_symbol", self.chips_per_symbol, minimum=2)
        _positive_integer("chips_per_burst", self.chips_per_burst)
        _positive_integer("samples_per_chip", self.samples_per_chip, minimum=2)
        if self.chips_per_symbol % 2 or self.chips_per_burst > self.chips_per_symbol // 2:
            raise ValueError("chips_per_symbol must be even and each half must fit a burst")
        if not np.isfinite(self.chip_rate_hz) or self.chip_rate_hz <= 0:
            raise ValueError("chip_rate_hz must be positive and finite")
        _validate_noise(self.snr_db, self.seed)

    @property
    def sample_rate_hz(self):
        return self.chip_rate_hz * self.samples_per_chip

    @property
    def samples_per_symbol(self):
        return self.chips_per_symbol * self.samples_per_chip


def _uwb_burst(cfg):
    pulse = gaussian(cfg.samples_per_chip, std=cfg.samples_per_chip / 6)
    pulse /= np.sqrt(np.sum(pulse ** 2))
    return np.tile(pulse, cfg.chips_per_burst)


def generate_uwb_waveform(cfg: UWBConfig, tx_bits=None):
    """Map bit pairs to early/late position and positive/negative burst polarity.

    One known early positive burst precedes the data for flat-channel gain
    estimation. tx_symbols holds flattened early/late reference amplitudes.
    """
    rng = np.random.default_rng(cfg.seed)
    bit_count = cfg.n_symbols * 2
    bits = (rng.integers(0, 2, bit_count, dtype=np.uint8) if tx_bits is None
            else _bits(tx_bits, bit_count))
    positions = bits[0::2]
    polarities = 1 - 2 * bits[1::2].astype(float)
    burst = _uwb_burst(cfg)
    half_symbol = cfg.samples_per_symbol // 2
    blocks = np.zeros((cfg.n_symbols + 1, cfg.samples_per_symbol), dtype=complex)
    blocks[0, :burst.size] = burst
    sample_indices = positions[:, None].astype(int) * half_symbol + np.arange(burst.size)
    blocks[np.arange(1, cfg.n_symbols + 1)[:, None], sample_indices] = polarities[:, None] * burst
    reference = np.zeros((cfg.n_symbols, 2), dtype=complex)
    reference[np.arange(cfg.n_symbols), positions] = polarities
    clean = blocks.ravel()
    return {"config": cfg, "tx_bits": bits, "tx_symbols": reference.ravel(),
            "burst_template": burst, "time_signal_clean": clean,
            "time_signal": _add_awgn(clean, cfg.snr_db, rng)}


def demodulate_uwb(rx_signal, cfg: UWBConfig, tx_bits=None, tx_symbols=None):
    """Matched-filter BPM-BPSK recovery using training-only complex gain.

    Assumes a flat channel and exact burst timing. rx_symbols is the flattened
    pair of soft early/late amplitudes, before hard position/polarity decisions.
    """
    samples = _samples(rx_signal, (cfg.n_symbols + 1) * cfg.samples_per_symbol)
    blocks = samples.reshape(cfg.n_symbols + 1, cfg.samples_per_symbol)
    burst = _uwb_burst(cfg)
    energy = np.sum(np.abs(burst) ** 2)
    half_symbol = cfg.samples_per_symbol // 2
    early = blocks[:, :burst.size] @ burst.conj() / energy
    late = blocks[:, half_symbol:half_symbol + burst.size] @ burst.conj() / energy
    channel = early[0]
    if np.abs(channel) < 1e-12:
        raise ValueError("Training burst has zero or unusable channel gain")
    scores = np.column_stack([early[1:], late[1:]]) / channel
    positions = np.argmax(np.abs(scores), axis=1)
    polarities = scores[np.arange(cfg.n_symbols), positions].real < 0
    bits = np.column_stack([positions, polarities]).astype(np.uint8).ravel()
    result = WirelessResult(bits, scores.ravel(), cfg.sample_rate_hz,
                            diagnostics={"early": scores[:, 0], "late": scores[:, 1],
                                         "channel_est": channel})
    return _measure_result(result, tx_bits, tx_symbols)


def plot_wireless_result(waveform, result: WirelessResult):
    """Return a Matplotlib figure without showing it or switching backends."""
    import matplotlib.pyplot as plt

    cfg = waveform["config"]
    if isinstance(cfg, WiFi7Config):
        label = f"Wi-Fi 7-inspired OFDM, {cfg.bandwidth_mhz} MHz, {cfg.modulation}"
    elif isinstance(cfg, BluetoothConfig):
        label = f"Bluetooth {cfg.phy} GFSK"
    elif isinstance(cfg, UWBConfig):
        label = "UWB BPM-BPSK pulse model"
    else:
        raise ValueError("Unsupported wireless configuration")
    snr_label = "no added noise" if cfg.snr_db is None else f"SNR {cfg.snr_db:g} dB"
    evm_label = "N/A" if result.evm_rms is None else f"{result.evm_rms:.4f}%"
    ber_label = "N/A" if result.ber is None else f"{result.ber:.2e}"
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    figure.suptitle(f"{label}\nUncoded simulation, {snr_label}; EVM {evm_label}; BER {ber_label}")

    samples = waveform["time_signal"]
    shown = min(2048, samples.size)
    time_us = np.arange(shown) / cfg.sample_rate_hz * 1e6
    axes[0, 0].plot(time_us, samples[:shown].real, linewidth=0.8, label="I")
    axes[0, 0].plot(time_us, samples[:shown].imag, linewidth=0.8, alpha=0.7, label="Q")
    axes[0, 0].set(title="IQ waveform", xlabel="Time (us)", ylabel="Amplitude")
    axes[0, 0].legend()

    frequency, density = welch(samples, fs=cfg.sample_rate_hz,
                                nperseg=min(1024, samples.size), return_onesided=False)
    axes[0, 1].plot(np.fft.fftshift(frequency) / 1e6,
                    10 * np.log10(np.maximum(np.fft.fftshift(density), 1e-30)))
    axes[0, 1].set(title="Baseband power spectrum", xlabel="Frequency (MHz)",
                   ylabel="PSD (dB/Hz)")

    if isinstance(cfg, BluetoothConfig):
        frequency = result.diagnostics["frequency_hz"][:256]
        axes[1, 0].plot(np.arange(frequency.size) / cfg.sample_rate_hz * 1e6, frequency / 1e3)
        axes[1, 0].axhline(cfg.frequency_deviation_hz / 1e3, color="gray", linestyle="--")
        axes[1, 0].axhline(-cfg.frequency_deviation_hz / 1e3, color="gray", linestyle="--")
        axes[1, 0].set(title="GFSK frequency discriminator", xlabel="Time (us)",
                       ylabel="Frequency deviation (kHz)")
        decisions = result.diagnostics["decision_metric"][:80]
        axes[1, 1].step(np.arange(decisions.size), decisions, where="mid", label="Received")
        axes[1, 1].step(np.arange(decisions.size), waveform["tx_symbols"][:decisions.size].real,
                        where="mid", linestyle="--", alpha=0.6, label="Transmitted levels")
        axes[1, 1].set(title="GFSK bit decision metrics (not QAM EVM)",
                       xlabel="Bit index", ylabel="Normalized frequency")
        axes[1, 1].legend()
    else:
        if isinstance(cfg, WiFi7Config):
            symbols = result.rx_symbols[:10000]
            axes[1, 0].scatter(symbols.real, symbols.imag, s=3, alpha=0.5)
            axes[1, 0].set(title="Equalized QAM constellation", xlabel="I", ylabel="Q")
        else:
            axes[1, 0].scatter(result.diagnostics["early"].real,
                               result.diagnostics["late"].real, s=12, alpha=0.6)
            axes[1, 0].scatter([1, -1, 0, 0], [0, 0, 1, -1], marker="+", color="red")
            axes[1, 0].set(title="BPM-BPSK matched-filter amplitudes",
                           xlabel="Early amplitude", ylabel="Late amplitude")
        axes[1, 0].set_aspect("equal", adjustable="box")
        reference = waveform["tx_symbols"]
        errors = np.abs(result.rx_symbols - reference) / np.sqrt(np.mean(np.abs(reference) ** 2))
        axes[1, 1].plot(errors[:1500] * 100, linewidth=0.7)
        axes[1, 1].set(title="Soft-symbol error magnitude", xlabel="Decision component index",
                       ylabel="Error (% of reference RMS)")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure