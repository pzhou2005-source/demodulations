"""5G NR OFDM Demodulator.

Receives time-domain IQ samples and recovers:
  - Coarse CFO estimation (CP-based) and correction
  - Resource grid via OFDM demodulation
  - Channel estimate from DMRS (LS + frequency-domain smoothing)
  - Equalised data symbols (ZF or MMSE)
  - Hard-decision bits
  - EVM (RMS, per-symbol, per-subcarrier), BER
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from nr5g_waveform import (
    NR5GConfig, ofdm_demodulate, qam_demodulate_hard, generate_dmrs,
    _qam_constellation, _cp_lengths,
)


@dataclass
class DemodResult:
    """Container for demodulation results."""
    rx_grid: np.ndarray
    channel_est: np.ndarray
    eq_grid: np.ndarray
    rx_symbols: np.ndarray
    rx_bits: np.ndarray
    evm_per_symbol: np.ndarray
    evm_per_subcarrier: np.ndarray
    evm_rms: float
    cfo_est_hz: float = 0.0
    noise_var_est: float = 0.0
    ber: Optional[float] = None
    constellation_ref: Optional[np.ndarray] = None
    evm_peak: Optional[float] = None


# ---------------------------------------------------------------------------
# CFO estimation (CP-based)
# ---------------------------------------------------------------------------
def estimate_cfo_cp(signal: np.ndarray, n_fft: int,
                    cp_lengths: list[int], n_symbols: int,
                    sample_rate_hz: float) -> float:
    """Estimate carrier frequency offset from cyclic prefix correlation.

    Uses the phase rotation between CP and its copy at end of each symbol.
    """
    acc = 0.0 + 0j
    pos = 0
    count = 0
    for sym_idx in range(min(n_symbols, len(cp_lengths))):
        cp_len = cp_lengths[sym_idx % len(cp_lengths)]
        if pos + cp_len + n_fft > len(signal):
            break
        cp_part = signal[pos:pos + cp_len]
        end_part = signal[pos + n_fft:pos + n_fft + cp_len]
        acc += np.sum(end_part * np.conj(cp_part))
        count += 1
        pos += cp_len + n_fft

    if count == 0:
        return 0.0
    phase = np.angle(acc)
    cfo_hz = phase / (2 * np.pi * n_fft) * sample_rate_hz
    return float(cfo_hz)


def correct_cfo(signal: np.ndarray, cfo_hz: float,
                sample_rate_hz: float) -> np.ndarray:
    """Remove carrier frequency offset from time-domain signal."""
    t = np.arange(len(signal)) / sample_rate_hz
    return signal * np.exp(-1j * 2 * np.pi * cfo_hz * t)


# ---------------------------------------------------------------------------
# Channel estimation
# ---------------------------------------------------------------------------
def channel_estimate_dmrs(rx_grid: np.ndarray, cfg: NR5GConfig,
                          smoothing_taps: int = 5) -> np.ndarray:
    """LS channel estimation from DMRS with optional frequency-domain smoothing.

    Interpolates across subcarriers, optionally smooths with a moving-average
    filter, then applies the same estimate to all symbols in the slot
    (flat-fading assumption per slot).
    """
    if cfg.channel_estimation not in ("linear", "flat"):
        raise ValueError("channel_estimation must be 'linear' or 'flat'")
    n_sym, n_sc = rx_grid.shape
    h_est = np.ones((n_sym, n_sc), dtype=complex)

    for slot in range(cfg.n_slots):
        sym_global = slot * cfg.symbols_per_slot + cfg.dmrs_symbol
        dmrs_ref = generate_dmrs(n_sc, slot, cfg.dmrs_symbol, cfg.cell_id)

        rx_dmrs = rx_grid[sym_global, 0::2]
        h_dmrs = rx_dmrs / dmrs_ref

        sc_dmrs = np.arange(0, n_sc, 2)
        sc_all = np.arange(n_sc)
        h_full = np.interp(sc_all, sc_dmrs, h_dmrs.real) + \
                 1j * np.interp(sc_all, sc_dmrs, h_dmrs.imag)

        # frequency-domain moving-average smoothing
        if smoothing_taps > 1:
            kernel = np.ones(smoothing_taps) / smoothing_taps
            h_full = np.convolve(h_full, kernel, mode="same")

        sym_start = slot * cfg.symbols_per_slot
        sym_end = sym_start + cfg.symbols_per_slot
        for s in range(sym_start, min(sym_end, n_sym)):
            h_est[s, :] = h_full

    return h_est


def estimate_noise_variance(rx_grid: np.ndarray, h_est: np.ndarray,
                            cfg: NR5GConfig) -> float:
    """Estimate noise variance from DMRS residuals."""
    noise_samples = []
    for slot in range(cfg.n_slots):
        sym_global = slot * cfg.symbols_per_slot + cfg.dmrs_symbol
        dmrs_ref = generate_dmrs(cfg.n_sc, slot, cfg.dmrs_symbol, cfg.cell_id)
        rx_dmrs = rx_grid[sym_global, 0::2]
        h_on_dmrs = h_est[sym_global, 0::2]
        residual = rx_dmrs - h_on_dmrs * dmrs_ref
        noise_samples.append(residual)
    all_noise = np.concatenate(noise_samples)
    return float(np.mean(np.abs(all_noise) ** 2))


# ---------------------------------------------------------------------------
# Equalisation
# ---------------------------------------------------------------------------
def equalise_zf(rx_grid: np.ndarray, h_est: np.ndarray) -> np.ndarray:
    """Zero-forcing equalisation."""
    return rx_grid / h_est


def equalise_mmse(rx_grid: np.ndarray, h_est: np.ndarray,
                  noise_var: float) -> np.ndarray:
    """MMSE equalisation: W = H* / (|H|² + σ²)."""
    h_conj = np.conj(h_est)
    h_pow = np.abs(h_est) ** 2
    return (h_conj * rx_grid) / (h_pow + noise_var)


# ---------------------------------------------------------------------------
# EVM
# ---------------------------------------------------------------------------
def compute_evm(tx_symbols: np.ndarray, rx_symbols: np.ndarray) -> float:
    """RMS EVM in percent."""
    err = rx_symbols - tx_symbols
    return float(np.sqrt(np.mean(np.abs(err) ** 2) /
                         np.mean(np.abs(tx_symbols) ** 2)) * 100)


def compute_evm_per_subcarrier(tx_grid: np.ndarray, eq_grid: np.ndarray,
                               cfg: NR5GConfig,
                               data_positions: list) -> np.ndarray:
    """EVM per subcarrier, averaged across OFDM symbols."""
    n_sc = cfg.n_sc
    evm_acc = np.zeros(n_sc)
    count = np.zeros(n_sc)

    for sym_idx, kind in data_positions:
        if kind == "odd":
            scs = np.arange(1, n_sc, 2)
        else:
            scs = np.arange(n_sc)
        tx_slice = tx_grid[sym_idx, scs]
        rx_slice = eq_grid[sym_idx, scs]
        err2 = np.abs(rx_slice - tx_slice) ** 2
        ref2 = np.abs(tx_slice) ** 2
        np.add.at(evm_acc, scs, err2)
        np.add.at(count, scs, ref2)

    mask = count > 0
    evm_sc = np.zeros(n_sc)
    evm_sc[mask] = np.sqrt(evm_acc[mask] / count[mask]) * 100
    return evm_sc


# ---------------------------------------------------------------------------
# Full demodulation pipeline
# ---------------------------------------------------------------------------
def demodulate_nr5g(rx_signal: np.ndarray, cfg: NR5GConfig,
                     tx_bits: Optional[np.ndarray] = None,
                     tx_symbols: Optional[np.ndarray] = None,
                     tx_grid: Optional[np.ndarray] = None,
                     data_positions: Optional[list] = None,
                     equaliser: str = "mmse",
                     cfo_correct: bool = True) -> DemodResult:
    """Full 5G NR demodulation pipeline.

    Args:
        rx_signal: time-domain IQ samples
        cfg: NR5GConfig used for generation
        tx_bits: original bits for BER calculation
        tx_symbols: original symbols for EVM
        tx_grid: original resource grid for per-subcarrier EVM
        data_positions: list of (sym_idx, 'all'|'odd')
        equaliser: 'zf' or 'mmse'
        cfo_correct: estimate and correct CFO before OFDM demod
    """
    cp = [length for slot in range(cfg.n_slots)
          for length in _cp_lengths(cfg.n_fft, cfg.mu, slot)]
    n_sym_total = cfg.symbols_per_slot * cfg.n_slots
    sample_rate_hz = cfg.sample_rate_mhz * 1e6

    # 0) CFO estimation & correction
    cfo_est = 0.0
    sig = rx_signal
    if cfo_correct:
        cfo_est = estimate_cfo_cp(sig, cfg.n_fft, cp, n_sym_total, sample_rate_hz)
        if abs(cfo_est) > 0.5:
            sig = correct_cfo(sig, cfo_est, sample_rate_hz)

    # 1) OFDM demodulate
    rx_grid = ofdm_demodulate(sig, cfg.n_fft, cfg.n_sc, cp, n_sym_total)

    # 2) Channel estimation
    h_est = channel_estimate_dmrs(rx_grid, cfg)

    # 3) Noise variance estimation
    noise_var = estimate_noise_variance(rx_grid, h_est, cfg)

    # 4) Equalisation
    if equaliser == "mmse":
        eq_grid = equalise_mmse(rx_grid, h_est, noise_var)
    else:
        eq_grid = equalise_zf(rx_grid, h_est)

    # 5) Extract data symbols
    rx_data = []
    if data_positions is not None:
        for sym_idx, kind in data_positions:
            if kind == "odd":
                rx_data.append(eq_grid[sym_idx, 1::2])
            else:
                rx_data.append(eq_grid[sym_idx, :])
    else:
        for slot in range(cfg.n_slots):
            for sym_local in range(cfg.symbols_per_slot):
                sym_global = slot * cfg.symbols_per_slot + sym_local
                if sym_local == cfg.dmrs_symbol:
                    rx_data.append(eq_grid[sym_global, 1::2])
                else:
                    rx_data.append(eq_grid[sym_global, :])
    rx_symbols = np.concatenate(rx_data)

    # 6) Hard-decision demod
    rx_bits = qam_demodulate_hard(rx_symbols, cfg.bits_per_symbol)

    # 7) EVM per OFDM symbol
    evm_per_sym = np.zeros(n_sym_total)
    if tx_symbols is not None:
        pos = 0
        for i, (sym_idx, kind) in enumerate(data_positions or []):
            n = cfg.n_sc // 2 if kind == "odd" else cfg.n_sc
            if pos + n <= len(tx_symbols):
                evm_per_sym[sym_idx] = compute_evm(
                    tx_symbols[pos:pos + n], rx_symbols[pos:pos + n])
            pos += n

    evm_rms = compute_evm(tx_symbols, rx_symbols) if tx_symbols is not None else 0.0
    evm_peak = None
    if tx_symbols is not None:
        evm_peak = float(np.max(np.abs(rx_symbols - tx_symbols)) /
                         np.sqrt(np.mean(np.abs(tx_symbols) ** 2)) * 100.0)

    # 8) EVM per subcarrier
    evm_per_sc = np.zeros(cfg.n_sc)
    if tx_grid is not None and data_positions is not None:
        evm_per_sc = compute_evm_per_subcarrier(tx_grid, eq_grid, cfg, data_positions)

    # 9) BER
    ber = None
    if tx_bits is not None:
        n = min(len(tx_bits), len(rx_bits))
        ber = float(np.mean(tx_bits[:n] != rx_bits[:n]))

    constellation_ref = _qam_constellation(cfg.bits_per_symbol)

    return DemodResult(
        rx_grid=rx_grid,
        channel_est=h_est,
        eq_grid=eq_grid,
        rx_symbols=rx_symbols,
        rx_bits=rx_bits,
        evm_per_symbol=evm_per_sym,
        evm_per_subcarrier=evm_per_sc,
        evm_rms=evm_rms,
        cfo_est_hz=cfo_est,
        noise_var_est=noise_var,
        ber=ber,
        constellation_ref=constellation_ref,
        evm_peak=evm_peak,
    )


if __name__ == "__main__":
    from nr5g_waveform import generate_nr5g_waveform

    cfg = NR5GConfig(mu=1, bw_mhz=20, n_rb=51, modulation="64QAM",
                     n_slots=2, snr_db=30, seed=42)
    tx = generate_nr5g_waveform(cfg)
    result = demodulate_nr5g(
        tx["time_signal"], cfg,
        tx_bits=tx["tx_bits"],
        tx_symbols=tx["tx_symbols"],
        tx_grid=tx["resource_grid"],
        data_positions=tx["data_positions"],
        equaliser="mmse",
    )
    print(f"CFO est:  {result.cfo_est_hz:.1f} Hz")
    print(f"Noise σ²: {result.noise_var_est:.2e}")
    print(f"EVM RMS:  {result.evm_rms:.2f}%")
    print(f"BER:      {result.ber:.2e}" if result.ber is not None else "BER: N/A")
    print(f"Rx syms:  {len(result.rx_symbols)}")
