"""5G NR OFDM Demodulator.

Receives time-domain IQ samples and recovers:
  - Resource grid via OFDM demodulation
  - Channel estimate from DMRS
  - Equalised data symbols
  - Hard-decision bits
  - EVM, BER, constellation plots
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from nr5g_waveform import (
    NR5GConfig, ofdm_demodulate, qam_demodulate_hard, generate_dmrs,
    _qam_constellation, _cp_lengths,
)


@dataclass
class DemodResult:
    """Container for demodulation results."""
    rx_grid: np.ndarray             # received resource grid [n_sym x n_sc]
    channel_est: np.ndarray         # channel estimate [n_sym x n_sc]
    eq_grid: np.ndarray             # equalised resource grid
    rx_symbols: np.ndarray          # equalised data symbols (flat)
    rx_bits: np.ndarray             # hard-decision bits
    evm_per_symbol: np.ndarray      # EVM per OFDM symbol (%)
    evm_rms: float                  # RMS EVM (%)
    ber: Optional[float] = None     # BER (if tx_bits provided)
    constellation_ref: Optional[np.ndarray] = None
    evm_peak: Optional[float] = None


def channel_estimate_dmrs(rx_grid: np.ndarray, cfg: NR5GConfig) -> np.ndarray:
    """Least-squares channel estimation from DMRS, with linear interpolation.

    DMRS sits on even subcarriers of the DMRS symbol(s).
    Interpolate in frequency across subcarriers, then copy to all symbols
    (flat-fading assumption across one slot — good enough for static/AWGN).
    """
    n_sym, n_sc = rx_grid.shape
    h_est = np.ones((n_sym, n_sc), dtype=complex)

    for slot in range(cfg.n_slots):
        sym_global = slot * cfg.symbols_per_slot + cfg.dmrs_symbol
        # known DMRS
        dmrs_ref = generate_dmrs(n_sc, slot, cfg.dmrs_symbol, cfg.cell_id)
        # LS estimate on DMRS subcarriers (even)
        rx_dmrs = rx_grid[sym_global, 0::2]
        h_dmrs = rx_dmrs / dmrs_ref  # LS: H = Y / X

        # interpolate to all subcarriers (linear)
        sc_dmrs = np.arange(0, n_sc, 2)
        sc_all = np.arange(n_sc)
        h_full = np.interp(sc_all, sc_dmrs, h_dmrs.real) + \
                 1j * np.interp(sc_all, sc_dmrs, h_dmrs.imag)

        # apply to all symbols in this slot
        sym_start = slot * cfg.symbols_per_slot
        sym_end = sym_start + cfg.symbols_per_slot
        for s in range(sym_start, min(sym_end, n_sym)):
            h_est[s, :] = h_full

    return h_est


def equalise_zf(rx_grid: np.ndarray, h_est: np.ndarray) -> np.ndarray:
    """Zero-forcing equalisation."""
    return rx_grid / h_est


def compute_evm(tx_symbols: np.ndarray, rx_symbols: np.ndarray) -> float:
    """RMS EVM in percent."""
    err = rx_symbols - tx_symbols
    evm = np.sqrt(np.mean(np.abs(err) ** 2) / np.mean(np.abs(tx_symbols) ** 2)) * 100
    return evm


def demodulate_nr5g(rx_signal: np.ndarray, cfg: NR5GConfig,
                     tx_bits: Optional[np.ndarray] = None,
                     tx_symbols: Optional[np.ndarray] = None,
                     data_positions: Optional[list] = None) -> DemodResult:
    """Full 5G NR demodulation pipeline.

    Args:
        rx_signal: time-domain IQ samples
        cfg: NR5GConfig used for generation
        tx_bits: (optional) original bits for BER calculation
        tx_symbols: (optional) original symbols for EVM
        data_positions: (optional) list of (sym_idx, 'all'|'odd')
    """
    cp = _cp_lengths(cfg.n_fft, cfg.mu) * cfg.n_slots
    n_sym_total = cfg.symbols_per_slot * cfg.n_slots

    # 1) OFDM demodulate
    rx_grid = ofdm_demodulate(rx_signal, cfg.n_fft, cfg.n_sc, cp, n_sym_total)

    # 2) Channel estimation
    h_est = channel_estimate_dmrs(rx_grid, cfg)

    # 3) Equalisation (ZF)
    eq_grid = equalise_zf(rx_grid, h_est)

    # 4) Extract data symbols
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

    # 5) Hard-decision demod
    rx_bits = qam_demodulate_hard(rx_symbols, cfg.bits_per_symbol)

    # 6) EVM per OFDM symbol
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

    # 7) BER
    ber = None
    if tx_bits is not None:
        n = min(len(tx_bits), len(rx_bits))
        ber = np.mean(tx_bits[:n] != rx_bits[:n])

    constellation_ref = _qam_constellation(cfg.bits_per_symbol)

    return DemodResult(
        rx_grid=rx_grid,
        channel_est=h_est,
        eq_grid=eq_grid,
        rx_symbols=rx_symbols,
        rx_bits=rx_bits,
        evm_per_symbol=evm_per_sym,
        evm_rms=evm_rms,
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
        data_positions=tx["data_positions"],
    )
    print(f"EVM RMS:  {result.evm_rms:.2f}%")
    print(f"BER:      {result.ber:.2e}" if result.ber is not None else "BER: N/A")
    print(f"Rx syms:  {len(result.rx_symbols)}")
