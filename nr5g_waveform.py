"""5G NR Waveform Generator (OFDM downlink, single-component carrier).

Generates a 5G NR-compliant PDSCH waveform with:
  - Configurable numerology (μ = 0..3)
  - QAM modulation (QPSK, 16-QAM, 64-QAM, 256-QAM)
  - DMRS insertion (Type 1, single-symbol)
  - CP-OFDM modulation
  - Optional AWGN channel
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 3GPP TS 38.211 tables
# ---------------------------------------------------------------------------
QAM_MAP = {
    "QPSK": 2,
    "16QAM": 4,
    "64QAM": 6,
    "256QAM": 8,
}

NUMEROLOGY = {
    0: {"scs_khz": 15,  "slots_per_subframe": 1,  "symbols_per_slot": 14},
    1: {"scs_khz": 30,  "slots_per_subframe": 2,  "symbols_per_slot": 14},
    2: {"scs_khz": 60,  "slots_per_subframe": 4,  "symbols_per_slot": 14},
    3: {"scs_khz": 120, "slots_per_subframe": 8,  "symbols_per_slot": 14},
}

# Normal CP lengths (in samples at Nfft rate) per TS 38.211 Table 5.3.1-1
# First symbol of each half-slot has extended CP.
def _cp_lengths(nfft: int, mu: int) -> list[int]:
    """Return list of CP lengths for one slot (14 symbols, normal CP)."""
    base = nfft * 144 // 2048
    ext  = nfft * 512 // 2048  # extended portion for symbol 0 & 7
    kappa = 1 << mu  # Ts/Tc scaling
    cp = []
    for sym in range(14):
        if sym in (0, 7):
            cp.append(base + ext // kappa)
        else:
            cp.append(base)
    return cp


@dataclass
class NR5GConfig:
    """Configuration for a 5G NR downlink waveform."""
    mu: int = 1                     # numerology index (SCS = 15 * 2^mu kHz)
    bw_mhz: float = 20.0           # channel bandwidth in MHz
    n_rb: int = 51                  # number of resource blocks
    modulation: str = "64QAM"       # modulation order
    n_slots: int = 2                # number of slots to generate
    n_fft: int = 2048               # FFT size
    sample_rate_mhz: float = 61.44  # sampling rate in MHz
    dmrs_symbol: int = 2            # OFDM symbol index carrying DMRS
    cell_id: int = 0                # physical cell ID (for DMRS sequence)
    snr_db: Optional[float] = None  # add AWGN if set
    cfo_hz: float = 0.0             # carrier frequency offset impairment
    channel_taps: Optional[list] = None  # multipath channel [complex coeffs]
    channel_delays: Optional[list] = None  # tap delays in samples
    seed: int = 42                  # RNG seed for data & noise

    @property
    def scs_khz(self) -> int:
        return NUMEROLOGY[self.mu]["scs_khz"]

    @property
    def n_sc(self) -> int:
        return self.n_rb * 12

    @property
    def symbols_per_slot(self) -> int:
        return NUMEROLOGY[self.mu]["symbols_per_slot"]

    @property
    def bits_per_symbol(self) -> int:
        return QAM_MAP[self.modulation]


# ---------------------------------------------------------------------------
# Modulation / demodulation helpers
# ---------------------------------------------------------------------------
def _qam_constellation(order_bits: int) -> np.ndarray:
    """Gray-coded QAM constellation, average power ≈ 1."""
    M = 1 << order_bits
    m = int(np.sqrt(M))
    idx = np.arange(M)
    # Gray-code mapping per axis
    def _gray(n, bits):
        return n ^ (n >> 1)
    re = np.array([_gray(i % m, order_bits // 2) for i in idx])
    im = np.array([_gray(i // m, order_bits // 2) for i in idx])
    # map to symmetric grid
    pts = (2 * re - m + 1) + 1j * (2 * im - m + 1)
    pts /= np.sqrt(np.mean(np.abs(pts) ** 2))  # normalise power
    return pts


def qam_modulate(bits: np.ndarray, order_bits: int) -> np.ndarray:
    """Map bit vector to QAM symbols."""
    constellation = _qam_constellation(order_bits)
    n_sym = len(bits) // order_bits
    indices = np.zeros(n_sym, dtype=int)
    for b in range(order_bits):
        indices += bits[b::order_bits].astype(int) << b
    return constellation[indices]


def qam_demodulate_hard(symbols: np.ndarray, order_bits: int) -> np.ndarray:
    """Hard-decision QAM demodulation → bit array."""
    constellation = _qam_constellation(order_bits)
    # nearest-neighbour
    dist = np.abs(symbols[:, None] - constellation[None, :])
    indices = np.argmin(dist, axis=1)
    bits = np.zeros(len(symbols) * order_bits, dtype=np.uint8)
    for b in range(order_bits):
        bits[b::order_bits] = (indices >> b) & 1
    return bits


# ---------------------------------------------------------------------------
# DMRS sequence (gold-sequence based, simplified)
# ---------------------------------------------------------------------------
def _gold_sequence(length: int, c_init: int) -> np.ndarray:
    """TS 38.211 §5.2.1 pseudo-random sequence generator."""
    Nc = 1600
    total = length + Nc
    x1 = np.zeros(total, dtype=np.uint8)
    x2 = np.zeros(total, dtype=np.uint8)
    # init x1
    x1[0] = 1
    # init x2 from c_init
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    for n in range(total - 31):
        x1[n + 31] = (x1[n + 3] + x1[n]) % 2
        x2[n + 31] = (x2[n + 3] + x2[n + 2] + x2[n + 1] + x2[n]) % 2
    c = np.zeros(length, dtype=np.uint8)
    for n in range(length):
        c[n] = (x1[n + Nc] + x2[n + Nc]) % 2
    return c


def generate_dmrs(n_sc: int, slot: int, symbol: int, cell_id: int) -> np.ndarray:
    """Generate DMRS symbols for one OFDM symbol (Type 1, config type 1)."""
    # c_init per TS 38.211 §7.4.1.1.1
    c_init = ((1 << 17) * (14 * slot + symbol + 1) * (2 * cell_id + 1)
              + 2 * cell_id) % (1 << 31)
    # DMRS on even subcarriers → n_sc // 2 complex symbols
    n_dmrs = n_sc // 2
    seq = _gold_sequence(2 * n_dmrs, c_init)
    r = (1 / np.sqrt(2)) * ((1 - 2 * seq[0::2]) + 1j * (1 - 2 * seq[1::2]))
    return r


# ---------------------------------------------------------------------------
# OFDM modulator
# ---------------------------------------------------------------------------
def ofdm_modulate(resource_grid: np.ndarray, n_fft: int, cp_lengths: list[int]) -> np.ndarray:
    """OFDM modulate a resource grid [n_symbols x n_sc] → time-domain samples."""
    n_symbols, n_sc = resource_grid.shape
    samples = []
    for sym_idx in range(n_symbols):
        freq = np.zeros(n_fft, dtype=complex)
        # map subcarriers centred around DC
        half = n_sc // 2
        freq[1:half + 1] = resource_grid[sym_idx, half:]
        freq[n_fft - half:] = resource_grid[sym_idx, :half]
        td = np.fft.ifft(freq) * np.sqrt(n_fft)
        cp_len = cp_lengths[sym_idx % len(cp_lengths)]
        symbol_with_cp = np.concatenate([td[-cp_len:], td])
        samples.append(symbol_with_cp)
    return np.concatenate(samples)


def ofdm_demodulate(signal: np.ndarray, n_fft: int, n_sc: int,
                     cp_lengths: list[int], n_symbols: int) -> np.ndarray:
    """OFDM demodulate time-domain signal → resource grid [n_symbols x n_sc]."""
    grid = np.zeros((n_symbols, n_sc), dtype=complex)
    pos = 0
    for sym_idx in range(n_symbols):
        cp_len = cp_lengths[sym_idx % len(cp_lengths)]
        pos += cp_len  # skip CP
        td = signal[pos:pos + n_fft]
        if len(td) < n_fft:
            break
        freq = np.fft.fft(td) / np.sqrt(n_fft)
        half = n_sc // 2
        grid[sym_idx, half:] = freq[1:half + 1]
        grid[sym_idx, :half] = freq[n_fft - half:]
        pos += n_fft
    return grid


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------
def generate_nr5g_waveform(cfg: NR5GConfig):
    """Generate a 5G NR waveform and return all intermediate data.

    Returns dict with keys:
        tx_bits, tx_symbols, resource_grid, dmrs_symbols, dmrs_positions,
        time_signal, config
    """
    rng = np.random.default_rng(cfg.seed)
    cp = _cp_lengths(cfg.n_fft, cfg.mu)
    n_sym_total = cfg.symbols_per_slot * cfg.n_slots
    n_sc = cfg.n_sc

    resource_grid = np.zeros((n_sym_total, n_sc), dtype=complex)
    all_tx_bits = []
    all_tx_symbols = []
    dmrs_all = {}
    data_positions = []

    for slot in range(cfg.n_slots):
        for sym_local in range(cfg.symbols_per_slot):
            sym_global = slot * cfg.symbols_per_slot + sym_local
            if sym_local == cfg.dmrs_symbol:
                # DMRS symbol — even subcarriers carry DMRS, odd carry data
                dmrs = generate_dmrs(n_sc, slot, sym_local, cfg.cell_id)
                resource_grid[sym_global, 0::2] = dmrs
                dmrs_all[sym_global] = dmrs
                # data on odd subcarriers
                n_data = n_sc // 2
                bits = rng.integers(0, 2, n_data * cfg.bits_per_symbol, dtype=np.uint8)
                syms = qam_modulate(bits, cfg.bits_per_symbol)
                resource_grid[sym_global, 1::2] = syms
                all_tx_bits.append(bits)
                all_tx_symbols.append(syms)
                data_positions.append((sym_global, "odd"))
            else:
                # pure data symbol
                bits = rng.integers(0, 2, n_sc * cfg.bits_per_symbol, dtype=np.uint8)
                syms = qam_modulate(bits, cfg.bits_per_symbol)
                resource_grid[sym_global] = syms
                all_tx_bits.append(bits)
                all_tx_symbols.append(syms)
                data_positions.append((sym_global, "all"))

    tx_bits = np.concatenate(all_tx_bits)
    tx_symbols = np.concatenate(all_tx_symbols)

    # OFDM modulate
    cp_full = cp * cfg.n_slots  # repeat CP pattern for all slots
    time_signal = ofdm_modulate(resource_grid, cfg.n_fft, cp_full)

    # apply multipath channel
    time_signal_ch = time_signal.copy()
    if cfg.channel_taps is not None:
        taps = np.array(cfg.channel_taps, dtype=complex)
        delays = cfg.channel_delays or list(range(len(taps)))
        h = np.zeros(max(delays) + 1, dtype=complex)
        for coeff, d in zip(taps, delays):
            h[d] = coeff
        time_signal_ch = np.convolve(time_signal, h)[:len(time_signal)]

    # apply CFO
    if cfg.cfo_hz != 0.0:
        t = np.arange(len(time_signal_ch)) / (cfg.sample_rate_mhz * 1e6)
        time_signal_ch = time_signal_ch * np.exp(1j * 2 * np.pi * cfg.cfo_hz * t)

    # add AWGN
    time_signal_noisy = time_signal_ch.copy()
    if cfg.snr_db is not None:
        sig_power = np.mean(np.abs(time_signal_ch) ** 2)
        noise_power = sig_power / (10 ** (cfg.snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            rng.standard_normal(len(time_signal_ch))
            + 1j * rng.standard_normal(len(time_signal_ch))
        )
        time_signal_noisy = time_signal_ch + noise

    return {
        "tx_bits": tx_bits,
        "tx_symbols": tx_symbols,
        "resource_grid": resource_grid,
        "dmrs_symbols": dmrs_all,
        "data_positions": data_positions,
        "time_signal_clean": time_signal,
        "time_signal": time_signal_noisy,
        "cp_lengths": cp_full,
        "config": cfg,
    }


def save_waveform_iq(filepath: str, signal: np.ndarray, fmt: str = "binary"):
    """Save IQ waveform to file. Formats: 'binary' (interleaved float32), 'csv'."""
    if fmt == "binary":
        iq = np.zeros(2 * len(signal), dtype=np.float32)
        iq[0::2] = signal.real.astype(np.float32)
        iq[1::2] = signal.imag.astype(np.float32)
        iq.tofile(filepath)
    elif fmt == "csv":
        data = np.column_stack([signal.real, signal.imag])
        np.savetxt(filepath, data, delimiter=",", header="I,Q", comments="")
    else:
        raise ValueError(f"Unknown format: {fmt}")


if __name__ == "__main__":
    cfg = NR5GConfig(mu=1, bw_mhz=20, n_rb=51, modulation="64QAM",
                     n_slots=2, snr_db=30, seed=42)
    result = generate_nr5g_waveform(cfg)
    print(f"Generated {len(result['time_signal'])} samples, "
          f"{len(result['tx_bits'])} bits, "
          f"{len(result['tx_symbols'])} QAM symbols")
    save_waveform_iq("nr5g_waveform.bin", result["time_signal"])
    print("Saved to nr5g_waveform.bin")
