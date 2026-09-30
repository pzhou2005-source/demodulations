"""Comparison and correlation between Python demodulator and VSA 89600 results.

Produces side-by-side metrics, plots, and correlation analysis:
  - EVM RMS / Peak comparison
  - Constellation overlay
  - EVM vs OFDM symbol / subcarrier
  - Symbol-level cross-correlation (timing, phase, amplitude alignment)
  - Per-OFDM-symbol and per-subcarrier correlation
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from nr5g_waveform import NR5GConfig, generate_nr5g_waveform, save_waveform_iq
from nr5g_demod import demodulate_nr5g, DemodResult
from vsa_89600 import VSAResult, simulate_vsa_result


# ---------------------------------------------------------------------------
# Correlation engine
# ---------------------------------------------------------------------------
@dataclass
class CorrelationResult:
    """Results from correlating Python demod against VSA 89600."""
    sample_offset: int
    phase_offset_rad: float
    amplitude_ratio: float
    correlation_peak: float
    symbol_correlation: float
    evm_of_difference: float
    per_symbol_corr: np.ndarray
    per_subcarrier_corr: np.ndarray
    aligned_py_symbols: np.ndarray
    aligned_vsa_symbols: np.ndarray
    is_simulated: bool = False


def _complex_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised complex correlation coefficient |ρ| ∈ [0, 1]."""
    num = np.abs(np.sum(a * np.conj(b)))
    den = np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))
    return float(num / den) if den > 0 else 0.0


def correlate_symbol_streams(py_symbols: np.ndarray,
                             vsa_i: np.ndarray, vsa_q: np.ndarray,
                             max_lag: int = 50) -> tuple[int, float]:
    """Cross-correlate two symbol streams to find timing offset.

    Positive offset → VSA is delayed relative to Python.
    """
    vsa_symbols = vsa_i + 1j * vsa_q
    n = min(len(py_symbols), len(vsa_symbols))
    if n == 0:
        return 0, 0.0

    a = py_symbols[:n] / np.sqrt(np.mean(np.abs(py_symbols[:n]) ** 2))
    b = vsa_symbols[:n] / np.sqrt(np.mean(np.abs(vsa_symbols[:n]) ** 2))

    best_lag, best_corr = 0, 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = _complex_corr(a[:n - lag], b[lag:n])
        else:
            c = _complex_corr(a[-lag:n], b[:n + lag])
        if c > best_corr:
            best_corr = c
            best_lag = lag

    return best_lag, best_corr


def correlate_results(py_result: DemodResult, vsa_result: VSAResult,
                      cfg: NR5GConfig) -> CorrelationResult:
    """Full correlation: timing align → phase/amp correct → per-sym/sc metrics."""
    py_sym = py_result.rx_symbols
    vsa_sym = vsa_result.symbols_i + 1j * vsa_result.symbols_q

    # 1) timing offset
    offset, xcorr_peak = correlate_symbol_streams(
        py_sym, vsa_result.symbols_i, vsa_result.symbols_q)

    # 2) align
    n = min(len(py_sym), len(vsa_sym))
    if offset >= 0:
        a_py = py_sym[:n - offset]
        a_vsa = vsa_sym[offset:n]
    else:
        a_py = py_sym[-offset:n]
        a_vsa = vsa_sym[:n + offset]

    # 3) phase & amplitude correction
    phase_off = float(np.angle(np.sum(a_vsa * np.conj(a_py))))
    rms_py = np.sqrt(np.mean(np.abs(a_py) ** 2))
    rms_vsa = np.sqrt(np.mean(np.abs(a_vsa) ** 2))
    amp_ratio = float(rms_vsa / rms_py) if rms_py > 0 else 1.0
    a_vsa_corr = (a_vsa / amp_ratio) * np.exp(-1j * phase_off)

    # 4) overall symbol correlation
    sym_corr = _complex_corr(a_py, a_vsa_corr)

    # 5) EVM of the difference
    err = a_vsa_corr - a_py
    evm_diff = np.sqrt(np.mean(np.abs(err) ** 2) /
                       np.mean(np.abs(a_py) ** 2)) * 100.0

    # 6) per-OFDM-symbol correlation
    n_sym = cfg.symbols_per_slot * cfg.n_slots
    sym_sizes = []
    for slot in range(cfg.n_slots):
        for sym_local in range(cfg.symbols_per_slot):
            if sym_local == cfg.dmrs_symbol:
                sym_sizes.append(cfg.n_sc // 2)
            else:
                sym_sizes.append(cfg.n_sc)

    per_sym_corr = np.zeros(n_sym)
    pos = 0
    for i, sz in enumerate(sym_sizes):
        end = min(pos + sz, len(a_py))
        if end > pos and end <= len(a_vsa_corr):
            per_sym_corr[i] = _complex_corr(a_py[pos:end], a_vsa_corr[pos:end])
        pos = end

    # 7) per-subcarrier correlation (averaged across symbols)
    n_sc = cfg.n_sc
    per_sc_acc = np.zeros(n_sc, dtype=complex)
    per_sc_cnt = np.zeros(n_sc)
    pos = 0
    for i, sz in enumerate(sym_sizes):
        sym_local = i % cfg.symbols_per_slot
        if sym_local == cfg.dmrs_symbol:
            scs = np.arange(1, n_sc, 2)[:sz]
        else:
            scs = np.arange(n_sc)[:sz]
        for j, sc in enumerate(scs):
            idx = pos + j
            if idx < len(a_py) and idx < len(a_vsa_corr):
                per_sc_acc[sc] += a_py[idx] * np.conj(a_vsa_corr[idx])
                per_sc_cnt[sc] += 1
        pos += sz
    mask = per_sc_cnt > 0
    per_sc_corr = np.zeros(n_sc)
    per_sc_corr[mask] = np.abs(per_sc_acc[mask]) / per_sc_cnt[mask]
    sc_max = np.max(per_sc_corr) if np.any(mask) else 1.0
    if sc_max > 0:
        per_sc_corr /= sc_max

    return CorrelationResult(
        sample_offset=offset,
        phase_offset_rad=phase_off,
        amplitude_ratio=amp_ratio,
        correlation_peak=xcorr_peak,
        symbol_correlation=sym_corr,
        evm_of_difference=evm_diff,
        per_symbol_corr=per_sym_corr,
        per_subcarrier_corr=per_sc_corr,
        aligned_py_symbols=a_py,
        aligned_vsa_symbols=a_vsa_corr,
        is_simulated=vsa_result.is_simulated,
    )


def plot_correlation(corr: CorrelationResult, cfg: NR5GConfig,
                     output_dir: str = "results"):
    """Generate correlation analysis plots."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    reference_label = "VSA (SIMULATED)" if corr.is_simulated else "VSA 89600"
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Python vs {reference_label} Correlation — μ={cfg.mu}, {cfg.modulation}",
                 fontsize=13, fontweight="bold")

    # 1) aligned constellation overlay
    ax = axes[0, 0]
    ax.scatter(corr.aligned_py_symbols.real, corr.aligned_py_symbols.imag,
               s=1, alpha=0.3, c="steelblue", label="Python")
    ax.scatter(corr.aligned_vsa_symbols.real, corr.aligned_vsa_symbols.imag,
               s=1, alpha=0.3, c="darkorange", label=f"{reference_label} (aligned)")
    ax.set_title(f"Constellation Overlay\nρ={corr.symbol_correlation:.4f}")
    ax.set_xlabel("I"); ax.set_ylabel("Q")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # 2) symbol error scatter
    ax = axes[0, 1]
    err = corr.aligned_vsa_symbols - corr.aligned_py_symbols
    ax.scatter(err.real, err.imag, s=1, alpha=0.3, c="red")
    ax.set_title(f"Symbol Difference (VSA − Python)\nEVM={corr.evm_of_difference:.3f}%")
    ax.set_xlabel("ΔI"); ax.set_ylabel("ΔQ")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    # 3) magnitude comparison
    ax = axes[0, 2]
    n_show = min(200, len(corr.aligned_py_symbols))
    ax.plot(np.abs(corr.aligned_py_symbols[:n_show]), label="Python", alpha=0.7)
    ax.plot(np.abs(corr.aligned_vsa_symbols[:n_show]), label=reference_label, alpha=0.7)
    ax.set_title("Symbol Magnitude (first 200)")
    ax.set_xlabel("Symbol Index"); ax.set_ylabel("|symbol|")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4) per-OFDM-symbol correlation
    ax = axes[1, 0]
    ax.bar(range(len(corr.per_symbol_corr)), corr.per_symbol_corr, color="steelblue")
    ax.axhline(corr.symbol_correlation, color="red", linestyle="--",
               label=f"Overall ρ={corr.symbol_correlation:.4f}")
    ax.set_title("Correlation per OFDM Symbol")
    ax.set_xlabel("OFDM Symbol Index"); ax.set_ylabel("|ρ|")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 5) per-subcarrier correlation
    ax = axes[1, 1]
    ax.plot(corr.per_subcarrier_corr, linewidth=0.8, color="steelblue")
    ax.set_title("Correlation per Subcarrier")
    ax.set_xlabel("Subcarrier Index"); ax.set_ylabel("Normalised correlation")
    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

    # 6) summary text box
    ax = axes[1, 2]
    ax.axis("off")
    txt = (
        f"Symbol offset:     {corr.sample_offset}\n"
        f"Phase offset:      {np.degrees(corr.phase_offset_rad):.2f}°\n"
        f"Amplitude ratio:   {corr.amplitude_ratio:.4f}\n"
        f"XCorr peak:        {corr.correlation_peak:.6f}\n"
        f"Symbol corr |ρ|:   {corr.symbol_correlation:.6f}\n"
        f"EVM of diff:       {corr.evm_of_difference:.3f}%\n"
        f"Per-sym corr min:  {np.min(corr.per_symbol_corr):.4f}\n"
        f"Per-sym corr max:  {np.max(corr.per_symbol_corr):.4f}"
    )
    ax.text(0.1, 0.5, txt, transform=ax.transAxes, fontsize=11,
            verticalalignment="center", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_title("Correlation Summary")

    plt.tight_layout()
    fig_path = out / "correlation_plots.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Correlation plots saved to {fig_path}")
    return fig_path


# ---------------------------------------------------------------------------
# Original comparison (now includes correlation)
# ---------------------------------------------------------------------------
def compare_results(py_result: DemodResult, vsa_result: VSAResult,
                    cfg: NR5GConfig, output_dir: str = "results"):
    """Generate comparison report, plots, and correlation analysis."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    reference_label = "VSA (SIMULATED)" if vsa_result.is_simulated else "VSA 89600"
    peak_text = f"{py_result.evm_peak:.3f}" if py_result.evm_peak is not None else "N/A"

    # ---- Summary table ----
    lines = [
        "=" * 60,
        "  5G NR Demodulation Comparison",
        "=" * 60,
        f"  Reference: {reference_label}",
        ("  SIMULATED: same Python demodulator with synthetic metric offsets.\n"
         "  Not an independent VSA measurement or validation."
         if vsa_result.is_simulated else "  Reference values supplied by external VSA."),
        f"  Config: μ={cfg.mu}, BW={cfg.bw_mhz}MHz, {cfg.n_rb}RB, {cfg.modulation}",
        f"  Slots: {cfg.n_slots}, SNR: {cfg.snr_db}dB",
        "-" * 60,
        f"  {'Metric':<30s} {'Python':>12s} {reference_label:>12s}",
        "-" * 60,
        f"  {'EVM RMS (%)':<30s} {py_result.evm_rms:>12.3f} {vsa_result.evm_rms:>12.3f}",
        f"  {'EVM Peak (%)':<30s} {peak_text:>12s} {vsa_result.evm_peak:>12.3f}",
        f"  {'Freq Error (Hz)':<30s} {'N/A':>12s} {vsa_result.freq_error_hz:>12.1f}",
    ]
    if py_result.ber is not None:
        lines.append(f"  {'BER':<30s} {py_result.ber:>12.2e} {'N/A':>12s}")
    lines += [
        f"  {'# Data symbols':<30s} {len(py_result.rx_symbols):>12d} {len(vsa_result.symbols_i):>12d}",
        "-" * 60,
        f"  EVM difference: {abs(py_result.evm_rms - vsa_result.evm_rms):.3f}%",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print(report)
    (out / "comparison_report.txt").write_text(report, encoding="utf-8")

    # ---- Comparison plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"5G NR Demod Comparison ({reference_label}) — μ={cfg.mu}, {cfg.modulation}, SNR={cfg.snr_db}dB",
                 fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.scatter(py_result.rx_symbols.real, py_result.rx_symbols.imag,
               s=1, alpha=0.3, c="steelblue")
    if py_result.constellation_ref is not None:
        ax.scatter(py_result.constellation_ref.real, py_result.constellation_ref.imag,
                   s=40, c="red", marker="+", linewidths=1.5, label="Ideal")
    ax.set_title("Python Demod — Constellation")
    ax.set_xlabel("I"); ax.set_ylabel("Q")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[0, 1]
    if len(vsa_result.symbols_i) > 0:
        ax.scatter(vsa_result.symbols_i, vsa_result.symbols_q,
                   s=1, alpha=0.3, c="darkorange")
        if py_result.constellation_ref is not None:
            ax.scatter(py_result.constellation_ref.real,
                       py_result.constellation_ref.imag,
                       s=40, c="red", marker="+", linewidths=1.5, label="Ideal")
    ax.set_title(f"{reference_label} — Constellation")
    ax.set_xlabel("I"); ax.set_ylabel("Q")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 0]
    n_sym = len(py_result.evm_per_symbol)
    ax.bar(np.arange(n_sym) - 0.15, py_result.evm_per_symbol, width=0.3,
           label="Python", color="steelblue", alpha=0.8)
    if len(vsa_result.evm_per_symbol) >= n_sym:
        ax.bar(np.arange(n_sym) + 0.15, vsa_result.evm_per_symbol[:n_sym],
             width=0.3, label=reference_label, color="darkorange", alpha=0.8)
    ax.set_title("EVM vs OFDM Symbol")
    ax.set_xlabel("OFDM Symbol Index"); ax.set_ylabel("EVM (%)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    labels = ["EVM RMS (%)", "EVM Peak (%)"]
    py_vals = [py_result.evm_rms, py_result.evm_peak if py_result.evm_peak is not None else np.nan]
    vsa_vals = [vsa_result.evm_rms, vsa_result.evm_peak]
    x = np.arange(len(labels))
    ax.bar(x - 0.15, py_vals, 0.3, label="Python", color="steelblue")
    ax.bar(x + 0.15, vsa_vals, 0.3, label=reference_label, color="darkorange")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("EVM Summary Comparison")
    ax.set_ylabel("EVM (%)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out / "comparison_plots.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nPlots saved to {fig_path}")

    # ---- Correlation analysis ----
    corr = correlate_results(py_result, vsa_result, cfg)
    print(f"\n{'— Correlation Analysis —':^60s}")
    print(f"  Symbol offset:      {corr.sample_offset}")
    print(f"  Phase offset:       {np.degrees(corr.phase_offset_rad):.2f}°")
    print(f"  Amplitude ratio:    {corr.amplitude_ratio:.4f}")
    print(f"  XCorr peak:         {corr.correlation_peak:.6f}")
    print(f"  Symbol correlation: {corr.symbol_correlation:.6f}")
    print(f"  EVM of difference:  {corr.evm_of_difference:.3f}%")
    plot_correlation(corr, cfg, output_dir)
    (out / "correlation_report.txt").write_text(
        f"reference={reference_label}\n"
        f"offset={corr.sample_offset}\n"
        f"phase_deg={np.degrees(corr.phase_offset_rad):.4f}\n"
        f"amp_ratio={corr.amplitude_ratio:.6f}\n"
        f"xcorr_peak={corr.correlation_peak:.6f}\n"
        f"symbol_corr={corr.symbol_correlation:.6f}\n"
        f"evm_diff={corr.evm_of_difference:.4f}\n",
        encoding="utf-8",
    )

    return report


def run_full_comparison(use_real_vsa: bool = False, snr_db: float = 30.0,
                        modulation: str = "64QAM", mu: int = 1,
                        output_dir: str = "results"):
    """End-to-end: generate → demod → VSA → compare + correlate."""
    cfg = NR5GConfig(
        mu=mu, bw_mhz=20, n_rb=51, modulation=modulation,
        n_slots=2, snr_db=snr_db, seed=42,
    )

    print("Generating 5G NR waveform...")
    tx = generate_nr5g_waveform(cfg)
    iq_file = str(Path(output_dir) / "nr5g_waveform.bin")
    Path(output_dir).mkdir(exist_ok=True)
    save_waveform_iq(iq_file, tx["time_signal"])
    print(f"  Saved {len(tx['time_signal'])} samples → {iq_file}")

    print("Running Python demodulator...")
    py_result = demodulate_nr5g(
        tx["time_signal"], cfg,
        tx_bits=tx["tx_bits"],
        tx_symbols=tx["tx_symbols"],
        data_positions=tx["data_positions"],
    )
    print(f"  EVM RMS = {py_result.evm_rms:.3f}%, BER = {py_result.ber:.2e}")

    if use_real_vsa:
        from vsa_89600 import run_vsa_demod
        print("Running VSA 89600 demodulation...")
        vsa_result = run_vsa_demod(
            iq_file, cfg.sample_rate_mhz * 1e6,
            cfg_mu=cfg.mu, cfg_bw_mhz=cfg.bw_mhz,
            cfg_n_rb=cfg.n_rb, cfg_modulation=cfg.modulation,
            cfg_cell_id=cfg.cell_id,
        )
    else:
        print("Simulating VSA 89600 results (offline mode)...")
        vsa_result = simulate_vsa_result(tx["time_signal"], cfg)
    reference_label = "VSA SIMULATED" if vsa_result.is_simulated else "VSA"
    print(f"  [{reference_label}] EVM RMS = {vsa_result.evm_rms:.3f}%")

    report = compare_results(py_result, vsa_result, cfg, output_dir)
    return py_result, vsa_result, report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare Python demodulation with VSA 89600")
    parser.add_argument("--real-vsa", action="store_true", help="Require a real VSA measurement")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    run_full_comparison(use_real_vsa=args.real_vsa, output_dir=args.output_dir)
