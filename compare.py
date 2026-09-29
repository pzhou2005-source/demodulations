"""Comparison between Python demodulator and VSA 89600 results.

Produces side-by-side metrics and plots:
  - EVM RMS / Peak comparison
  - Constellation overlay
  - EVM vs OFDM symbol
  - EVM vs subcarrier
  - BER (Python demod only)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from nr5g_waveform import NR5GConfig, generate_nr5g_waveform, save_waveform_iq
from nr5g_demod import demodulate_nr5g, DemodResult
from vsa_89600 import VSAResult, simulate_vsa_result


def compare_results(py_result: DemodResult, vsa_result: VSAResult,
                    cfg: NR5GConfig, output_dir: str = "results"):
    """Generate comparison report and plots."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    # ---- Summary table ----
    lines = [
        "=" * 60,
        "  5G NR Demodulation Comparison",
        "=" * 60,
        f"  Config: μ={cfg.mu}, BW={cfg.bw_mhz}MHz, {cfg.n_rb}RB, {cfg.modulation}",
        f"  Slots: {cfg.n_slots}, SNR: {cfg.snr_db}dB",
        "-" * 60,
        f"  {'Metric':<30s} {'Python':>12s} {'VSA 89600':>12s}",
        "-" * 60,
        f"  {'EVM RMS (%)':<30s} {py_result.evm_rms:>12.3f} {vsa_result.evm_rms:>12.3f}",
        f"  {'EVM Peak (%)':<30s} {np.max(py_result.evm_per_symbol):>12.3f} {vsa_result.evm_peak:>12.3f}",
        f"  {'Freq Error (Hz)':<30s} {'N/A':>12s} {vsa_result.freq_error_hz:>12.1f}",
        f"  {'BER':<30s} {py_result.ber:>12.2e} {'N/A':>12s}" if py_result.ber is not None else "",
        f"  {'# Data symbols':<30s} {len(py_result.rx_symbols):>12d} {len(vsa_result.symbols_i):>12d}",
        "-" * 60,
        f"  EVM difference: {abs(py_result.evm_rms - vsa_result.evm_rms):.3f}%",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print(report)
    (out / "comparison_report.txt").write_text(report)

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"5G NR Demod Comparison — μ={cfg.mu}, {cfg.modulation}, SNR={cfg.snr_db}dB",
                 fontsize=13, fontweight="bold")

    # 1) Constellation: Python
    ax = axes[0, 0]
    ax.scatter(py_result.rx_symbols.real, py_result.rx_symbols.imag,
               s=1, alpha=0.3, c="steelblue")
    if py_result.constellation_ref is not None:
        ax.scatter(py_result.constellation_ref.real, py_result.constellation_ref.imag,
                   s=40, c="red", marker="+", linewidths=1.5, label="Ideal")
    ax.set_title("Python Demod — Constellation")
    ax.set_xlabel("I"); ax.set_ylabel("Q")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 2) Constellation: VSA
    ax = axes[0, 1]
    if len(vsa_result.symbols_i) > 0:
        ax.scatter(vsa_result.symbols_i, vsa_result.symbols_q,
                   s=1, alpha=0.3, c="darkorange")
        if py_result.constellation_ref is not None:
            ax.scatter(py_result.constellation_ref.real,
                       py_result.constellation_ref.imag,
                       s=40, c="red", marker="+", linewidths=1.5, label="Ideal")
    ax.set_title("VSA 89600 — Constellation")
    ax.set_xlabel("I"); ax.set_ylabel("Q")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 3) EVM vs OFDM symbol
    ax = axes[1, 0]
    n_sym = len(py_result.evm_per_symbol)
    ax.bar(np.arange(n_sym) - 0.15, py_result.evm_per_symbol, width=0.3,
           label="Python", color="steelblue", alpha=0.8)
    if len(vsa_result.evm_per_symbol) >= n_sym:
        ax.bar(np.arange(n_sym) + 0.15, vsa_result.evm_per_symbol[:n_sym],
               width=0.3, label="VSA", color="darkorange", alpha=0.8)
    ax.set_title("EVM vs OFDM Symbol")
    ax.set_xlabel("OFDM Symbol Index"); ax.set_ylabel("EVM (%)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4) EVM summary bar chart
    ax = axes[1, 1]
    labels = ["EVM RMS (%)", "EVM Peak (%)"]
    py_vals = [py_result.evm_rms, np.max(py_result.evm_per_symbol)]
    vsa_vals = [vsa_result.evm_rms, vsa_result.evm_peak]
    x = np.arange(len(labels))
    ax.bar(x - 0.15, py_vals, 0.3, label="Python", color="steelblue")
    ax.bar(x + 0.15, vsa_vals, 0.3, label="VSA 89600", color="darkorange")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("EVM Summary Comparison")
    ax.set_ylabel("EVM (%)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out / "comparison_plots.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nPlots saved to {fig_path}")

    return report


def run_full_comparison(use_real_vsa: bool = False, snr_db: float = 30.0,
                        modulation: str = "64QAM", mu: int = 1,
                        output_dir: str = "results"):
    """End-to-end: generate → demod → VSA → compare."""
    cfg = NR5GConfig(
        mu=mu, bw_mhz=20, n_rb=51, modulation=modulation,
        n_slots=2, snr_db=snr_db, seed=42,
    )

    # 1) Generate waveform
    print("Generating 5G NR waveform...")
    tx = generate_nr5g_waveform(cfg)
    iq_file = str(Path(output_dir) / "nr5g_waveform.bin")
    Path(output_dir).mkdir(exist_ok=True)
    save_waveform_iq(iq_file, tx["time_signal"])
    print(f"  Saved {len(tx['time_signal'])} samples → {iq_file}")

    # 2) Python demodulation
    print("Running Python demodulator...")
    py_result = demodulate_nr5g(
        tx["time_signal"], cfg,
        tx_bits=tx["tx_bits"],
        tx_symbols=tx["tx_symbols"],
        data_positions=tx["data_positions"],
    )
    print(f"  EVM RMS = {py_result.evm_rms:.3f}%, BER = {py_result.ber:.2e}")

    # 3) VSA 89600
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
    print(f"  [VSA] EVM RMS = {vsa_result.evm_rms:.3f}%")

    # 4) Compare
    report = compare_results(py_result, vsa_result, cfg, output_dir)
    return py_result, vsa_result, report


if __name__ == "__main__":
    run_full_comparison(use_real_vsa=False, snr_db=30, modulation="64QAM")
