# %% [markdown]
# # 5G NR Demodulation — Interactive Scratch Pad
# Run each cell with `Ctrl+Enter` (or `Shift+Enter` to advance).
# Every step shows intermediate results inline.

# %% 1. Configuration
import numpy as np
import matplotlib.pyplot as plt
from nr5g_waveform import NR5GConfig, NUMEROLOGY, QAM_MAP

cfg = NR5GConfig(
    mu=1,               # 30 kHz SCS
    bw_mhz=20.0,        # 20 MHz bandwidth
    n_rb=51,             # 51 resource blocks
    modulation="64QAM",
    n_slots=2,
    snr_db=30,
    cfo_hz=150.0,        # 150 Hz carrier frequency offset
    channel_taps=[1.0, 0.3+0.1j, 0.05],  # 3-tap multipath
    channel_delays=[0, 3, 7],
    seed=42,
)

print(f"Numerology μ={cfg.mu}: SCS={cfg.scs_khz} kHz, {cfg.symbols_per_slot} sym/slot")
print(f"Subcarriers: {cfg.n_sc}  ({cfg.n_rb} RB × 12)")
print(f"FFT size: {cfg.n_fft}, Sample rate: {cfg.sample_rate_mhz} MHz")
print(f"Modulation: {cfg.modulation} ({cfg.bits_per_symbol} bits/sym)")
print(f"CFO: {cfg.cfo_hz} Hz, Channel: {len(cfg.channel_taps)}-tap multipath")

# %% 2. Generate 5G NR waveform
from nr5g_waveform import generate_nr5g_waveform

tx = generate_nr5g_waveform(cfg)

print(f"TX bits:      {len(tx['tx_bits'])}")
print(f"TX symbols:   {len(tx['tx_symbols'])}")
print(f"IQ samples:   {len(tx['time_signal'])}")
print(f"Data positions: {len(tx['data_positions'])} OFDM symbols carry data")

# %% 3. Inspect the resource grid
resource_grid = tx["resource_grid"]
print(f"Resource grid shape: {resource_grid.shape}  (symbols × subcarriers)")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].imshow(np.abs(resource_grid), aspect="auto", interpolation="none")
axes[0].set_title("Resource Grid — Magnitude")
axes[0].set_xlabel("Subcarrier"); axes[0].set_ylabel("OFDM Symbol")
axes[1].imshow(np.angle(resource_grid), aspect="auto", interpolation="none", cmap="twilight")
axes[1].set_title("Resource Grid — Phase")
axes[1].set_xlabel("Subcarrier"); axes[1].set_ylabel("OFDM Symbol")
plt.tight_layout(); plt.show()

# %% 4. TX constellation (before OFDM)
from nr5g_waveform import _qam_constellation

ref = _qam_constellation(cfg.bits_per_symbol)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(tx["tx_symbols"].real, tx["tx_symbols"].imag, s=2, alpha=0.3, label="TX data")
ax.scatter(ref.real, ref.imag, s=60, c="red", marker="+", linewidths=2, label="Ideal")
ax.set_title(f"TX Constellation — {cfg.modulation}")
ax.set_xlabel("I"); ax.set_ylabel("Q")
ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend()
plt.show()

# %% 5. Time-domain waveform & spectrum
time_signal = tx["time_signal"]

fig, axes = plt.subplots(2, 1, figsize=(14, 6))

t_us = np.arange(len(time_signal)) / (cfg.sample_rate_mhz)  # µs
axes[0].plot(t_us[:2000], time_signal[:2000].real, linewidth=0.5)
axes[0].set_title("Time-domain I (first 2000 samples)")
axes[0].set_xlabel("Time (µs)"); axes[0].set_ylabel("Amplitude")

freq = np.fft.fftshift(np.fft.fft(time_signal[:cfg.n_fft]))
f_mhz = np.linspace(-cfg.sample_rate_mhz / 2, cfg.sample_rate_mhz / 2, cfg.n_fft)
axes[1].plot(f_mhz, 20 * np.log10(np.abs(freq) + 1e-12), linewidth=0.5)
axes[1].set_title("Spectrum (single OFDM symbol)")
axes[1].set_xlabel("Frequency (MHz)"); axes[1].set_ylabel("Power (dB)")
axes[1].set_ylim(bottom=-40)

plt.tight_layout(); plt.show()

# %% 6. OFDM demodulation
from nr5g_waveform import ofdm_demodulate, _cp_lengths

cp = _cp_lengths(cfg.n_fft, cfg.mu) * cfg.n_slots
n_sym_total = cfg.symbols_per_slot * cfg.n_slots
rx_grid = ofdm_demodulate(time_signal, cfg.n_fft, cfg.n_sc, cp, n_sym_total)

print(f"RX grid shape: {rx_grid.shape}")

fig, ax = plt.subplots(figsize=(14, 4))
ax.imshow(np.abs(rx_grid), aspect="auto", interpolation="none")
ax.set_title("Received Resource Grid — Magnitude")
ax.set_xlabel("Subcarrier"); ax.set_ylabel("OFDM Symbol")
plt.tight_layout(); plt.show()

# %% 7. Channel estimation from DMRS
from nr5g_demod import channel_estimate_dmrs

h_est = channel_estimate_dmrs(rx_grid, cfg)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(np.abs(h_est[0, :]), linewidth=0.8)
axes[0].set_title("Channel Magnitude (1st symbol)")
axes[0].set_xlabel("Subcarrier"); axes[0].set_ylabel("|H|")
axes[1].plot(np.angle(h_est[0, :]), linewidth=0.8)
axes[1].set_title("Channel Phase (1st symbol)")
axes[1].set_xlabel("Subcarrier"); axes[1].set_ylabel("∠H (rad)")
plt.tight_layout(); plt.show()

print(f"Channel est mean |H| = {np.mean(np.abs(h_est)):.4f}")

# %% 8. Equalisation
from nr5g_demod import equalise_zf

eq_grid = equalise_zf(rx_grid, h_est)

fig, ax = plt.subplots(figsize=(14, 4))
err = np.abs(eq_grid - resource_grid)
ax.imshow(err, aspect="auto", interpolation="none", cmap="hot")
ax.set_title("Equalisation Error (|eq - tx| per RE)")
ax.set_xlabel("Subcarrier"); ax.set_ylabel("OFDM Symbol")
plt.colorbar(ax.images[0], ax=ax); plt.tight_layout(); plt.show()

# %% 9. Full demodulation pipeline (MMSE + CFO correction)
from nr5g_demod import demodulate_nr5g

result = demodulate_nr5g(
    time_signal, cfg,
    tx_bits=tx["tx_bits"],
    tx_symbols=tx["tx_symbols"],
    tx_grid=tx["resource_grid"],
    data_positions=tx["data_positions"],
    equaliser="mmse",
    cfo_correct=True,
)

print(f"CFO estimate:  {result.cfo_est_hz:.1f} Hz  (true: {cfg.cfo_hz} Hz)")
print(f"Noise var est: {result.noise_var_est:.2e}")
print(f"EVM RMS:       {result.evm_rms:.3f}%")
print(f"BER:           {result.ber:.2e}")
print(f"RX symbols:    {len(result.rx_symbols)}")

# %% 10. RX constellation after equalisation
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(result.rx_symbols.real, result.rx_symbols.imag,
           s=2, alpha=0.3, c="steelblue", label="RX equalised")
ax.scatter(ref.real, ref.imag, s=60, c="red", marker="+", linewidths=2, label="Ideal")
ax.set_title(f"RX Constellation — {cfg.modulation}, SNR={cfg.snr_db}dB\n"
             f"EVM={result.evm_rms:.2f}%, BER={result.ber:.1e}")
ax.set_xlabel("I"); ax.set_ylabel("Q")
ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend()
plt.show()

# %% 11. EVM per OFDM symbol
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(result.evm_per_symbol)), result.evm_per_symbol, color="steelblue")
ax.axhline(result.evm_rms, color="red", linestyle="--", label=f"RMS = {result.evm_rms:.2f}%")
ax.set_title("EVM per OFDM Symbol")
ax.set_xlabel("OFDM Symbol Index"); ax.set_ylabel("EVM (%)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# %% 11b. EVM per subcarrier
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(result.evm_per_subcarrier, linewidth=0.8, color="steelblue")
ax.axhline(result.evm_rms, color="red", linestyle="--", label=f"RMS = {result.evm_rms:.2f}%")
ax.set_title("EVM per Subcarrier")
ax.set_xlabel("Subcarrier Index"); ax.set_ylabel("EVM (%)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# %% 11c. ZF vs MMSE comparison
result_zf = demodulate_nr5g(
    time_signal, cfg,
    tx_bits=tx["tx_bits"], tx_symbols=tx["tx_symbols"],
    tx_grid=tx["resource_grid"], data_positions=tx["data_positions"],
    equaliser="zf", cfo_correct=True,
)
print(f"ZF:   EVM={result_zf.evm_rms:.3f}%, BER={result_zf.ber:.2e}")
print(f"MMSE: EVM={result.evm_rms:.3f}%, BER={result.ber:.2e}")
print(f"MMSE advantage: {result_zf.evm_rms - result.evm_rms:.3f}% EVM reduction")

# %% 12. VSA 89600 simulation & comparison
from vsa_89600 import simulate_vsa_result
from compare import compare_results, correlate_results, plot_correlation

vsa_result = simulate_vsa_result(time_signal, cfg)
print(f"[VSA sim] EVM RMS: {vsa_result.evm_rms:.3f}%, Peak: {vsa_result.evm_peak:.3f}%")

# %% 13. Correlate Python vs VSA 89600
corr = correlate_results(result, vsa_result, cfg)

print(f"Symbol offset:      {corr.sample_offset}")
print(f"Phase offset:       {np.degrees(corr.phase_offset_rad):.2f}°")
print(f"Amplitude ratio:    {corr.amplitude_ratio:.4f}")
print(f"XCorr peak:         {corr.correlation_peak:.6f}")
print(f"Symbol correlation: {corr.symbol_correlation:.6f}")
print(f"EVM of difference:  {corr.evm_of_difference:.3f}%")

# %% 14. Correlation plots — constellation overlay & error scatter
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.scatter(corr.aligned_py_symbols.real, corr.aligned_py_symbols.imag,
           s=1, alpha=0.3, c="steelblue", label="Python")
ax.scatter(corr.aligned_vsa_symbols.real, corr.aligned_vsa_symbols.imag,
           s=1, alpha=0.3, c="darkorange", label="VSA (aligned)")
ax.set_title(f"Constellation Overlay — ρ={corr.symbol_correlation:.4f}")
ax.set_xlabel("I"); ax.set_ylabel("Q")
ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

ax = axes[1]
err = corr.aligned_vsa_symbols - corr.aligned_py_symbols
ax.scatter(err.real, err.imag, s=1, alpha=0.3, c="red")
ax.set_title(f"Symbol Error — EVM={corr.evm_of_difference:.3f}%")
ax.set_xlabel("ΔI"); ax.set_ylabel("ΔQ")
ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

ax = axes[2]
n_show = min(200, len(corr.aligned_py_symbols))
ax.plot(np.abs(corr.aligned_py_symbols[:n_show]), label="Python", alpha=0.7)
ax.plot(np.abs(corr.aligned_vsa_symbols[:n_show]), label="VSA", alpha=0.7)
ax.set_title("Symbol Magnitude (first 200)")
ax.set_xlabel("Symbol Index"); ax.set_ylabel("|symbol|")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# %% 15. Per-OFDM-symbol and per-subcarrier correlation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.bar(range(len(corr.per_symbol_corr)), corr.per_symbol_corr, color="steelblue")
ax.axhline(corr.symbol_correlation, color="red", linestyle="--",
           label=f"Overall ρ={corr.symbol_correlation:.4f}")
ax.set_title("Correlation per OFDM Symbol")
ax.set_xlabel("OFDM Symbol Index"); ax.set_ylabel("|ρ|")
ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(corr.per_subcarrier_corr, linewidth=0.8, color="steelblue")
ax.set_title("Correlation per Subcarrier")
ax.set_xlabel("Subcarrier Index"); ax.set_ylabel("Normalised correlation")
ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# %% 16. Side-by-side comparison (saves to file)
report = compare_results(result, vsa_result, cfg, output_dir="results")

# %% 17. Sweep SNR — ZF vs MMSE
snr_range = [10, 15, 20, 25, 30, 40]
evm_zf_list, evm_mmse_list, evm_vsa_list = [], [], []

for snr in snr_range:
    c = NR5GConfig(mu=1, bw_mhz=20, n_rb=51, modulation="64QAM",
                   n_slots=2, snr_db=snr, cfo_hz=150, seed=42,
                   channel_taps=[1.0, 0.3+0.1j, 0.05], channel_delays=[0, 3, 7])
    t = generate_nr5g_waveform(c)
    r_zf = demodulate_nr5g(t["time_signal"], c,
                           tx_bits=t["tx_bits"], tx_symbols=t["tx_symbols"],
                           tx_grid=t["resource_grid"], data_positions=t["data_positions"],
                           equaliser="zf")
    r_mmse = demodulate_nr5g(t["time_signal"], c,
                             tx_bits=t["tx_bits"], tx_symbols=t["tx_symbols"],
                             tx_grid=t["resource_grid"], data_positions=t["data_positions"],
                             equaliser="mmse")
    v = simulate_vsa_result(t["time_signal"], c)
    evm_zf_list.append(r_zf.evm_rms)
    evm_mmse_list.append(r_mmse.evm_rms)
    evm_vsa_list.append(v.evm_rms)
    print(f"SNR={snr:3d} dB → ZF={r_zf.evm_rms:.3f}%, MMSE={r_mmse.evm_rms:.3f}%, VSA={v.evm_rms:.3f}%")

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(snr_range, evm_zf_list, "^--", label="Python ZF")
ax.semilogy(snr_range, evm_mmse_list, "o-", label="Python MMSE")
ax.semilogy(snr_range, evm_vsa_list, "s:", label="VSA 89600 (sim)")
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("EVM RMS (%)")
ax.set_title("EVM vs SNR — ZF vs MMSE vs VSA"); ax.legend(); ax.grid(True, which="both", alpha=0.3)
plt.tight_layout(); plt.show()

# %% 18. Sweep modulation order
for mod in ["QPSK", "16QAM", "64QAM", "256QAM"]:
    c = NR5GConfig(mu=1, bw_mhz=20, n_rb=51, modulation=mod,
                   n_slots=2, snr_db=30, cfo_hz=150, seed=42,
                   channel_taps=[1.0, 0.3+0.1j, 0.05], channel_delays=[0, 3, 7])
    t = generate_nr5g_waveform(c)
    r = demodulate_nr5g(t["time_signal"], c,
                        tx_bits=t["tx_bits"], tx_symbols=t["tx_symbols"],
                        tx_grid=t["resource_grid"], data_positions=t["data_positions"],
                        equaliser="mmse")
    print(f"{mod:>6s}: EVM={r.evm_rms:.3f}%, BER={r.ber:.1e}, CFO_est={r.cfo_est_hz:.1f}Hz")
