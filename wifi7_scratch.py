# %% [markdown]
# # Wi-Fi 7-Inspired OFDM Scratch Pad
# A small uncoded EHT-numerology model for exploring OFDM, pilots, QAM,
# equalization, and EVM. It is not a complete IEEE 802.11be PPDU or a
# conformance/interoperability implementation.
# Run the cells in order. The receiver assumes the first sample is aligned.

# %% 1. Configuration
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (project_root / "wireless_phy.py").is_file():
    project_root = project_root / "nr5g_demod"
if not (project_root / "wireless_phy.py").is_file():
    raise FileNotFoundError("Open the nr5g_demod project folder before running this script")
project_root = project_root.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np

from wireless_phy import WiFi7Config, generate_wifi7_waveform, demodulate_wifi7
from wireless_phy import plot_wireless_result

wifi_cfg = WiFi7Config(
    bandwidth_mhz=20,
    modulation="64QAM",
    guard_interval_us=0.8,
    n_symbols=4,
    snr_db=30.0,
    seed=42,
)

print(f"Bandwidth: {wifi_cfg.bandwidth_mhz} MHz")
print(f"Sample rate / FFT: {wifi_cfg.sample_rate_hz / 1e6:.1f} Msps / {wifi_cfg.n_fft}")
print(f"Subcarrier spacing: {wifi_cfg.sample_rate_hz / wifi_cfg.n_fft / 1e3:.3f} kHz")
print(f"Guard interval: {wifi_cfg.guard_interval_us} us; modulation: {wifi_cfg.modulation}")
print(f"Bits per data QAM symbol: {wifi_cfg.bits_per_symbol}")

# %% 2. Generate the training symbol and data waveform
wifi_waveform = generate_wifi7_waveform(wifi_cfg)

print(f"Transmit bits: {wifi_waveform['tx_bits'].size}")
print(f"Transmit QAM symbols: {wifi_waveform['tx_symbols'].size}")
print(f"Resource grid: {wifi_waveform['resource_grid'].shape} (OFDM symbols x FFT bins)")
print(f"Active tones: {wifi_waveform['active_bins'].size}")
print(f"Pilot tones per data symbol: {wifi_waveform['pilot_bins'].size}")
print(f"IQ samples: {wifi_waveform['time_signal'].size}")

# %% 3. Inspect the frequency-domain allocation
resource_grid = wifi_waveform["resource_grid"]
fig, ax = plt.subplots(figsize=(12, 4))
ax.imshow(np.abs(resource_grid), aspect="auto", origin="lower", interpolation="none")
ax.set(title="Generic OFDM Allocation: Training, Pilots, and Data",
       xlabel="FFT bin", ylabel="OFDM symbol")
plt.tight_layout()
plt.show()

# %% 4. Demodulate and measure against the known payload
wifi_result = demodulate_wifi7(
    wifi_waveform["time_signal"],
    wifi_cfg,
    tx_bits=wifi_waveform["tx_bits"],
    tx_symbols=wifi_waveform["tx_symbols"],
)

print(f"BER: {wifi_result.ber:.3e}")
print(f"RMS EVM: {wifi_result.evm_rms:.3f}%")
print(f"Recovered bits: {wifi_result.rx_bits.size}")
print("Known payload bits are used only to calculate BER, not to decode the signal.")

# %% 5. Inspect channel and pilot tracking
channel_est = wifi_result.diagnostics["channel_est"]
pilot_gain = wifi_result.diagnostics["pilot_gain"]
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(np.abs(channel_est))
axes[0].set(title="Training-Based Channel Estimate", xlabel="Active-tone index",
            ylabel="Channel magnitude")
axes[1].plot(np.abs(pilot_gain), marker="o")
axes[1].set(title="Per-Symbol Pilot Gain", xlabel="Data OFDM symbol index",
            ylabel="Gain magnitude")
for axis in axes:
    axis.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% 6. Check common gain and phase correction
gain_phase_signal = wifi_waveform["time_signal"] * 0.75 * np.exp(0.8j)
gain_phase_result = demodulate_wifi7(
    gain_phase_signal,
    wifi_cfg,
    tx_bits=wifi_waveform["tx_bits"],
    tx_symbols=wifi_waveform["tx_symbols"],
)
print(f"BER after complex gain: {gain_phase_result.ber:.3e}")
print(f"RMS EVM after complex gain: {gain_phase_result.evm_rms:.3f}%")

# %% 7. Plot waveform, spectrum, constellation, and symbol error
wifi_figure = plot_wireless_result(wifi_waveform, wifi_result)
plt.show()