# %% [markdown]
# # UWB BPM-BPSK Scratch Pad
# Explore burst-position modulation and pulse polarity with a matched-filter
# receiver. This is an aligned teaching model, not an IEEE 802.15.4z packet,
# secure timestamp sequence, ranging, or conformance implementation.
# Each payload symbol uses two bits: early/late burst position and polarity.

# %% 1. Configuration and bit mapping
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

from wireless_phy import UWBConfig, generate_uwb_waveform, demodulate_uwb
from wireless_phy import plot_wireless_result

uwb_cfg = UWBConfig(
    n_symbols=8,
    chips_per_symbol=64,
    chips_per_burst=8,
    samples_per_chip=4,
    chip_rate_hz=499.2e6,
    snr_db=20.0,
    seed=42,
)
tx_bits = np.array([0, 0, 0, 1, 1, 0, 1, 1,
                    0, 0, 1, 1, 0, 1, 1, 0], dtype=np.uint8)

print(f"Chip rate: {uwb_cfg.chip_rate_hz / 1e6:.1f} MHz")
print(f"Sample rate: {uwb_cfg.sample_rate_hz / 1e9:.4f} Gsps")
print(f"Samples per burst symbol: {uwb_cfg.samples_per_symbol}")
print("Bit pair: first bit selects early(0)/late(1); second selects + (0)/- (1) polarity.")
print("Payload pairs:", tx_bits.reshape(-1, 2).tolist())

# %% 2. Generate the training burst and BPM-BPSK payload
uwb_waveform = generate_uwb_waveform(uwb_cfg, tx_bits=tx_bits)
blocks = uwb_waveform["time_signal_clean"].reshape(
    uwb_cfg.n_symbols + 1, uwb_cfg.samples_per_symbol)

print(f"Burst template samples: {uwb_waveform['burst_template'].size}")
print(f"Training plus payload blocks: {blocks.shape}")
print(f"Transmit bits: {uwb_waveform['tx_bits'].size}")

# %% 3. Inspect training and payload bursts
shown = min(4, blocks.shape[0])
time_us = np.arange(uwb_cfg.samples_per_symbol) / uwb_cfg.sample_rate_hz * 1e6
fig, ax = plt.subplots(figsize=(12, 4))
for block_index in range(shown):
    label = "training" if block_index == 0 else f"payload {block_index - 1}"
    ax.plot(time_us, blocks[block_index].real, label=label)
ax.set(title="UWB Pulse Blocks (I component)", xlabel="Time within block (us)",
       ylabel="Amplitude")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% 4. Matched-filter demodulation
uwb_result = demodulate_uwb(
    uwb_waveform["time_signal"],
    uwb_cfg,
    tx_bits=uwb_waveform["tx_bits"],
    tx_symbols=uwb_waveform["tx_symbols"],
)

print(f"BER: {uwb_result.ber:.3e}")
print(f"Matched-filter RMS EVM: {uwb_result.evm_rms:.3f}%")
print(f"Training gain estimate: {uwb_result.diagnostics['channel_est']:.4f}")
print("EVM is defined on the early/late matched-filter amplitudes, not on a QAM constellation.")

# %% 5. Compare early and late matched-filter scores
early = uwb_result.diagnostics["early"]
late = uwb_result.diagnostics["late"]
for index, (early_score, late_score) in enumerate(zip(early, late)):
    print(f"symbol {index}: early={early_score.real:+.3f}, late={late_score.real:+.3f}, "
          f"bits={uwb_result.rx_bits[2 * index:2 * index + 2].tolist()}")

# %% 6. Plot IQ, spectrum, matched-filter amplitudes, and soft-symbol error
uwb_figure = plot_wireless_result(uwb_waveform, uwb_result)
plt.show()