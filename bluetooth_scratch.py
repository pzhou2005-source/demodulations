# %% [markdown]
# # Bluetooth LE and BR/EDR EDR Scratch Pad
# Explore uncoded LE 1M/2M GFSK and payload-only EDR2M/EDR3M differential PSK.
# These aligned baseband models are not complete Bluetooth packets. The EDR
# cells can use the 89600 Digital Demod option to correlate symbol-state output.
# The LE receiver uses a frequency discriminator; QAM-style EVM does not apply.

# %% 1. Configure the LE PHY
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

from wireless_phy import BluetoothConfig, generate_bluetooth_waveform
from wireless_phy import demodulate_bluetooth, plot_wireless_result

bluetooth_cfg = BluetoothConfig(
    phy="LE2M",
    n_bits=128,
    samples_per_symbol=8,
    snr_db=20.0,
    seed=42,
)

print(f"PHY: {bluetooth_cfg.phy}")
print(f"Symbol rate: {bluetooth_cfg.symbol_rate_hz / 1e6:.1f} Msymbol/s")
print(f"Sample rate: {bluetooth_cfg.sample_rate_hz / 1e6:.1f} Msps")
print(f"Frequency deviation: +/-{bluetooth_cfg.frequency_deviation_hz / 1e3:.1f} kHz")
print(f"Samples per bit: {bluetooth_cfg.samples_per_symbol}")

# %% 2. Generate the Gaussian-filtered continuous-phase waveform
bluetooth_waveform = generate_bluetooth_waveform(bluetooth_cfg)
print(f"Transmit bits: {bluetooth_waveform['tx_bits'].size}")
print(f"IQ samples: {bluetooth_waveform['time_signal'].size}")
print(f"Clean envelope range: {np.abs(bluetooth_waveform['time_signal_clean']).min():.6f} .. "
      f"{np.abs(bluetooth_waveform['time_signal_clean']).max():.6f}")
print("First 32 bits:", bluetooth_waveform["tx_bits"][:32].tolist())

# %% 3. Discriminate frequency and integrate one decision per bit
bluetooth_result = demodulate_bluetooth(
    bluetooth_waveform["time_signal"],
    bluetooth_cfg,
    tx_bits=bluetooth_waveform["tx_bits"],
)

frequency_hz = bluetooth_result.diagnostics["frequency_hz"]
decision_metric = bluetooth_result.diagnostics["decision_metric"]
print(f"BER: {bluetooth_result.ber:.3e}")
print(f"QAM-style EVM: {bluetooth_result.evm_rms}")
print("First 16 frequency decisions (kHz):",
      np.round(frequency_hz[:16] / 1e3, 1).tolist())
print("First 16 normalized bit metrics:", np.round(decision_metric[:16], 3).tolist())

# %% 4. Check constant gain and phase invariance
gain_phase_signal = bluetooth_waveform["time_signal"] * 1.5 * np.exp(1.2j)
gain_phase_result = demodulate_bluetooth(
    gain_phase_signal,
    bluetooth_cfg,
    tx_bits=bluetooth_waveform["tx_bits"],
)
print(f"BER after constant gain and phase rotation: {gain_phase_result.ber:.3e}")
print("The discriminator uses adjacent-sample phase change, so constant phase cancels.")

# %% 5. Plot IQ, spectrum, discriminator output, and bit metrics
bluetooth_figure = plot_wireless_result(bluetooth_waveform, bluetooth_result)
plt.show()

# %% 6. Generate and locally demodulate BR/EDR EDR2M and EDR3M payloads
# This model begins with a known phase-reference symbol; it omits packet framing,
# access code, header, guard, whitening, CRC, FEC, and the preceding GFSK section.
from wireless_phy import BluetoothEDRConfig, generate_bluetooth_edr_waveform
from wireless_phy import demodulate_bluetooth_edr

edr_runs = {}
for edr_phy in ("EDR2M", "EDR3M"):
    edr_cfg = BluetoothEDRConfig(phy=edr_phy, n_symbols=512, snr_db=None, seed=17)
    edr_waveform = generate_bluetooth_edr_waveform(edr_cfg)
    edr_result = demodulate_bluetooth_edr(
        edr_waveform["time_signal"], edr_cfg,
        tx_bits=edr_waveform["tx_bits"], tx_symbols=edr_waveform["tx_symbols"],
    )
    edr_runs[edr_phy] = (edr_cfg, edr_waveform, edr_result)
    print(f"{edr_phy}: local BER={edr_result.ber:.3e}, "
          f"differential-symbol EVM={edr_result.evm_rms:.3f}%")

# %% 7. Correlate EDR symbol states with the real Keysight 89600 VSA
# Requires the VSA Digital Demod option. The VSA reports a GapData warning for
# these payload-only recordings; the warning is printed alongside correlation.
from compare import correlate_symbol_streams
from wireless_phy import bluetooth_edr_bits_from_state_indices
from wireless_phy import bluetooth_edr_symbols_from_state_indices
from vsa_89600 import run_vsa_edr_demod

for edr_phy, (edr_cfg, edr_waveform, edr_result) in edr_runs.items():
    vsa_edr_result = run_vsa_edr_demod(
        edr_waveform["time_signal"], edr_cfg, center_freq_hz=2.44e9, visible=True)
    vsa_symbols = bluetooth_edr_symbols_from_state_indices(
        vsa_edr_result.symbol_indices, edr_cfg)
    offset, correlation = correlate_symbol_streams(
        edr_result.rx_symbols, vsa_symbols.real, vsa_symbols.imag, max_lag=None)
    python_start = max(0, -offset)
    vsa_start = max(0, offset)
    matched = min(len(edr_result.rx_symbols) - python_start,
                  len(vsa_symbols) - vsa_start)
    python_states = edr_result.diagnostics["symbol_indices"][python_start:python_start + matched]
    vsa_states = vsa_edr_result.symbol_indices[vsa_start:vsa_start + matched]
    state_ber = np.mean(python_states != vsa_states)
    bits_per_symbol = edr_cfg.bits_per_symbol
    python_bits = edr_result.rx_bits[python_start * bits_per_symbol:
                                     (python_start + matched) * bits_per_symbol]
    vsa_bits = bluetooth_edr_bits_from_state_indices(vsa_states, edr_cfg)
    bit_ber = np.mean(python_bits != vsa_bits)
    print(f"{edr_phy} VSA status: {vsa_edr_result.measurement_status}")
    print(f"  matched symbols={matched}, offset={offset}, correlation={correlation:.9f}")
    print(f"  state BER={state_ber:.3e}, payload BER={bit_ber:.3e}")