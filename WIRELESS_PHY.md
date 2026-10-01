# Wireless PHY Simulations

Run cells **19-21** in `nr5g_scratch.py` with VS Code's Jupyter Interactive
Window. Each cell generates a waveform, demodulates it, prints metrics, and
plots IQ samples, spectrum, and receiver outputs. These cells can also run
independently of the NR cells.

Dependencies: `numpy`, `scipy`, and `matplotlib`; tests additionally use `pytest`.
Select the workspace `.venv` kernel in Jupyter; SciPy is installed there.

## Supported Models

| Model | Configuration | Generation / demodulation |
| --- | --- | --- |
| Wi-Fi 7-inspired OFDM | `WiFi7Config` | `generate_wifi7_waveform` / `demodulate_wifi7` |
| Bluetooth LE GFSK | `BluetoothConfig` | `generate_bluetooth_waveform` / `demodulate_bluetooth` |
| UWB BPM-BPSK pulses | `UWBConfig` | `generate_uwb_waveform` / `demodulate_uwb` |

All functions are in `wireless_phy.py`. Generators accept an optional binary
payload of exactly the configured length and return clean/noisy IQ, transmitted
bits, reference symbols, and configuration. Receivers decode without known
payloads; reference arguments only enable BER and EVM measurements.

```python
from wireless_phy import WiFi7Config, generate_wifi7_waveform, demodulate_wifi7

cfg = WiFi7Config(bandwidth_mhz=80, modulation="4096QAM", snr_db=60)
waveform = generate_wifi7_waveform(cfg)
result = demodulate_wifi7(
    waveform["time_signal"], cfg,
    tx_bits=waveform["tx_bits"], tx_symbols=waveform["tx_symbols"],
)
print(result.ber, result.evm_rms)
```

## Scope and Assumptions

- **Wi-Fi:** 20/40/80/160/320 MHz, 78.125 kHz subcarrier spacing,
  0.8/1.6/3.2 us guard intervals, QPSK through 4096-QAM. Uses a generic
  DC-null allocation, comb pilots, and a known training symbol. It does not
  implement EHT PPDU framing, standard RU/pilot maps or bit labeling, FEC,
  MIMO, puncturing, or multi-link operation. `flat` estimation assumes a
  frequency-flat channel; `per_subcarrier` estimates each active tone.
- **Bluetooth:** uncoded LE1M/LE2M GFSK with BT=0.5 and modulation index 0.5.
  Does not implement packet acquisition, whitening, CRC, hopping, LE Coded,
  or BR/EDR. BER and frequency-discriminator outputs are available;
  `evm_rms` is intentionally `None`, not a fabricated QAM EVM value.
- **UWB:** early/late burst position plus positive/negative polarity,
  Gaussian chip pulses, and a known training burst for coherent matched
  filtering. Default chip rate is 499.2 MHz. EVM measures the soft early/late
  amplitude pair. This is not an IEEE 802.15.4z packet, spreading-code, time
  hopping, secure timestamp sequence (STS), ranging, or conformance model.
- Captures must have the exact configured length and known starting sample.
  Timing acquisition, residual carrier-offset recovery, and sample-clock
  tracking are not implemented. Bluetooth tolerates constant phase/gain;
  Wi-Fi and UWB estimate gain from training rather than from known payloads.
- SNR is average time-domain sample SNR, not Eb/N0; SNR numbers and EVM
  definitions are not directly comparable between these different models.
- No VSA hardware comparison is provided for these new protocols. A passing
  self-test establishes model consistency, not standards interoperability.

## Validation

```text
python -m pytest -q test_wireless_phy.py
python -m pytest -q test_nr5g.py test_wireless_phy.py
```

Tests cover noiseless recovery, gain and phase, known bit/pulse patterns,
high-SNR scaling, input validation, and Matplotlib rendering. The VS Code
task **Validate wireless PHY simulations** runs both test files in the workspace
`.venv`, with a headless plotting backend for tests only.