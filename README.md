# NR Demodulation Comparison

Python CP-OFDM/PDSCH waveform generation, pilot-based demodulation, and comparison
with Keysight 89600 VSA. The implemented waveform is an uncoded, single-layer
PDSCH allocation with type-1 DMRS on symbol 2 and no SSB or control channels.
It is not a complete NR base station or a conformance-test implementation.

## Setup

Use Python 3.11 or newer:

```powershell
python -m pip install numpy scipy matplotlib pytest
python -m pytest -q
```

The optional independent waveform-reference test requires `py3gpp`:

```powershell
python -m pip install py3gpp
```

## Real VSA Comparison

On Windows, install Keysight 89600 VSA with 5G NR analysis and the .NET bridge:

```powershell
python -m pip install pythonnet
python -X utf8 compare.py --real-vsa
```

The adapter discovers the Keysight installation under Program Files. Set
`VSA_INSTALL_DIR` to the `89600 VSA Software` directory for another location.
It connects to a running VSA or attempts a bounded startup, creates a separate
measurement, and removes that measurement when finished. Start VSA manually if
hardware-discovery or license dialogs prevent automated startup.

The verified setup is 20 MHz, 51 RB, 30 kHz SCS, 64QAM, port 1000. The generator
provides a 10 ms recording; VSA analyzes one subframe using PDSCH-DMRS time
cross-correlation. The MAT recording carries sample-rate and center-frequency
metadata. The adapter uses the installed .NET API, not legacy COM ProgIDs.
Do not infer unavailable NR support from an idle `License.IsValid` flag alone:
selecting the NR measurement and querying `IsLicensed` confirmed availability.

The comparison removes only the configured DMRS positions from VSA's native IQ
grid, searches the full Python data stream for the measured interval, and fits
one constant amplitude/phase correction. Real comparisons require correlation
of at least 0.99 and overlap of the complete VSA capture. No simulated fallback
is allowed. Python and VSA summary EVM are compared on the matched interval;
the Python BER covers the complete generated recording. SNR refers to average
time-domain sample power, not active-subcarrier power.

With seed 42, 30 dB SNR, and the default Python FFT window, a verified real run
correlated 16,524 data symbols at approximately 0.999982. Matched RMS EVM was
2.300% (Python) and 2.298% (VSA), with 0.603% EVM between aligned streams.
Different FFT windows, filtering, timing, and estimation can leave a nonzero
difference even when the transmitted symbols agree.

VSA reported `MeasurementDone, GapData, Visible` on that run, with finite EVM and
full symbol overlap. The warning is retained in the report and saved arrays;
this result is not a claim that acquisition was warning-free. VSA's native
per-symbol RMS trace includes DMRS, unlike Python's data-only per-symbol EVM.

## Outputs

The default `results` directory contains comparison/correlation text reports,
plots, the IQ recording, and `comparison_data.npz`. The NPZ stores untouched
VSA symbols, the native grid, aligned arrays, matched transmit references,
correlation, offset, provenance, and acquisition status. Use `--output-dir` to
keep separate runs. Do not interpret older reports as the result of a failed run.

Running without `--real-vsa` explicitly uses the same Python demodulator with
synthetic metric offsets. Those outputs are labeled `SIMULATED` and are not
independent validation.