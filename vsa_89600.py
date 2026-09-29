"""Keysight VSA 89600 automation for 5G NR demodulation.

Controls VSA 89600 via COM/DCOM to:
  1. Load an IQ waveform file (recorded or generated)
  2. Configure 5G NR demodulation
  3. Extract EVM, constellation, and per-subcarrier results
  4. Return results for comparison with our Python demodulator

Requirements:
  - Keysight 89600 VSA software installed
  - win32com (pywin32) for COM automation
  - The VSA must be licensed for 5G NR demod (option BHN)
"""

import numpy as np
import time
import os
from dataclasses import dataclass
from typing import Optional

try:
    import win32com.client
    HAS_COM = True
except ImportError:
    HAS_COM = False


@dataclass
class VSAResult:
    """Results extracted from VSA 89600."""
    evm_rms: float                   # RMS EVM (%)
    evm_peak: float                  # peak EVM (%)
    freq_error_hz: float             # frequency error
    symbols_i: np.ndarray            # constellation I values
    symbols_q: np.ndarray            # constellation Q values
    evm_per_symbol: np.ndarray       # EVM per OFDM symbol (%)
    evm_per_subcarrier: np.ndarray   # EVM per subcarrier (%)
    raw_trace_data: dict             # all trace data keyed by trace name


class VSA89600:
    """Wrapper around Keysight VSA 89600 COM automation."""

    def __init__(self, visible: bool = True):
        if not HAS_COM:
            raise RuntimeError("pywin32 not installed. Run: pip install pywin32")
        self.app = None
        self.meas = None
        self._visible = visible

    def connect(self):
        """Launch or connect to VSA 89600."""
        try:
            self.app = win32com.client.Dispatch("AgtVsa.Application")
        except Exception:
            self.app = win32com.client.Dispatch("AgtVsa.Application.1")
        self.app.Visible = self._visible
        self.app.IsRunning = True
        self.meas = self.app.Measurements.SelectedItem
        time.sleep(1)

    def disconnect(self):
        if self.app:
            self.app = None
            self.meas = None

    # ------------------------------------------------------------------
    # Input configuration
    # ------------------------------------------------------------------
    def load_recording(self, filepath: str, sample_rate_hz: float,
                       center_freq_hz: float = 3.5e9):
        """Load IQ recording file into VSA."""
        filepath = os.path.abspath(filepath)
        inp = self.meas.Input
        inp.Recording.FileName = filepath
        inp.Recording.SampleRate = sample_rate_hz
        inp.Recording.CenterFrequency = center_freq_hz
        inp.DataFrom = 2  # 2 = Recording
        self._restart_meas()

    def load_iq_binary(self, filepath: str, sample_rate_hz: float,
                       center_freq_hz: float = 3.5e9,
                       data_type: str = "float32"):
        """Load raw IQ binary file.
        
        The file should contain interleaved I,Q float32 samples.
        """
        filepath = os.path.abspath(filepath)
        inp = self.meas.Input
        inp.Recording.FileName = filepath
        inp.Recording.SampleRate = sample_rate_hz
        inp.Recording.CenterFrequency = center_freq_hz
        # Set format: 32-bit float interleaved IQ
        inp.Recording.DataFormat = 3  # IQ interleaved
        inp.Recording.NumberFormat = 1  # 32-bit float
        inp.DataFrom = 2
        self._restart_meas()

    # ------------------------------------------------------------------
    # 5G NR demod setup
    # ------------------------------------------------------------------
    def setup_nr5g_demod(self, mu: int = 1, bw_mhz: float = 20.0,
                          n_rb: int = 51, modulation: str = "64QAM",
                          cell_id: int = 0, duplex: str = "TDD"):
        """Configure VSA for 5G NR downlink demodulation."""
        demod = self.meas.DDEmod  # digital demod object

        # Set measurement type to 5G NR
        self.meas.MeasurementType = 48  # 5G NR

        # access 5G NR demod properties
        nr = self.meas.DDEmod
        
        # Standard properties via SCPI-style commands
        self._send_scpi(f":DDEMod:NR5G:SUBCarrier:SPACing {15 * (2 ** mu)}")
        self._send_scpi(f":DDEMod:NR5G:BANDwidth {bw_mhz}E6")
        self._send_scpi(f":DDEMod:NR5G:RBCount {n_rb}")
        self._send_scpi(f":DDEMod:NR5G:DUPLex {duplex}")
        self._send_scpi(f":DDEMod:NR5G:CELL:ID {cell_id}")

        # Modulation mapping
        mod_map = {
            "QPSK": "QPSK", "16QAM": "QAM16",
            "64QAM": "QAM64", "256QAM": "QAM256"
        }
        self._send_scpi(f":DDEMod:NR5G:MODulation {mod_map.get(modulation, 'QAM64')}")

        self._restart_meas()
        time.sleep(2)

    # ------------------------------------------------------------------
    # Trace / result extraction
    # ------------------------------------------------------------------
    def get_results(self) -> VSAResult:
        """Extract demodulation results from VSA traces."""
        traces = {}

        # Standard traces for 5G NR demod
        trace_names = [
            "EVM Summary",
            "Constellation",
            "EVM vs Symbol",
            "EVM vs Subcarrier",
        ]

        for name in trace_names:
            try:
                tr = self._find_trace(name)
                if tr is not None:
                    data = self._read_trace_data(tr)
                    traces[name] = data
            except Exception:
                pass

        # Extract key metrics
        evm_rms = self._get_scalar("EvmRms") or 0.0
        evm_peak = self._get_scalar("EvmPeak") or 0.0
        freq_error = self._get_scalar("FrequencyError") or 0.0

        # Constellation data
        const_data = traces.get("Constellation", np.array([]))
        if len(const_data) > 0 and const_data.ndim == 2:
            sym_i = const_data[:, 0]
            sym_q = const_data[:, 1]
        elif np.iscomplexobj(const_data):
            sym_i = const_data.real
            sym_q = const_data.imag
        else:
            sym_i = np.array([])
            sym_q = np.array([])

        evm_sym = traces.get("EVM vs Symbol", np.array([]))
        evm_sc = traces.get("EVM vs Subcarrier", np.array([]))

        return VSAResult(
            evm_rms=evm_rms,
            evm_peak=evm_peak,
            freq_error_hz=freq_error,
            symbols_i=sym_i,
            symbols_q=sym_q,
            evm_per_symbol=evm_sym if len(evm_sym) else np.array([]),
            evm_per_subcarrier=evm_sc if len(evm_sc) else np.array([]),
            raw_trace_data=traces,
        )

    # ------------------------------------------------------------------
    # SCPI / helper methods
    # ------------------------------------------------------------------
    def _send_scpi(self, cmd: str):
        """Send SCPI command to VSA."""
        try:
            self.app.SCPI.Parse(cmd)
        except AttributeError:
            try:
                self.meas.RemoteCommand(cmd)
            except Exception:
                pass

    def _restart_meas(self):
        """Restart measurement."""
        try:
            self.meas.Restart()
        except Exception:
            pass
        time.sleep(0.5)

    def _find_trace(self, name: str):
        """Find a trace by name."""
        for i in range(self.meas.Traces.Count):
            tr = self.meas.Traces.Item(i)
            if name.lower() in tr.Title.lower():
                return tr
        return None

    def _read_trace_data(self, trace) -> np.ndarray:
        """Read numeric data from a trace."""
        data = trace.DoubleData
        return np.array(data)

    def _get_scalar(self, name: str) -> Optional[float]:
        """Get a scalar result from the summary table."""
        try:
            return float(self.app.SCPI.Parse(f":FETCh:DDEMod:NR5G:{name}?"))
        except Exception:
            try:
                return float(self.meas.RemoteCommand(f":FETCh:DDEMod:NR5G:{name}?"))
            except Exception:
                return None


def run_vsa_demod(iq_filepath: str, sample_rate_hz: float,
                  cfg_mu: int = 1, cfg_bw_mhz: float = 20.0,
                  cfg_n_rb: int = 51, cfg_modulation: str = "64QAM",
                  cfg_cell_id: int = 0,
                  center_freq_hz: float = 3.5e9) -> VSAResult:
    """One-shot: load waveform into VSA 89600, demod, return results."""
    vsa = VSA89600(visible=True)
    try:
        vsa.connect()
        vsa.load_iq_binary(iq_filepath, sample_rate_hz, center_freq_hz)
        vsa.setup_nr5g_demod(mu=cfg_mu, bw_mhz=cfg_bw_mhz, n_rb=cfg_n_rb,
                              modulation=cfg_modulation, cell_id=cfg_cell_id)
        time.sleep(3)  # let VSA settle
        result = vsa.get_results()
        return result
    finally:
        vsa.disconnect()


# ---------------------------------------------------------------------------
# Simulated VSA results for offline development / testing
# ---------------------------------------------------------------------------
def simulate_vsa_result(rx_signal: np.ndarray, cfg) -> VSAResult:
    """Simulate VSA-like results using our own demodulator (for offline testing).
    
    This lets you run the comparison pipeline without actual VSA hardware.
    """
    from nr5g_demod import demodulate_nr5g
    from nr5g_waveform import generate_nr5g_waveform

    # re-generate TX for reference
    tx = generate_nr5g_waveform(cfg)
    result = demodulate_nr5g(
        rx_signal, cfg,
        tx_bits=tx["tx_bits"],
        tx_symbols=tx["tx_symbols"],
        data_positions=tx["data_positions"],
    )
    # add small random offset to simulate VSA measurement differences
    rng = np.random.default_rng(99)
    evm_offset = rng.uniform(0.05, 0.3)

    return VSAResult(
        evm_rms=result.evm_rms + evm_offset,
        evm_peak=np.max(result.evm_per_symbol) + evm_offset * 2,
        freq_error_hz=rng.uniform(-5, 5),
        symbols_i=result.rx_symbols.real,
        symbols_q=result.rx_symbols.imag,
        evm_per_symbol=result.evm_per_symbol + rng.uniform(0, 0.1, len(result.evm_per_symbol)),
        evm_per_subcarrier=np.zeros(cfg.n_sc),
        raw_trace_data={},
    )


if __name__ == "__main__":
    from nr5g_waveform import NR5GConfig, generate_nr5g_waveform, save_waveform_iq

    cfg = NR5GConfig(mu=1, bw_mhz=20, n_rb=51, modulation="64QAM",
                     n_slots=2, snr_db=30, seed=42)
    tx = generate_nr5g_waveform(cfg)
    save_waveform_iq("nr5g_waveform.bin", tx["time_signal"])

    # simulate VSA result (offline)
    vsa_res = simulate_vsa_result(tx["time_signal"], cfg)
    print(f"[VSA sim] EVM RMS: {vsa_res.evm_rms:.2f}%")
    print(f"[VSA sim] Freq Error: {vsa_res.freq_error_hz:.1f} Hz")
