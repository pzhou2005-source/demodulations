"""Keysight VSA 89600 automation for 5G NR demodulation.

Controls VSA 89600 via its .NET API to:
  1. Load an IQ waveform file (recorded or generated)
  2. Configure 5G NR demodulation
  3. Extract EVM, constellation, and per-subcarrier results
  4. Return results for comparison with our Python demodulator

Requirements:
  - Keysight 89600 VSA software installed
    - pythonnet for .NET automation
  - The VSA must be licensed for 5G NR demod (option BHN)
"""

import numpy as np
import time
import os
import sys
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _pdsch_data_symbols(symbols: np.ndarray, n_sc: int) -> np.ndarray:
    if symbols.size == 0 or symbols.size % (14 * n_sc):
        raise ValueError("VSA IQ grid must contain complete 14-symbol slots")
    grid = symbols.reshape(-1, n_sc)
    data_mask = np.ones(grid.shape, dtype=bool)
    data_mask[2::14, 0::2] = False
    return grid[data_mask]


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
    is_simulated: bool = False
    measurement_status: str = ""


class VSA89600:
    """Wrapper around Keysight VSA 89600 .NET automation."""

    def __init__(self, visible: bool = True):
        self.app = None
        self.meas = None
        self._api = None
        self._nr_api = None
        self._original_measurement = None
        self._visible = visible

    def connect(self):
        """Launch or connect to VSA 89600."""
        try:
            import clr
        except ImportError as exc:
            raise RuntimeError("Install pythonnet to use the VSA .NET API") from exc
        install_dir = os.environ.get("VSA_INSTALL_DIR")
        if install_dir:
            software = Path(install_dir)
        else:
            installations = sorted(Path(os.environ.get("ProgramFiles", "C:/Program Files")).glob(
                "Keysight/89600 Software */89600 VSA Software"))
            if not installations:
                raise RuntimeError("VSA not found; set VSA_INSTALL_DIR to its software directory")
            software = installations[-1]
        interfaces = software / "Interfaces"
        for directory in (software, interfaces):
            if str(directory) not in sys.path:
                sys.path.append(str(directory))
        clr.AddReference(str(interfaces / "Agilent.SA.Vsa.Interfaces.dll"))
        clr.AddReference(str(interfaces / "Agilent.SA.Vsa.NewRadio.Interfaces.dll"))
        self._api = importlib.import_module("Agilent.SA.Vsa")
        self._nr_api = importlib.import_module("Agilent.SA.Vsa.NewRadio")
        self.app = self._api.ApplicationFactory.Create()
        if self.app is None:
            self.app = self._api.ApplicationFactory.Create(
                True, None, None, -1, 60, self._api.CreateOptions.Use64Bit,
                "/nosearchhardware")
        if self.app is None:
            raise RuntimeError("VSA startup did not complete")
        self.app.IsVisible = self._visible
        self._original_measurement = self.app.Measurements.SelectedItem
        self.meas = self.app.Measurements.Create()
        self.meas.Name = "Python NR comparison"

    def disconnect(self):
        if self.app is not None:
            if self.meas is not None:
                self.app.Measurements.Remove(self.meas)
            if self._original_measurement is not None:
                self.app.Measurements.SelectedItem = self._original_measurement
            self.app = None
            self.meas = None

    # ------------------------------------------------------------------
    # Input configuration
    # ------------------------------------------------------------------
    def load_recording(self, filepath: str, sample_rate_hz: float,
                       center_freq_hz: float = 3.5e9):
        """Load IQ recording file into VSA."""
        filepath = os.path.abspath(filepath)
        self.meas.Input.Recording.RecallFile(filepath, "MAT")
        self.meas.Input.DataFrom = self._api.DataSource.Recording
        self.meas.Input.Recording.IsPlayLoop = True
        actual_rate = self.meas.Input.Recording.PlaySampleRate
        if not np.isclose(actual_rate, sample_rate_hz):
            raise RuntimeError(f"VSA sample rate {actual_rate} differs from {sample_rate_hz}")

    def load_iq_binary(self, filepath: str, sample_rate_hz: float,
                       center_freq_hz: float = 3.5e9,
                       data_type: str = "float32"):
        """Load raw IQ binary file.
        
        The file should contain interleaved I,Q float32 samples.
        """
        from scipy.io import savemat

        samples = np.fromfile(filepath, dtype=data_type)
        if samples.size == 0 or samples.size % 2:
            raise ValueError("IQ file must contain nonempty interleaved I,Q pairs")
        signal = samples[0::2] + 1j * samples[1::2]
        recording = Path(filepath).with_suffix(".mat")
        savemat(recording, {"Y": signal, "XDelta": 1.0 / sample_rate_hz,
                            "XStart": 0.0, "InputCenter": center_freq_hz,
                            "InputZoom": 1}, oned_as="column")
        self.load_recording(str(recording), sample_rate_hz, center_freq_hz)

    # ------------------------------------------------------------------
    # 5G NR demod setup
    # ------------------------------------------------------------------
    def setup_nr5g_demod(self, mu: int = 1, bw_mhz: float = 20.0,
                          n_rb: int = 51, modulation: str = "64QAM",
                          cell_id: int = 0, duplex: str = "TDD"):
        """Configure VSA for 5G NR downlink demodulation."""
        import clr

        api = self._nr_api
        nr = api.MeasurementExtension.CastToExtensionType(
            self.meas.SetMeasurementExtension(clr.GetClrType(api.MeasurementExtension)))
        nr.Preset(api.StandardProfile.Default)
        carrier = nr.PrimaryComponentCarrier
        carrier.MaximumTransmissionBandwidth = getattr(
            api.MaximumTransmissionBandwidth, f"FROne{bw_mhz:g}MHz")
        carrier.IsAutoCellIdentity = False
        carrier.CellIdentity = cell_id
        carrier.PrimarySSBlock.IsSSBlockEnabled = False
        carrier.PdcchCollection.Clear()
        carrier.CsirsCollection.Clear()
        carrier.SymbolPhaseCompensationSource = api.SymbolPhaseCompensationSource.Disabled
        carrier.DownlinkSynchronizationSource = api.SynchronizationSource.PdschDmrs
        carrier.SynchronizationTechnique = api.SynchronizationTechnique.TimeCrossCorrelation
        bwp = carrier.PrimaryDownlinkBwp
        bwp.Numerology = getattr(api.Numerology, "Mu2NormalCP" if mu == 2 else f"Mu{mu}")
        bwp.RBOffset = 0
        bwp.RBNumber = n_rb
        pdsch = bwp.PrimaryPdsch
        pdsch.IsPdschEnabled = True
        pdsch.RBOffset = 0
        pdsch.RBNumber = n_rb
        pdsch.FirstSymbolIndex = 0
        pdsch.LastSymbolIndex = 13
        pdsch.SetAllocatedSlotIndexes(0, 10 * (2 ** mu))
        pdsch.McsTable = api.McsTable.Qam256 if modulation == "256QAM" else api.McsTable.Qam64
        pdsch.Mcs = {"QPSK": 0, "16QAM": 10, "64QAM": 17, "256QAM": 20}[modulation]
        expected_modulation = getattr(api.ModulationFormat, {
            "QPSK": "Qpsk", "16QAM": "Qam16", "64QAM": "Qam64",
            "256QAM": "Qam256"}[modulation])
        if pdsch.Modulation != expected_modulation:
            raise RuntimeError(f"VSA modulation mismatch: {pdsch.Modulation}")
        pdsch.MappingType = api.MappingType.TypeA
        pdsch.DmrsConfigType = api.DmrsConfigType.Type1
        pdsch.DmrsAddPos = api.DmrsAddPosition.Pos0
        pdsch.DmrsDuration = api.DmrsDuration.SingleSymbol
        pdsch.DmrsMappingTypeAPosition = api.DmrsMappingTypeAPosition.Pos2
        pdsch.DmrsNidSource = api.DmrsNidSource.FromCellId
        pdsch.Nscid = 0
        pdsch.ReservedCdmGroupNumber = api.ReservedDmrsCdmGroupNumber.One
        pdsch.IsPtrsEnabled = False
        pdsch.IsDataIncludedInRmsEvm = True
        pdsch.IsDmrsIncludedInRmsEvm = False
        nr.SynchronizationPrefersPdschDmrs = True
        nr.AnalysisStartBoundary = api.AnalysisBoundaryType.Subframe
        nr.AcquisitionMode = api.AcquisitionMode.Reduced
        nr.ResultLength = 1
        nr.SubframeInterval = 1
        nr.FirstSlotIndex = 0
        nr.IsEvmReportedInDB = False
        nr.IsEvmConformanceCompliant = False
        nr.IsEqualizerTrainingMovingAverageEnabled = False
        nr.EqualizerTrainingSignalBasis = api.EqualizerTrainingSignalBasis.RS
        nr.EqualizerTrainingTimeBasis = api.EqualizerTrainingTimeBasis.Slot
        self.meas.IsContinuous = False
        self._nr = nr
        self._n_sc = n_rb * 12

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
        scalars = {name: self._get_scalar(name)
                   for name in ("EvmRms", "EvmPeak", "FrequencyError")}
        invalid = [name for name, value in scalars.items()
                   if value is None or not np.isfinite(value)]
        if invalid:
            raise RuntimeError("Missing or invalid VSA measurements: " + ", ".join(invalid))
        evm_rms = scalars["EvmRms"]
        evm_peak = scalars["EvmPeak"]
        freq_error = scalars["FrequencyError"]

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

        if not len(sym_i) or not np.all(np.isfinite(sym_i + 1j * sym_q)):
            raise RuntimeError("VSA returned no valid constellation; comparison is unavailable")

        data_symbols = _pdsch_data_symbols(sym_i + 1j * sym_q, self._n_sc)
        sym_i, sym_q = data_symbols.real, data_symbols.imag
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
            measurement_status=f"{self.meas.Status.Value}; {self.meas.Message}",
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
            except Exception as exc:
                raise RuntimeError(f"VSA command failed: {cmd}") from exc

    def _restart_meas(self):
        """Restart measurement."""
        for name in ("Summary1", "IQ Meas1", "IQ Ref1", "RMS Error Vector Time1",
                     "RMS Error Vector Spectrum1"):
            self.meas.IsCalculateMeasurementData(name, True)
        self.meas.Input.Recording.PlayPosition = 0
        self.meas.Restart()
        self.meas.WaitForMeasurementDone(30000)
        print(f"VSA status: {self.meas.Status.Value}; {self.meas.Message}")

    def _find_trace(self, name: str):
        """Find a trace by name."""
        data_names = {"EVM Summary": "Summary1", "Constellation": "IQ Meas1",
                      "EVM vs Symbol": "RMS Error Vector Time1",
                      "EVM vs Subcarrier": "RMS Error Vector Spectrum1"}
        return self.meas.MeasurementData(data_names[name])

    def _read_trace_data(self, trace) -> np.ndarray:
        """Read numeric data from a trace."""
        try:
            data = np.asarray(list(trace.DoubleData), dtype=float)
            if trace.IsComplex:
                data = data[0::2] + 1j * data[1::2]
            return data
        finally:
            trace.Dispose()

    def _get_scalar(self, name: str) -> Optional[float]:
        """Get a scalar result from the summary table."""
        summary = self.meas.MeasurementData("Summary1")
        if summary is None:
            return None
        try:
            name = {"EvmRms": "EVM", "EvmPeak": "EVMPk"}.get(name, name)
            names = list(summary.SummaryNames)
            if name not in names:
                return None
            value = summary.Summary(name)
            print(f"VSA {name}: {value} [{summary.SummaryUnit(name)}]")
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        finally:
            summary.Dispose()


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
        vsa._restart_meas()
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
        evm_peak=result.evm_peak + evm_offset * 2,
        freq_error_hz=rng.uniform(-5, 5),
        symbols_i=result.rx_symbols.real,
        symbols_q=result.rx_symbols.imag,
        evm_per_symbol=result.evm_per_symbol + rng.uniform(0, 0.1, len(result.evm_per_symbol)),
        evm_per_subcarrier=np.zeros(cfg.n_sc),
        raw_trace_data={},
        is_simulated=True,
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
