# %% [markdown]
# # 5G NR Demodulation — Interactive Scratch Pad
# 中文学习导读：这是交互式实验入口，具体算法在导入的模块中实现。
# “# %%”把脚本分成单元；点击 Run Cell，可在 Jupyter Interactive Window 中逐段运行。
# 第 1–18 单元应按顺序执行；后面的单元会使用前面创建的变量。
# 第 19–21 单元分别演示 Wi-Fi、Bluetooth、UWB，可独立运行。
# 主线：比特 → QAM 复数符号 → 频域资源网格 → IFFT/循环前缀 → 时域 IQ → 解调。
# 三个不同概念：一个 QAM 符号表示若干比特；一个 OFDM 符号包含许多子载波；
# 一个 IQ 采样点是这些子载波叠加后的时域复数值，不能把三者混为一谈。
# 这里的 NR 是简化的物理层实验，不是包含完整信道编码、同步和协议栈的接收机。

# %% 1. Configuration
# 第 1 步：配置实验。首次学习可手动改为 QPSK、n_slots=1、snr_db=None，观察无噪声恢复。
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (project_root / "nr5g_waveform.py").is_file():
    project_root = project_root / "nr5g_demod"
if not (project_root / "nr5g_waveform.py").is_file():
    raise FileNotFoundError("Open the nr5g_demod project folder before running this script")
project_root = project_root.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np  # NumPy 提供数组、复数、FFT 等运算；np 是常用的简写名称。
import matplotlib.pyplot as plt  # Matplotlib 的绘图接口；plt 用于创建和显示图形。
from nr5g_waveform import NR5GConfig, NUMEROLOGY, QAM_MAP  # 导入配置类及参数表；本脚本未直接使用后两个表。

cfg = NR5GConfig(  # 创建配置对象；后续用 cfg.属性名 读取参数。
    mu=1,               # 子载波间隔 SCS=15×2^mu kHz；mu=1 时为 30 kHz。
    bw_mhz=20.0,        # 标称信道带宽 20 MHz；不等于采样率，也不等于所有有效子载波的总宽度。
    n_rb=51,             # 51 个资源块，每个 RB 有 12 个子载波，总计 612 个。
    modulation="64QAM",  # 64 个星座点；每个数据 QAM 符号携带 log2(64)=6 个比特。
    n_slots=2,  # 生成两个时隙；正常循环前缀下，每个时隙有 14 个 OFDM 符号。
    snr_db=30,  # 按时域平均采样功率添加 30 dB AWGN；None 表示不加噪声，不是 0 dB。
    cfo_hz=150.0,        # 残余载波频偏（Hz），由完整接收链估计并校正。
    channel_taps=[1.0, 0.3 + 0.1j, 0.05],  # 3 径多径信道的复数系数。
    channel_delays=[0, 3, 7],  # 各径相对主径的采样点延迟。
    seed=42,  # 固定随机种子，便于在相同配置下复现数据和噪声。
)  # 未显式指定的参数使用类的默认值，例如 n_fft=2048、sample_rate_mhz=61.44。

print(f"Numerology μ={cfg.mu}: SCS={cfg.scs_khz} kHz, {cfg.symbols_per_slot} sym/slot")  # f 字符串会把花括号中的表达式替换成值。
print(f"Subcarriers: {cfg.n_sc}  ({cfg.n_rb} RB × 12)")  # n_sc 是按 RB 数量计算出的有效子载波数。
print(f"FFT size: {cfg.n_fft}, Sample rate: {cfg.sample_rate_mhz} MHz")  # FFT 包含有效载波和空载波；本配置 Fs/NFFT=30 kHz。
print(f"Modulation: {cfg.modulation} ({cfg.bits_per_symbol} bits/sym)")  # 此处 bits/sym 指每个 QAM 符号的比特数。
print(f"CFO: {cfg.cfo_hz} Hz, Channel: {len(cfg.channel_taps)}-tap multipath")  # 本轮额外注入的频偏和多径损伤参数。

# %% 2. Generate 5G NR waveform
# 第 2 步：构建发射信号。先理解返回的字典，再进入函数研究内部实现。
from nr5g_waveform import generate_nr5g_waveform  # 导入波形发生器：生成比特、映射 QAM、插入 DMRS、执行 OFDM 并加噪声。

tx = generate_nr5g_waveform(cfg)  # tx 是 dict，不是一条数组；用字符串键取出各阶段结果。
# tx["tx_bits"]：原始数据比特；tx["tx_symbols"]：加噪声之前的理想数据 QAM 符号。
# tx["resource_grid"]：含数据和 DMRS 导频的二维频域网格；tx["time_signal"]：加噪后的时域 IQ。
# tx["time_signal_clean"]：未加噪时域 IQ；tx["cp_lengths"]：每个 OFDM 符号的循环前缀长度。
# tx["data_positions"]：形如 (符号行号, "all"/"odd") 的列表，说明哪些资源元素承载数据。

print(f"TX bits:      {len(tx['tx_bits'])}")  # len 求数组长度；当前配置应有 99144 个数据比特。
print(f"TX symbols:   {len(tx['tx_symbols'])}")  # 当前有 16524 个数据 QAM 符号，每个对应 6 比特；不含 DMRS。
print(f"IQ samples:   {len(tx['time_signal'])}")  # 当前有 61440 个复数采样点，包含循环前缀，对应 1 ms。
print(f"Data positions: {len(tx['data_positions'])} OFDM symbols carry data")  # 当前为 28 行；列表长度不是 QAM 符号数量。

# %% 3. Inspect the resource grid
# 第 3 步：看频域网格。一个资源元素 RE = 一个 OFDM 符号时段上的一个子载波。
resource_grid = tx["resource_grid"]  # 取出网格；当前形状为 (28, 612)，行是时间，列是子载波。
print(f"Resource grid shape: {resource_grid.shape}  (symbols × subcarriers)")  # shape 返回各轴长度，而不是总元素数。

fig, axes = plt.subplots(1, 2, figsize=(14, 4))  # 创建整幅图 fig 和两个子图 axes；figsize 的单位是英寸。
axes[0].imshow(np.abs(resource_grid), aspect="auto", interpolation="none")  # abs 对复数取模；热力图显示每个 RE 的幅度，不做插值平滑。
axes[0].set_title("Resource Grid — Magnitude")  # 左图是幅度；64-QAM 的不同点本来就有不同幅度，不一定是失真。
axes[0].set_xlabel("Subcarrier"); axes[0].set_ylabel("OFDM Symbol")  # 横轴为子载波索引，纵轴为 OFDM 符号索引；分号分隔两条语句。
axes[1].imshow(np.angle(resource_grid), aspect="auto", interpolation="none", cmap="twilight")  # angle 返回复数相位，单位为弧度；循环色表适合相位。
axes[1].set_title("Resource Grid — Phase")  # 右图展示数据和导频在各个 RE 上的相位。
axes[1].set_xlabel("Subcarrier"); axes[1].set_ylabel("OFDM Symbol")  # 两图使用同样的时间/频率索引，便于对照。
plt.tight_layout(); plt.show()  # 自动调整子图间距，然后把图显示到当前绘图后端。

# %% 3b. Typical impairment patterns on the resource grid
# 第 3b 步：人为注入几种常见损伤，对照热力图学会"一眼认出"每种问题的视觉特征；真实信号不会同时出现这么多问题。
n_sym, n_sc_grid = resource_grid.shape  # 取出网格尺寸，用于定位要注入损伤的行/列；n_sc_grid 避免跟 cfg.n_sc 混淆。

error_grids = {}  # 字典收集每种损伤对应的网格，键是子图标题。

dead_sc = resource_grid.copy()  # copy 避免污染原始 resource_grid，后面步骤仍需要干净的参考网格。
dead_sc[:, n_sc_grid // 2] = 0  # 整列清零：模拟坏子载波或陷波滤波器——热力图上会看到一条竖直黑线。
error_grids["Dead subcarrier"] = dead_sc

dead_sym = resource_grid.copy()
dead_sym[n_sym // 2, :] = 0  # 整行清零：模拟符号丢失或门控错误——热力图上会看到一条水平黑线。
error_grids["Dead OFDM symbol"] = dead_sym

freq_offset = resource_grid.copy()
row_phase = np.exp(1j * 2 * np.pi * 0.015 * np.arange(n_sym))  # 残余频偏：相位随符号序号（时间）线性增长，同一符号内各子载波相同。
freq_offset *= row_phase[:, None]  # [:, None] 把一维相位转成列向量，按行广播相乘。
error_grids["Residual freq offset"] = freq_offset

timing_offset = resource_grid.copy()
col_phase = np.exp(1j * 2 * np.pi * 0.01 * np.arange(n_sc_grid))  # 采样定时偏差：相位随子载波（频率）线性增长，同一子载波在各符号上相同。
timing_offset *= col_phase[None, :]  # [None, :] 把一维相位转成行向量，按列广播相乘；这正是你在 FFT 窗口偏移实验里要找的那种斜坡。
error_grids["Timing / CP offset"] = timing_offset

iq_imbalance = resource_grid + 0.15 * np.conj(resource_grid[:, ::-1])  # IQ 不平衡产生镜像泄漏：以直流为中心翻转子载波索引后叠加一个衰减共轭副本。
error_grids["IQ imbalance (mirror image)"] = iq_imbalance

interferer = resource_grid.copy()
interferer[:, n_sc_grid // 4] += 6.0  # 窄带干扰：只在某一根子载波上叠加一个幅度很大的常数音，其余子载波不受影响。
error_grids["Narrowband interferer"] = interferer

fig, axes = plt.subplots(len(error_grids), 2, figsize=(12, 4 * len(error_grids)))  # 每种损伤占一行，左列幅度、右列相位。
for row, (name, grid) in enumerate(error_grids.items()):  # enumerate 同时给出行号和字典里的 (标题, 网格) 对。
    axes[row, 0].imshow(np.abs(grid), aspect="auto", interpolation="none")
    axes[row, 0].set_title(f"{name} — Magnitude")
    axes[row, 0].set_xlabel("Subcarrier"); axes[row, 0].set_ylabel("OFDM Symbol")
    axes[row, 1].imshow(np.angle(grid), aspect="auto", interpolation="none", cmap="twilight")
    axes[row, 1].set_title(f"{name} — Phase")
    axes[row, 1].set_xlabel("Subcarrier"); axes[row, 1].set_ylabel("OFDM Symbol")
plt.tight_layout(); plt.show()  # 对照各行记忆特征：黑线=丢失，渐变色带=相位斜坡（时间方向=频偏，频率方向=定时偏差），对称亮斑=镜像泄漏，单点极亮=干扰音。

# %% 3c. Isolating phase ramps from frequency/timing offset
# 第 3c 步：原始数据相位本身随机铺满整个色环，叠加的常数/线性相位会被这种随机性掩盖，热力图上肉眼看不出来；
# 真实接收机诊断频偏/定时偏差时，会先除掉已知的发射相位，只留下误差因子，再按时间或频率画成曲线——这里复现这一步。
freq_err_factor = np.angle(freq_offset * np.conj(resource_grid))  # 乘以共轭消去随机数据相位，只留下人为叠加的 row_phase。
timing_err_factor = np.angle(timing_offset * np.conj(resource_grid))  # 同理，只留下人为叠加的 col_phase。

fig, axes = plt.subplots(1, 2, figsize=(14, 4))  # 左图看频偏随时间的斜坡，右图看定时偏差随频率的斜坡。
axes[0].plot(np.unwrap(freq_err_factor[:, 0]), marker="o", markersize=3)  # 固定第 0 根子载波，看误差相位随符号序号变化；unwrap 去掉 ±π 跳变，露出真实斜率。
axes[0].set_title("Residual freq offset — phase vs OFDM symbol (error only)")
axes[0].set_xlabel("OFDM Symbol"); axes[0].set_ylabel("Phase error (rad)")
axes[1].plot(np.unwrap(timing_err_factor[0, :]), marker="o", markersize=2)  # 固定第 0 个符号，看误差相位随子载波变化。
axes[1].set_title("Timing / CP offset — phase vs subcarrier (error only)")
axes[1].set_xlabel("Subcarrier"); axes[1].set_ylabel("Phase error (rad)")
plt.tight_layout(); plt.show()  # 两条线应是干净的直线：左图斜率来自频偏，右图斜率来自定时偏差；这正是真实接收机做频偏/定时估计的方法。

# %% 3d. Full demodulation of each impairment type
# 第 3d 步：把每种损伤网格送入完整接收链（DMRS 信道估计 → ZF 均衡 → 判决），用 EVM/BER 量化影响，而不是只看热力图颜色。
from nr5g_demod import channel_estimate_dmrs, equalise_zf, compute_evm  # 复用第 7–9 步用到的同一组接收函数。
from nr5g_waveform import qam_demodulate_hard  # 硬判决，用于计算 BER。

grids_to_demod = {"No impairment (baseline)": resource_grid, **error_grids}  # 基线放最前，便于和各损伤对比。

demod_rows = []  # 收集 (名称, EVM, BER)，供打印和画图使用。
for name, grid in grids_to_demod.items():
    h_est = channel_estimate_dmrs(grid, cfg)  # 仍从同一批 DMRS 位置估计信道；损伤越靠近导频，估计越容易被污染。
    eq_grid = equalise_zf(grid, h_est)  # 逐 RE 除以估计增益，不区分数据和导频 RE。
    rx_data = [eq_grid[sym_idx, 1::2] if kind == "odd" else eq_grid[sym_idx, :]  # 按发射端相同的数据 RE 布局抽取。
               for sym_idx, kind in tx["data_positions"]]
    rx_symbols = np.concatenate(rx_data)  # 展平成与 tx["tx_symbols"] 对齐的一维数组。
    evm = compute_evm(tx["tx_symbols"], rx_symbols)  # 与真实发射符号比较，不是与基线比较。
    bits = qam_demodulate_hard(rx_symbols, cfg.bits_per_symbol)  # 最近邻判决得到比特。
    ber = np.mean(tx["tx_bits"][:len(bits)] != bits)  # 按比特逐位比较，可能为 0。
    demod_rows.append((name, evm, ber))
    print(f"{name:28s} EVM={evm:7.3f}%  BER={ber:.2e}")  # 直接对照数字，看哪种损伤实际影响最大。

fig, ax = plt.subplots(figsize=(9, 4))  # 条形图把各损伤的 EVM 排成一行，便于横向比较。
ax.bar([row[0] for row in demod_rows], [row[1] for row in demod_rows], color="indianred")
ax.set_title("EVM after full demodulation, per impairment")  # 标题强调这是过完整接收链后的结果，不是注入前的理论值。
ax.set_ylabel("EVM (%)")
ax.tick_params(axis="x", rotation=30)  # 标签较长，倾斜避免重叠。
for label in ax.get_xticklabels():
    label.set_horizontalalignment("right")
plt.tight_layout(); plt.show()  # 基线条应接近第 9 步的 EVM；其余条体现各损伤经信道估计/均衡后的实际放大或抑制效果。

# %% 4. TX constellation (before OFDM)
# 第 4 步：QAM 星座是单个数据符号的 I/Q 分布，不是时域 OFDM 采样点的分布。
from nr5g_waveform import _qam_constellation  # 导入内部星座生成函数；名称前的下划线表示内部辅助接口。

ref = _qam_constellation(cfg.bits_per_symbol)  # 生成全部理想星座点；64-QAM 有 64 点，整套星座平均功率归一化为 1。

fig, ax = plt.subplots(figsize=(6, 6))  # 创建单个星座子图；ax 是一个坐标轴对象。
ax.scatter(tx["tx_symbols"].real, tx["tx_symbols"].imag, s=2, alpha=0.3, label="TX data")  # 复数实部为 I、虚部为 Q；s 控制点面积，alpha 控制透明度。
ax.scatter(ref.real, ref.imag, s=60, c="red", marker="+", linewidths=2, label="Ideal")  # 用红色加号叠加全部理想位置；无噪声数据点应落在这些位置。
ax.set_title(f"TX Constellation — {cfg.modulation}")  # 在标题中显示当前调制方式。
ax.set_xlabel("I"); ax.set_ylabel("Q")  # I 为同相分量，Q 为正交分量。
ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend()  # 横纵轴等比例，避免星座被拉伸；显示网格和图例。
plt.show()  # 在交互窗口显示星座图。

# %% 5. Time-domain waveform & spectrum
# 第 5 步：同一个信号的时域与频域视角；这一段是粗略频谱展示，不是精确的功率谱测量。
time_signal = tx["time_signal"]  # 取加噪后的复数 IQ；它由许多 QAM 子载波经 IFFT 叠加而成。

fig, axes = plt.subplots(2, 1, figsize=(14, 6))  # 创建上下两个子图，上面看时间波形，下面看频谱。

t_us = np.arange(len(time_signal)) / (cfg.sample_rate_mhz)  # 采样索引除以 MHz 采样率，得到微秒时间；arange 生成 0 到 N-1。
axes[0].plot(t_us[:2000], time_signal[:2000].real, linewidth=0.5)  # [:2000] 取前 2000 点；这里只画实部 I，Q 仍保留在原始数组中。
axes[0].set_title("Time-domain I (first 2000 samples)")  # 限制点数便于观察，不改变实际解调使用的完整信号。
axes[0].set_xlabel("Time (µs)"); axes[0].set_ylabel("Amplitude")  # 横轴是物理时间，纵轴是线性幅度。

# 这里从记录起点取 NFFT 点，包含循环前缀；并非严格“去 CP 后完整有效符号”的 FFT。
freq = np.fft.fftshift(np.fft.fft(time_signal[:cfg.n_fft]))  # FFT 转到频域，fftshift 把直流移到数组中间；freq 实际保存复数频谱。
f_mhz = np.linspace(-cfg.sample_rate_mhz / 2, cfg.sample_rate_mhz / 2, cfg.n_fft)  # 近似频率轴；精确 FFT 频点应由 fftfreq 生成，正端点不含 Fs/2。
axes[1].plot(f_mhz, 20 * np.log10(np.abs(freq) + 1e-12), linewidth=0.5)  # 幅度转 dB；小量避免 log(0)，此处未作 FFT/阻抗/带宽归一化。
axes[1].set_title("Spectrum (single OFDM symbol)")  # 标题是示意；不要将这段粗略频谱当作标准解调网格。
axes[1].set_xlabel("Frequency (MHz)"); axes[1].set_ylabel("Power (dB)")  # 纵轴标签虽写 Power，但这里不是校准后的 dBm 或 dB/Hz。
axes[1].set_ylim(bottom=-40)  # 只限制显示下界，不裁剪频谱数据。

plt.tight_layout(); plt.show()  # 调整布局并显示时域和频域两幅图。

# %% 6. OFDM demodulation
# 第 6 步：按已知符号边界去 CP、做 FFT、抽取有效子载波；这里没有自动包检测或频偏捕获。
from nr5g_waveform import ofdm_demodulate, _cp_lengths  # 导入 OFDM 逆变换和正常循环前缀长度计算函数。

cp = _cp_lengths(cfg.n_fft, cfg.mu) * cfg.n_slots  # Python 列表乘法会重复列表；当前 mu=1 各时隙 CP 相同，这样写适用。
# 学习提醒：mu=2/3 时长 CP 的分布依赖时隙号，不应重复首时隙；发生器返回的 tx["cp_lengths"] 更通用。
n_sym_total = cfg.symbols_per_slot * cfg.n_slots  # 14×2=28 个 OFDM 符号，决定输出网格的行数。
rx_grid = ofdm_demodulate(time_signal, cfg.n_fft, cfg.n_sc, cp, n_sym_total)  # 恢复 (28, 612) 接收网格；本调用使用默认 FFT 偏移 0。
# 若修改 cfg.fft_window_offset，此处不会自动传递它；第 9 步的完整解调器才会读取该配置。

print(f"RX grid shape: {rx_grid.shape}")  # 检查接收网格维度是否与发射网格一致。

fig, ax = plt.subplots(figsize=(14, 4))  # 创建接收网格热力图。
ax.imshow(np.abs(rx_grid), aspect="auto", interpolation="none")  # 查看各 RE 的幅度；此时还没有进行信道均衡。
ax.set_title("Received Resource Grid — Magnitude")  # 将该图与第 3 步的发射幅度图对照。
ax.set_xlabel("Subcarrier"); ax.set_ylabel("OFDM Symbol")  # 横轴子载波，纵轴 OFDM 符号。
plt.tight_layout(); plt.show()  # 显示去 CP、FFT 后的结果。

# %% 7. Channel estimation from DMRS
# 第 7 步：利用已知 DMRS 导频估计 H；接收模型可写成 Y=H×X+N。
from nr5g_demod import channel_estimate_dmrs  # 导入导频信道估计；在导频上用接收值除以已知发射值。

h_est = channel_estimate_dmrs(rx_grid, cfg)  # H_hat=Y_DMRS/X_DMRS；默认 linear 在频率方向插值，并在时隙内复用估计。
# cfg.channel_estimation="flat" 会平均一个时隙内的导频，适用于频率平坦信道，不能泛用于多径信道。
# 当前只加 AWGN，没有额外多径失真，因此真实 H=1；估计值偏离 1 主要来自导频噪声。

fig, axes = plt.subplots(1, 2, figsize=(14, 4))  # 左侧看信道增益，右侧看信道相位。
axes[0].plot(np.abs(h_est[0, :]), linewidth=0.8)  # [0, :] 取第一个 OFDM 符号对应的全部子载波；索引从 0 开始。
axes[0].set_title("Channel Magnitude (1st symbol)")  # 本实验的估计幅度应围绕 1 波动。
axes[0].set_xlabel("Subcarrier"); axes[0].set_ylabel("|H|")  # |H| 是幅度增益，不是功率增益。
axes[1].plot(np.angle(h_est[0, :]), linewidth=0.8)  # 相位估计通常围绕 0 波动；angle 返回弧度。
axes[1].set_title("Channel Phase (1st symbol)")  # 显示估计相移随子载波的位置变化。
axes[1].set_xlabel("Subcarrier"); axes[1].set_ylabel("∠H (rad)")  # 横轴是索引，不是 MHz；纵轴是弧度。
plt.tight_layout(); plt.show()  # 显示信道估计图。

print(f"Channel est mean |H| = {np.mean(np.abs(h_est)):.4f}")  # 对所有 RE 的估计幅度取平均；:.4f 表示保留四位小数。

# %% 8. Equalisation
# 第 8 步：均衡器尝试逆转信道影响，而不是直接用发射数据替换接收数据。
# 一般来说，数据 OFDM 符号离 DMRS 越近，信道估计越新，均衡后相对发射网格的偏差通常越小；离 DMRS 越远，信道变化造成的估计误差可能越大。
# 注意：本例假设每个时隙内信道不随时间变化，并把同一 DMRS 信道估计用于该时隙的所有符号，因此不会单独体现这种距离趋势。
from nr5g_demod import equalise_zf  # ZF 是零迫均衡；逐个 RE 计算 X_hat=Y/H_hat。

eq_grid = equalise_zf(rx_grid, h_est)  # 除掉估计增益和相位；H_hat 很小时会放大噪声，这是 ZF 的局限。

fig, ax = plt.subplots(figsize=(14, 4))  # 创建均衡残差图。
err = np.abs(eq_grid - resource_grid)  # 逐个 RE 计算复数差的模；只有仿真已知发射网格时才能这样直接检查。
ax.imshow(err, aspect="auto", interpolation="none", cmap="hot")  # 用颜色展示误差，亮色通常代表更大误差。
ax.set_title("Equalisation Error (|eq - tx| per RE)")  # 这是线性绝对误差，尚未按参考功率归一化成百分比 EVM。
ax.set_xlabel("Subcarrier"); ax.set_ylabel("OFDM Symbol")  # 定位误差发生在哪个子载波、哪个 OFDM 符号。
plt.colorbar(ax.images[0], ax=ax); plt.tight_layout(); plt.show()  # 为热力图添加数值色条，整理布局并显示。

# %% 9. Full demodulation pipeline (MMSE + CFO correction)
# 第 9 步：调用完整接收链，含 CFO 估计/校正和 MMSE 均衡，把第 6–8 步及数据抽取、判决、误差统计统一执行。
from nr5g_demod import demodulate_nr5g  # 返回 DemodResult 对象，使用 result.属性名 获取结果。

result = demodulate_nr5g(  # 完整解调函数会重新执行 CFO 校正、FFT 和均衡，并非直接读取上面的 eq_grid。
    time_signal, cfg,  # 输入完整 IQ 采样数组及接收参数；假定记录起点与时隙已对齐。
    tx_bits=tx["tx_bits"],  # 发射比特仅用于计算 BER，不参与接收比特判决。
    tx_symbols=tx["tx_symbols"],  # 理想数据符号用于计算 EVM，不用于强行校正接收数据。
    tx_grid=tx["resource_grid"],  # 原始发射网格，用于按子载波统计 EVM。
    data_positions=tx["data_positions"],  # 数据 RE 位置：跳过 DMRS，在导频符号行只提取奇数列数据。
    equaliser="mmse",  # MMSE 均衡在估计噪声功率后比 ZF 更能抑制噪声放大。
    cfo_correct=True,  # 先估计并校正残余载波频偏，再做 OFDM 解调。
)  # 返回的 rx_symbols 是均衡后的软符号；rx_bits 是按最近星座点得到的硬判决比特。

print(f"CFO estimate:  {result.cfo_est_hz:.1f} Hz  (true: {cfg.cfo_hz} Hz)")  # 与配置中注入的真实频偏对比，验证估计精度。
print(f"Noise var est: {result.noise_var_est:.2e}")  # MMSE 均衡所用的噪声方差估计。
print(f"EVM RMS:       {result.evm_rms:.3f}%")  # EVM=100×sqrt(mean(|RX-TX|²)/mean(|TX|²))；越小越接近参考。
print(f"BER:           {result.ber:.2e}")  # BER=错误比特数/比较比特数；:.2e 使用科学计数法，0 BER 不代表 EVM 必须为 0。
print(f"RX symbols:    {len(result.rx_symbols)}")  # 这里只统计数据 QAM 符号，不包括 DMRS 或时域 CP 采样点。

# %% 10. RX constellation after equalisation
# 第 10 步：观察软符号散布；不能先硬判决再画图，否则真实误差会被“吸附”到理想点而隐藏。
fig, ax = plt.subplots(figsize=(6, 6))  # 创建方形画布。
ax.scatter(result.rx_symbols.real, result.rx_symbols.imag,  # 均衡后的 I、Q 是散点坐标。
           s=2, alpha=0.3, c="steelblue", label="RX equalised")  # 用小而半透明的蓝点显示噪声散布。
ax.scatter(ref.real, ref.imag, s=60, c="red", marker="+", linewidths=2, label="Ideal")  # 理想点作为位置参考，不覆盖接收数据。
ax.set_title(f"RX Constellation — {cfg.modulation}, SNR={cfg.snr_db}dB\n"  # \n 在图标题中换行。
             f"EVM={result.evm_rms:.2f}%, BER={result.ber:.1e}")  # 括号内相邻字符串自动拼接；第二行显示误差指标。
ax.set_xlabel("I"); ax.set_ylabel("Q")  # 星座坐标轴分别为实部与虚部。
ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend()  # 保持几何比例，方便观察旋转、缩放和散布。
plt.show()  # 显示图形；降低 SNR 后通常可以看到更大的点云。

# %% 11. EVM per OFDM symbol
# 第 11 步：按时间分组看数据 EVM，可发现某个 OFDM 符号比其他符号更差。
fig, ax = plt.subplots(figsize=(10, 4))  # 创建柱状图。
ax.bar(range(len(result.evm_per_symbol)), result.evm_per_symbol, color="steelblue")  # 每根柱子对应一个 OFDM 符号的数据 RE 的 RMS EVM。
ax.axhline(result.evm_rms, color="red", linestyle="--", label=f"RMS = {result.evm_rms:.2f}%")  # 画全体数据的 RMS EVM；它不是各柱高的简单算术平均。
ax.set_title("EVM per OFDM Symbol")  # 这里的 Symbol 是 OFDM 符号，不是单个 QAM 数据点。
ax.set_xlabel("OFDM Symbol Index"); ax.set_ylabel("EVM (%)")  # 横轴为 OFDM 符号索引，纵轴为百分比误差。
ax.legend(); ax.grid(True, alpha=0.3)  # 显示红色参考线的含义和淡网格。
plt.tight_layout(); plt.show()  # 显示按符号统计的误差。

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

# %% 12. Real VSA 89600 measurement & comparison
# 第 12 步：调用真实 89600 VSA 软件分析新生成的 10 ms IQ 记录；需要已安装并授权的 5G NR 测量功能，不会退回模拟结果。
# 此处使用独立生成的 VSA 兼容记录，不使用前面步骤中带 CFO 和多径的 2 ms time_signal。
import sys
from datetime import datetime
from pathlib import Path

project_root = Path.cwd()
if not (project_root / "compare.py").is_file():
    project_root = project_root / "nr5g_demod"
if not (project_root / "compare.py").is_file():
    raise FileNotFoundError("Open the nr5g_demod project folder before running this cell")
project_root = project_root.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from compare import run_full_comparison, correlate_results

vsa_output_dir = project_root / "results" / f"real_vsa_step12_{datetime.now():%Y%m%d_%H%M%S}"
real_py_result, vsa_result, vsa_report = run_full_comparison(
    use_real_vsa=True,
    snr_db=cfg.snr_db,
    modulation=cfg.modulation,
    mu=cfg.mu,
    output_dir=str(vsa_output_dir),
    seed=cfg.seed,
    fft_window_offset=cfg.fft_window_offset,
)
print(f"Real VSA report: {vsa_report}")

# %% 13. Correlate Python vs VSA 89600
# 第 13 步：先对齐符号序列，再补偿统一的相位/幅度差，最后统计残差和相关系数。
corr = correlate_results(real_py_result, vsa_result, cfg)  # 使用真实 VSA 捕获对应的 Python 结果；cfg 的两个时隙对应 VSA 分析的一个子帧。

print(f"Symbol offset:      {corr.sample_offset}")  # 名字虽叫 sample_offset，这里单位是数据 QAM 符号；正值表示 VSA 序列相对延迟。
print(f"Phase offset:       {np.degrees(corr.phase_offset_rad):.2f}°")  # 将估计的整体相位偏移从弧度转换成角度。
print(f"Amplitude ratio:    {corr.amplitude_ratio:.4f}")  # 比值为匹配区间内 VSA 符号 RMS 幅度/Python 符号 RMS 幅度。
print(f"XCorr peak:         {corr.correlation_peak:.6f}")  # 延迟搜索得到的归一化复相关峰值，不是未经归一化的乘积和。
print(f"Symbol correlation: {corr.symbol_correlation:.6f}")  # 补偿后的相关幅度，范围约为 0–1；常量相位/增益差不会降低该指标。
print(f"EVM of difference:  {corr.evm_of_difference:.3f}%")  # 两个接收结果间的归一化差异，不是各自相对于发射符号的 EVM。

# %% 14. Correlation plots — constellation overlay & error scatter
# 第 14 步：用三种图观察真实 VSA 与 Python 的对齐结果。
fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 左：星座叠加；中：复数差；右：前若干符号的幅度。

ax = axes[0]  # 选择左侧子图；后续 ax 调用都作用于该子图。
ax.scatter(corr.aligned_py_symbols.real, corr.aligned_py_symbols.imag,  # 只画已经按延迟匹配的 Python 符号。
           s=1, alpha=0.3, c="steelblue", label="Python")  # 蓝色点标识 Python 结果。
ax.scatter(corr.aligned_vsa_symbols.real, corr.aligned_vsa_symbols.imag,  # 画经整体幅度和相位补偿后的真实 VSA 符号。
           s=1, alpha=0.3, c="darkorange", label="VSA (aligned)")  # 橙色可能覆盖蓝色；重合本身不说明两套算法独立。
ax.set_title(f"Constellation Overlay — ρ={corr.symbol_correlation:.4f}")  # 标题只保留四位小数，显示 1.0000 不保证数学上完全相等。
ax.set_xlabel("I"); ax.set_ylabel("Q")  # 星座的复平面坐标。
ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)  # 等比例并标明两种点的来源。

ax = axes[1]  # 切换到中间子图。
err = corr.aligned_vsa_symbols - corr.aligned_py_symbols  # err 在这里改为一维复数残差，不再是第 8 步的二维幅度误差。
ax.scatter(err.real, err.imag, s=1, alpha=0.3, c="red")  # 若两者完全相同，差值点应集中在原点。
ax.set_title(f"Symbol Error — EVM={corr.evm_of_difference:.3f}%")  # 展示接收结果之间的差异 EVM。
ax.set_xlabel("ΔI"); ax.set_ylabel("ΔQ")  # Δ 表示两侧在 I/Q 分量上的差值。
ax.set_aspect("equal"); ax.grid(True, alpha=0.3)  # 等比例显示残差方向和大小。

ax = axes[2]  # 切换到右侧子图。
n_show = min(200, len(corr.aligned_py_symbols))  # 最多画 200 个数据符号；min 防止数据不足时越界。
ax.plot(np.abs(corr.aligned_py_symbols[:n_show]), label="Python", alpha=0.7)  # 查看 Python 符号的幅度随数据索引变化。
ax.plot(np.abs(corr.aligned_vsa_symbols[:n_show]), label="VSA", alpha=0.7)  # 叠加真实 VSA 幅度；该图不显示相位差。
ax.set_title("Symbol Magnitude (first 200)")  # 截取短区间便于看清两条曲线是否同步。
ax.set_xlabel("Symbol Index"); ax.set_ylabel("|symbol|")  # 横轴为扁平化后的数据 QAM 索引，并非 OFDM 符号行号。
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)  # 标注曲线来源。
plt.tight_layout(); plt.show()  # 显示三幅相关分析图。

# %% 15. Per-OFDM-symbol and per-subcarrier correlation
# 第 15 步：把整体相关性拆成“随时间”和“随频率”的视角，以定位局部不一致。
fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 左图按 OFDM 符号分组，右图按子载波分组。

ax = axes[0]  # 选择时间方向的相关性图。
ax.bar(range(len(corr.per_symbol_corr)), corr.per_symbol_corr, color="steelblue")  # 每个 OFDM 符号内，用匹配数据 RE 计算归一化相关。
ax.axhline(corr.symbol_correlation, color="red", linestyle="--",  # 用整体相关作为水平参考线。
           label=f"Overall ρ={corr.symbol_correlation:.4f}")  # 为该参考线设置图例文字。
ax.set_title("Correlation per OFDM Symbol")  # 未匹配到的 OFDM 符号可为 NaN，表示没有比较数据，不是相关性为零。
ax.set_xlabel("OFDM Symbol Index"); ax.set_ylabel("|ρ|")  # |ρ| 表示复相关系数的幅度。
ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)  # 留出顶部空间；接近 1 的微小差异在此尺度下不明显。

ax = axes[1]  # 选择频率方向的相关性图。
ax.plot(corr.per_subcarrier_corr, linewidth=0.8, color="steelblue")  # 汇总同一子载波在多个符号时段的数据，再用双方能量归一化。
ax.set_title("Correlation per Subcarrier")  # 可以观察某些频率位置是否特别不一致。
ax.set_xlabel("Subcarrier Index"); ax.set_ylabel("Normalised correlation")  # 横轴是有效子载波索引，不是所有 FFT 点。
ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)  # 使用 0–1 的相关尺度并显示淡网格。
plt.tight_layout(); plt.show()  # 显示分组相关图。

# %% 16. Real VSA comparison report
# 第 16 步：显示第 12 步真实 VSA 测量已生成并保存的报告。
report = vsa_report
print(report)

# %% 17. Sweep SNR — ZF vs MMSE
# 第 17 步：扫描 SNR，同时对比 ZF 与 MMSE 均衡器，以及模拟 VSA 的 EVM；这是多次独立调用，不是同一接收机的时间序列。
from vsa_89600 import simulate_vsa_result

snr_range = [10, 15, 20, 25, 30, 40]  # 待测试的时域采样 SNR，单位 dB，不是 Eb/N0。
evm_zf_list, evm_mmse_list, evm_vsa_list = [], [], []  # 分别收集 ZF、MMSE 和模拟 VSA 的 EVM。

for snr in snr_range:  # 逐个取出 SNR；缩进的语句都会重复执行。
    c = NR5GConfig(mu=1, bw_mhz=20, n_rb=51, modulation="64QAM",  # c 是本次扫描的独立配置，不会修改前面的 cfg。
                   n_slots=2, snr_db=snr, cfo_hz=150, seed=42,  # 保持调制、频偏和种子不变，仅改变噪声强度，便于公平比较。
                   channel_taps=[1.0, 0.3 + 0.1j, 0.05], channel_delays=[0, 3, 7])  # 同样注入 3 径多径信道。
    t = generate_nr5g_waveform(c)  # t 与 tx 一样是字典，但对应当前 SNR 的波形。
    r_zf = demodulate_nr5g(t["time_signal"], c,  # 用 ZF 均衡解调同一条波形。
                           tx_bits=t["tx_bits"], tx_symbols=t["tx_symbols"],
                           tx_grid=t["resource_grid"], data_positions=t["data_positions"],
                           equaliser="zf")
    r_mmse = demodulate_nr5g(t["time_signal"], c,  # 用 MMSE 均衡解调同一条波形，便于直接比较。
                             tx_bits=t["tx_bits"], tx_symbols=t["tx_symbols"],
                             tx_grid=t["resource_grid"], data_positions=t["data_positions"],
                             equaliser="mmse")
    v = simulate_vsa_result(t["time_signal"], c)  # 使用同一条波形生成模拟对照，不是另一台仪器的测量。
    evm_zf_list.append(r_zf.evm_rms)  # append 把一个标量追加到列表末尾，顺序与 snr_range 一致。
    evm_mmse_list.append(r_mmse.evm_rms)
    evm_vsa_list.append(v.evm_rms)  # 保存模拟指标，便于与 Python 曲线同图显示。
    print(f"SNR={snr:3d} dB → ZF={r_zf.evm_rms:.3f}%, MMSE={r_mmse.evm_rms:.3f}%, VSA={v.evm_rms:.3f}%")  # :3d 让整数占三列，便于对齐输出。

fig, ax = plt.subplots(figsize=(8, 5))  # 创建 SNR 扫描图。
ax.semilogy(snr_range, evm_zf_list, "^--", label="Python ZF")  # ^-- 表示三角点加虚线。
ax.semilogy(snr_range, evm_mmse_list, "o-", label="Python MMSE")  # o- 表示圆点加实线。
ax.semilogy(snr_range, evm_vsa_list, "s:", label="VSA 89600 (sim)")  # s: 表示方点加点线；这是模拟对照曲线。
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("EVM RMS (%)")  # SNR 越高表示相对噪声越小；EVM 通常随之下降。
ax.set_title("EVM vs SNR — ZF vs MMSE vs VSA"); ax.legend(); ax.grid(True, which="both", alpha=0.3)  # both 显示对数轴的主、次网格。
plt.tight_layout(); plt.show()  # 高 SNR 区，噪声主导的 EVM 通常每增加 10 dB 约下降到原来的 1/sqrt(10)。

# %% 18. Sweep modulation order
# 第 18 步：固定 30 dB SNR、150 Hz 频偏和多径信道，比较不同星座阶数；阶数越高，单位平均功率下点间距越小。
for mod in ["QPSK", "16QAM", "64QAM", "256QAM"]:  # 四种调制每个数据符号分别承载 2、4、6、8 比特。
    c = NR5GConfig(mu=1, bw_mhz=20, n_rb=51, modulation=mod,  # 本轮仅选择不同的调制阶数。
                   n_slots=2, snr_db=30, cfo_hz=150, seed=42,  # 固定参数便于比较，但不同阶数消耗的随机比特数不同，噪声 realization 不保证相同。
                   channel_taps=[1.0, 0.3 + 0.1j, 0.05], channel_delays=[0, 3, 7])
    t = generate_nr5g_waveform(c)  # 根据新的每符号比特数生成数据、资源网格和 IQ。
    r = demodulate_nr5g(t["time_signal"], c,  # 接收机必须使用对应的调制配置进行星座判决。
                        tx_bits=t["tx_bits"], tx_symbols=t["tx_symbols"],  # 本轮发射参考不能使用上一轮的参考。
                        tx_grid=t["resource_grid"], data_positions=t["data_positions"],  # 保持数据 RE 顺序与发射端一致。
                        equaliser="mmse")  # 继续使用 MMSE 均衡。
    print(f"{mod:>6s}: EVM={r.evm_rms:.3f}%, BER={r.ber:.1e}, CFO_est={r.cfo_est_hz:.1f}Hz")  # >6s 表示字符串右对齐；相近 EVM 下，高阶调制仍可能有更多误码。

# %% 19. Wi-Fi 7-inspired OFDM simulation (not an EHT packet)
# 第 19 步：Wi-Fi 7 风格的未编码 OFDM 教学模型，不是可直接与商用设备互通的 EHT 数据包。
# 使用通用导频/训练符号布局，没有实现完整 EHT 帧、标准 RU 映射、FEC、MIMO 或多链路操作。
import matplotlib.pyplot as plt  # 再次导入是为了让本单元可独立运行；重复导入不会重建整个模块。
from wireless_phy import WiFi7Config, generate_wifi7_waveform, demodulate_wifi7, plot_wireless_result  # 分别导入配置、发射、接收和通用绘图入口。

wifi_cfg = WiFi7Config(bandwidth_mhz=20, modulation="4096QAM", snr_db=50)  # 4096-QAM 每点携带 12 比特；密集星座需要较高 SNR。
wifi_waveform = generate_wifi7_waveform(wifi_cfg)  # 返回波形字典；先发送已知训练符号，再发送含梳状导频的数据 OFDM 符号。
wifi_result = demodulate_wifi7(wifi_waveform["time_signal"], wifi_cfg,  # FFT 后先用训练估计信道，再利用导频修正各数据符号的整体增益。
                               wifi_waveform["tx_bits"], wifi_waveform["tx_symbols"])  # 已知数据只用于 BER/EVM 统计，不帮助接收机猜比特。
print(f"Wi-Fi OFDM model: EVM={wifi_result.evm_rms:.4f}%, BER={wifi_result.ber:.2e}")  # 输出软星座误差与硬判决误码率。
wifi_figure = plot_wireless_result(wifi_waveform, wifi_result)  # 生成 IQ、功率谱、QAM 星座和软符号误差四幅子图；函数不主动 show。
plt.show()  # 显示 Wi-Fi 图；模型假设采样起点已对齐，不执行真实包捕获。

# %% 20. Bluetooth LE GFSK simulation (uncoded, aligned payload)
# 第 20 步：蓝牙 LE 用 GFSK，以平滑后的瞬时频率变化表示比特，而不是使用 QAM 星座点。
# 本模型不含包同步、白化、CRC、跳频、LE Coded 或 BR/EDR；默认无剩余载波频偏。
import matplotlib.pyplot as plt  # 保证此单元能够独立绘图。
from wireless_phy import BluetoothConfig, generate_bluetooth_waveform, demodulate_bluetooth, plot_wireless_result  # 导入 LE 配置和 GFSK 收发函数。

bluetooth_cfg = BluetoothConfig(phy="LE2M", snr_db=20)  # LE2M 的符号率是 2 Msymbol/s；每个符号承载一个未编码比特。
bluetooth_waveform = generate_bluetooth_waveform(bluetooth_cfg)  # 比特经高斯滤波变成频偏，再积分为连续相位，生成恒包络复数信号。
bluetooth_result = demodulate_bluetooth(bluetooth_waveform["time_signal"], bluetooth_cfg,  # 相邻 IQ 的相位差给出频率估计，再按比特周期平均并判决。
                                       bluetooth_waveform["tx_bits"])  # 参考比特仅用于计算 BER；无需 QAM 参考星座。
print(f"Bluetooth GFSK model: BER={bluetooth_result.ber:.2e}; QAM-style EVM is not applicable")  # 这里 evm_rms 为 None；不能套用 QAM 星座 EVM。
bluetooth_figure = plot_wireless_result(bluetooth_waveform, bluetooth_result)  # 展示 IQ、功率谱、鉴频输出及归一化比特判决量。
plt.show()  # 观察频率如何随 0/1 改变；接收判决量不是 QAM 符号。

# %% 21. UWB BPM-BPSK pulse simulation (not an IEEE 802.15.4z packet)
# 第 21 步：用脉冲突发的早/晚位置承载一比特，用正/负极性承载另一比特。
# 这是 BPM-BPSK 教学模型，不含标准帧、扩频码、跳时、STS、安全测距或完整 IEEE 802.15.4z 功能。
import matplotlib.pyplot as plt  # 保证此单元可以独立运行并显示图形。
from wireless_phy import UWBConfig, generate_uwb_waveform, demodulate_uwb, plot_wireless_result  # 导入脉冲模型的配置、发生器、匹配滤波接收器和绘图函数。

uwb_cfg = UWBConfig(snr_db=20)  # 其余参数取默认值，包括 499.2 MHz 码片率；采样率还要乘每码片采样点数。
uwb_waveform = generate_uwb_waveform(uwb_cfg)  # 先放一个已知训练突发，再按每对比特选择早/晚窗口和脉冲极性。
uwb_result = demodulate_uwb(uwb_waveform["time_signal"], uwb_cfg,  # 在早、晚窗口分别与脉冲模板做匹配滤波，再用训练估计的增益校正。
                           uwb_waveform["tx_bits"], uwb_waveform["tx_symbols"])  # 参考符号是早/晚软幅度对，不是 QAM 点；参考仅用于误差测量。
print(f"UWB pulse model: matched-filter EVM={uwb_result.evm_rms:.4f}%, BER={uwb_result.ber:.2e}")  # 这是模型定义的匹配滤波 EVM，不是标准一致性指标。
uwb_figure = plot_wireless_result(uwb_waveform, uwb_result)  # 绘制 IQ、谱、早/晚幅度散点和误差；理想幅度对为 (±1,0) 或 (0,±1)。
plt.show()  # 显示最后一组图；不同协议的采样 SNR 和 EVM 定义不能直接横向等同。
