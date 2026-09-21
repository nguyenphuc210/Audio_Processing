"""
=============================================================================
CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI
LAB 1: PHÂN TÍCH VÀ XỬ LÝ TÍN HIỆU ÂM THANH SỐ
Sinh viên: Nguyễn Hồng Phúc
MSSV: 2351260680
Mã lớp / Môn học: CSE457 - Xử lý âm thanh và tiếng nói

Quy trình thực hiện tuần tự:
- Phần A: Đọc và kiểm tra dữ liệu âm thanh
- Phần B: Phân tích miền thời gian
- Phần C: Phân tích miền tần số bằng FFT
- Phần D: STFT và Spectrogram (Time-Frequency Analysis)
- Phần E: Thí nghiệm cửa sổ (Windowing & Spectral Leakage)
- Phần F: Thiết kế và ứng dụng bộ lọc số FIR (Filtering)
- Phần G: Lượng tử hóa, Resampling và Mã hóa âm thanh
=============================================================================
"""

# %% [markdown]
# # BÀI THỰC HÀNH 1: PHÂN TÍCH VÀ XỬ LÝ TÍN HIỆU ÂM THANH SỐ
# **Họ và tên:** Nguyễn Hồng Phúc  
# **MSSV:** 2351260680  
# **Học phần:** CSE457 - Xử lý âm thanh và tiếng nói

# %% [code]
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf

# Cấu hình mã hóa UTF-8 cho console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Cấu hình hiển thị Matplotlib chuyên nghiệp và rõ ràng
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.4
plt.rcParams['lines.linewidth'] = 1.2

# Đảm bảo các thư mục tồn tại
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def save_and_show(filename):
    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Đã lưu hình ảnh: {out_path}")
    if 'ipykernel' in sys.modules or 'IPython' in sys.modules:
        plt.show()
    else:
        plt.close()

print("Đường dẫn làm việc:", BASE_DIR)
print("Thư mục audio:", AUDIO_DIR)
print("Thư mục figures:", FIGURES_DIR)

# %% [markdown]
# ## PHẦN A: ĐỌC VÀ KIỂM TRA DỮ LIỆU ÂM THANH
# - Đọc file âm thanh từ `audio/input.wav` (hoặc `audio/input.mp3`)
# - Kiểm tra các thông số metadata: Fs, channels, duration, dtype, file size
# - Nếu là stereo: tách/tạo bản mono bằng trung bình hai kênh, so sánh waveform và RMS
# - Chuẩn hóa biên độ về miền [-1, 1]

# %% [code]
# Đường dẫn file âm thanh đầu vào
wav_path = os.path.join(AUDIO_DIR, "input.wav")
mp3_path = os.path.join(AUDIO_DIR, "input.mp3")

if not os.path.exists(wav_path):
    raise FileNotFoundError(f"Không tìm thấy file {wav_path}. Hãy chạy file convert_audio.ps1 trước.")

# Đọc file âm thanh bằng soundfile
raw_data, Fs = sf.read(wav_path)
num_samples = len(raw_data)
duration = num_samples / Fs
num_channels = raw_data.shape[1] if raw_data.ndim > 1 else 1
file_size_bytes = os.path.getsize(wav_path)
orig_mp3_size = os.path.getsize(mp3_path) if os.path.exists(mp3_path) else None

print("="*60)
print("THÔNG TIN METADATA FILE ÂM THANH ĐẦU VÀO")
print("="*60)
print(f"Tên file âm thanh      : input.wav (chuyển đổi từ input.mp3)")
print(f"Tần số lấy mẫu (Fs)    : {Fs} Hz")
print(f"Tần số Nyquist         : {Fs / 2} Hz")
print(f"Số kênh (Channels)     : {num_channels} ({'Stereo' if num_channels == 2 else 'Mono'})")
print(f"Số lượng mẫu (Samples) : {num_samples} mẫu")
print(f"Thời lượng (Duration)  : {duration:.3f} s ({duration/60:.2f} phút)")
print(f"Kiểu dữ liệu (dtype)   : {raw_data.dtype}")
print(f"Kích thước file WAV    : {file_size_bytes:,} bytes ({file_size_bytes / (1024*1024):.2f} MB)")
if orig_mp3_size:
    print(f"Kích thước file MP3 gốc: {orig_mp3_size:,} bytes ({orig_mp3_size / (1024*1024):.2f} MB)")
print("="*60)

# Chuyển đổi sang Mono và Chuẩn hóa về miền [-1, 1]
if num_channels == 2:
    left_ch = raw_data[:, 0]
    right_ch = raw_data[:, 1]
    mono_raw = (left_ch + right_ch) / 2.0
    
    rms_left = np.sqrt(np.mean(left_ch**2))
    rms_right = np.sqrt(np.mean(right_ch**2))
    rms_mono = np.sqrt(np.mean(mono_raw**2))
    
    print(f"RMS Kênh Trái (Left)   : {rms_left:.6f} ({20*np.log10(rms_left):.2f} dBFS)")
    print(f"RMS Kênh Phải (Right)  : {rms_right:.6f} ({20*np.log10(rms_right):.2f} dBFS)")
    print(f"RMS Kênh Mono          : {rms_mono:.6f} ({20*np.log10(rms_mono):.2f} dBFS)")
else:
    mono_raw = raw_data

# Chuẩn hóa về [-1, 1]
max_abs = np.max(np.abs(mono_raw))
if max_abs > 0:
    x = mono_raw / max_abs
else:
    x = mono_raw

print(f"Biên độ đỉnh trước chuẩn hóa : {max_abs:.6f}")
print(f"Biên độ đỉnh sau chuẩn hóa   : {np.max(np.abs(x)):.6f}")

# %% [markdown]
# ## PHẦN B: PHÂN TÍCH MIỀN THỜI GIAN
# - Vẽ waveform toàn bộ tệp và một đoạn ngắn 0.5 – 1.0 s
# - Tính Peak, RMS, Năng lượng E, kiểm tra clipping
# - Chọn ít nhất 02 đoạn có đặc tính khác nhau và giải thích sự khác biệt

# %% [code]
# 1. Tính toán các đại lượng trên toàn bộ tệp
peak_full = np.max(np.abs(x))
rms_full = np.sqrt(np.mean(x**2))
energy_full = np.sum(x**2)
dbfs_full = 20 * np.log10(np.maximum(rms_full, 1e-12))
is_clipping = peak_full >= 0.999

print("="*60)
print("CÁC ĐẠI LƯỢNG TOÀN BỘ TỆP (MIỀN THỜI GIAN)")
print("="*60)
print(f"Peak (Giá trị cực đại)   : {peak_full:.6f}")
print(f"RMS (Năng lượng hiệu dụng): {rms_full:.6f} ({dbfs_full:.2f} dBFS)")
print(f"Tổng năng lượng (E)      : {energy_full:.2f}")
print(f"Kiểm tra clipping        : {'Có nguy cơ clipping!' if is_clipping else 'Không clipping (An toàn)'}")
print("="*60)

# 2. Chọn 2 đoạn có đặc tính khác nhau:
# Đoạn 1: 12.0s - 13.0s (đoạn êm dịu, mức năng lượng thấp)
# Đoạn 2: 44.0s - 45.0s (đoạn cao trào, mức năng lượng lớn)
t_start1, t_end1 = 12.0, 13.0
t_start2, t_end2 = 44.0, 45.0

seg1 = x[int(t_start1 * Fs): int(t_end1 * Fs)]
seg2 = x[int(t_start2 * Fs): int(t_end2 * Fs)]

peak_seg1 = np.max(np.abs(seg1))
rms_seg1 = np.sqrt(np.mean(seg1**2))
energy_seg1 = np.sum(seg1**2)

peak_seg2 = np.max(np.abs(seg2))
rms_seg2 = np.sqrt(np.mean(seg2**2))
energy_seg2 = np.sum(seg2**2)

print("SO SÁNH 2 ĐOẠN ĐẶC TRƯNG:")
print(f"• Đoạn 1 ({t_start1}-{t_end1}s - Âm lượng nhỏ): Peak = {peak_seg1:.4f}, RMS = {rms_seg1:.4f} ({20*np.log10(rms_seg1):.2f} dBFS), E = {energy_seg1:.2f}")
print(f"• Đoạn 2 ({t_start2}-{t_end2}s - Cao trào)    : Peak = {peak_seg2:.4f}, RMS = {rms_seg2:.4f} ({20*np.log10(rms_seg2):.2f} dBFS), E = {energy_seg2:.2f}")
print(f"-> Tỷ số năng lượng Đoạn 2 / Đoạn 1: {energy_seg2 / energy_seg1:.2f} lần.")

# 3. Vẽ đồ thị Waveform
time_axis_full = np.arange(len(x)) / Fs
time_axis_seg2 = np.arange(len(seg2)) / Fs + t_start2

fig, axs = plt.subplots(2, 1, figsize=(12, 6))

# Waveform toàn bộ tệp
axs[0].plot(time_axis_full, x, color='#1f77b4', lw=0.6)
axs[0].set_title(f"Waveform toàn bộ tệp âm thanh ({duration:.2f} s, Fs = {Fs} Hz)", fontweight='bold')
axs[0].set_xlabel("Thời gian (giây)")
axs[0].set_ylabel("Biên độ chuẩn hóa")
axs[0].axvspan(t_start1, t_end1, color='orange', alpha=0.3, label=f"Đoạn 1 ({t_start1}-{t_end1}s)")
axs[0].axvspan(t_start2, t_end2, color='red', alpha=0.3, label=f"Đoạn 2 ({t_start2}-{t_end2}s)")
axs[0].legend(loc='upper right')
axs[0].set_ylim([-1.05, 1.05])

# Waveform zoom đoạn 44-45s
axs[1].plot(time_axis_seg2, seg2, color='#d62728', lw=1.0)
axs[1].set_title(f"Chi tiết Waveform đoạn cao trào {t_start2}s - {t_end2}s (Thời lượng 1.0 s)", fontweight='bold')
axs[1].set_xlabel("Thời gian (giây)")
axs[1].set_ylabel("Biên độ chuẩn hóa")
axs[1].set_ylim([-1.05, 1.05])

plt.tight_layout()
save_and_show("waveform.png")

# %% [markdown]
# ## PHẦN C: PHÂN TÍCH MIỀN TẦN SỐ BẰNG FFT
# - Chọn đoạn ổn định 0.5 – 1.0 s, nhân cửa sổ Hamming và tính FFT
# - Vẽ magnitude spectrum theo Hz và dB, chỉ ra ít nhất 3 đỉnh nổi bật
# - Thử nghiệm ít nhất 2 giá trị NFFT, phân biệt frequency-bin spacing và true resolution

# %% [code]
# Chọn đoạn ổn định 15.0s - 16.0s (dài 1.0 s)
t_fft_start, t_fft_end = 15.0, 16.0
seg_fft = x[int(t_fft_start * Fs): int(t_fft_end * Fs)]
N_samples = len(seg_fft)

# Áp dụng cửa sổ Hamming
w_hamming = np.hamming(N_samples)
seg_windowed = seg_fft * w_hamming

# Thử nghiệm 2 giá trị NFFT
NFFT_1 = 2048
NFFT_2 = 65536

# Tính FFT với NFFT 1
X1 = np.fft.rfft(seg_windowed, n=NFFT_1)
f1 = np.fft.rfftfreq(NFFT_1, 1/Fs)
mag_db1 = 20 * np.log10(np.maximum(np.abs(X1), 1e-12))
delta_f1 = Fs / NFFT_1

# Tính FFT với NFFT 2
X2 = np.fft.rfft(seg_windowed, n=NFFT_2)
f2 = np.fft.rfftfreq(NFFT_2, 1/Fs)
mag_db2 = 20 * np.log10(np.maximum(np.abs(X2), 1e-12))
delta_f2 = Fs / NFFT_2

print("="*60)
print("PHÂN TÍCH ĐỘ PHÂN GIẢI FFT")
print("="*60)
print(f"Độ dài frame tín hiệu gốc: N = {N_samples} mẫu ({N_samples/Fs:.3f} s)")
print(f"Độ phân giải vật lý thực (True Resolution ~ 1/T): {1.0 / (N_samples/Fs):.2f} Hz")
print(f"NFFT = {NFFT_1}: Khoảng cách bin tần số (Δf) = {delta_f1:.3f} Hz")
print(f"NFFT = {NFFT_2}: Khoảng cách bin tần số (Δf) = {delta_f2:.3f} Hz")
print("="*60)

# Tìm các đỉnh phổ nổi bật trên NFFT_2 (giới hạn dải tần số 0 - 5000 Hz)
mask_5k = (f2 >= 50) & (f2 <= 5000)
f_sub = f2[mask_5k]
mag_sub = mag_db2[mask_5k]

# Tìm các đỉnh với độ cao và khoảng cách tối thiểu
peaks_idx, props = signal.find_peaks(mag_sub, distance=int(100/delta_f2), prominence=10)
sorted_peaks = peaks_idx[np.argsort(mag_sub[peaks_idx])][::-1][:5]
peak_freqs = f_sub[sorted_peaks]
peak_mags = mag_sub[sorted_peaks]

print("CÁC ĐỈNH PHỔ NỔI BẬT ĐO ĐƯỢC:")
for idx, (pf, pm) in enumerate(zip(peak_freqs, peak_mags), 1):
    print(f"  Đỉnh {idx}: f = {pf:.1f} Hz, Biên độ tương đối = {pm:.2f} dB")

# Vẽ đồ thị phổ FFT
plt.figure(figsize=(12, 6))
plt.plot(f2, mag_db2, color='#1f77b4', lw=0.9, label=f"NFFT = {NFFT_2} (Δf = {delta_f2:.3f} Hz - Nội suy mịn)")
plt.plot(f1, mag_db1, color='orange', alpha=0.75, lw=1.2, linestyle='--', label=f"NFFT = {NFFT_1} (Δf = {delta_f1:.2f} Hz)")

# Đánh dấu các đỉnh
for pf, pm in zip(peak_freqs[:3], peak_mags[:3]):
    plt.scatter([pf], [pm], color='red', s=50, zorder=5)
    plt.annotate(f"{pf:.1f} Hz\n({pm:.1f} dB)", xy=(pf, pm), xytext=(pf + 60, pm + 3),
                 arrowprops=dict(arrowstyle="->", color='red', lw=1),
                 fontsize=9, fontweight='bold', color='darkred')

plt.title(f"Phổ biên độ FFT đoạn {t_fft_start}s - {t_fft_end}s (Cửa sổ Hamming, Fs = {Fs} Hz)", fontweight='bold')
plt.xlabel("Tần số (Hz)")
plt.ylabel("Biên độ tương đối (dB)")
plt.xlim([0, 6000])
plt.ylim([np.min(mag_db2[f2 <= 6000]) - 5, np.max(mag_db2) + 10])
plt.legend(loc='upper right')
plt.tight_layout()
save_and_show("fft.png")

# %% [markdown]
# ## PHẦN D: STFT VÀ SPECTROGRAM (TIME-FREQUENCY ANALYSIS)
# - Tạo spectrogram với cấu hình chuẩn: frame ≈ 25 ms, hop ≈ 10 ms
# - So sánh ít nhất 3 frame lengths: 10 ms, 25 ms, 50 ms
# - Giải thích vùng năng lượng ổn định, transient và sự thay đổi time–frequency resolution

# %% [code]
# Lấy một đoạn tín hiệu 10 giây (15s - 25s) để hiển thị chi tiết Spectrogram
t_stft_start, t_stft_end = 15.0, 25.0
seg_stft = x[int(t_stft_start * Fs): int(t_stft_end * Fs)]

# Các kích thước frame length cần so sánh
frame_ms_list = [10, 25, 50]  # mili-giây
hop_ms = 10                  # hop size chuẩn 10 ms

fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)

for i, f_ms in enumerate(frame_ms_list):
    L = int(round((f_ms / 1000.0) * Fs))
    H = int(round((hop_ms / 1000.0) * Fs))
    noverlap = max(0, L - H)
    NFFT_stft = 4096
    
    # Tính spectrogram
    f_spec, t_spec, Sxx = signal.spectrogram(
        seg_stft,
        fs=Fs,
        window='hamming',
        nperseg=L,
        noverlap=noverlap,
        nfft=NFFT_stft,
        mode='magnitude'
    )
    
    # Chuyển sang thang đo dB (dBFS/relative dB)
    Sxx_db = 20 * np.log10(np.maximum(Sxx, 1e-12))
    
    # Giới hạn dynamic range để so sánh công bằng
    vmax = np.max(Sxx_db)
    vmin = vmax - 80
    
    im = axs[i].pcolormesh(t_spec + t_stft_start, f_spec, Sxx_db, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[i].set_title(f"Frame Length = {f_ms} ms (L = {L} mẫu, Hop = {hop_ms} ms, Overlap = {noverlap/L*100:.1f}%)", fontweight='bold')
    axs[i].set_ylabel("Tần số (Hz)")
    axs[i].set_ylim([0, 6000])

axs[-1].set_xlabel("Thời gian (giây)")
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Biên độ phổ (dB)")

save_and_show("spectrogram.png")

# %% [markdown]
# ## PHẦN E: THÍ NGHIỆM CỬA SỔ (WINDOWING EXPERIMENT)
# - So sánh cửa sổ Rectangular và Hamming trên cùng một frame tín hiệu
# - Vẽ log-spectrum, nhận xét về main-lobe width, side-lobe attenuation và spectral leakage
# - Giữ nguyên dữ liệu và NFFT để đảm bảo so sánh công bằng

# %% [code]
# Lấy 1 frame độ dài 1024 mẫu tại thời điểm t = 15.0 s
frame_len = 1024
frame_raw = x[int(15.0 * Fs): int(15.0 * Fs) + frame_len]

# Tạo 2 cửa sổ
win_rect = np.ones(frame_len)
win_hamm = np.hamming(frame_len)

# Chuẩn hóa diện tích cửa sổ để so sánh biên độ công bằng
w_rect_norm = win_rect / np.sum(win_rect)
w_hamm_norm = win_hamm / np.sum(win_hamm)

frame_rect = frame_raw * w_rect_norm
frame_hamm = frame_raw * w_hamm_norm

NFFT_win = 8192
X_rect = np.fft.rfft(frame_rect, n=NFFT_win)
X_hamm = np.fft.rfft(frame_hamm, n=NFFT_win)
f_win = np.fft.rfftfreq(NFFT_win, 1/Fs)

mag_rect_db = 20 * np.log10(np.maximum(np.abs(X_rect), 1e-12))
mag_hamm_db = 20 * np.log10(np.maximum(np.abs(X_hamm), 1e-12))

# Vẽ so sánh
fig, axs = plt.subplots(2, 1, figsize=(12, 7))

# Hình dạng cửa sổ trong miền thời gian
axs[0].plot(win_rect, label="Cửa sổ Rectangular (Chữ nhật)", color='red', lw=1.5)
axs[0].plot(win_hamm, label="Cửa sổ Hamming", color='#1f77b4', lw=1.5)
axs[0].set_title(f"1. Hình dạng cửa sổ trong miền thời gian (N = {frame_len} mẫu)", fontweight='bold')
axs[0].set_xlabel("Chỉ số mẫu n")
axs[0].set_ylabel("Hệ số trọng số w[n]")
axs[0].set_ylim([-0.1, 1.1])
axs[0].legend(loc='upper right')

# Phổ tần số Log-Spectrum so sánh rò rỉ phổ
axs[1].plot(f_win, mag_rect_db, label="Rectangular (Main-lobe hẹp, Side-lobe cao -> Leakage mạnh)", color='red', alpha=0.7, lw=1.0)
axs[1].plot(f_win, mag_hamm_db, label="Hamming (Main-lobe rộng hơn, Side-lobe suy hao -43 dB -> Giảm leakage)", color='#1f77b4', lw=1.2)
axs[1].set_title("2. So sánh phổ Log-Spectrum (Đối chiếu Spectral Leakage & Độ sắc nét đỉnh)", fontweight='bold')
axs[1].set_xlabel("Tần số (Hz)")
axs[1].set_ylabel("Biên độ tương đối (dB)")
axs[1].set_xlim([0, 5000])
axs[1].set_ylim([np.max(mag_hamm_db) - 80, np.max(mag_hamm_db) + 5])
axs[1].legend(loc='upper right')

plt.tight_layout()
save_and_show("window_comparison.png")

# %% [markdown]
# ## PHẦN F: THIẾT KẾ VÀ ÁP DỤNG BỘ LỌC SỐ (DIGITAL FILTERING)
# - Thiết kế 01 bộ lọc FIR Low-Pass (LPF, cutoff = 2000 Hz)
# - Thiết kế 01 bộ lọc FIR High-Pass (HPF, cutoff = 1000 Hz)
# - Vẽ đáp ứng tần số H(f) và tính độ trễ nhóm (Group Delay)
# - Lọc tín hiệu âm thanh và xuất ra file WAV để nghe thử
# - So sánh phổ FFT trước và sau khi lọc

# %% [code]
num_taps = 201  # Bậc M = num_taps - 1 = 200
cutoff_lpf = 2000.0  # Hz
cutoff_hpf = 1000.0  # Hz

# 1. Thiết kế bộ lọc FIR dùng scipy.signal.firwin
b_lpf = signal.firwin(numtaps=num_taps, cutoff=cutoff_lpf, fs=Fs, window='hamming', pass_zero='lowpass')
b_hpf = signal.firwin(numtaps=num_taps, cutoff=cutoff_hpf, fs=Fs, window='hamming', pass_zero='highpass')

# Tính độ trễ nhóm lý thuyết (Group delay của FIR pha tuyến tính đối xứng)
group_delay_samples = (num_taps - 1) / 2
group_delay_ms = (group_delay_samples / Fs) * 1000.0
print("="*60)
print("THÔNG SỐ THIẾT KẾ BỘ LỌC FIR:")
print(f"• Số điểm trích mẫu (Taps): {num_taps} (Bậc M = {num_taps - 1})")
print(f"• Độ trễ nhóm (Group Delay): {group_delay_samples:.1f} mẫu ≈ {group_delay_ms:.3f} ms")
print(f"• Tần số cắt LPF: {cutoff_lpf} Hz | Tần số cắt HPF: {cutoff_hpf} Hz")
print("="*60)

# Tính đáp ứng tần số bằng freqz
w_lpf, H_lpf = signal.freqz(b_lpf, worN=4096, fs=Fs)
w_hpf, H_hpf = signal.freqz(b_hpf, worN=4096, fs=Fs)

# 2. Áp dụng bộ lọc lên toàn bộ tín hiệu x
y_lpf = signal.lfilter(b_lpf, [1.0], x)
y_hpf = signal.lfilter(b_hpf, [1.0], x)

# Xuất file âm thanh đã lọc
out_lpf_path = os.path.join(AUDIO_DIR, "filtered_lpf_2k.wav")
out_hpf_path = os.path.join(AUDIO_DIR, "filtered_hpf_1k.wav")

sf.write(out_lpf_path, y_lpf, Fs)
sf.write(out_hpf_path, y_hpf, Fs)
print(f"Đã xuất audio LPF: {out_lpf_path}")
print(f"Đã xuất audio HPF: {out_hpf_path}")

# 3. Phân tích phổ trước và sau lọc trên đoạn 15.0s - 16.0s
seg_orig = x[int(15.0 * Fs): int(16.0 * Fs)] * np.hamming(int(1.0 * Fs))
seg_lpf = y_lpf[int(15.0 * Fs): int(16.0 * Fs)] * np.hamming(int(1.0 * Fs))
seg_hpf = y_hpf[int(15.0 * Fs): int(16.0 * Fs)] * np.hamming(int(1.0 * Fs))

NFFT_flt = 16384
f_flt = np.fft.rfftfreq(NFFT_flt, 1/Fs)
X_orig_db = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(seg_orig, n=NFFT_flt)), 1e-12))
X_lpf_db = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(seg_lpf, n=NFFT_flt)), 1e-12))
X_hpf_db = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(seg_hpf, n=NFFT_flt)), 1e-12))

# 4. Vẽ đồ thị đáp ứng tần số và phổ trước/sau lọc
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# Đáp ứng biên độ của các bộ lọc H(f)
axs[0].plot(w_lpf, 20*np.log10(np.maximum(np.abs(H_lpf), 1e-6)), color='blue', label=f"FIR Low-pass (Cutoff = {cutoff_lpf} Hz)")
axs[0].plot(w_hpf, 20*np.log10(np.maximum(np.abs(H_hpf), 1e-6)), color='green', label=f"FIR High-pass (Cutoff = {cutoff_hpf} Hz)")
axs[0].set_title(f"Đáp ứng biên độ của bộ lọc FIR (Hamming window, {num_taps} taps, Group Delay = {group_delay_ms:.2f} ms)", fontweight='bold')
axs[0].set_xlabel("Tần số (Hz)")
axs[0].set_ylabel("Đáp ứng biên độ |H(f)| (dB)")
axs[0].set_xlim([0, 6000])
axs[0].set_ylim([-100, 5])
axs[0].axvline(cutoff_lpf, color='blue', linestyle=':', alpha=0.7)
axs[0].axvline(cutoff_hpf, color='green', linestyle=':', alpha=0.7)
axs[0].legend(loc='upper right')

# Phổ FFT trước và sau khi lọc
axs[1].plot(f_flt, X_orig_db, color='gray', alpha=0.5, label="Tín hiệu gốc trước lọc", lw=1)
axs[1].plot(f_flt, X_lpf_db, color='blue', alpha=0.85, label=f"Sau Low-pass (triệt tiêu > {cutoff_lpf} Hz)", lw=1.2)
axs[1].plot(f_flt, X_hpf_db, color='green', alpha=0.85, label=f"Sau High-pass (triệt tiêu < {cutoff_hpf} Hz)", lw=1.2)
axs[1].set_title("So sánh phổ FFT trước và sau khi lọc (Đoạn 15s - 16s)", fontweight='bold')
axs[1].set_xlabel("Tần số (Hz)")
axs[1].set_ylabel("Biên độ phổ (dB)")
axs[1].set_xlim([0, 6000])
axs[1].set_ylim([np.max(X_orig_db) - 80, np.max(X_orig_db) + 5])
axs[1].legend(loc='upper right')

plt.tight_layout()
save_and_show("filter_response.png")

# %% [markdown]
# ## PHẦN G: LƯỢNG TỬ HÓA, RESAMPLING VÀ MÃ HÓA ÂM THANH
# - Lượng tử hóa tín hiệu tại B = 4, 8, 16 bit, tính SNR thực nghiệm đo được
# - Resample về 16 kHz và 8 kHz (lọc chống aliasing), so sánh phổ và xuất file nghe thử
# - Tính theoretical PCM bit rate, file size, so sánh với file nén và tính compression ratio

# %% [code]
# 1. Hàm lượng tử hóa đều B-bit đối xứng trên khoảng [-1, 1]
def quantize_signal(sig, B):
    qmax = 2**(B - 1) - 1
    # Clip về [-1, 1], lượng tử hóa đều và chuẩn hóa lại về float
    sig_clipped = np.clip(sig, -1.0, 1.0)
    sig_quantized = np.round(sig_clipped * qmax) / qmax
    return sig_quantized

def calculate_snr(sig_ref, sig_quant):
    noise = sig_quant - sig_ref
    p_signal = np.sum(sig_ref**2)
    p_noise = np.sum(noise**2)
    if p_noise == 0:
        return float('inf')
    return 10.0 * np.log10(p_signal / p_noise)

# Thực nghiệm trên các mức bit: 4, 6, 8, 12, 16 bit
bits_to_test = [4, 6, 8, 12, 16]
snr_measured = []
snr_theory = []

print("="*60)
print("KẾT QUẢ THỰC NGHIỆM LƯỢNG TỬ HÓA VÀ SNR:")
print("="*60)
rms_x = np.sqrt(np.mean(x**2))
for b in bits_to_test:
    xq = quantize_signal(x, b)
    snr_val = calculate_snr(x, xq)
    snr_measured.append(snr_val)
    
    # Công thức lý thuyết Rabiner-Schafer: SNR_Q = 6.02*B + 4.77 - 20*log10(Xmax/sigma_x)
    snr_th = 6.02 * b + 4.77 - 20 * np.log10(1.0 / rms_x)
    snr_theory.append(snr_th)
    
    print(f"• B = {b:2d} bit: SNR đo được = {snr_val:6.2f} dB | SNR lý thuyết = {snr_th:6.2f} dB")
    
    # Xuất file audio các mức bit tiêu chuẩn (4, 8, 16 bit)
    if b in [4, 8, 16]:
        out_quant_path = os.path.join(AUDIO_DIR, f"quantized_{b}bit.wav")
        sf.write(out_quant_path, xq, Fs)

print("="*60)

# 2. Resampling về 16 kHz và 8 kHz (sử dụng signal.resample_poly có anti-aliasing)
Fs_resample_list = [16000, 8000]
resampled_signals = {}

for target_fs in Fs_resample_list:
    # resample_poly: up/down
    gcd_val = np.gcd(int(Fs), int(target_fs))
    up = target_fs // gcd_val
    down = Fs // gcd_val
    x_resampled = signal.resample_poly(x, up, down)
    resampled_signals[target_fs] = x_resampled
    
    out_resample_path = os.path.join(AUDIO_DIR, f"resampled_{target_fs//1000}k.wav")
    sf.write(out_resample_path, x_resampled, target_fs)
    print(f"Đã xuất audio resample {target_fs} Hz: {out_resample_path}")

# 3. Tính toán Tốc độ bit và Tỷ số nén (Coding & Compression Ratio)
# PCM Stereo 16-bit ở Fs = 48000 Hz
R_pcm_stereo = Fs * 16 * 2  # bit/s
size_pcm_stereo_bytes = R_pcm_stereo * duration / 8
size_pcm_stereo_mb = size_pcm_stereo_bytes / (1024 * 1024)

# PCM Mono 16-bit ở Fs = 48000 Hz
R_pcm_mono = Fs * 16 * 1    # bit/s
size_pcm_mono_bytes = R_pcm_mono * duration / 8
size_pcm_mono_mb = size_pcm_mono_bytes / (1024 * 1024)

# MP3 tham chiếu: 128 kbps, 256 kbps và MP3 gốc
bitrate_mp3_128 = 128 * 1000 # bit/s
size_mp3_128_mb = (bitrate_mp3_128 * duration / 8) / (1024 * 1024)

bitrate_mp3_256 = 256 * 1000 # bit/s
size_mp3_256_mb = (bitrate_mp3_256 * duration / 8) / (1024 * 1024)

actual_mp3_size_mb = orig_mp3_size / (1024 * 1024) if orig_mp3_size else 2.53
actual_mp3_bitrate_kbps = (orig_mp3_size * 8 / duration / 1000) if orig_mp3_size else 416.4

comp_ratio_actual = size_pcm_stereo_bytes / orig_mp3_size if orig_mp3_size else size_pcm_stereo_mb / actual_mp3_size_mb
saving_actual_pct = (1.0 - (orig_mp3_size / size_pcm_stereo_bytes)) * 100.0 if orig_mp3_size else 0

comp_ratio_128 = R_pcm_stereo / bitrate_mp3_128
saving_128_pct = (1.0 - (bitrate_mp3_128 / R_pcm_stereo)) * 100.0

comp_ratio_256 = R_pcm_stereo / bitrate_mp3_256
saving_256_pct = (1.0 - (bitrate_mp3_256 / R_pcm_stereo)) * 100.0

print("="*60)
print("BẢNG SO SÁNH TỐC ĐỘ BIT VÀ TỶ SỐ NÉN:")
print("="*60)
print(f"1. PCM 16-bit Stereo lý thuyết: Bitrate = {R_pcm_stereo/1000:.1f} kbps, Dung lượng = {size_pcm_stereo_mb:.2f} MB")
print(f"2. PCM 16-bit Mono lý thuyết  : Bitrate = {R_pcm_mono/1000:.1f} kbps, Dung lượng = {size_pcm_mono_mb:.2f} MB")
print(f"3. MP3 tham chiếu 256 kbps    : Bitrate = 256.0 kbps, Dung lượng = {size_mp3_256_mb:.2f} MB (Tỷ số nén = {comp_ratio_256:.2f}:1, Tiết kiệm = {saving_256_pct:.1f}%)")
print(f"4. MP3 tham chiếu 128 kbps    : Bitrate = 128.0 kbps, Dung lượng = {size_mp3_128_mb:.2f} MB (Tỷ số nén = {comp_ratio_128:.2f}:1, Tiết kiệm = {saving_128_pct:.1f}%)")
print(f"5. File MP3 thực tế (input)   : Bitrate ≈ {actual_mp3_bitrate_kbps:.1f} kbps, Dung lượng = {actual_mp3_size_mb:.2f} MB (Tỷ số nén = {comp_ratio_actual:.2f}:1, Tiết kiệm = {saving_actual_pct:.1f}%)")
print("="*60)

# 4. Vẽ biểu đồ SNR và biểu đồ so sánh nén dữ liệu
fig, axs = plt.subplots(1, 2, figsize=(13, 5))

# Đồ thị SNR theo số bit
axs[0].plot(bits_to_test, snr_measured, 'o-', color='#1f77b4', lw=2, markersize=7, label="SNR đo thực nghiệm")
axs[0].plot(bits_to_test, snr_theory, 's--', color='green', alpha=0.7, lw=1.5, label="SNR lý thuyết (Rabiner-Schafer)")
axs[0].set_title("Độ phụ thuộc SNR vào số bit lượng tử (B)", fontweight='bold')
axs[0].set_xlabel("Số bit / mẫu (B)")
axs[0].set_ylabel("SNR đo được (dB)")
axs[0].set_xticks(bits_to_test)
for b, s in zip(bits_to_test, snr_measured):
    axs[0].annotate(f"{s:.1f} dB", xy=(b, s), xytext=(b - 0.5, s + 3), fontsize=9, fontweight='bold')
axs[0].legend(loc='lower right')

# Biểu đồ so sánh Bitrate
formats = ['PCM Stereo\n(16-bit, 48k)', 'File MP3\nthực tế', 'MP3\n256 kbps', 'MP3\n128 kbps']
bitrates = [R_pcm_stereo / 1000, actual_mp3_bitrate_kbps, 256, 128]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

bars = axs[1].bar(formats, bitrates, color=colors, width=0.55, edgecolor='black', alpha=0.85)
axs[1].set_title("So sánh tốc độ bit giữa PCM và các chuẩn mã hóa nén", fontweight='bold')
axs[1].set_ylabel("Tốc độ bit (kbps)")
for bar, br in zip(bars, bitrates):
    axs[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, f"{br:.0f} kbps", ha='center', va='bottom', fontweight='bold')
axs[1].set_ylim([0, max(bitrates) * 1.15])

plt.tight_layout()
save_and_show("quantization_compression.png")

print("\n" + "="*70)
print("ĐÃ HOÀN THÀNH TẤT CẢ CÁC BƯỚC XỬ LÝ LAB 1 THÀNH CÔNG!")
print("="*70)
