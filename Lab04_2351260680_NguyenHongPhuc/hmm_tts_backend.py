"""
hmm_tts_backend.py
Mô-đun tổng hợp tiếng nói tham số dựa trên mô hình HMM (HMM-based Speech Synthesis / Source-Filter model)
Phục vụ thực hành Lab 4 - CSE457 Xử lý âm thanh và tiếng nói.
"""

import numpy as np
import scipy.signal as signal

# Từ điển ngữ âm đơn giản cho các từ trong miền hẹp "thông báo xếp hàng"
# Mỗi từ ánh xạ thành chuỗi âm/state với tham số formant (F1, F2, F3) và voiced/unvoiced
PHONEME_TABLE = {
    "mời": [
        {"name": "m", "voiced": True, "f1": 300, "f2": 1100, "f3": 2500, "dur_ms": 70},
        {"name": "ơi", "voiced": True, "f1": 480, "f2": 1250, "f3": 2400, "dur_ms": 170},
    ],
    "khách": [
        {"name": "kh", "voiced": False, "f1": 400, "f2": 1600, "f3": 2600, "dur_ms": 90},
        {"name": "a", "voiced": True, "f1": 750, "f2": 1300, "f3": 2500, "dur_ms": 150},
        {"name": "ch", "voiced": False, "f1": 350, "f2": 2100, "f3": 2800, "dur_ms": 60},
    ],
    "hàng": [
        {"name": "h", "voiced": False, "f1": 500, "f2": 1400, "f3": 2500, "dur_ms": 60},
        {"name": "ang", "voiced": True, "f1": 700, "f2": 1200, "f3": 2400, "dur_ms": 190},
    ],
    "số": [
        {"name": "s", "voiced": False, "f1": 350, "f2": 1800, "f3": 3000, "dur_ms": 110},
        {"name": "ô", "voiced": True, "f1": 450, "f2": 950, "f3": 2400, "dur_ms": 170},
    ],
    "không": [
        {"name": "kh", "voiced": False, "f1": 400, "f2": 1600, "f3": 2600, "dur_ms": 70},
        {"name": "ông", "voiced": True, "f1": 450, "f2": 900, "f3": 2350, "dur_ms": 180},
    ],
    "một": [
        {"name": "m", "voiced": True, "f1": 300, "f2": 1100, "f3": 2500, "dur_ms": 60},
        {"name": "ô", "voiced": True, "f1": 460, "f2": 1000, "f3": 2400, "dur_ms": 110},
        {"name": "t", "voiced": False, "f1": 300, "f2": 1700, "f3": 2700, "dur_ms": 40},
    ],
    "hai": [
        {"name": "h", "voiced": False, "f1": 500, "f2": 1400, "f3": 2500, "dur_ms": 50},
        {"name": "ai", "voiced": True, "f1": 680, "f2": 1800, "f3": 2600, "dur_ms": 200},
    ],
    "ba": [
        {"name": "b", "voiced": True, "f1": 250, "f2": 1100, "f3": 2400, "dur_ms": 50},
        {"name": "a", "voiced": True, "f1": 780, "f2": 1300, "f3": 2500, "dur_ms": 190},
    ],
    "bốn": [
        {"name": "b", "voiced": True, "f1": 250, "f2": 1100, "f3": 2400, "dur_ms": 50},
        {"name": "ôn", "voiced": True, "f1": 460, "f2": 1050, "f3": 2400, "dur_ms": 180},
    ],
    "năm": [
        {"name": "n", "voiced": True, "f1": 300, "f2": 1400, "f3": 2500, "dur_ms": 60},
        {"name": "ăm", "voiced": True, "f1": 650, "f2": 1300, "f3": 2450, "dur_ms": 180},
    ],
    "sáu": [
        {"name": "s", "voiced": False, "f1": 350, "f2": 1800, "f3": 3000, "dur_ms": 90},
        {"name": "au", "voiced": True, "f1": 680, "f2": 1100, "f3": 2400, "dur_ms": 180},
    ],
    "bảy": [
        {"name": "b", "voiced": True, "f1": 250, "f2": 1100, "f3": 2400, "dur_ms": 50},
        {"name": "ay", "voiced": True, "f1": 600, "f2": 1600, "f3": 2550, "dur_ms": 170},
    ],
    "tám": [
        {"name": "t", "voiced": False, "f1": 300, "f2": 1700, "f3": 2700, "dur_ms": 60},
        {"name": "am", "voiced": True, "f1": 700, "f2": 1250, "f3": 2450, "dur_ms": 180},
    ],
    "chín": [
        {"name": "ch", "voiced": False, "f1": 350, "f2": 2100, "f3": 2800, "dur_ms": 70},
        {"name": "in", "voiced": True, "f1": 320, "f2": 2200, "f3": 2900, "dur_ms": 190},
    ],
    "đến": [
        {"name": "đ", "voiced": True, "f1": 250, "f2": 1700, "f3": 2600, "dur_ms": 50},
        {"name": "ên", "voiced": True, "f1": 420, "f2": 1900, "f3": 2600, "dur_ms": 180},
    ],
    "quầy": [
        {"name": "qu", "voiced": True, "f1": 350, "f2": 850, "f3": 2300, "dur_ms": 70},
        {"name": "ây", "voiced": True, "f1": 550, "f2": 1500, "f3": 2500, "dur_ms": 180},
    ],
    "pau": [
        {"name": "pau", "voiced": False, "f1": 200, "f2": 500, "f3": 1500, "dur_ms": 100},
    ]
}

DIGIT_MAP = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
}


def normalize_text_for_hmm(text: str):
    text = text.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "")
    tokens = []
    for tok in text.split():
        if tok.isdigit():
            tokens.extend([DIGIT_MAP[c] for c in tok])
        else:
            tokens.append(tok)
    return tokens


def synthesize_hmm(text: str, duration_scale: float = 1.0, f0_scale: float = 1.0, sr: int = 16000):
    """
    Sinh chuỗi tham số âm học (trajectories) và resynthesize waveform bằng HMM Source-Filter.
    
    Tham số:
        text: Câu văn bản đầu vào.
        duration_scale: Tỉ lệ co giãn độ dài (duration scaling, vd 0.8 hoặc 1.2).
        f0_scale: Tỉ lệ nâng/hạ tần số cơ bản (F0 scaling, vd 0.9 hoặc 1.1).
        sr: Tần số lấy mẫu (mặc định 16000 Hz).
        
    Trả về:
        y_hmm: waveform âm thanh tổng hợp (numpy float32).
        sr: Tần số lấy mẫu (16000).
        tracks: Dict chứa các trajectory:
            - 'f0': Quỹ đạo F0 theo frame (Hz).
            - 'f1', 'f2', 'f3': Quỹ đạo formant theo frame (Hz).
            - 'energy': Năng lượng theo frame.
            - 'states': Chuỗi tên trạng thái / âm vị.
            - 'frame_times': Trục thời gian tính bằng giây.
    """
    tokens = normalize_text_for_hmm(text)
    frame_shift_ms = 10.0  # 10 ms mỗi frame âm học (chuẩn HMM/HTS)
    frame_shift_samples = int(sr * frame_shift_ms / 1000.0)

    # 1. Phonetic Analysis -> State Sequence
    states_info = []
    for i, tok in enumerate(tokens):
        phonemes = PHONEME_TABLE.get(tok, [{"name": tok, "voiced": True, "f1": 500, "f2": 1500, "f3": 2500, "dur_ms": 150}])
        for ph in phonemes:
            # Scale duration
            dur_frames = max(3, int(round((ph["dur_ms"] * duration_scale) / frame_shift_ms)))
            states_info.append({
                "name": f"{tok}:{ph['name']}",
                "voiced": ph["voiced"],
                "f1": ph["f1"],
                "f2": ph["f2"],
                "f3": ph["f3"],
                "num_frames": dur_frames
            })
        if i < len(tokens) - 1:
            # Ngắt ngắn giữa các từ (short pause 3 frame ~ 30 ms)
            states_info.append({
                "name": "pause",
                "voiced": False,
                "f1": 200, "f2": 500, "f3": 1500,
                "num_frames": 3
            })

    total_frames = sum(s["num_frames"] for s in states_info)
    f0_target = np.zeros(total_frames, dtype=np.float32)
    f1_target = np.zeros(total_frames, dtype=np.float32)
    f2_target = np.zeros(total_frames, dtype=np.float32)
    f3_target = np.zeros(total_frames, dtype=np.float32)
    voiced_flag = np.zeros(total_frames, dtype=bool)
    state_names = []

    # 2. Xây dựng tham số bậc 0 (Means theo từng state)
    cur = 0
    base_f0 = 135.0 * f0_scale  # Tần số cơ bản cơ sở nam/nữ ~ 135 Hz
    for s in states_info:
        n = s["num_frames"]
        state_names.extend([s["name"]] * n)
        if s["voiced"]:
            voiced_flag[cur:cur + n] = True
            # Tạo đường lượn tự nhiên F0 (intonation declination trend)
            pos_ratio = cur / max(1, total_frames)
            declination = 1.0 - 0.15 * pos_ratio
            f0_target[cur:cur + n] = base_f0 * declination
        else:
            voiced_flag[cur:cur + n] = False
            f0_target[cur:cur + n] = 0.0

        f1_target[cur:cur + n] = s["f1"]
        f2_target[cur:cur + n] = s["f2"]
        f3_target[cur:cur + n] = s["f3"]
        cur += n

    # 3. Dynamic feature smoothing (Ràng buộc Delta tương đương trong HMM generation)
    # Áp dụng bộ lọc trơn làm mềm bước nhảy tham số giữa các state
    smooth_window = signal.windows.gaussian(7, std=1.8)
    smooth_window /= smooth_window.sum()

    f1_smooth = np.convolve(f1_target, smooth_window, mode='same')
    f2_smooth = np.convolve(f2_target, smooth_window, mode='same')
    f3_smooth = np.convolve(f3_target, smooth_window, mode='same')

    # F0 smooth chỉ trên các vùng voiced
    f0_smooth = np.zeros_like(f0_target)
    voiced_indices = np.where(voiced_flag)[0]
    if len(voiced_indices) > 0:
        f0_voiced = f0_target[voiced_indices]
        # Thêm biến thiên vi ngữ điệu (micro-prosody)
        t_voiced = np.linspace(0, 4 * np.pi, len(voiced_indices))
        micro_f0 = f0_voiced + 3.0 * np.sin(t_voiced)
        f0_smooth[voiced_indices] = micro_f0

    # 4. Source-Filter Resynthesis (Sinh waveform từ tham số âm học)
    total_samples = total_frames * frame_shift_samples
    excitation = np.zeros(total_samples, dtype=np.float32)

    # Nguồn kích thích (Excitation): Xung pitch đối với Voiced, Nhiễu Gaussian đối với Unvoiced
    phase = 0.0
    for frame_idx in range(total_frames):
        start_samp = frame_idx * frame_shift_samples
        end_samp = start_samp + frame_shift_samples
        f0_val = f0_smooth[frame_idx]
        if f0_val > 40:
            # Voiced: pulse train
            period_samples = sr / f0_val
            for s_idx in range(start_samp, end_samp):
                phase += 1.0
                if phase >= period_samples:
                    phase -= period_samples
                    excitation[s_idx] = 1.0  # Glottal impulse
        else:
            # Unvoiced: White noise
            excitation[start_samp:end_samp] = np.random.normal(0, 0.05, frame_shift_samples)

    # Bộ lọc đường dẫn thanh âm (Vocal Tract Resonant Filter theo F1, F2, F3)
    waveform = np.zeros(total_samples, dtype=np.float32)
    # Băng thông formant chuẩn
    bw1, bw2, bw3 = 80.0, 110.0, 140.0

    # Lọc block-by-block với tham số biến thiên theo frame
    zi1 = np.zeros(2)
    zi2 = np.zeros(2)
    zi3 = np.zeros(2)

    def resonator_coeffs(freq, bw, fs):
        r = np.exp(-np.pi * bw / fs)
        theta = 2.0 * np.pi * freq / fs
        a = [1.0, -2.0 * r * np.cos(theta), r * r]
        # Gain chuẩn hóa
        b = [1.0 - 2.0 * r * np.cos(theta) + r * r]
        return b, a

    for frame_idx in range(total_frames):
        start_samp = frame_idx * frame_shift_samples
        end_samp = start_samp + frame_shift_samples
        exc_block = excitation[start_samp:end_samp]

        f1_val = max(150.0, f1_smooth[frame_idx])
        f2_val = max(f1_val + 100.0, f2_smooth[frame_idx])
        f3_val = max(f2_val + 100.0, f3_smooth[frame_idx])

        b1, a1 = resonator_coeffs(f1_val, bw1, sr)
        b2, a2 = resonator_coeffs(f2_val, bw2, sr)
        b3, a3 = resonator_coeffs(f3_val, bw3, sr)

        y1, zi1 = signal.lfilter(b1, a1, exc_block, zi=zi1)
        y2, zi2 = signal.lfilter(b2, a2, exc_block, zi=zi2)
        y3, zi3 = signal.lfilter(b3, a3, exc_block, zi=zi3)

        # Tổng hợp 3 formant song song + unvoiced booster
        block_out = 0.5 * y1 + 0.3 * y2 + 0.15 * y3
        if not voiced_flag[frame_idx]:
            block_out = exc_block * 0.4
        waveform[start_samp:end_samp] = block_out

    # Chuẩn hóa biên độ
    peak = np.max(np.abs(waveform)) + 1e-6
    if peak > 0:
        waveform = 0.85 * (waveform / peak)

    frame_times = np.arange(total_frames) * (frame_shift_ms / 1000.0)

    tracks = {
        "f0": f0_smooth,
        "f0_raw": f0_target,
        "f1": f1_smooth,
        "f2": f2_smooth,
        "f3": f3_smooth,
        "energy": np.array([np.sqrt(np.mean(waveform[i*frame_shift_samples:(i+1)*frame_shift_samples]**2)) for i in range(total_frames)]),
        "states": state_names,
        "frame_times": frame_times,
        "voiced_flag": voiced_flag
    }

    return waveform, sr, tracks


if __name__ == "__main__":
    import soundfile as sf
    test_text = "Mời khách hàng số 23 đến quầy 5"
    y, sr, tr = synthesize_hmm(test_text)
    print(f"HMM Synthesis completed: duration={len(y)/sr:.2f}s, frames={len(tr['f0'])}")
    sf.write("hmm_test_output.wav", y, sr)
    print("Saved hmm_test_output.wav")

