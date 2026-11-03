"""
neural_tts_backend.py
Mô-đun wrapper cho Neural TTS (Mô hình Text-to-Speech học sâu hiện đại)
Phục vụ thực hành Lab 4 - CSE457 Xử lý âm thanh và tiếng nói.
Hỗ trợ lưu đệm offline (cache) để đảm bảo tuân thủ nguyên tắc "không tải mô hình ngẫu nhiên trong giờ Lab".
"""

import os
import time
import hashlib
import asyncio
import subprocess
import numpy as np
import soundfile as sf
import librosa

CACHE_DIR = os.path.join(os.path.dirname(__file__), "neural_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


async def _synthesize_edge_tts(text: str, out_wav_path: str, voice: str = "vi-VN-HoaiMyNeural"):
    import edge_tts
    tmp_mp3 = out_wav_path + ".tmp.mp3"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(tmp_mp3)

    # Chuyển đổi sang wav 16kHz mono PCM 16-bit bằng ffmpeg
    cmd = [
        "ffmpeg", "-y", "-i", tmp_mp3,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        out_wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if os.path.exists(tmp_mp3):
        os.remove(tmp_mp3)


def synthesize(text: str, voice: str = "vi-VN-HoaiMyNeural", target_sr: int = 16000):
    """
    Inference Neural TTS cho câu văn bản tiếng Việt.
    
    Tham số:
        text: Chuỗi văn bản tiếng Việt cần tổng hợp.
        voice: Giọng đọc nơ-ron (mặc định vi-VN-HoaiMyNeural).
        target_sr: Tần số lấy mẫu mục tiêu (16000 Hz).
        
    Trả về:
        y_nn: Mảng numpy chứa waveform âm thanh (float32).
        sr: Tần số lấy mẫu (16000).
        extra: Dict chứa:
            - 'mel_spectrogram': Ma trận Mel-spectrogram (dB).
            - 'latency_ms': Thời gian sinh âm thanh (mili-giây).
            - 'duration_s': Thời lượng audio (giây).
            - 'rtf': Real-Time Factor (latency / duration).
            - 'text': Văn bản gốc.
    """
    text_clean = text.strip()
    text_hash = hashlib.md5(f"{text_clean}_{voice}".encode("utf-8")).hexdigest()[:12]
    cached_wav = os.path.join(CACHE_DIR, f"neural_{text_hash}.wav")

    start_time = time.perf_counter()

    if os.path.exists(cached_wav):
        # Đã có trong bộ nhớ đệm offline
        y_nn, sr = sf.read(cached_wav, dtype="float32")
    else:
        # Tổng hợp và lưu vào bộ nhớ đệm
        try:
            asyncio.run(_synthesize_edge_tts(text_clean, cached_wav, voice))
            y_nn, sr = sf.read(cached_wav, dtype="float32")
        except Exception as e:
            # Fallback nếu không có kết nối internet: dùng pyttsx3 offline
            import pyttsx3
            engine = pyttsx3.init()
            tmp_fallback = cached_wav + ".tmp_sapi.wav"
            engine.save_to_file(text_clean, tmp_fallback)
            engine.runAndWait()
            cmd = [
                "ffmpeg", "-y", "-i", tmp_fallback,
                "-ar", str(target_sr), "-ac", "1", "-c:a", "pcm_s16le",
                cached_wav
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(tmp_fallback):
                os.remove(tmp_fallback)
            y_nn, sr = sf.read(cached_wav, dtype="float32")

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000.0

    # Đảm bảo đúng target_sr và mono
    if y_nn.ndim > 1:
        y_nn = y_nn.mean(axis=1)
    if sr != target_sr:
        y_nn = librosa.resample(y_nn, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    duration_s = len(y_nn) / sr
    rtf = (latency_ms / 1000.0) / max(1e-4, duration_s)

    # Trích xuất Mel-spectrogram (80 mel bands, chuẩn Mel-scale TTS như Tacotron2/FastSpeech)
    S = librosa.feature.melspectrogram(y=y_nn, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    mel_db = librosa.power_to_db(S, ref=np.max)

    extra = {
        "mel_spectrogram": mel_db,
        "latency_ms": latency_ms,
        "duration_s": duration_s,
        "rtf": rtf,
        "text": text_clean,
        "voice": voice
    }

    return y_nn, sr, extra


if __name__ == "__main__":
    t = "Mời khách hàng số 23 đến quầy 5"
    y, sr, extra = synthesize(t)
    print(f"Neural TTS synthesized: {extra['duration_s']:.2f}s, Latency={extra['latency_ms']:.1f}ms, RTF={extra['rtf']:.3f}")
    print(f"Mel-spectrogram shape: {extra['mel_spectrogram'].shape}")

