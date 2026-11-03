import os
import glob
import re
import shutil
import subprocess
import soundfile as sf

dataset_dir = os.path.abspath("./dataset")
input_m4a_lab4 = os.path.abspath("../input_m4a/input_Lab4")
os.makedirs(input_m4a_lab4, exist_ok=True)

m4a_files = sorted(glob.glob(os.path.join(dataset_dir, "*.m4a")))
print(f"Found {len(m4a_files)} m4a files in {dataset_dir}")

converted = []
for src in m4a_files:
    fname = os.path.basename(src)
    # Copy to backup folder in input_m4a
    shutil.copy2(src, os.path.join(input_m4a_lab4, fname))
    
    # Parse word and number, e.g. bay1.m4a -> word='bay', num=1
    m = re.match(r"^([a-zA-Z]+)(\d+)\.m4a$", fname)
    if not m:
        print(f"Skipping unexpected filename: {fname}")
        continue
    word, num_str = m.groups()
    num = int(num_str)
    
    # Destination folder: dataset/<word>/
    word_dir = os.path.join(dataset_dir, word)
    os.makedirs(word_dir, exist_ok=True)
    
    # Target wav name: e.g. bay_01.wav
    dst_wav = os.path.join(word_dir, f"{word}_{num:02d}.wav")
    
    # Convert via ffmpeg (16kHz, mono, 16-bit PCM WAV)
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        dst_wav
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(f"Error converting {fname}: {res.stderr.decode(errors='ignore')}")
    else:
        # Verify with soundfile
        data, sr = sf.read(dst_wav)
        info = sf.info(dst_wav)
        converted.append((word, f"{word}_{num:02d}.wav", len(data)/sr, info.subtype))
        # Remove original loose m4a from dataset folder
        os.remove(src)

print(f"\nSuccessfully converted and organized {len(converted)} files into dataset folders:")
for word, wname, dur, subtype in converted:
    print(f"  dataset/{word}/{wname}: duration={dur:.2f}s, format={subtype}")

print("\nBackup saved to:", input_m4a_lab4)

