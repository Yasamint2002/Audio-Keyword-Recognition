import os
import librosa
import numpy as np
from scipy.io import wavfile

def augment_audio(file_path, output_dir, num_augments=10):
    y, sr = librosa.load(file_path, sr=16000)
    
    for i in range(1, num_augments + 1):
        # تغییر سرعت
        speed_rate = np.random.uniform(low=0.9, high=1.1)
        y_speed = librosa.effects.time_stretch(y, speed_rate)
        
        # افزودن نویز
        noise = np.random.randn(len(y_speed))
        y_noise = y_speed + 0.005 * noise
        
        # تغییر تن صدا
        pitch_factor = np.random.uniform(low=-1, high=1)
        y_pitch = librosa.effects.pitch_shift(y_noise, sr, n_steps=pitch_factor)
        
        # ذخیره فایل افزایش یافته
        base_name = os.path.basename(file_path).replace('.wav', '')
        augmented_file_path = os.path.join(output_dir, f"{base_name}_aug_{i}.wav")
        wavfile.write(augmented_file_path, sr, (y_pitch * 32767).astype(np.int16))

# مسیر داده‌ها
data_dir = "data"

# لیست کلمات
words = ["balee", "nah", "salam", "khodahafez", "lotfan", "tashakor", "bebakhshid", "komak", "tavaqof", "boro",
         "chap", "rast", "bal", "paein", "shoroe", "payan", "baz", "baste", "roshan", "khamosh"]

# افزایش داده‌ها برای هر کلمه
for word in words:
    input_dir = os.path.join(data_dir, word)
    output_dir = os.path.join(data_dir, word)
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith(".wav"):
            file_path = os.path.join(input_dir, file)
            augment_audio(file_path, output_dir)