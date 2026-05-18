import os
import uuid
import librosa
import numpy as np
import soundfile as sf

from scipy.signal import butter, lfilter, hilbert
from scipy import signal

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ※プロジェクトフォルダ内に "static" フォルダがないとエラーになるので注意
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):

    unique_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{unique_id}_{file.filename}")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    base_name = os.path.splitext(input_path)[0]
    output_path = f"{base_name}_enhanced.wav"

    TARGET_SR = 96000

    print("Loading audio...")
    # 安全のため最大60秒に制限しつつ読み込み
    y, sr = librosa.load(input_path, sr=None, mono=False, duration=60.0)

    if y.ndim == 1:
        y = np.vstack([y, y])

    print("Upsampling (Memory-Safe Mode)...")
    # 💡 signal.resample をやめて、メモリ消費が極めて少ない resample_poly に変更！
    # 元のサンプリングレート（sr）から 96000（TARGET_SR）への比率を計算して変換します
    gcd = np.gcd(TARGET_SR, sr)
    up = TARGET_SR // gcd
    down = sr // gcd
    
    left = signal.resample_poly(y[0], up, down)
    right = signal.resample_poly(y[1], up, down)

    def highpass(data, cutoff=5000, fs=96000, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high')
        return lfilter(b, a, data)

    left_high = highpass(left)
    right_high = highpass(right)

    print("Generating harmonics...")
    left_harm = np.tanh(left_high * 3.5) * 0.12
    right_harm = np.tanh(right_high * 3.5) * 0.12

    def excite(signal_in):
        analytic = hilbert(signal_in)
        envelope = np.abs(analytic)
        airy = np.sin(signal_in * 25.0) * envelope * 0.015
        return airy

    left_air = excite(left_high)
    right_air = excite(right_high)

    stereo_boost = 1.08
    mid = (left + right) * 0.5
    side = (left - right) * 0.5 * stereo_boost

    left_wide = mid + side
    right_wide = mid - side

    enhanced_left = left_wide + left_harm + left_air
    enhanced_right = right_wide + right_harm + right_air

    enhanced = np.vstack([enhanced_left, enhanced_right])
    peak = np.max(np.abs(enhanced))

    if peak > 0:
        enhanced = enhanced / peak * 0.98

    enhanced = enhanced.T

    print("Saving...")
    sf.write(output_path, enhanced, TARGET_SR, subtype='PCM_24')

    return FileResponse(
        output_path,
        media_type='audio/wav',
        filename=os.path.basename(output_path)
    )