import os
import uuid
import librosa
import numpy as np
import soundfile as sf
import shutil

from scipy.signal import butter, lfilter, hilbert
from scipy import signal

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

def remove_file(path: str):
    if os.path.exists(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)

def lowpass_filter(data, cutoff=120, fs=96000, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    # 🟢 メモリ節約のためfloat32でフィルター計算
    return lfilter(b, a, data.astype(np.float32))

@app.post("/upload")
async def upload_audio(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    mode: str = Form("2ch_96_24")
):
    unique_id = str(uuid.uuid4())
    task_dir = os.path.join(UPLOAD_DIR, unique_id)
    os.makedirs(task_dir, exist_ok=True)

    input_path = os.path.join(task_dir, file.filename)
    with open(input_path, "wb") as f:
        f.write(await file.read())

    base_name = os.path.splitext(file.filename)[0]
    orig_wav_path = os.path.join(task_dir, f"{base_name}_original.wav")
    enhanced_wav_path = os.path.join(task_dir, f"{base_name}_enhanced.wav")
    zip_output_path = os.path.join(UPLOAD_DIR, f"{unique_id}_result")

    TARGET_SR = 192000 if mode == "2ch_192_32" else 96000

    print("Loading audio...")
    # 512MB制限対策（最大60秒、さらに最初からfloat32型で読み込んでメモリを半分にする）
    y, sr = librosa.load(input_path, sr=None, mono=False, duration=60.0, dtype=np.float32)
    if y.ndim == 1:
        y = np.vstack([y, y])

    sf.write(orig_wav_path, y.T, sr, subtype='PCM_16')

    print("Upsampling...")
    num_samples = int(y.shape[1] * TARGET_SR / sr)
    # 🟢 処理をfloat32に固定してメモリ爆発を防ぐ
    left = signal.resample(y[0], num_samples).astype(np.float32)
    right = signal.resample(y[1], num_samples).astype(np.float32)

    def highpass(data, cutoff=5000, fs=96000, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high')
        return lfilter(b, a, data.astype(np.float32))

    left_high = highpass(left, fs=TARGET_SR)
    right_high = highpass(right, fs=TARGET_SR)

    left_harm = (np.tanh(left_high * 3.5) * 0.12).astype(np.float32)
    right_harm = (np.tanh(right_high * 3.5) * 0.12).astype(np.float32)

    def excite(signal_in):
        analytic = hilbert(signal_in)
        envelope = np.abs(analytic).astype(np.float32)
        return (np.sin(signal_in * 25.0) * envelope * 0.015).astype(np.float32)

    left_air = excite(left_high)
    right_air = excite(right_high)

    stereo_boost = 1.08
    mid = ((left + right) * 0.5).astype(np.float32)
    side = ((left - right) * 0.5 * stereo_boost).astype(np.float32)

    if mode.startswith("5.1ch"):
        print("Processing 5.1ch Surround...")
        center = (mid * 0.7).astype(np.float32)
        lfe = lowpass_filter(mid, fs=TARGET_SR)
        
        delay_samples = int(TARGET_SR * 0.015)
        ls_rear = (np.roll(side, delay_samples) * 0.5).astype(np.float32)
        rs_rear = (np.roll(-side, delay_samples) * 0.5).astype(np.float32)

        if mode == "5.1ch_96_24_pseudo":
            front_l = mid + side + left_harm + left_air + (ls_rear * 0.3)
            front_r = mid - side + right_harm + right_air - (rs_rear * 0.3)
        else:
            front_l = mid + side + left_harm + left_air
            front_r = mid - side + right_harm + right_air
            ls_rear = ls_rear * 0.7
            rs_rear = rs_rear * 0.7

        enhanced_5_1 = np.vstack([front_l, front_r, center, lfe, ls_rear, rs_rear])
        peak = np.max(np.abs(enhanced_5_1))
        if peak > 0:
            enhanced_5_1 = enhanced_5_1 / peak * 0.98
            
        sf.write(enhanced_wav_path, enhanced_5_1.T, TARGET_SR, subtype='PCM_24')

    else:
        print("Processing 2ch Stereo...")
        front_l = mid + side + left_harm + left_air
        front_r = mid - side + right_harm + right_air
        enhanced_2ch = np.vstack([front_l, front_r])
        
        peak = np.max(np.abs(enhanced_2ch))
        if peak > 0:
            enhanced_2ch = enhanced_2ch / peak * 0.98

        if mode == "2ch_192_32":
            sf.write(enhanced_wav_path, enhanced_2ch.T, TARGET_SR, subtype='FLOAT')
        else:
            sf.write(enhanced_wav_path, enhanced_2ch.T, TARGET_SR, subtype='PCM_24')

    print("Zipping results...")
    shutil.make_archive(zip_output_path, 'zip', task_dir)
    final_zip = f"{zip_output_path}.zip"

    background_tasks.add_task(remove_file, task_dir)
    background_tasks.add_task(remove_file, final_zip)

    return FileResponse(
        final_zip,
        media_type='application/zip',
        filename=f"{base_name}_compressed.zip"
    )