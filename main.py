import os
import uuid
import librosa
import numpy as np
import soundfile as sf
import shutil
import soxr

from scipy.signal import butter, lfilter, hilbert
from scipy import signal as scipy_signal

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
    return lfilter(b, a, data.astype(np.float32))

def highpass(data, cutoff=5000, fs=96000, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    return lfilter(b, a, data.astype(np.float32))

def excite(signal_in):
    analytic = hilbert(signal_in)
    envelope = np.abs(analytic).astype(np.float32)
    return (np.sin(signal_in * 25.0) * envelope * 0.015).astype(np.float32)

@app.post("/upload")
async def upload_audio(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    mode: str = Form("2ch_96_24_soxr"),
    start_time: float = Form(0.0)  # 🟢 フロントからカット開始位置（秒）を受け取る
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

    is_scipy = "_scipy" in mode
    is_192 = "192" in mode
    is_51ch = "5.1ch" in mode
    
    TARGET_SR = 192000 if is_192 else 96000

    print(f"Loading audio (Engine: {'Scipy' if is_scipy else 'Soxr'}, Start: {start_time}s)...")
    
    # 🟢 Scipyモードなら指定された開始位置から最大60秒だけをピンポイントロード！
    if is_scipy:
        y, sr = librosa.load(input_path, sr=None, mono=False, offset=start_time, duration=60.0, dtype=np.float32)
    else:
        # Soxrモードは今まで通りフル尺
        y, sr = librosa.load(input_path, sr=None, mono=False, dtype=np.float32)

    if y.ndim == 1:
        y = np.vstack([y, y])

    sf.write(orig_wav_path, y.T, sr, subtype='PCM_16')

    if is_scipy:
        print("Upsampling with ultra-precise Scipy (FFT)...")
        num_samples = int(y.shape[1] * TARGET_SR / sr)
        left = scipy_signal.resample(y[0], num_samples).astype(np.float32)
        right = scipy_signal.resample(y[1], num_samples).astype(np.float32)
    else:
        print("Upsampling with lightweight Soxr (Kaiser)...")
        left = soxr.resample(y[0], sr, TARGET_SR, quality='HQ').astype(np.float32)
        right = soxr.resample(y[1], sr, TARGET_SR, quality='HQ').astype(np.float32)
        
    del y  

    print("Processing audio effects...")
    left_high = highpass(left, fs=TARGET_SR)
    right_high = highpass(right, fs=TARGET_SR)

    left_harm = np.tanh(left_high * 3.5) * 0.12
    right_harm = np.tanh(right_high * 3.5) * 0.12

    left_air = excite(left_high)
    right_air = excite(right_high)

    stereo_boost = 1.08
    mid = (left + right) * 0.5
    side = (left - right) * 0.5 * stereo_boost

    subtype_str = 'FLOAT' if is_192 else 'PCM_24'

    if is_51ch:
        print("Creating 5.1ch Surround Matrix...")
        center = mid * 0.7
        lfe = lowpass_filter(mid, fs=TARGET_SR)
        
        delay_samples = int(TARGET_SR * 0.015)
        ls_rear = np.roll(side, delay_samples) * 0.5
        rs_rear = np.roll(-side, delay_samples) * 0.5

        if "pseudo" in mode:
            front_l = mid + side + left_harm + left_air + (ls_rear * 0.3)
            front_r = mid - side + right_harm + right_air - (rs_rear * 0.3)
        else:
            front_l = mid + side + left_harm + left_air
            front_r = mid - side + right_harm + right_air
            ls_rear = ls_rear * 0.7
            rs_rear = rs_rear * 0.7

        enhanced_data = np.vstack([front_l, front_r, center, lfe, ls_rear, rs_rear])
    else:
        print("Creating 2ch Stereo Matrix...")
        front_l = mid + side + left_harm + left_air
        front_r = mid - side + right_harm + right_air
        enhanced_data = np.vstack([front_l, front_r])
        
    peak = np.max(np.abs(enhanced_data))
    if peak > 0:
        enhanced_data = enhanced_data / peak * 0.98

    print("Writing enhanced WAV...")
    sf.write(enhanced_wav_path, enhanced_data.T, TARGET_SR, subtype=subtype_str)

    print("Packaging final zip...")
    shutil.make_archive(zip_output_path, 'zip', task_dir)
    final_zip = f"{zip_output_path}.zip"

    background_tasks.add_task(remove_file, task_dir)
    background_tasks.add_task(remove_file, final_zip)

    return FileResponse(final_zip, media_type='application/zip', filename=f"{base_name}_compressed.zip")