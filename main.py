import os
import uuid
import librosa
import numpy as np
import soundfile as sf
import shutil

from scipy.signal import butter, lfilter, hilbert
from scipy import signal

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

# 処理後にサーバー内のゴミファイルを自動削除する関数
def remove_file(path: str):
    if os.path.exists(path):
        os.remove(path)
    # フォルダなら丸ごと消す
    elif os.path.isdir(path):
        shutil.rmtree(path)

@app.post("/upload")
async def upload_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    unique_id = str(uuid.uuid4())
    task_dir = os.path.join(UPLOAD_DIR, unique_id)
    os.makedirs(task_dir, exist_ok=True)

    input_path = os.path.join(task_dir, file.filename)
    with open(input_path, "wb") as f:
        f.write(await file.read())

    base_name = os.path.splitext(file.filename)[0]
    
    # 書き出しファイルパスの設定
    orig_wav_path = os.path.join(task_dir, f"{base_name}_original.wav")
    enhanced_wav_path = os.path.join(task_dir, f"{base_name}_enhanced.wav")
    zip_output_path = os.path.join(UPLOAD_DIR, f"{unique_id}_result")

    TARGET_SR = 96000

    print("Loading audio (Max 60s for memory safety)...")
    # 512MB制限対策として最大60秒に制限
    y, sr = librosa.load(input_path, sr=None, mono=False, duration=60.0)

    if y.ndim == 1:
        y = np.vstack([y, y])

    # 元の音声をそのままWAVとして保存（聴き比べ用）
    sf.write(orig_wav_path, y.T, sr, subtype='PCM_16')

    print("Upsampling...")
    num_samples = int(y.shape[1] * TARGET_SR / sr)
    left = signal.resample(y[0], num_samples)
    right = signal.resample(y[1], num_samples)

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

    # エフェクト音源を保存
    sf.write(enhanced_wav_path, enhanced.T, TARGET_SR, subtype='PCM_24')

    print("Zipping results...")
    # 聴き比べ用とダウンロード用にフォルダごとZIP圧縮する
    shutil.make_archive(zip_output_path, 'zip', task_dir)
    final_zip = f"{zip_output_path}.zip"

    # メモリ解放と、送信後の自動ゴミお掃除をタスクに登録
    background_tasks.add_task(remove_file, task_dir)
    background_tasks.add_task(remove_file, final_zip)

    return FileResponse(
        final_zip,
        media_type='application/zip',
        filename=f"{base_name}_compressed.zip"
    )