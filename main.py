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

# 重低音抽出用のローパスフィルター
def lowpass_filter(data, cutoff=120, fs=96000, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return lfilter(b, a, data)

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

    # 🟢 サンプリングレートの分岐（①案の192kHzに対応）
    TARGET_SR = 192000 if mode == "2ch_192_32" else 96000

    # 512MB制限対策（最大60秒）
    y, sr = librosa.load(input_path, sr=None, mono=False, duration=60.0)
    if y.ndim == 1:
        y = np.vstack([y, y])

    # オリジナル音源の保存（聴き比べ用）
    sf.write(orig_wav_path, y.T, sr, subtype='PCM_16')

    # アップサンプリング計算
    num_samples = int(y.shape[1] * TARGET_SR / sr)
    left = signal.resample(y[0], num_samples)
    right = signal.resample(y[1], num_samples)

    # 高音域の倍音生成（Exciter）
    def highpass(data, cutoff=5000, fs=96000, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high')
        return lfilter(b, a, data)

    left_high = highpass(left, fs=TARGET_SR)
    right_high = highpass(right, fs=TARGET_SR)

    left_harm = np.tanh(left_high * 3.5) * 0.12
    right_harm = np.tanh(right_high * 3.5) * 0.12

    def excite(signal_in):
        analytic = hilbert(signal_in)
        envelope = np.abs(analytic)
        return np.sin(signal_in * 25.0) * envelope * 0.015

    left_air = excite(left_high)
    right_air = excite(right_high)

    # 基本のステレオ拡張
    stereo_boost = 1.08
    mid = (left + right) * 0.5
    side = (left - right) * 0.5 * stereo_boost

    # チャンネル生成・書き出し分岐
    if mode.startswith("5.1ch"):
        print("Processing 5.1ch Surround...")
        # 5.1chの割り当て: [L, R, C, LFE, Ls, Rs]
        center = mid * 0.7
        lfe = lowpass_filter(mid, fs=TARGET_SR) * 1.2
        
        # サラウンドリア成分（15msディレイ）
        delay_samples = int(TARGET_SR * 0.015)
        ls_rear = np.roll(side, delay_samples) * 0.5
        rs_rear = np.roll(-side, delay_samples) * 0.5

        if mode == "5.1ch_96_24_pseudo":
            # 疑似：ステレオ可モード
            front_l = mid + side + left_harm + left_air + (ls_rear * 0.3)
            front_r = mid - side + right_harm + right_air - (rs_rear * 0.3)
        else:
            # 🟢 ③案：音響機材用（映画館風シアター・サラウンド）
            # リアの残響感を高め、フロントは芯のある音に調整
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

        # 🟢 ①案（32bit float）か 通常の24bitかを判定して書き出し
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