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

    print("Loading bypass track...")
    y_orig, sr_orig = librosa.load(input_path, sr=None, mono=False, dtype=np.float32)
    if y_orig.ndim == 1:
        y_orig = np.vstack([y_orig, y_orig])
    sf.write(orig_wav_path, y_orig.T, sr_orig, subtype='PCM_16')
    
    total_duration = librosa.get_duration(y=y_orig, sr=sr_orig)
    del y_orig

    print(f"Total Duration: {total_duration:.2f}s. Processing in seamless chunks...")
    
    # 🟢 ガタつきを無くすための厳密なチャンク設計
    CHUNK_SEC = 4.0      
    OVERLAP_SEC = 0.5    # クロスフェードの幅を少しタイト(0.5秒)にして安定化
    
    fade_len = int(OVERLAP_SEC * TARGET_SR)
    fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
    fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)

    out_channels = 6 if mode.startswith("5.1ch") else 2
    subtype_str = 'FLOAT' if mode == "2ch_192_32" else 'PCM_24'
    
    prev_overlap_data = None
    current_time = 0.0

    with sf.SoundFile(enhanced_wav_path, mode='w', samplerate=TARGET_SR, channels=out_channels, subtype=subtype_str) as outfile:
        
        while current_time < total_duration:
            # 🟢 進捗パーセントをサーバーログに出力（フロント連携の布石）
            progress_percent = min(100, int((current_time / total_duration) * 100))
            print(f"PROGRESS_STATUS:{progress_percent}% (Time: {current_time:.1f}s / {total_duration:.1f}s)")

            # 境界線のバグを防ぐため、ジャストな位置からダブり分を安全に拡張して読み込む
            start_read = max(0.0, current_time - OVERLAP_SEC)
            end_read = min(total_duration, current_time + CHUNK_SEC + OVERLAP_SEC)
            duration_to_read = end_read - start_read

            if duration_to_read <= 0:
                break

            y_chunk, sr = librosa.load(input_path, sr=None, mono=False, offset=start_read, duration=duration_to_read, dtype=np.float32)
            if y_chunk.ndim == 1:
                y_chunk = np.vstack([y_chunk, y_chunk])

            # リサンプリング
            num_samples = int(y_chunk.shape[1] * TARGET_SR / sr)
            left = signal.resample(y_chunk[0], num_samples).astype(np.float32)
            right = signal.resample(y_chunk[1], num_samples).astype(np.float32)

            # エフェクト
            left_high = highpass(left, fs=TARGET_SR)
            right_high = highpass(right, fs=TARGET_SR)
            left_harm = np.tanh(left_high * 3.5) * 0.12
            right_harm = np.tanh(right_high * 3.5) * 0.12
            left_air = excite(left_high)
            right_air = excite(right_high)

            stereo_boost = 1.08
            mid = (left + right) * 0.5
            side = (left - right) * 0.5 * stereo_boost

            if out_channels == 6:
                center = mid * 0.7
                lfe = lowpass_filter(mid, fs=TARGET_SR)
                delay_samples = int(TARGET_SR * 0.015)
                ls_rear = np.roll(side, delay_samples) * 0.5
                rs_rear = np.roll(-side, delay_samples) * 0.5

                if mode == "5.1ch_96_24_pseudo":
                    front_l = mid + side + left_harm + left_air + (ls_rear * 0.3)
                    front_r = mid - side + right_harm + right_air - (rs_rear * 0.3)
                else:
                    front_l = mid + side + left_harm + left_air
                    front_r = mid - side + right_harm + right_air
                    ls_rear = ls_rear * 0.7
                    rs_rear = rs_rear * 0.7
                
                chunk_enhanced = np.vstack([front_l, front_r, center, lfe, ls_rear, rs_rear])
            else:
                front_l = mid + side + left_harm + left_air
                front_r = mid - side + right_harm + right_air
                chunk_enhanced = np.vstack([front_l, front_r])

            chunk_enhanced = np.clip(chunk_enhanced, -0.98, 0.98)

            # 🟢 改修：飛び飛びバグを潰すためのジャスト切り出し
            # 読み込んだ全体のなかから、今回書き出すべき「本編（コア部分）」の位置を精密に計算
            actual_start_idx = int((current_time - start_read) * TARGET_SR)
            actual_core_len = int(min(CHUNK_SEC, total_duration - current_time) * TARGET_SR)
            
            core_data = chunk_enhanced[:, actual_start_idx : actual_start_idx + actual_core_len]

            if current_time == 0.0:
                # 最初のチャンク：次へ繋ぐお尻のfade_len分を引いて書き出し
                if core_data.shape[1] > fade_len:
                    prev_overlap_data = core_data[:, -fade_len:].copy()
                    outfile.write(core_data[:, :-fade_len].T)
                else:
                    outfile.write(core_data.T)
            else:
                # 2回目以降：前回の「お尻」と今回の「頭」をクロスフェードさせて結合
                if prev_overlap_data is not None and core_data.shape[1] >= fade_len:
                    current_head = core_data[:, :fade_len]
                    
                    # 要素数が完全に一致していることを確認してフェード合成
                    faded_connection = (prev_overlap_data * fade_out) + (current_head * fade_in)
                    outfile.write(faded_connection.T)
                    
                    # 残りの本編を書き出す。まだ次に続くならお尻をストック
                    if current_time + CHUNK_SEC < total_duration and core_data.shape[1] > fade_len:
                        prev_overlap_data = core_data[:, -fade_len:].copy()
                        outfile.write(core_data[:, fade_len:-fade_len].T)
                    else:
                        outfile.write(core_data[:, fade_len:].T)
                        prev_overlap_data = None
                else:
                    outfile.write(core_data.T)

            current_time += CHUNK_SEC

    print("PROGRESS_STATUS:100%")
    print("Packaging final zip...")
    shutil.make_archive(zip_output_path, 'zip', task_dir)
    final_zip = f"{zip_output_path}.zip"

    background_tasks.add_task(remove_file, task_dir)
    background_tasks.add_task(remove_file, final_zip)

    return FileResponse(final_zip, media_type='application/zip', filename=f"{base_name}_compressed.zip")