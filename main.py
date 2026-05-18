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

    # 🟢 1. オリジナル音源を丸ごと一度WAV化（聴き比べ用。ここはアップサンプリングしないので超軽量）
    print("Saving original bypass track...")
    y_orig, sr_orig = librosa.load(input_path, sr=None, mono=False, dtype=np.float32)
    if y_orig.ndim == 1:
        y_orig = np.vstack([y_orig, y_orig])
    sf.write(orig_wav_path, y_orig.T, sr_orig, subtype='PCM_16')
    
    total_duration = librosa.get_duration(y=y_orig, sr=sr_orig)
    del y_orig  # メモリ即時解放

    # 🟢 2. 本気の本気のチャンク処理スタート
    print(f"Starting Chunk Processing for {total_duration:.2f}s audio...")
    
    # 処理のパラメーター設定（秒単位）
    CHUNK_SEC = 4.0      # メインで進む秒数
    OVERLAP_SEC = 1.0    # 前後でダブらせる秒数
    
    current_time = 0.0
    is_first_chunk = True

    # クロスフェード用の窓関数（なめらかに繋ぐためのスロープ）
    fade_len = int(OVERLAP_SEC * TARGET_SR)
    fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
    fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)

    # チャンネル数決定
    out_channels = 6 if mode.startswith("5.1ch") else 2
    
    # 前のチャンクの「お尻（ダブり部分）」を記憶しておく変数
    prev_overlap_data = None

    # soundfileの「追記モード」用ファイルを開く
    subtype_str = 'FLOAT' if mode == "2ch_192_32" else 'PCM_24'
    
    with sf.SoundFile(enhanced_wav_path, mode='w', samplerate=TARGET_SR, channels=out_channels, subtype=subtype_str) as outfile:
        
        while current_time < total_duration:
            # 1回に読み込む範囲（前後のダブり分を余分に深く読み込む）
            start_read = max(0.0, current_time - OVERLAP_SEC)
            end_read = min(total_duration, current_time + CHUNK_SEC + OVERLAP_SEC)
            duration_to_read = end_read - start_read

            if duration_to_read <= 0:
                break

            # 部分読み込み（メモリ消費は常にこの数秒分だけ！）
            y_chunk, sr = librosa.load(input_path, sr=None, mono=False, offset=start_read, duration=duration_to_read, dtype=np.float32)
            if y_chunk.ndim == 1:
                y_chunk = np.vstack([y_chunk, y_chunk])

            # アップサンプリング（数秒分だけなので一瞬かつ超軽量）
            num_samples = int(y_chunk.shape[1] * TARGET_SR / sr)
            left = signal.resample(y_chunk[0], num_samples).astype(np.float32)
            right = signal.resample(y_chunk[1], num_samples).astype(np.float32)

            # エフェクト処理（倍音・空気感）
            left_high = highpass(left, fs=TARGET_SR)
            right_high = highpass(right, fs=TARGET_SR)
            left_harm = (np.tanh(left_high * 3.5) * 0.12).astype(np.float32)
            right_harm = (np.tanh(right_high * 3.5) * 0.12).astype(np.float32)
            left_air = excite(left_high)
            right_air = excite(right_high)

            # ステレオ拡張
            stereo_boost = 1.08
            mid = ((left + right) * 0.5).astype(np.float32)
            side = ((left - right) * 0.5 * stereo_boost).astype(np.float32)

            # 出力用波形マトリクスの組み立て
            if out_channels == 6:
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
                
                chunk_enhanced = np.vstack([front_l, front_r, center, lfe, ls_rear, rs_rear])
            else:
                front_l = mid + side + left_harm + left_air
                front_r = mid - side + right_harm + right_air
                chunk_enhanced = np.vstack([front_l, front_r])

            # 音割れ防止防止の簡易クリッピング
            chunk_enhanced = np.clip(chunk_enhanced, -0.98, 0.98)

            # 🟢 ここからクロスフェード合体ロジック
            # 最初と最後以外の「有効なコアデータ」を切り出す
            start_idx = int(OVERLAP_SEC * TARGET_SR) if current_time > 0 else 0
            end_idx = int((duration_to_read - OVERLAP_SEC) * TARGET_SR) if end_read < total_duration else chunk_enhanced.shape[1]
            
            core_data = chunk_enhanced[:, start_idx:end_idx]

            if is_first_chunk:
                # 最初のパーツはそのまま書き出す
                if core_data.shape[1] > fade_len and end_read < total_duration:
                    # 次のチャンクへ繋ぐお尻の1秒を記憶
                    prev_overlap_data = core_data[:, -fade_len:]
                    outfile.write(core_data[:, :-fade_len].T)
                else:
                    outfile.write(core_data.T)
                is_first_chunk = False
            else:
                # 2回目以降は、前回の「お尻」と今回の「頭」をクロスフェード
                if prev_overlap_data is not None and core_data.shape[1] >= fade_len:
                    current_head = core_data[:, :fade_len]
                    # フェードインとフェードアウトを掛け合わせて足す（これでプツプツが消える！）
                    faded_connection = (prev_overlap_data * fade_out) + (current_head * fade_in)
                    
                    # 結合したパーツを書き出し
                    outfile.write(faded_connection.T)
                    
                    if end_read < total_duration:
                        prev_overlap_data = core_data[:, -fade_len:]
                        outfile.write(core_data[:, fade_len:-fade_len].T)
                    else:
                        outfile.write(core_data[:, fade_len:].T)
                        prev_overlap_data = None
                else:
                    outfile.write(core_data.T)

            # 次のチャンクへ進む
            current_time += CHUNK_SEC

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