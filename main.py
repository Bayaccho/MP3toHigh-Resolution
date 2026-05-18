import os
import uuid
import librosa
import numpy as np
import soundfile as sf
import shutil
import asyncio

from scipy.signal import butter, lfilter
from scipy import signal

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# 進捗状況を一時的に保存する辞書
progress_store = {}

@app.get("/")
def home():
    return FileResponse("static/index.html")

# リアルタイムで進捗率をブラウザに届けるエンドポイント (SSE)
@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    async def event_generator():
        while True:
            prog = progress_store.get(task_id, 0)
            yield f"data: {prog}\n\n"
            if prog >= 100:
                # 終わったら少し待って辞書から削除
                await asyncio.sleep(2)
                if task_id in progress_store:
                    del progress_store[task_id]
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

def remove_file(path: str):
    if os.path.exists(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)

# フィルター関数（チャンク用に初期状態を維持できるように改良）
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    return b, a

@app.post("/upload")
async def upload_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    progress_store[task_id] = 0  # 進捗初期化

    task_dir = os.path.join(UPLOAD_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    input_path = os.path.join(task_dir, file.filename)
    with open(input_path, "wb") as f:
        f.write(await file.read())

    base_name = os.path.splitext(file.filename)[0]
    orig_wav_path = os.path.join(task_dir, f"{base_name}_original.wav")
    enhanced_wav_path = os.path.join(task_dir, f"{base_name}_enhanced.wav")
    zip_output_path = os.path.join(UPLOAD_DIR, f"{task_id}_result")

    # メモリを節約するため、まず音声の総長さとサンプリングレートだけを取得
    total_duration = librosa.get_duration(path=input_path)
    native_sr = librosa.get_samplerate(input_path)
    
    TARGET_SR = 96000
    CHUNK_SEC = 5.0  # 5秒ずつ処理
    OVERLAP_SEC = 0.1  # 前後100msをダブらせてノイズを防ぐ

    # フィルターの係数をあらかじめ計算
    b, a = butter_highpass(5000, TARGET_SR)

    # 聴き比べ用オリジナルWAVの作成
    # チャンクごとに読み込んで書き出すことでメモリを節約
    with sf.SoundFile(orig_wav_path, mode='w', samplerate=native_sr, channels=2, subtype='PCM_16') as out_orig:
        for block in librosa.stream(input_path, block_length=int(CHUNK_SEC), frame_length=2048, hop_length=512, mono=False):
            if block.ndim == 1:
                block = np.vstack([block, block])
            out_orig.write(block.T)

    # ハイレゾ化エフェクトの処理ループ
    progress_store[task_id] = 10  # 準備完了で10%

    current_time = 0.0
    
    with sf.SoundFile(enhanced_wav_path, mode='w', samplerate=TARGET_SR, channels=2, subtype='PCM_24') as out_enh:
        while current_time < total_duration:
            # 進捗計算 (10%〜90%の間で進む)
            pct = int(10 + (current_time / total_duration) * 80)
            progress_store[task_id] = min(pct, 90)

            # オーバーラップを含めた読み込み範囲の計算
            start = max(0.0, current_time - OVERLAP_SEC)
            end = min(total_duration, current_time + CHUNK_SEC + OVERLAP_SEC)
            duration = end - start

            # 5秒＋前後のダブり分だけをピンポイントで読み込み（メモリ消費は数MBだけ！）
            y_chunk, sr = librosa.load(input_path, sr=native_sr, mono=False, offset=start, duration=duration)
            if y_chunk.ndim == 1:
                y_chunk = np.vstack([y_chunk, y_chunk])

            # アップサンプリング
            num_samples = int(y_chunk.shape[1] * TARGET_SR / sr)
            left = signal.resample(y_chunk[0], num_samples)
            right = signal.resample(y_chunk[1], num_samples)

            # エフェクト処理（ハイパス ➔ 倍音生成）
            left_high = lfilter(b, a, left)
            right_high = lfilter(b, a, right)

            left_harm = np.tanh(left_high * 3.5) * 0.12
            right_harm = np.tanh(right_high * 3.5) * 0.12

            # ステレオ幅の拡張
            mid = (left + right) * 0.5
            side = (left - right) * 0.5 * 1.08
            left_wide = mid + side
            right_wide = mid - side

            enhanced_left = left_wide + left_harm
            enhanced_right = right_wide + right_harm
            enhanced_chunk = np.vstack([enhanced_left, enhanced_right])

            # 音割れ防止
            peak = np.max(np.abs(enhanced_chunk))
            if peak > 1.0:
                enhanced_chunk = enhanced_chunk / peak * 0.98

            # ダブらせて読み込んだ分を、書き出すときにカットして繋ぎ目を滑らかにする
            cut_start = int((current_time - start) * TARGET_SR)
            cut_end = cut_start + int(min(CHUNK_SEC, total_duration - current_time) * TARGET_SR)
            final_chunk = enhanced_chunk[:, cut_start:cut_end]

            # 音声ファイル（HDD）に追記
            out_enh.write(final_chunk.T)
            
            current_time += CHUNK_SEC
            await asyncio.sleep(0.01) # サーバーが他の処理もできるように一瞬譲る

    progress_store[task_id] = 95  # 圧縮前で95%

    # ZIP圧縮してまとめる
    shutil.make_archive(zip_output_path, 'zip', task_dir)
    final_zip = f"{zip_output_path}.zip"

    progress_store[task_id] = 100  # 100%完了！

    background_tasks.add_task(remove_file, task_dir)
    background_tasks.add_task(remove_file, final_zip)

    # フロント側に、進捗確認用としてtask_idをヘッダーに乗せてファイルを返す
    return FileResponse(
        final_zip,
        media_type='application/zip',
        filename=f"{base_name}_compressed.zip",
        headers={"X-Task-ID": task_id}
    )