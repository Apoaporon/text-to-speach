"""
test_sample1 の Docstring
"""
import atexit

# PyTorchの終了時クリーンアップエラーを回避
# 必ずimport前に設定する
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# torchcodecを無効化してtorchaudioを使用
os.environ["PYANNOTE_AUDIO_USE_TORCHAUDIO"] = "1"

from dotenv import load_dotenv  # noqa: E402
from pyannote.audio import Pipeline  # noqa: E402
from pydub import AudioSegment  # noqa: E402


def cleanup_handler():
    """PyTorch終了時のクラッシュを防ぐための処理"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):  # torch未インストールまたはCUDA無効時
        pass
    # 正常終了を強制(デストラクタ呼び出しをスキップ)
    os._exit(0)


# プログラム終了時にクリーンアップハンドラを登録
atexit.register(cleanup_handler)


def main():
    # .envファイルから環境変数を読み込み
    load_dotenv()
    
    # 環境変数から Hugging Face トークン取得
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError("環境変数 HUGGINGFACE_TOKEN が設定されていません。")

    # pyannote の話者分離パイプラインをロード (最新の3.1を使用)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    # 入力音声ファイル
    from pathlib import Path

    import torchaudio
    
    audio_path = str(Path("wav_file/sample.wav").resolve())
    
    # 音声を事前にメモリにロード (torchaudio 2.1.0はtorchcodec不要)
    waveform, sample_rate = torchaudio.load(audio_path)
    audio_dict = {
        "waveform": waveform,
        "sample_rate": sample_rate
    }

    # 話者分離を実行
    diarization_output = pipeline(audio_dict)

    # 元音声を読み込み（切り出し用）
    audio = AudioSegment.from_wav(audio_path)

    # DiarizeOutputからAnnotationオブジェクトを取得
    annotation = diarization_output.speaker_diarization
    
    print(f"Found {len(list(annotation.itertracks()))} speech segments")
    
    # 話者ごとに区間を切り出して保存
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)

        audio_segment = audio[start_ms:end_ms]
        out_name = f"{speaker}_{start_ms}_{end_ms}.wav"
        audio_segment.export(out_name, format="wav")
        print(f"saved: {out_name} (duration: {segment.end - segment.start:.2f}s)")


if __name__ == "__main__":
    main()
