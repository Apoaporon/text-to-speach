"""
音声ファイルを指定秒数ごとに分割するスクリプト
"""
import os
from pathlib import Path

from pydub import AudioSegment


def split_audio_by_seconds(input_file: str, segment_duration: int = 5, output_dir: str = "split_segments"):
    """
    音声ファイルを指定秒数ごとに分割する
    
    Args:
        input_file: 入力音声ファイルのパス
        segment_duration: 分割する秒数 (デフォルト: 5秒)
        output_dir: 出力ディレクトリ (デフォルト: split_segments)
    """
    # 入力ファイルの存在確認
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 音声ファイルを読み込み
    print(f"音声ファイルを読み込み中: {input_file}")
    audio = AudioSegment.from_wav(input_file)
    
    # 音声の総時間（ミリ秒）
    total_duration_ms = len(audio)
    total_duration_sec = total_duration_ms / 1000
    
    print(f"総時間: {total_duration_sec:.2f}秒")
    print(f"{segment_duration}秒ごとに分割します...")
    
    # セグメントの長さ（ミリ秒）
    segment_duration_ms = segment_duration * 1000
    
    # ファイル名の準備
    input_path = Path(input_file)
    base_name = input_path.stem
    
    # 分割処理
    segment_count = 0
    for start_ms in range(0, total_duration_ms, segment_duration_ms):
        end_ms = min(start_ms + segment_duration_ms, total_duration_ms)
        
        # セグメントを切り出し
        segment = audio[start_ms:end_ms]
        
        # 出力ファイル名
        output_file = os.path.join(
            output_dir,
            f"{base_name}_segment_{segment_count:04d}_{start_ms}_{end_ms}.wav"
        )
        
        # 保存
        segment.export(output_file, format="wav")
        duration = (end_ms - start_ms) / 1000
        print(f"保存: {output_file} (長さ: {duration:.2f}秒)")
        
        segment_count += 1
    
    print(f"\n完了: {segment_count}個のセグメントを作成しました")


def main():
    """メイン処理"""
    # 設定
    input_file = "wav_file/sample.wav"  # 入力ファイルパス
    segment_duration = 5  # 分割する秒数
    output_dir = "split_segments"  # 出力ディレクトリ
    
    # 分割実行
    split_audio_by_seconds(input_file, segment_duration, output_dir)


if __name__ == "__main__":
    main()
