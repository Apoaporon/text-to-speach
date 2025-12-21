"""
音声ファイルを一括で分割するスクリプト
"""
import argparse
from pathlib import Path

from src.audio_silence_processor import AudioSilenceProcessor


def split_directory(
    input_dir: str,
    output_dir: str,
    split_mode: str = 'duration',
    duration_seconds: float = 30.0,
    silence_thresh_db: float = -40.0,
    min_voice_duration: float = 0.5,
    min_silence_duration: float = 1.0,
    pattern: str = "*.wav"
) -> None:
    """
    ディレクトリ内の音声ファイルを一括分割
    
    Args:
        input_dir: 入力ディレクトリ
        output_dir: 出力ディレクトリ
        split_mode: 分割モード ('duration' or 'vad')
        duration_seconds: 分割する秒数（durationモード用）
        silence_thresh_db: 無音閾値（VADモード用）
        min_voice_duration: 最小音声区間長（VADモード用）
        min_silence_duration: 最小無音区間長（VADモード用）
        pattern: ファイルパターン
    """
    processor = AudioSilenceProcessor(verbose=True)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 対象ファイルを取得
    audio_files = list(input_path.glob(pattern))
    
    if not audio_files:
        print(f"❌ 音声ファイルが見つかりません: {input_dir}/{pattern}")
        return
    
    print(f"\n{'=' * 80}")
    print(f"【一括分割処理】")
    print(f"{'=' * 80}")
    print(f"入力ディレクトリ: {input_dir}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"分割モード: {split_mode}")
    if split_mode == 'duration':
        print(f"分割秒数: {duration_seconds}秒")
    else:
        print(f"無音閾値: {silence_thresh_db} dB")
        print(f"最小音声区間: {min_voice_duration}秒")
        print(f"最小無音区間: {min_silence_duration}秒")
    print(f"対象ファイル数: {len(audio_files)}個")
    print(f"{'=' * 80}\n")
    
    success_count = 0
    total_segments = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n{'=' * 80}")
        print(f"処理中 [{i}/{len(audio_files)}]: {audio_file.name}")
        print(f"{'=' * 80}")
        
        # ファイルごとの出力ディレクトリを作成
        file_output_dir = output_path / audio_file.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if split_mode == 'duration':
                # 指定秒数ごとに分割
                split_files = processor.split_by_duration(
                    str(audio_file),
                    str(file_output_dir),
                    duration_seconds=duration_seconds,
                    prefix=audio_file.stem
                )
            else:
                # VAD（無音区間）で分割
                split_files = processor.split_by_vad(
                    str(audio_file),
                    str(file_output_dir),
                    silence_thresh_db=silence_thresh_db,
                    min_voice_duration=min_voice_duration,
                    min_silence_duration=min_silence_duration,
                    prefix=audio_file.stem
                )
            
            if split_files:
                success_count += 1
                total_segments += len(split_files)
                print(f"✅ {len(split_files)}個のセグメントに分割完了")
            else:
                print(f"⚠️ 分割に失敗しました")
                
        except Exception as e:
            print(f"❌ エラー: {e}")
    
    print(f"\n{'=' * 80}")
    print(f"【処理完了】")
    print(f"{'=' * 80}")
    print(f"成功: {success_count}/{len(audio_files)}ファイル")
    print(f"総セグメント数: {total_segments}個")
    print(f"出力先: {output_dir}")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="音声ファイルを一括で分割",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 30秒ごとに分割
  python split_audio_batch.py downloads/soshina audio/output/split --mode duration --duration 30

  # 無音区間で分割
  python split_audio_batch.py downloads/soshina audio/output/split --mode vad --min-silence 1.0

  # m4aファイルを処理
  python split_audio_batch.py downloads/soshina audio/output/split --pattern "*.m4a" --duration 30
        """
    )
    
    parser.add_argument("input_dir", help="入力ディレクトリ")
    parser.add_argument("output_dir", help="出力ディレクトリ")
    parser.add_argument("--mode", choices=['duration', 'vad'], default='duration', 
                        help="分割モード (デフォルト: duration)")
    parser.add_argument("--pattern", default="*.wav", 
                        help="ファイルパターン (デフォルト: *.wav)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="[durationモード] 分割する秒数 (デフォルト: 30.0)")
    parser.add_argument("--silence-thresh", type=float, default=-40.0,
                        help="[VADモード] 無音閾値 dBFS (デフォルト: -40.0)")
    parser.add_argument("--min-voice", type=float, default=0.5,
                        help="[VADモード] 最小音声区間長 秒 (デフォルト: 0.5)")
    parser.add_argument("--min-silence", type=float, default=1.0,
                        help="[VADモード] 最小無音区間長 秒 (デフォルト: 1.0)")
    
    args = parser.parse_args()
    
    split_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split_mode=args.mode,
        duration_seconds=args.duration,
        silence_thresh_db=args.silence_thresh,
        min_voice_duration=args.min_voice,
        min_silence_duration=args.min_silence,
        pattern=args.pattern
    )


if __name__ == "__main__":
    main()
