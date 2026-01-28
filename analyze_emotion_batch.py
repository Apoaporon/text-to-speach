"""
音声ファイルの感情を一括分析するスクリプト

指定されたディレクトリ内の音声ファイルを読み込み、
各ファイルの感情を分析して結果をJSON/CSV形式で保存します。
"""
import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

from src.emotion_recognizer import EmotionRecognizer


# サポートされる音声ファイル拡張子
SUPPORTED_EXTENSIONS = {'.wav', '.flac', '.ogg', '.opus', '.mp3', '.aiff', '.aif', '.aifc'}


def get_audio_files(
    directory: Path,
    recursive: bool = False,
    pattern: str = "*.wav"
) -> List[Path]:
    """
    ディレクトリから音声ファイルを取得

    Args:
        directory: 検索するディレクトリ
        recursive: サブディレクトリも検索するか
        pattern: ファイルパターン（recursiveがFalseの場合のみ使用）

    Returns:
        音声ファイルパスのリスト
    """
    audio_files = []
    
    if recursive:
        # 再帰的に全ての音声ファイルを検索
        for ext in SUPPORTED_EXTENSIONS:
            audio_files.extend(directory.rglob(f'*{ext}'))
    else:
        # パターンに一致するファイルのみ検索
        audio_files = list(directory.glob(pattern))
    
    # 重複を除去してソート
    audio_files = sorted(set(audio_files))
    
    return audio_files


def get_audio_duration(file_path: Path) -> float:
    """
    音声ファイルの長さを取得（秒）

    Args:
        file_path: 音声ファイルパス

    Returns:
        音声の長さ（秒）、エラーの場合は0.0
    """
    try:
        info = sf.info(str(file_path))
        return info.duration
    except Exception as e:
        print(f"⚠️  ファイル読み込みエラー: {file_path.name} - {e}")
        return 0.0


def calculate_summary(results: List[Dict]) -> Dict:
    """
    感情分析結果のサマリー統計を計算

    Args:
        results: 感情認識結果のリスト

    Returns:
        サマリー統計の辞書
    """
    # エラーがない結果のみを抽出
    valid_results = [r for r in results if r['error'] is None]
    
    if not valid_results:
        return {
            'total_duration': 0.0,
            'emotion_distribution': {},
            'average_confidence': 0.0,
            'total_processing_time': 0.0
        }
    
    # 感情分布の集計
    emotions_count = Counter(r['dominant_emotion'] for r in valid_results)
    
    # 合計時間の計算
    total_duration = sum(r['duration'] for r in valid_results)
    total_processing_time = sum(r['processing_time'] for r in valid_results)
    
    # 平均信頼度の計算
    average_confidence = np.mean([r['confidence'] for r in valid_results])
    
    # 感情ごとの統計
    emotion_stats = {}
    for emotion in set(r['dominant_emotion'] for r in valid_results):
        emotion_results = [r for r in valid_results if r['dominant_emotion'] == emotion]
        emotion_stats[emotion] = {
            'count': len(emotion_results),
            'percentage': len(emotion_results) / len(valid_results) * 100,
            'average_confidence': np.mean([r['confidence'] for r in emotion_results]),
            'total_duration': sum(r['duration'] for r in emotion_results)
        }
    
    return {
        'total_files': len(results),
        'success_count': len(valid_results),
        'error_count': len(results) - len(valid_results),
        'total_duration': total_duration,
        'total_processing_time': total_processing_time,
        'average_processing_speed': total_duration / total_processing_time if total_processing_time > 0 else 0,
        'emotion_distribution': dict(emotions_count),
        'emotion_stats': emotion_stats,
        'average_confidence': float(average_confidence)
    }


def save_results_json(
    results: List[Dict],
    output_file: Path,
    summary: Dict
) -> None:
    """
    結果をJSON形式で保存

    Args:
        results: 感情認識結果のリスト
        output_file: 出力ファイルパス
        summary: サマリー統計
    """
    output_data = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0'
        },
        'summary': summary,
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ JSON結果を保存: {output_file}")


def save_results_csv(
    results: List[Dict],
    output_file: Path
) -> None:
    """
    結果をCSV形式で保存

    Args:
        results: 感情認識結果のリスト
        output_file: 出力ファイルパス
    """
    if not results:
        print("⚠️  保存する結果がありません")
        return
    
    # 全ての感情ラベルを取得
    emotion_labels = set()
    for result in results:
        if result['emotions']:
            emotion_labels.update(result['emotions'].keys())
    emotion_labels = sorted(emotion_labels)
    
    # CSVヘッダー
    fieldnames = [
        'filename',
        'filepath',
        'duration',
        'dominant_emotion',
        'confidence',
        'processing_time',
        'error'
    ] + emotion_labels
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {
                'filename': result['filename'],
                'filepath': result['filepath'],
                'duration': result['duration'],
                'dominant_emotion': result['dominant_emotion'],
                'confidence': result['confidence'],
                'processing_time': result['processing_time'],
                'error': result['error'] or ''
            }
            # 各感情のスコアを追加
            for emotion in emotion_labels:
                row[emotion] = result['emotions'].get(emotion, 0.0)
            
            writer.writerow(row)
    
    print(f"✅ CSV結果を保存: {output_file}")


def analyze_directory(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.wav",
    recursive: bool = False,
    model_name: str = None,
    device: str = None,
    output_format: str = "both"
) -> None:
    """
    ディレクトリ内の音声ファイルを一括で感情分析

    Args:
        input_dir: 入力ディレクトリ
        output_dir: 出力ディレクトリ
        pattern: ファイルパターン（recursiveがFalseの場合のみ使用）
        recursive: サブディレクトリも検索するか
        model_name: 使用するモデル名
        device: 使用するデバイス（'cuda', 'cpu', Noneで自動）
        output_format: 出力形式（'json', 'csv', 'both'）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 音声ファイルを取得
    print(f"\n🔍 音声ファイルを検索中...")
    audio_files = get_audio_files(input_path, recursive=recursive, pattern=pattern)
    
    if not audio_files:
        print(f"❌ 音声ファイルが見つかりません")
        if not recursive:
            print(f"   検索パターン: {input_dir}/{pattern}")
        else:
            print(f"   検索ディレクトリ: {input_dir} (再帰的)")
        return
    
    print(f"\n{'=' * 80}")
    print(f"【音声感情分析 - 一括処理】")
    print(f"{'=' * 80}")
    print(f"入力ディレクトリ: {input_dir}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"対象ファイル数: {len(audio_files)}個")
    if not recursive:
        print(f"ファイルパターン: {pattern}")
    else:
        print(f"検索モード: 再帰的")
    print(f"出力形式: {output_format}")
    print(f"{'=' * 80}\n")
    
    # 感情認識器の初期化
    recognizer = EmotionRecognizer(
        model_name=model_name,
        device=device,
        verbose=True
    )
    
    # デバイス情報の表示
    device_info = recognizer.get_device_info()
    print(f"\n{'=' * 80}")
    print("デバイス情報:")
    print(f"{'=' * 80}")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    print(f"{'=' * 80}\n")
    
    # 一括処理の実行
    results = []
    
    print(f"\n🎭 感情分析を開始します...\n")
    
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        print("⚠️  tqdmがインストールされていません。プログレスバーなしで実行します。")
        use_tqdm = False
    
    iterator = tqdm(audio_files, desc="感情分析処理中") if use_tqdm else audio_files
    
    for i, audio_file in enumerate(iterator, 1):
        if not use_tqdm:
            print(f"\n{'=' * 80}")
            print(f"処理中 [{i}/{len(audio_files)}]: {audio_file.name}")
            print(f"{'=' * 80}")
        
        try:
            # 音声の長さを取得
            duration = get_audio_duration(audio_file)
            
            # 感情認識を実行
            result = recognizer.recognize_emotion(str(audio_file))
            
            # 結果に追加情報を付加
            result['filename'] = audio_file.name
            result['filepath'] = str(audio_file)
            result['duration'] = duration
            
            results.append(result)
            
            if not use_tqdm and result['error'] is None:
                print(f"✅ 支配的感情: {result['dominant_emotion']} (信頼度: {result['confidence']:.2%})")
                print(f"   処理時間: {result['processing_time']:.3f}秒")
            
        except Exception as e:
            print(f"❌ エラー: {e}")
            results.append({
                'filename': audio_file.name,
                'filepath': str(audio_file),
                'duration': 0.0,
                'emotions': {},
                'dominant_emotion': None,
                'confidence': 0.0,
                'processing_time': 0.0,
                'error': str(e)
            })
    
    # サマリー統計の計算
    summary = calculate_summary(results)
    
    # 結果の保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_format in ['json', 'both']:
        json_file = output_path / f"emotion_analysis_{timestamp}.json"
        save_results_json(results, json_file, summary)
    
    if output_format in ['csv', 'both']:
        csv_file = output_path / f"emotion_analysis_{timestamp}.csv"
        save_results_csv(results, csv_file)
    
    # 結果サマリーの表示
    print(f"\n{'=' * 80}")
    print("【分析結果サマリー】")
    print(f"{'=' * 80}")
    print(f"総ファイル数: {summary['total_files']}個")
    print(f"成功: {summary['success_count']}個")
    print(f"エラー: {summary['error_count']}個")
    print(f"総音声時間: {summary['total_duration']:.1f}秒 ({summary['total_duration']/60:.1f}分)")
    print(f"総処理時間: {summary['total_processing_time']:.1f}秒")
    if summary['total_processing_time'] > 0:
        print(f"処理速度: {summary['average_processing_speed']:.2f}x (リアルタイム比)")
    print(f"平均信頼度: {summary['average_confidence']:.2%}")
    
    print(f"\n【感情分布】")
    if summary['emotion_distribution']:
        for emotion, count in sorted(
            summary['emotion_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            percentage = count / summary['success_count'] * 100
            bar = '█' * int(percentage / 2)
            print(f"  {emotion:10s}: {count:4d}個 ({percentage:5.1f}%) {bar}")
            
            # 詳細統計
            if emotion in summary['emotion_stats']:
                stats = summary['emotion_stats'][emotion]
                print(f"              平均信頼度: {stats['average_confidence']:.2%}, "
                      f"合計時間: {stats['total_duration']:.1f}秒")
    else:
        print("  感情データなし")
    
    print(f"{'=' * 80}\n")
    print(f"✅ 処理完了！")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="音声ファイルの感情を一括分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使い方（*.wavファイルを分析）
  python analyze_emotion_batch.py input_dir output_dir

  # 特定のパターンのファイルを分析
  python analyze_emotion_batch.py input_dir output_dir --pattern "*.flac"

  # サブディレクトリも含めて再帰的に分析
  python analyze_emotion_batch.py input_dir output_dir --recursive

  # JSON形式のみで出力
  python analyze_emotion_batch.py input_dir output_dir --format json

  # CPUを強制的に使用
  python analyze_emotion_batch.py input_dir output_dir --device cpu

  # カスタムモデルを使用
  python analyze_emotion_batch.py input_dir output_dir --model "your-model-name"
        """
    )
    
    parser.add_argument(
        "input_dir",
        help="入力ディレクトリ（音声ファイルが含まれるフォルダ）"
    )
    parser.add_argument(
        "output_dir",
        help="出力ディレクトリ（分析結果を保存するフォルダ）"
    )
    parser.add_argument(
        "--pattern",
        default="*.wav",
        help="ファイルパターン（デフォルト: *.wav）"
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="サブディレクトリも再帰的に検索"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="使用するモデル名（デフォルト: Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition）"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="使用するデバイス（デフォルト: auto）"
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "both"],
        default="both",
        help="出力形式（デフォルト: both）"
    )
    
    args = parser.parse_args()
    
    # デバイスの設定
    device = None if args.device == "auto" else args.device
    
    # 分析実行
    analyze_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        recursive=args.recursive,
        model_name=args.model,
        device=device,
        output_format=args.format
    )


if __name__ == "__main__":
    main()
