"""
指定フォルダ配下の音声ファイルの時間を算出し、合計時間と20%区間ごとの時間配分を表示する

このスクリプトは以下の機能を提供します:
- 指定フォルダ配下の全音声ファイルを再帰的に検索
- 各ファイルの長さ（分秒）を算出
- 合計時間を計算
- 全体を20%ずつの区間に分割し、各区間の音声時間を表示
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from datetime import timedelta

try:
    import japanize_matplotlib
except ImportError:
    pass  # japanize_matplotlibがない場合は手動設定を使用


# サポートする音声ファイル拡張子
SUPPORTED_EXTENSIONS = {'.wav', '.flac', '.ogg', '.opus', '.mp3', '.aiff', '.aif', '.aifc'}


def find_audio_files(directory: Path, recursive: bool = True) -> List[Path]:
    """
    指定ディレクトリから音声ファイルを検索する
    
    Args:
        directory: 検索するディレクトリパス
        recursive: 再帰的に検索するか（デフォルト: True）
    
    Returns:
        検出された音声ファイルのパスリスト
    """
    audio_files = []
    
    if recursive:
        for ext in SUPPORTED_EXTENSIONS:
            audio_files.extend(directory.rglob(f'*{ext}'))
    else:
        for ext in SUPPORTED_EXTENSIONS:
            audio_files.extend(directory.glob(f'*{ext}'))
    
    return sorted(audio_files)


def get_audio_duration(file_path: Path) -> Optional[float]:
    """
    音声ファイルの長さ（秒）を取得する
    
    Args:
        file_path: 音声ファイルのパス
    
    Returns:
        音声の長さ（秒）。取得失敗時はNone
    """
    try:
        info = sf.info(str(file_path))
        return info.duration
    except Exception as e:
        print(f"⚠️  ファイル読み込みエラー: {file_path.name} - {e}")
        return None


def format_time(seconds: float) -> str:
    """
    秒数を「X分Y秒」形式にフォーマットする
    
    Args:
        seconds: 秒数
    
    Returns:
        フォーマットされた時間文字列
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}分{secs}秒"


def format_time_detailed(seconds: float) -> str:
    """
    秒数を「X時間Y分Z秒」形式にフォーマットする
    
    Args:
        seconds: 秒数
    
    Returns:
        フォーマットされた時間文字列
    """
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = int(td.total_seconds() % 60)
    
    if hours > 0:
        return f"{hours}時間{minutes}分{secs}秒"
    elif minutes > 0:
        return f"{minutes}分{secs}秒"
    else:
        return f"{secs}秒"


def calculate_percentile_distribution(
    file_durations: List[Tuple[Path, float]], 
    total_duration: float
) -> List[Dict[str, any]]:
    """
    ファイル数を20%ずつの区間に分割し、各区間の音声時間を算出する
    
    Args:
        file_durations: (ファイルパス, 長さ) のタプルリスト
        total_duration: 合計時間（秒）
    
    Returns:
        各区間の情報を含む辞書のリスト
    """
    percentiles = [0, 20, 40, 60, 80, 100]
    distribution = []
    total_files = len(file_durations)
    
    for i in range(len(percentiles) - 1):
        # 各区間のファイルインデックス範囲を計算
        start_idx = int(total_files * percentiles[i] / 100)
        end_idx = int(total_files * percentiles[i + 1] / 100)
        
        # 最後の区間は端数を含める
        if i == len(percentiles) - 2:
            end_idx = total_files
        
        # 区間内のファイルと時間を集計
        segment_files = file_durations[start_idx:end_idx]
        segment_duration = sum(duration for _, duration in segment_files)
        
        distribution.append({
            'range': f"{percentiles[i]}%-{percentiles[i + 1]}%",
            'duration': segment_duration,
            'file_count': len(segment_files),
            'files': segment_files
        })
    
    return distribution


def visualize_distribution(
    distribution: List[Dict[str, any]],
    total_duration: float,
    output_path: str = 'audio_duration_analysis.png'
) -> None:
    """
    20%区間ごとの時間配分を円グラフで視覚化する
    
    Args:
        distribution: 20%区間ごとの配分情報
        total_duration: 合計時間（秒）
        output_path: 出力画像のパス
    """
    # 日本語フォント設定
    matplotlib.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # データ準備
    ranges = [seg['range'] for seg in distribution]
    durations = [seg['duration'] for seg in distribution]
    
    # カラーパレット（matplotlibのカラーマップを使用）
    cmap = plt.cm.get_cmap('tab10')
    colors = [cmap(i / len(ranges)) for i in range(len(ranges))]
    
    # 図の作成
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.suptitle('🎵 音声ファイル時間配分分析', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 円グラフ: 時間配分（割合と実際の時間を表示）
    wedges, texts, autotexts = ax.pie(
        durations,
        labels=ranges,
        autopct=lambda pct: f'{pct:.1f}%\n{format_time(pct * total_duration / 100)}',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 11, 'weight': 'bold'},
        pctdistance=0.85,
        wedgeprops=dict(edgecolor='white', linewidth=3)
    )
    
    # タイトル
    ax.set_title('⏱️ 20%区間ごとの時間配分', fontsize=14, fontweight='bold', pad=20)
    
    # 凡例に詳細情報を追加
    legend_labels = [
        f"{seg['range']}: {format_time(seg['duration'])} ({seg['file_count']}個)"
        for seg in distribution
    ]
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=10, title='📊 詳細', title_fontsize=12)
    
    # 統計情報テキスト
    stats_text = f"""📈 総合統計
━━━━━━━━━━━━━━━━━━
総合計時間: {format_time_detailed(total_duration)}
総ファイル数: {sum(seg['file_count'] for seg in distribution)}個
平均時間/区間: {format_time(total_duration / len(ranges))}
平均ファイル数/区間: {sum(seg['file_count'] for seg in distribution) / len(distribution):.1f}個"""
    
    fig.text(0.15, 0.02, stats_text, fontsize=10, 
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n📊 グラフを保存しました: {output_path}")


def display_results(
    audio_files: List[Path],
    file_durations: List[Tuple[Path, float]],
    total_duration: float,
    distribution: List[Dict[str, any]],
    verbose: bool = False
) -> None:
    """
    結果を視覚的に表示する
    
    Args:
        audio_files: 検出された全音声ファイルのリスト
        file_durations: 有効な(ファイルパス, 長さ)のタプルリスト
        total_duration: 合計時間（秒）
        distribution: 20%区間ごとの配分情報
        verbose: 詳細表示するか
    """
    print("\n" + "=" * 70)
    print("📊 音声ファイル時間算出結果")
    print("=" * 70)
    
    print(f"\n✅ 検出されたファイル数: {len(audio_files)}個")
    print(f"✅ 有効なファイル数: {len(file_durations)}個")
    
    if len(file_durations) < len(audio_files):
        failed_count = len(audio_files) - len(file_durations)
        print(f"❌ 読み込み失敗: {failed_count}個")
    
    print("\n" + "=" * 70)
    print(f"⏱️  総合計時間: {format_time_detailed(total_duration)}")
    print(f"   ({total_duration:.2f}秒 / {total_duration/60:.2f}分 / {total_duration/3600:.2f}時間)")
    print("=" * 70)
    
    # 詳細表示: 個別ファイルの時間
    if verbose and file_durations:
        print("\n" + "-" * 70)
        print("📁 個別ファイルの長さ:")
        print("-" * 70)
        for file_path, duration in file_durations:
            print(f"  • {file_path.name}: {format_time(duration)} ({duration:.2f}秒)")
    
    # 20%区間ごとの配分
    if distribution:
        print("\n" + "-" * 70)
        print("📊 20%区間ごとの時間配分:")
        print("-" * 70)
        for segment in distribution:
            percentage = (segment['duration'] / total_duration * 100) if total_duration > 0 else 0
            print(f"\n  {segment['range']}:")
            print(f"    時間: {format_time(segment['duration'])} ({segment['duration']:.2f}秒)")
            print(f"    割合: {percentage:.1f}%")
            print(f"    ファイル数: {segment['file_count']}個")
            
            if verbose and segment['files']:
                print(f"    ファイル:")
                for file_path, duration in segment['files']:
                    print(f"      - {file_path.name} ({format_time(duration)})")
    
    print("\n" + "=" * 70)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="指定フォルダ配下の音声ファイルの時間を算出し、合計時間と20%区間ごとの時間配分を表示します"
    )
    parser.add_argument(
        'directory',
        type=str,
        help='音声ファイルが含まれるディレクトリパス'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='サブディレクトリを検索しない（指定ディレクトリのみ検索）'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='詳細な情報を表示（個別ファイルと区間ごとのファイルリスト）'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='時間配分をグラフで視覚化（PNG画像として保存）'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='audio_duration_analysis.png',
        help='視覚化画像の出力パス（デフォルト: audio_duration_analysis.png）'
    )
    
    args = parser.parse_args()
    
    # ディレクトリの存在確認
    directory = Path(args.directory)
    if not directory.exists():
        print(f"❌ エラー: ディレクトリが見つかりません: {directory}")
        sys.exit(1)
    
    if not directory.is_dir():
        print(f"❌ エラー: 指定されたパスはディレクトリではありません: {directory}")
        sys.exit(1)
    
    print(f"\n🔍 音声ファイルを検索中: {directory}")
    print(f"   再帰検索: {'無効' if args.no_recursive else '有効'}")
    
    # 音声ファイルの検索
    audio_files = find_audio_files(directory, recursive=not args.no_recursive)
    
    if not audio_files:
        print(f"\n⚠️  音声ファイルが見つかりませんでした")
        print(f"   サポート形式: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(0)
    
    print(f"✅ {len(audio_files)}個の音声ファイルを検出")
    
    # 各ファイルの長さを取得
    print(f"\n⏱️  音声ファイルの長さを算出中...")
    file_durations = []
    
    for audio_file in audio_files:
        duration = get_audio_duration(audio_file)
        if duration is not None:
            file_durations.append((audio_file, duration))
    
    if not file_durations:
        print(f"\n❌ エラー: 有効な音声ファイルがありませんでした")
        sys.exit(1)
    
    # 合計時間を算出
    total_duration = sum(duration for _, duration in file_durations)
    
    # 20%区間ごとの配分を算出
    distribution = calculate_percentile_distribution(file_durations, total_duration)
    
    # 結果表示
    display_results(audio_files, file_durations, total_duration, distribution, args.verbose)
    
    # 視覚化
    if args.visualize:
        try:
            visualize_distribution(distribution, total_duration, args.output)
        except Exception as e:
            print(f"\n⚠️  グラフの生成に失敗しました: {e}")
    
    print(f"\n✅ 処理が完了しました\n")


if __name__ == "__main__":
    main()
