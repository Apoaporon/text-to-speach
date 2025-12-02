"""
音声のレベル系（音量・クリッピング）を分析・修正するスクリプト
"""
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from scipy import signal


class AudioLevelProcessor:
    """音声レベル処理クラス"""

    def __init__(self, verbose: bool = True):
        """
        初期化

        Args:
            verbose: 処理状況を出力するかどうか
        """
        self.verbose = verbose

    def _print(self, message: str) -> None:
        """verboseがTrueの場合のみ出力"""
        if self.verbose:
            print(message)

    def analyze_audio_levels(
        self,
        file_path: str
    ) -> Optional[Dict[str, float]]:
        """
        音声ファイルのレベル情報を詳細に分析する
        
        Args:
            file_path: 音声ファイルのパス
        
        Returns:
            レベル情報の辞書
            {
                'peak_db': ピークレベル(dBFS),
                'rms_db': RMSレベル(dBFS),
                'lufs': ラウドネス(LUFS推定値),
                'crest_factor': クレストファクター(dB),
                'clipping_samples': クリッピングサンプル数,
                'clipping_percentage': クリッピング率(%),
                'dynamic_range': ダイナミックレンジ(dB),
                'headroom': ヘッドルーム(dB)
            }
        """
        try:
            # 音声データを読み込み
            y, sr = librosa.load(file_path, sr=None, mono=False)
            
            # ステレオの場合は平均化
            if y.ndim > 1:
                y = np.mean(y, axis=0)
            
            # ピークレベル（dBFS）
            peak_amplitude = np.max(np.abs(y))
            peak_db = 20 * np.log10(peak_amplitude) if peak_amplitude > 0 else -np.inf
            
            # RMSレベル（dBFS）
            rms = np.sqrt(np.mean(y**2))
            rms_db = 20 * np.log10(rms) if rms > 0 else -np.inf
            
            # ラウドネス（簡易LUFS推定）
            # 実際のLUFSはEBU R128に準拠した計算が必要だが、ここでは簡易版
            lufs_estimate = rms_db - 23.0  # 簡易的な推定値
            
            # クレストファクター（ピークとRMSの比）
            crest_factor = peak_db - rms_db
            
            # クリッピング検出（振幅が0.99以上のサンプル）
            clipping_threshold = 0.99
            clipping_samples = np.sum(np.abs(y) >= clipping_threshold)
            clipping_percentage = (clipping_samples / len(y)) * 100
            
            # ダイナミックレンジ（ピークとノイズフロアの差）
            # ノイズフロアは最小RMS（1秒ごとのRMS最小値）で推定
            frame_length = sr  # 1秒フレーム
            hop_length = sr // 2
            frame_rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            noise_floor = np.percentile(frame_rms, 5)  # 下位5%をノイズフロアとする
            noise_floor_db = 20 * np.log10(noise_floor) if noise_floor > 0 else -np.inf
            dynamic_range = peak_db - noise_floor_db
            
            # ヘッドルーム（0dBFSまでの余裕）
            headroom = 0.0 - peak_db
            
            return {
                'peak_db': round(peak_db, 2),
                'rms_db': round(rms_db, 2),
                'lufs': round(lufs_estimate, 2),
                'crest_factor': round(crest_factor, 2),
                'clipping_samples': int(clipping_samples),
                'clipping_percentage': round(clipping_percentage, 4),
                'dynamic_range': round(dynamic_range, 2),
                'headroom': round(headroom, 2)
            }
            
        except Exception as e:
            self._print(f"エラーが発生しました: {e}")
            return None

    def print_level_analysis(self, info: Dict[str, float], filename: str) -> None:
        """
        レベル分析結果を整形して出力する
        
        Args:
            info: レベル情報の辞書
            filename: ファイル名
        """
        self._print("\n" + "=" * 70)
        self._print(f"【音量・クリッピング分析】: {filename}")
        self._print("=" * 70)
        self._print(f"ピークレベル      : {info['peak_db']:+.2f} dBFS")
        self._print(f"RMSレベル         : {info['rms_db']:+.2f} dBFS")
        self._print(f"ラウドネス(推定)  : {info['lufs']:+.2f} LUFS")
        self._print(f"クレストファクター: {info['crest_factor']:+.2f} dB")
        self._print(f"ダイナミックレンジ: {info['dynamic_range']:+.2f} dB")
        self._print(f"ヘッドルーム      : {info['headroom']:+.2f} dB")
        self._print("-" * 70)
        self._print(f"クリッピング      : {info['clipping_samples']}サンプル ({info['clipping_percentage']:.4f}%)")
        
        # 推奨事項を表示
        self._print("-" * 70)
        self._print("【推奨事項】")
    
        # ピークレベルのチェック
        if info['peak_db'] > -1.0:
            self._print("⚠️  ピークが高すぎます（-1dBFS以上）→ ピークノーマライズ推奨（-1〜-3dBFS）")
        elif info['peak_db'] < -10.0:
            self._print("⚠️  ピークが低すぎます（-10dBFS以下）→ ゲインアップ推奨")
        else:
            self._print("✅ ピークレベルは適切です")
        
        # ラウドネスのチェック
        if info['lufs'] > -16.0:
            self._print("⚠️  ラウドネスが高すぎます → ラウドネスノーマライズ推奨（-20〜-16 LUFS）")
        elif info['lufs'] < -30.0:
            self._print("⚠️  ラウドネスが低すぎます → ラウドネスノーマライズ推奨")
        else:
            self._print("✅ ラウドネスは適切です")
        
        # クリッピングのチェック
        if info['clipping_percentage'] > 0.01:
            self._print(f"🔴 クリッピング検出！ ({info['clipping_percentage']:.4f}%) → デクリップ処理推奨")
        elif info['clipping_percentage'] > 0:
            self._print(f"⚠️  わずかなクリッピングあり ({info['clipping_percentage']:.4f}%)")
        else:
            self._print("✅ クリッピングなし")
        
        # ヘッドルームのチェック
        if info['headroom'] < 1.0:
            self._print("⚠️  ヘッドルームが不足 → ピークを下げることを推奨")
        
        self._print("=" * 70 + "\n")

    def normalize_peak(
        self,
        input_file: str,
        output_file: str,
        target_db: float = -1.0
    ) -> bool:
        """
        ピークノーマライズを実行する
        
        Args:
            input_file: 入力ファイルパス
            output_file: 出力ファイルパス
            target_db: 目標ピークレベル(dBFS) デフォルト: -1.0
        
        Returns:
            成功時True
        """
        try:
            self._print(f"ピークノーマライズ中: {Path(input_file).name} → 目標: {target_db} dBFS")
            
            # 音声データを読み込み
            y, sr = librosa.load(input_file, sr=None, mono=False)
            
            # 現在のピーク
            peak = np.max(np.abs(y))
            
            # 目標ピークに対するゲイン計算
            target_amplitude = 10 ** (target_db / 20)
            gain = target_amplitude / peak if peak > 0 else 1.0
            
            # ゲインを適用
            y_normalized = y * gain
            
            # 保存
            sf.write(output_file, y_normalized.T if y.ndim > 1 else y_normalized, sr)
            
            self._print(f"✅ 完了: {output_file}")
            self._print(f"   適用ゲイン: {20 * np.log10(gain):+.2f} dB")
            
            return True
            
        except Exception as e:
            self._print(f"エラー: {e}")
            return False

    def normalize_loudness(
        self,
        input_file: str,
        output_file: str,
        target_lufs: float = -20.0
    ) -> bool:
        """
        ラウドネスノーマライズを実行する（簡易版）
        
        Args:
            input_file: 入力ファイルパス
            output_file: 出力ファイルパス
            target_lufs: 目標ラウドネス(LUFS) デフォルト: -20.0
        
        Returns:
            成功時True
        """
        try:
            self._print(f"ラウドネスノーマライズ中: {Path(input_file).name} → 目標: {target_lufs} LUFS")
            
            # 音声データを読み込み
            y, sr = librosa.load(input_file, sr=None, mono=False)
            
            # ステレオの場合は平均化
            y_mono = np.mean(y, axis=0) if y.ndim > 1 else y
            
            # 現在のRMSレベル
            rms = np.sqrt(np.mean(y_mono**2))
            rms_db = 20 * np.log10(rms) if rms > 0 else -np.inf
            
            # 簡易LUFS推定
            current_lufs = rms_db - 23.0
            
            # 目標LUFSに対するゲイン計算
            gain_db = target_lufs - current_lufs
            gain = 10 ** (gain_db / 20)
            
            # ゲインを適用
            y_normalized = y * gain
            
            # クリッピング防止
            peak = np.max(np.abs(y_normalized))
            if peak > 0.99:
                safety_gain = 0.99 / peak
                y_normalized *= safety_gain
                self._print(f"   クリッピング防止のため追加調整: {20 * np.log10(safety_gain):+.2f} dB")
            
            # 保存
            sf.write(output_file, y_normalized.T if y.ndim > 1 else y_normalized, sr)
            
            self._print(f"✅ 完了: {output_file}")
            self._print(f"   適用ゲイン: {gain_db:+.2f} dB")
            
            return True
            
        except Exception as e:
            self._print(f"エラー: {e}")
            return False

    def declip_audio(
        self,
        input_file: str,
        output_file: str,
        threshold: float = 0.99
    ) -> bool:
        """
        クリッピング修正（デクリップ）を実行する
        
        Args:
            input_file: 入力ファイルパス
            output_file: 出力ファイルパス
            threshold: クリッピング判定閾値 デフォルト: 0.99
        
        Returns:
            成功時True
        """
        try:
            self._print(f"デクリップ処理中: {Path(input_file).name}")
            
            # 音声データを読み込み
            y, sr = librosa.load(input_file, sr=None, mono=False)
            
            # クリッピング検出
            is_clipped = np.abs(y) >= threshold
            clipped_samples = np.sum(is_clipped)
            
            if clipped_samples == 0:
                self._print("   クリッピングが検出されませんでした")
                return False
            
            self._print(f"   クリッピング検出: {clipped_samples}サンプル")
            
            # 簡易的なデクリップ処理（ローパスフィルタで平滑化）
            if y.ndim == 1:
                # モノラル
                sos = signal.butter(4, 0.95, 'low', output='sos')
                y_declipped = signal.sosfilt(sos, y)
            else:
                # ステレオ
                sos = signal.butter(4, 0.95, 'low', output='sos')
                y_declipped = np.array([signal.sosfilt(sos, y[ch]) for ch in range(y.shape[0])])
            
            # クリッピング部分のみ置き換え
            y_result = np.where(is_clipped, y_declipped, y)
            
            # 全体の音量を少し下げる（-3dB）
            y_result *= 10 ** (-3 / 20)
            
            # 保存
            sf.write(output_file, y_result.T if y.ndim > 1 else y_result, sr)
            
            self._print(f"✅ 完了: {output_file}")
            self._print(f"   注意: デクリップ処理は完全ではありません。可能なら元音源を使用してください。")
            
            return True
            
        except Exception as e:
            self._print(f"エラー: {e}")
            return False

    def batch_normalize_directory(
        self,
        input_dir: str,
        output_dir: str,
        mode: str = "peak",
        target_value: float = -1.0
    ) -> None:
        """
        ディレクトリ内の全ファイルを一括ノーマライズ
        
        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            mode: "peak" or "loudness"
            target_value: 目標値（peak: dBFS, loudness: LUFS）
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 対象ファイルを取得
        audio_files = list(input_path.glob("*.wav")) + list(input_path.glob("*.mp3")) + list(input_path.glob("*.m4a"))
        
        self._print(f"\n{len(audio_files)}個のファイルを処理します")
        self._print(f"モード: {mode}, 目標値: {target_value}\n")
        
        for i, audio_file in enumerate(audio_files, 1):
            self._print(f"[{i}/{len(audio_files)}]")
            output_file = output_path / audio_file.name
            
            if mode == "peak":
                self.normalize_peak(str(audio_file), str(output_file), target_value)
            elif mode == "loudness":
                self.normalize_loudness(str(audio_file), str(output_file), target_value)
            self._print("")


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="音声レベル処理")
    parser.add_argument("input", help="入力ファイルまたはディレクトリ")
    parser.add_argument("output", nargs="?", help="出力ファイルまたはディレクトリ（省略時は自動生成）")
    parser.add_argument("--mode", choices=["analyze", "peak", "loudness", "declip"], default="analyze",
                        help="処理モード: analyze(分析のみ), peak(ピークノーマライズ), loudness(ラウドネスノーマライズ), declip(デクリップ)")
    parser.add_argument("--target", type=float, help="目標値 (peak: dBFS, loudness: LUFS)")
    parser.add_argument("--batch", action="store_true", help="ディレクトリ一括処理モード")
    parser.add_argument("--pattern", default="*.wav", help="バッチ処理時のファイルパターン")
    parser.add_argument("--quiet", action="store_true", help="詳細出力を抑制")
    parser.add_argument("--output-dir", default="audio/output", help="出力ディレクトリ（デフォルト: audio/output）")
    
    args = parser.parse_args()
    
    # プロセッサーのインスタンスを作成
    processor = AudioLevelProcessor(verbose=not args.quiet)
    
    # 入力パスの確認
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"エラー: ファイルが見つかりません: {args.input}")
        exit(1)
    
    # 出力パスが指定されていない場合は自動生成
    if args.output is None:
        if args.batch or input_path.is_dir():
            args.output = args.output_dir
        else:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            args.output = str(output_dir / f"{args.mode}_{input_path.name}")
        
        if not args.quiet:
            print(f"出力先: {args.output}")
    
    # 分析モード
    if args.mode == "analyze":
        if input_path.is_file():
            info = processor.analyze_audio_levels(str(input_path))
            if info:
                processor.print_level_analysis(info, input_path.name)
        else:
            print("分析モードはファイル単位でのみ実行できます")
            exit(1)
    
    # バッチ処理
    elif args.batch or input_path.is_dir():
        if args.mode == "peak":
            target = args.target if args.target is not None else -3.0
            processor.batch_normalize_directory(args.input, args.output, "peak", target)
        elif args.mode == "loudness":
            target = args.target if args.target is not None else -16.0
            processor.batch_normalize_directory(args.input, args.output, "loudness", target)
        elif args.mode == "declip":
            # デクリップのバッチ処理
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            files = list(input_path.glob(args.pattern))
            for i, file_path in enumerate(files, 1):
                print(f"[{i}/{len(files)}] {file_path.name}")
                output_file = output_path / file_path.name
                processor.declip_audio(str(file_path), str(output_file))
    
    # 単一ファイル処理
    else:
        if args.mode == "peak":
            target = args.target if args.target is not None else -3.0
            processor.normalize_peak(args.input, args.output, target)
        elif args.mode == "loudness":
            target = args.target if args.target is not None else -16.0
            processor.normalize_loudness(args.input, args.output, target)
        elif args.mode == "declip":
            processor.declip_audio(args.input, args.output)


if __name__ == "__main__":
    main()
