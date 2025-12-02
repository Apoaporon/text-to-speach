"""
音声分析結果に基づいて自動的に前処理を実行するパイプライン
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from analyze_audio import analyze_audio_file, analyze_audio_levels, detect_noise_types
from audio_denoise import AudioDenoiser
from audio_eq_processor import AudioEQProcessor
from audio_level_processor import AudioLevelProcessor
from audio_silence_processor import AudioSilenceProcessor


class AudioProcessingPipeline:
    """音声前処理パイプラインクラス"""

    def __init__(self, verbose: bool = True):
        """
        初期化

        Args:
            verbose: 処理状況を出力するかどうか
        """
        self.verbose = verbose
        self.denoiser = AudioDenoiser(verbose=verbose)
        self.eq_processor = AudioEQProcessor(verbose=verbose)
        self.level_processor = AudioLevelProcessor(verbose=verbose)
        self.silence_processor = AudioSilenceProcessor(verbose=verbose)

    def _print(self, message: str) -> None:
        """verboseがTrueの場合のみ出力"""
        if self.verbose:
            print(message)

    def analyze_and_plan(self, input_file: str) -> Dict[str, any]:
        """
        音声ファイルを分析し、必要な前処理を判定する

        Args:
            input_file: 入力音声ファイルのパス

        Returns:
            分析結果と処理計画の辞書
            {
                'file_info': ファイル情報,
                'level_info': レベル情報,
                'noise_info': ノイズ情報,
                'processing_plan': 実行する処理のリスト
            }
        """
        self._print("\n" + "=" * 80)
        self._print(f"【音声分析開始】: {Path(input_file).name}")
        self._print("=" * 80)

        # 基本情報の取得
        file_info = analyze_audio_file(input_file)
        if file_info is None:
            raise ValueError(f"ファイルの分析に失敗しました: {input_file}")

        # レベル情報の取得
        level_info = analyze_audio_levels(input_file)
        if level_info is None:
            raise ValueError(f"レベル分析に失敗しました: {input_file}")

        # ノイズ検出
        noise_info = detect_noise_types(input_file)
        if noise_info is None:
            raise ValueError(f"ノイズ検出に失敗しました: {input_file}")

        # 処理計画の作成
        processing_plan = self._create_processing_plan(file_info, level_info, noise_info)

        self._print("\n" + "-" * 80)
        self._print("【分析結果サマリー】")
        self._print("-" * 80)
        self._print(f"長さ: {file_info['duration']}秒")
        self._print(f"サンプリングレート: {file_info['sample_rate']}Hz")
        self._print(f"チャンネル: {file_info['channel_type']}")
        self._print(f"ピークレベル: {level_info['peak_db']:.2f} dBFS")
        self._print(f"RMSレベル: {level_info['rms_db']:.2f} dBFS")
        self._print(f"クリッピング率: {level_info['clipping_percentage']:.4f}%")
        self._print(f"ホワイトノイズ: {'検出' if noise_info['has_white_noise'] else 'なし'}")
        self._print(f"ハムノイズ: {'検出' if noise_info['has_hum_noise'] else 'なし'}")
        self._print(f"クリックノイズ: {'検出' if noise_info['has_click_noise'] else 'なし'}")
        self._print(f"背景ノイズ: {'検出' if noise_info['has_background_noise'] else 'なし'}")

        return {
            'file_info': file_info,
            'level_info': level_info,
            'noise_info': noise_info,
            'processing_plan': processing_plan
        }

    def _create_processing_plan(
        self,
        file_info: Dict,
        level_info: Dict,
        noise_info: Dict
    ) -> List[Tuple[str, str, Dict]]:
        """
        分析結果から処理計画を作成

        Args:
            file_info: ファイル情報
            level_info: レベル情報
            noise_info: ノイズ情報

        Returns:
            処理リスト [(処理名, 関数名, パラメータ), ...]
        """
        plan = []

        # 1. 無音トリミング（先頭・末尾の無音を削除）
        # 常に実行する（データを整える意味で）
        plan.append((
            "無音トリミング",
            "trim_silence",
            {'silence_thresh_db': -40.0, 'min_silence_duration': 0.3}
        ))

        # 2. ノイズ除去処理

        # 2-1. クリックノイズ除去
        if noise_info['has_click_noise']:
            plan.append((
                "クリックノイズ除去",
                "remove_click_noise",
                {'threshold': 3.0}
            ))

        # 2-2. ハムノイズ除去
        if noise_info['has_hum_noise']:
            plan.append((
                "ハムノイズ除去",
                "remove_hum_noise",
                {'freq_50hz_strength': 0.9, 'freq_60hz_strength': 0.9}
            ))

        # 2-3. ホワイトノイズ除去
        if noise_info['has_white_noise']:
            # ノイズレベルに応じて強度を調整
            if noise_info['white_noise_level'] > -40:
                strength = 0.7  # 強めに除去
            elif noise_info['white_noise_level'] > -50:
                strength = 0.5  # 中程度
            else:
                strength = 0.3  # 弱め
            
            plan.append((
                "ホワイトノイズ除去",
                "remove_white_noise",
                {'noise_reduce_strength': strength}
            ))

        # 2-4. 背景ノイズ除去
        if noise_info['has_background_noise']:
            plan.append((
                "背景ノイズ除去",
                "reduce_background_noise",
                {'noise_reduction_strength': 0.5}
            ))

        # 3. EQ処理

        # 3-1. ハイパスフィルタ（低域ノイズカット）
        # サンプリングレートが高い場合や、ノイズがある場合に適用
        if file_info['sample_rate'] >= 44100 or noise_info['has_background_noise']:
            plan.append((
                "ハイパスフィルタ（低域カット）",
                "apply_highpass_filter",
                {'cutoff_freq': 80.0, 'order': 5}
            ))

        # 3-2. ボイスバンドEQ（音声帯域の強調）
        # 常に適用して音声を明瞭にする
        plan.append((
            "ボイスバンドEQ",
            "apply_voice_band_eq",
            {'boost_db': 3.0}
        ))

        # 3-3. ディエッサー（歯擦音の抑制）
        # サンプリングレートが高い場合に適用
        if file_info['sample_rate'] >= 44100:
            plan.append((
                "ディエッサー（歯擦音抑制）",
                "apply_deesser",
                {'threshold_db': -20.0, 'reduction_db': 6.0}
            ))

        # 4. レベル処理

        # 4-1. デクリップ（クリッピング修復）
        if level_info['clipping_percentage'] > 0.01:  # 0.01%以上クリッピングがある場合
            plan.append((
                "クリッピング修復",
                "declip_audio",
                {'threshold': 0.99}
            ))

        # 4-2. ノーマライゼーション
        # ピークレベルとラウドネスレベルで判定
        if level_info['peak_db'] < -6.0:
            # ピークが低すぎる場合はピークノーマライズ
            plan.append((
                "ピークノーマライズ",
                "normalize_peak",
                {'target_db': -3.0}
            ))
        elif abs(level_info['lufs'] - (-16.0)) > 3.0:
            # LUFSが-16から3dB以上離れている場合はラウドネスノーマライズ
            plan.append((
                "ラウドネスノーマライズ",
                "normalize_loudness",
                {'target_lufs': -16.0}
            ))

        # 5. 無音圧縮（長い無音を短縮）
        # 音声データの効率化のため常に実行
        plan.append((
            "長い無音の圧縮",
            "compress_long_silence",
            {'silence_thresh_db': -40.0, 'min_silence_duration': 0.5, 'keep_silence_duration': 0.2}
        ))

        return plan

    def execute_pipeline(
        self,
        input_file: str,
        output_file: str,
        analysis_result: Optional[Dict] = None,
        split_output: bool = False,
        split_params: Optional[Dict] = None
    ) -> bool:
        """
        パイプラインを実行

        Args:
            input_file: 入力ファイルパス
            output_file: 出力ファイルパス
            analysis_result: 分析結果（Noneの場合は自動分析）
            split_output: 最終処理として無音区間で分割するかどうか
            split_params: 分割処理のパラメータ

        Returns:
            成功時True
        """
        try:
            # 分析が未実施の場合は実行
            if analysis_result is None:
                analysis_result = self.analyze_and_plan(input_file)

            processing_plan = analysis_result['processing_plan']

            if not processing_plan:
                self._print("\n実行する処理がありません。")
                # 入力ファイルを出力ファイルにコピー
                import shutil
                shutil.copy2(input_file, output_file)
                return True

            self._print("\n" + "=" * 80)
            self._print("【処理計画】")
            self._print("=" * 80)
            for i, (name, func_name, params) in enumerate(processing_plan, 1):
                self._print(f"{i}. {name}")
                self._print(f"   関数: {func_name}")
                self._print(f"   パラメータ: {params}")
            self._print("=" * 80)

            # 一時ファイルを使用して順次処理
            temp_dir = Path(output_file).parent / "temp_pipeline"
            temp_dir.mkdir(parents=True, exist_ok=True)

            current_file = input_file

            for i, (name, func_name, params) in enumerate(processing_plan, 1):
                self._print(f"\n【処理 {i}/{len(processing_plan)}】: {name}")
                self._print("-" * 80)

                # 最後の処理の場合は出力ファイルに直接保存
                if i == len(processing_plan):
                    next_file = output_file
                else:
                    next_file = str(temp_dir / f"temp_{i:03d}.wav")

                # 処理の実行
                success = self._execute_single_process(
                    func_name,
                    current_file,
                    next_file,
                    params
                )

                if not success:
                    self._print(f"❌ 処理失敗: {name}")
                    return False

                self._print(f"✅ 完了: {name}")
                current_file = next_file

            # 一時ファイルの削除
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            # 分割処理（オプション）
            if split_output:
                self._print("\n" + "=" * 80)
                self._print("【最終処理: 無音区間で分割】")
                self._print("=" * 80)
                
                # 分割パラメータのデフォルト値
                if split_params is None:
                    split_params = {
                        'silence_thresh_db': -40.0,
                        'min_voice_duration': 0.5,
                        'min_silence_duration': 0.3
                    }
                
                # 出力ディレクトリを準備
                output_path = Path(output_file)
                split_dir = output_path.parent / "split_segments"
                split_dir.mkdir(parents=True, exist_ok=True)
                
                # プレフィックスを生成（元のファイル名から）
                prefix = output_path.stem
                
                # 分割実行
                split_files = self.silence_processor.split_by_vad(
                    output_file,
                    str(split_dir),
                    silence_thresh_db=split_params.get('silence_thresh_db', -40.0),
                    min_voice_duration=split_params.get('min_voice_duration', 0.5),
                    min_silence_duration=split_params.get('min_silence_duration', 0.3),
                    prefix=prefix
                )
                
                if split_files:
                    self._print(f"\n✅ {len(split_files)}個のファイルに分割完了")
                    self._print(f"出力ディレクトリ: {split_dir}")
                else:
                    self._print("\n⚠️ 分割処理に失敗しました")
                    return False

            self._print("\n" + "=" * 80)
            self._print("【パイプライン処理完了】")
            self._print("=" * 80)
            if split_output:
                self._print(f"分割ファイル出力先: {split_dir}")
            else:
                self._print(f"出力ファイル: {output_file}")

            return True

        except Exception as e:
            self._print(f"\n❌ エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _execute_single_process(
        self,
        func_name: str,
        input_file: str,
        output_file: str,
        params: Dict
    ) -> bool:
        """
        単一の処理を実行

        Args:
            func_name: 関数名
            input_file: 入力ファイル
            output_file: 出力ファイル
            params: パラメータ辞書

        Returns:
            成功時True
        """
        try:
            # 無音処理
            if func_name == "trim_silence":
                return self.silence_processor.trim_silence(input_file, output_file, **params)
            elif func_name == "compress_long_silence":
                return self.silence_processor.compress_long_silence(input_file, output_file, **params)

            # ノイズ除去
            elif func_name == "remove_white_noise":
                return self.denoiser.remove_white_noise(input_file, output_file, **params)
            elif func_name == "remove_hum_noise":
                return self.denoiser.remove_hum_noise(input_file, output_file, **params)
            elif func_name == "remove_click_noise":
                return self.denoiser.remove_click_noise(input_file, output_file, **params)
            elif func_name == "reduce_background_noise":
                return self.denoiser.reduce_background_noise(input_file, output_file, **params)

            # EQ処理
            elif func_name == "apply_highpass_filter":
                return self.eq_processor.apply_highpass_filter(input_file, output_file, **params)
            elif func_name == "apply_voice_band_eq":
                return self.eq_processor.apply_voice_band_eq(input_file, output_file, **params)
            elif func_name == "apply_deesser":
                return self.eq_processor.apply_deesser(input_file, output_file, **params)

            # レベル処理
            elif func_name == "declip_audio":
                return self.level_processor.declip_audio(input_file, output_file, **params)
            elif func_name == "normalize_peak":
                return self.level_processor.normalize_peak(input_file, output_file, **params)
            elif func_name == "normalize_loudness":
                return self.level_processor.normalize_loudness(input_file, output_file, **params)

            else:
                self._print(f"未知の関数名: {func_name}")
                return False

        except Exception as e:
            self._print(f"処理中にエラー: {e}")
            return False

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        file_pattern: str = "*.wav",
        split_output: bool = False,
        split_params: Optional[Dict] = None
    ) -> Dict[str, bool]:
        """
        ディレクトリ内の全ファイルを一括処理

        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            file_pattern: ファイルパターン（デフォルト: *.wav）

        Returns:
            処理結果の辞書 {ファイル名: 成功/失敗}
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = list(input_path.glob(file_pattern))
        
        if not files:
            self._print(f"処理対象のファイルが見つかりません: {input_dir}/{file_pattern}")
            return {}

        self._print(f"\n処理対象ファイル数: {len(files)}")
        
        results = {}
        
        for i, file_path in enumerate(files, 1):
            self._print(f"\n{'=' * 80}")
            self._print(f"処理中 [{i}/{len(files)}]: {file_path.name}")
            self._print(f"{'=' * 80}")
            
            output_file = output_path / file_path.name
            
            try:
                success = self.execute_pipeline(
                    str(file_path), 
                    str(output_file),
                    split_output=split_output,
                    split_params=split_params
                )
                results[file_path.name] = success
            except Exception as e:
                self._print(f"❌ エラー: {e}")
                results[file_path.name] = False

        # 結果サマリー
        success_count = sum(1 for v in results.values() if v)
        self._print(f"\n{'=' * 80}")
        self._print(f"【バッチ処理完了】")
        self._print(f"{'=' * 80}")
        self._print(f"成功: {success_count}/{len(results)}")
        self._print(f"失敗: {len(results) - success_count}/{len(results)}")
        
        return results


def main():
    """メイン実行例"""
    import argparse
    
    parser = argparse.ArgumentParser(description="音声処理パイプライン")
    parser.add_argument("input", help="入力ファイルまたはディレクトリ")
    parser.add_argument("output", nargs="?", help="出力ファイルまたはディレクトリ（省略時は自動生成）")
    parser.add_argument("--batch", action="store_true", help="ディレクトリ一括処理モード")
    parser.add_argument("--pattern", default="*.wav", help="バッチ処理時のファイルパターン")
    parser.add_argument("--quiet", action="store_true", help="詳細出力を抑制")
    parser.add_argument("--output-dir", default="audio/output", help="出力ディレクトリ（デフォルト: audio/output）")
    parser.add_argument("--split", action="store_true", help="最終処理として無音区間で分割する")
    parser.add_argument("--split-thresh", type=float, default=-40.0, help="分割時の無音閾値（dBFS, デフォルト: -40.0）")
    parser.add_argument("--min-voice", type=float, default=0.5, help="最小音声区間長（秒, デフォルト: 0.5）")
    parser.add_argument("--min-silence", type=float, default=0.3, help="最小無音区間長（秒, デフォルト: 0.3）")
    
    args = parser.parse_args()
    
    pipeline = AudioProcessingPipeline(verbose=not args.quiet)
    
    # 出力パスが指定されていない場合は自動生成
    if args.output is None:
        input_path = Path(args.input)
        if args.batch or input_path.is_dir():
            # ディレクトリ処理の場合
            args.output = args.output_dir
        else:
            # 単一ファイルの場合
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            args.output = str(output_dir / f"processed_{input_path.name}")
        
        if not args.quiet:
            print(f"出力先: {args.output}")
    
    # 分割パラメータの準備
    split_params = None
    if args.split:
        split_params = {
            'silence_thresh_db': args.split_thresh,
            'min_voice_duration': args.min_voice,
            'min_silence_duration': args.min_silence
        }
    
    if args.batch:
        # バッチ処理
        results = pipeline.process_directory(
            args.input, 
            args.output, 
            args.pattern,
            split_output=args.split,
            split_params=split_params
        )
        exit(0 if all(results.values()) else 1)
    else:
        # 単一ファイル処理
        success = pipeline.execute_pipeline(
            args.input, 
            args.output,
            split_output=args.split,
            split_params=split_params
        )
        exit(0 if success else 1)


if __name__ == "__main__":
    main()
