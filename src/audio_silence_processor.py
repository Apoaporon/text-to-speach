"""
音声ファイルの無音・不要区間を処理するスクリプト
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


class AudioSilenceProcessor:
    """音声無音処理クラス"""

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

    def trim_silence(
        self,
        input_file: str,
        output_file: str,
        silence_thresh_db: float = -40.0,
        min_silence_duration: float = 0.3
    ) -> bool:
        """
        先頭と末尾の無音をトリミング
        
        Args:
            input_file: 入力音声ファイルのパス
            output_file: 出力音声ファイルのパス
            silence_thresh_db: 無音判定の閾値（dBFS）
            min_silence_duration: 無音と判定する最小時間（秒）
        
        Returns:
            成功時True、失敗時False
        """
        try:
            self._print(f"先頭・末尾のトリミング処理中... (閾値: {silence_thresh_db} dB)")
            
            # 音声を読み込み
            y, sr = librosa.load(input_file, sr=None, mono=False)
            
            # ステレオの場合
            if y.ndim == 2:
                # 各チャンネルでトリミング位置を検出
                trimmed_channels = []
                for channel in y:
                    trimmed, _ = librosa.effects.trim(
                        channel,
                        top_db=-silence_thresh_db,
                        frame_length=2048,
                        hop_length=512
                    )
                    trimmed_channels.append(trimmed)
                
                # 最小長に合わせる
                min_len = min(len(ch) for ch in trimmed_channels)
                output_audio = np.array([ch[:min_len] for ch in trimmed_channels])
            else:
                # モノラルの場合
                output_audio, _ = librosa.effects.trim(
                    y,
                    top_db=-silence_thresh_db,
                    frame_length=2048,
                    hop_length=512
                )
            
            # 保存
            sf.write(output_file, output_audio.T if output_audio.ndim > 1 else output_audio, sr)
            
            original_duration = len(y[0] if y.ndim > 1 else y) / sr
            trimmed_duration = len(output_audio[0] if output_audio.ndim > 1 else output_audio) / sr
            removed_duration = original_duration - trimmed_duration
            
            self._print(f"  元の長さ: {original_duration:.2f}秒")
            self._print(f"  処理後: {trimmed_duration:.2f}秒")
            self._print(f"  削除: {removed_duration:.2f}秒")
            self._print(f"✅ 保存完了: {output_file}")
            return True
            
        except Exception as e:
            self._print(f"エラー: {e}")
            return False

    def compress_long_silence(
        self,
        input_file: str,
        output_file: str,
        silence_thresh_db: float = -40.0,
        max_silence_duration: float = 0.5,
        min_silence_to_compress: float = 1.0
    ) -> bool:
        """
        長い無音区間を圧縮
        
        Args:
            input_file: 入力音声ファイルのパス
            output_file: 出力音声ファイルのパス
            silence_thresh_db: 無音判定の閾値（dBFS）
            max_silence_duration: 圧縮後の最大無音時間（秒）
            min_silence_to_compress: 圧縮対象とする最小無音時間（秒）
        
        Returns:
            成功時True、失敗時False
        """
        try:
            self._print(f"長い無音の圧縮処理中... ({min_silence_to_compress}s以上→{max_silence_duration}sに圧縮)")
            
            # 音声を読み込み
            y, sr = librosa.load(input_file, sr=None, mono=True)
            
            # RMSエネルギーを計算
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            
            # フレームの時刻を計算
            frames = range(len(rms_db))
            times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
            
            # 無音判定
            is_silence = rms_db < silence_thresh_db
            
            # 区間を検出
            segments = []
            i = 0
            while i < len(is_silence):
                if is_silence[i]:
                    # 無音区間
                    start_time = times[i]
                    start_sample = librosa.time_to_samples(start_time, sr=sr)
                    
                    while i < len(is_silence) and is_silence[i]:
                        i += 1
                    
                    end_time = times[i-1] if i < len(times) else len(y) / sr
                    end_sample = librosa.time_to_samples(end_time, sr=sr)
                    
                    silence_duration = end_time - start_time
                    
                    # 長い無音は圧縮
                    if silence_duration > min_silence_to_compress:
                        compressed_samples = int(max_silence_duration * sr)
                        segments.append(y[start_sample:start_sample + compressed_samples])
                    else:
                        segments.append(y[start_sample:end_sample])
                else:
                    # 音声区間
                    start_time = times[i]
                    start_sample = librosa.time_to_samples(start_time, sr=sr)
                    
                    while i < len(is_silence) and not is_silence[i]:
                        i += 1
                    
                    end_time = times[i-1] if i < len(times) else len(y) / sr
                    end_sample = librosa.time_to_samples(end_time, sr=sr)
                    
                    segments.append(y[start_sample:end_sample])
            
            # 結合
            output_audio = np.concatenate(segments)
            
            # 元の音声がステレオの場合は、ステレオで保存
            y_original, sr_original = librosa.load(input_file, sr=None, mono=False)
            if y_original.ndim == 2:
                output_audio = np.array([output_audio, output_audio])
            
            # 保存
            sf.write(output_file, output_audio.T if output_audio.ndim > 1 else output_audio, sr)
            
            original_duration = len(y) / sr
            compressed_duration = len(segments[0] if isinstance(segments[0], np.ndarray) else output_audio) / sr
            if len(segments) > 0:
                compressed_duration = sum(len(seg) for seg in segments) / sr
            
            self._print(f"  元の長さ: {original_duration:.2f}秒")
            self._print(f"  処理後: {compressed_duration:.2f}秒")
            self._print(f"  短縮: {original_duration - compressed_duration:.2f}秒")
            self._print(f"✅ 保存完了: {output_file}")
            return True
            
        except Exception as e:
            self._print(f"エラー: {e}")
            return False

    def split_by_vad(
        self,
        input_file: str,
        output_dir: str,
        silence_thresh_db: float = -40.0,
        min_voice_duration: float = 0.5,
        min_silence_duration: float = 0.3,
        prefix: str = "segment"
    ) -> List[str]:
        """
        VAD（Voice Activity Detection）で音声区間ごとに分割
        
        Args:
            input_file: 入力音声ファイルのパス
            output_dir: 出力ディレクトリ
            silence_thresh_db: 無音判定の閾値（dBFS）
            min_voice_duration: 音声区間の最小時間（秒）
            min_silence_duration: 無音区間の最小時間（秒）
            prefix: 出力ファイル名のプレフィックス
        
        Returns:
            作成されたファイルのパスのリスト
        """
        try:
            self._print(f"VAD分割処理中... (閾値: {silence_thresh_db} dB)")
            
            # 出力ディレクトリを作成
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 音声を読み込み
            y, sr = librosa.load(input_file, sr=None, mono=True)
        
            # RMSエネルギーを計算
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            
            # フレームの時刻を計算
            frames = range(len(rms_db))
            times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
            
            # 無音判定
            is_silence = rms_db < silence_thresh_db
            
            # 音声区間を検出
            voice_segments = []
            i = 0
            while i < len(is_silence):
                if not is_silence[i]:
                    # 音声区間の開始
                    start_time = times[i]
                    start_sample = librosa.time_to_samples(start_time, sr=sr)
                    
                    while i < len(is_silence) and not is_silence[i]:
                        i += 1
                    
                    end_time = times[i-1] if i < len(times) else len(y) / sr
                    end_sample = librosa.time_to_samples(end_time, sr=sr)
                    
                    duration = end_time - start_time
                    
                    # 最小時間以上の音声区間のみ保存
                    if duration >= min_voice_duration:
                        voice_segments.append((start_sample, end_sample, start_time, end_time))
                else:
                    i += 1
            
            # 元の音声がステレオか確認
            y_original, sr_original = librosa.load(input_file, sr=None, mono=False)
            is_stereo = y_original.ndim == 2
            
            # 各音声区間を保存
            output_files = []
            for idx, (start_sample, end_sample, start_time, end_time) in enumerate(voice_segments, 1):
                if is_stereo:
                    segment_audio = y_original[:, start_sample:end_sample]
                else:
                    segment_audio = y[start_sample:end_sample]
                
                # ファイル名を生成
                duration_ms = int((end_time - start_time) * 1000)
                start_ms = int(start_time * 1000)
                output_file = output_path / f"{prefix}_{idx:04d}_{start_ms}ms_{duration_ms}ms.wav"
                
                # 保存
                sf.write(str(output_file), segment_audio.T if segment_audio.ndim > 1 else segment_audio, sr)
                output_files.append(str(output_file))
                
                self._print(f"  {idx}. {start_time:.2f}s - {end_time:.2f}s (長さ: {end_time - start_time:.2f}s) → {output_file.name}")
            
            self._print(f"✅ {len(output_files)}個のファイルに分割完了: {output_dir}")
            return output_files
            
        except Exception as e:
            self._print(f"エラー: {e}")
            return []

    def remove_all_silence(
        self,
        input_file: str,
        output_file: str,
        silence_thresh_db: float = -40.0
    ) -> bool:
        """
        すべての無音区間を削除（音声区間だけを結合）
        
        Args:
            input_file: 入力音声ファイルのパス
            output_file: 出力音声ファイルのパス
            silence_thresh_db: 無音判定の閾値（dBFS）
        
        Returns:
            成功時True、失敗時False
        """
        try:
            self._print(f"全無音区間の削除処理中... (閾値: {silence_thresh_db} dB)")
            
            # 音声を読み込み
            y, sr = librosa.load(input_file, sr=None, mono=True)
            
            # RMSエネルギーを計算
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            
            # フレームの時刻を計算
            frames = range(len(rms_db))
            times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
            
            # 無音判定
            is_silence = rms_db < silence_thresh_db
            
            # 音声区間のみ抽出
            voice_segments = []
            i = 0
            while i < len(is_silence):
                if not is_silence[i]:
                    start_time = times[i]
                    start_sample = librosa.time_to_samples(start_time, sr=sr)
                    
                    while i < len(is_silence) and not is_silence[i]:
                        i += 1
                    
                    end_time = times[i-1] if i < len(times) else len(y) / sr
                    end_sample = librosa.time_to_samples(end_time, sr=sr)
                    
                    voice_segments.append(y[start_sample:end_sample])
                else:
                    i += 1
            
            # 音声区間を結合
            if voice_segments:
                output_audio = np.concatenate(voice_segments)
            else:
                self._print("警告: 音声区間が見つかりませんでした")
                output_audio = y
            
            # 元の音声がステレオの場合
            y_original, sr_original = librosa.load(input_file, sr=None, mono=False)
            if y_original.ndim == 2:
                output_audio = np.array([output_audio, output_audio])
            
            # 保存
            sf.write(output_file, output_audio.T if output_audio.ndim > 1 else output_audio, sr)
            
            original_duration = len(y) / sr
            processed_duration = len(output_audio[0] if output_audio.ndim > 1 else output_audio) / sr
            
            self._print(f"  元の長さ: {original_duration:.2f}秒")
            self._print(f"  処理後: {processed_duration:.2f}秒")
            self._print(f"  削除: {original_duration - processed_duration:.2f}秒")
            self._print(f"✅ 保存完了: {output_file}")
            return True
            
        except Exception as e:
            self._print(f"エラー: {e}")
            return False

    def batch_silence_process_directory(
        self,
        input_dir: str,
        output_dir: str,
        process_type: str = "trim",
        silence_thresh_db: float = -40.0,
        **kwargs
    ) -> None:
        """
        ディレクトリ内の全音声ファイルに無音処理を適用
        
        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            process_type: 処理タイプ ('trim', 'compress', 'remove_all')
            silence_thresh_db: 無音判定の閾値（dBFS）
            **kwargs: 各処理関数への追加パラメータ
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f'*{ext}'))
        
        if not audio_files:
            self._print(f"音声ファイルが見つかりません: {input_dir}")
            return
        
        self._print(f"\n{len(audio_files)}個のファイルを処理します...")
        
        for i, audio_file in enumerate(audio_files, 1):
            self._print(f"\n[{i}/{len(audio_files)}] {audio_file.name}")
            
            if process_type == "trim":
                output_file = output_path / f"{audio_file.stem}_trimmed.wav"
                self.trim_silence(str(audio_file), str(output_file), silence_thresh_db, **kwargs)
            
            elif process_type == "compress":
                output_file = output_path / f"{audio_file.stem}_compressed.wav"
                self.compress_long_silence(str(audio_file), str(output_file), silence_thresh_db, **kwargs)
            
            elif process_type == "remove_all":
                output_file = output_path / f"{audio_file.stem}_no_silence.wav"
                self.remove_all_silence(str(audio_file), str(output_file), silence_thresh_db)
            
            else:
                self._print(f"不明な処理タイプ: {process_type}")
                return
        
        self._print(f"\n✅ 全ての処理が完了しました: {output_dir}")


def main():
    """メイン処理"""
    input_file = "downloads/audio/【粗品】最近のSNSニュース斬った【1人賛否】.wav"
    
    # プロセッサーのインスタンスを作成
    processor = AudioSilenceProcessor(verbose=True)
    
    print("=" * 70)
    print("【無音・不要区間処理の例】")
    print("=" * 70)
    
    if not os.path.exists(input_file):
        print(f"ファイルが見つかりません: {input_file}")
        return
    
    # 1. 先頭・末尾のトリミング
    print("\n1. 先頭・末尾のトリミング")
    print("-" * 70)
    output_file = "downloads/audio/【粗品】最近のSNSニュース斬った【1人賛否】_trimmed.wav"
    processor.trim_silence(input_file, output_file, silence_thresh_db=-40.0)
    
    # 2. 長い無音の圧縮
    print("\n2. 長い無音の圧縮")
    print("-" * 70)
    output_file = "downloads/audio/【粗品】最近のSNSニュース斬った【1人賛否】_compressed.wav"
    processor.compress_long_silence(
        input_file,
        output_file,
        silence_thresh_db=-40.0,
        max_silence_duration=0.5,
        min_silence_to_compress=1.0
    )
    
    # 3. VAD分割
    print("\n3. VAD（Voice Activity Detection）分割")
    print("-" * 70)
    output_dir = "downloads/audio_vad_segments"
    processor.split_by_vad(
        input_file,
        output_dir,
        silence_thresh_db=-40.0,
        min_voice_duration=0.5,
        min_silence_duration=0.3,
        prefix="voice_segment"
    )
    
    # 4. 全無音削除
    # print("\n4. 全無音区間の削除")
    # print("-" * 70)
    # output_file = "downloads/audio/【粗品】最近のSNSニュース斬った【1人賛否】_no_silence.wav"
    # processor.remove_all_silence(input_file, output_file, silence_thresh_db=-40.0)
    
    # バッチ処理の例
    # processor.batch_silence_process_directory(
    #     input_dir="downloads/audio",
    #     output_dir="downloads/audio_processed",
    #     process_type="trim",
    #     silence_thresh_db=-40.0
    # )


if __name__ == "__main__":
    main()
