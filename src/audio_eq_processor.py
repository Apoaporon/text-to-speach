"""
音声ファイルの周波数特性を調整する（EQ系処理）
"""
import os
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from scipy import signal


class AudioEQProcessor:
    """音声EQ処理クラス"""

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

    def apply_highpass_filter(
        self,
        input_file: str,
        output_file: str,
        cutoff_freq: float = 100.0,
        order: int = 5
    ) -> bool:
        """
        ハイパスフィルタを適用（低域カット）
        
        Args:
            input_file: 入力音声ファイルのパス
            output_file: 出力音声ファイルのパス
            cutoff_freq: カットオフ周波数（Hz）、デフォルト100Hz
            order: フィルタ次数（高いほど急峻）
        
        Returns:
            成功時True、失敗時False
        """
        try:
            self._print(f"ハイパスフィルタ処理中... (カットオフ: {cutoff_freq}Hz)")
            
            # 音声を読み込み
            y, sr = librosa.load(input_file, sr=None, mono=False)
            
            # ステレオの場合は各チャンネル処理
            if y.ndim == 1:
                y = y.reshape(1, -1)
            
            # バターワースハイパスフィルタを設計
            nyquist = sr / 2
            normal_cutoff = cutoff_freq / nyquist
            b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
            
            # 各チャンネルにフィルタ適用
            filtered_channels = []
            for channel in y:
                filtered = signal.filtfilt(b, a, channel)
                filtered_channels.append(filtered)
            
            # チャンネル結合
            if len(filtered_channels) == 1:
                output_audio = filtered_channels[0]
            else:
                output_audio = np.array(filtered_channels)
            
            # 保存
            sf.write(output_file, output_audio.T, sr)
            self._print(f"✅ 保存完了: {output_file}")
            return True
            
        except Exception as e:
            self._print(f"エラー: {e}")
            return False

    def apply_lowpass_filter(
        self,
        input_file: str,
        output_file: str,
        cutoff_freq: float = 16000.0,
        order: int = 5
    ) -> bool:
        """
        ローパスフィルタを適用（高域カット）
        
        Args:
            input_file: 入力音声ファイルのパス
            output_file: 出力音声ファイルのパス
            cutoff_freq: カットオフ周波数（Hz）、デフォルト16000Hz
            order: フィルタ次数
        
        Returns:
            成功時True、失敗時False
        """
        try:
            self._print(f"ローパスフィルタ処理中... (カットオフ: {cutoff_freq}Hz)")
            
            # 音声を読み込み
            y, sr = librosa.load(input_file, sr=None, mono=False)
            
            # サンプリングレートのチェック
            if cutoff_freq >= sr / 2:
                self._print(f"警告: カットオフ周波数がナイキスト周波数を超えています")
                cutoff_freq = sr / 2 * 0.9
            
            # ステレオの場合は各チャンネル処理
            if y.ndim == 1:
                y = y.reshape(1, -1)
            
            # バターワースローパスフィルタを設計
            nyquist = sr / 2
            normal_cutoff = cutoff_freq / nyquist
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            
            # 各チャンネルにフィルタ適用
            filtered_channels = []
            for channel in y:
                filtered = signal.filtfilt(b, a, channel)
                filtered_channels.append(filtered)
            
            # チャンネル結合
            if len(filtered_channels) == 1:
                output_audio = filtered_channels[0]
            else:
                output_audio = np.array(filtered_channels)
            
            # 保存
            sf.write(output_file, output_audio.T, sr)
            self._print(f"✅ 保存完了: {output_file}")
            return True
            
        except Exception as e:
            self._print(f"エラー: {e}")
            return False

    def apply_voice_band_eq(
        self,
        input_file: str,
        output_file: str,
        boost_db: float = 3.0,
        freq_center: float = 2000.0,
        bandwidth: float = 2.0
    ) -> bool:
        """
        音声帯域にEQブーストを適用
        
        Args:
            input_file: 入力音声ファイルのパス
            output_file: 出力音声ファイルのパス
            boost_db: ブースト量（dB）、デフォルト3.0dB
            freq_center: 中心周波数（Hz）、デフォルト2000Hz
            bandwidth: バンド幅（オクターブ）、デフォルト2.0
        
        Returns:
            成功時True、失敗時False
        """
        try:
            self._print(f"音声帯域EQ処理中... (中心: {freq_center}Hz, ブースト: +{boost_db}dB)")
            
            # 音声を読み込み
            y, sr = librosa.load(input_file, sr=None, mono=False)
            
            # ステレオの場合は各チャンネル処理
            if y.ndim == 1:
                y = y.reshape(1, -1)
            
            # ピーキングEQフィルタを設計
            Q = bandwidth  # Q値（帯域幅）
            gain_linear = 10 ** (boost_db / 20)
            w0 = 2 * np.pi * freq_center / sr
            
            alpha = np.sin(w0) / (2 * Q)
            
            # ピーキングフィルタの係数
            b0 = 1 + alpha * gain_linear
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha * gain_linear
            a0 = 1 + alpha / gain_linear
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha / gain_linear
            
            # 正規化
            b = np.array([b0, b1, b2]) / a0
            a = np.array([1, a1 / a0, a2 / a0])
            
            # 各チャンネルにフィルタ適用
            filtered_channels = []
            for channel in y:
                filtered = signal.filtfilt(b, a, channel)
                filtered_channels.append(filtered)
            
            # チャンネル結合
            if len(filtered_channels) == 1:
                output_audio = filtered_channels[0]
            else:
                output_audio = np.array(filtered_channels)
            
            # 保存
            sf.write(output_file, output_audio.T, sr)
            self._print(f"✅ 保存完了: {output_file}")
            return True
            
        except Exception as e:
            self._print(f"エラー: {e}")
            return False

    def apply_deesser(
        self,
        input_file: str,
        output_file: str,
        threshold_db: float = -20.0,
        ratio: float = 3.0,
        freq_start: float = 6000.0,
        freq_end: float = 10000.0
    ) -> bool:
        """
        ディエッサーを適用（シビランス抑制）
        
        Args:
            input_file: 入力音声ファイルのパス
            output_file: 出力音声ファイルのパス
            threshold_db: 圧縮開始閾値（dB）
            ratio: 圧縮比率
            freq_start: シビランス帯域の開始周波数（Hz）
            freq_end: シビランス帯域の終了周波数（Hz）
        
        Returns:
            成功時True、失敗時False
        """
        try:
            self._print(f"ディエッサー処理中... ({freq_start}-{freq_end}Hz を抑制)")
            
            # 音声を読み込み
            y, sr = librosa.load(input_file, sr=None, mono=False)
            
            # ステレオの場合は各チャンネル処理
            if y.ndim == 1:
                y = y.reshape(1, -1)
            
            # サンプリングレートチェック
            if freq_end >= sr / 2:
                self._print(f"警告: 終了周波数がナイキスト周波数を超えています")
                freq_end = sr / 2 * 0.9
            
            processed_channels = []
            for channel in y:
                # STFTでシビランス帯域を検出
                stft = librosa.stft(channel, n_fft=2048, hop_length=512)
                magnitude = np.abs(stft)
                phase = np.angle(stft)
                
                freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
                
                # シビランス帯域のインデックス
                sib_idx_start = np.argmin(np.abs(freqs - freq_start))
                sib_idx_end = np.argmin(np.abs(freqs - freq_end))
                
                # シビランス帯域のエネルギーを測定
                sib_energy = np.mean(magnitude[sib_idx_start:sib_idx_end, :], axis=0)
                sib_energy_db = 20 * np.log10(sib_energy + 1e-10)
                
                # 閾値を超えた部分を圧縮
                compression_gain = np.ones_like(sib_energy_db)
                over_threshold = sib_energy_db > threshold_db
                compression_gain[over_threshold] = 1.0 / ratio
                
                # 圧縮ゲインを適用
                for i in range(sib_idx_start, sib_idx_end):
                    magnitude[i, :] *= compression_gain
                
                # ISTFTで時間領域に戻す
                processed = librosa.istft(magnitude * np.exp(1j * phase), hop_length=512, length=len(channel))
                processed_channels.append(processed)
            
            # チャンネル結合
            if len(processed_channels) == 1:
                output_audio = processed_channels[0]
            else:
                output_audio = np.array(processed_channels)
            
            # 保存
            sf.write(output_file, output_audio.T, sr)
            self._print(f"✅ 保存完了: {output_file}")
            return True
            
        except Exception as e:
            self._print(f"エラー: {e}")
            return False

    def apply_full_eq_chain(
        self,
        input_file: str,
        output_file: str,
        highpass_cutoff: float = 100.0,
        lowpass_cutoff: Optional[float] = None,
        voice_boost_db: float = 2.0,
        apply_deesser_flag: bool = False
    ) -> bool:
        """
        フルEQチェーンを適用（一括処理）
        
        Args:
            input_file: 入力音声ファイルのパス
            output_file: 出力音声ファイルのパス
            highpass_cutoff: ハイパスフィルタのカットオフ周波数（Hz）
            lowpass_cutoff: ローパスフィルタのカットオフ周波数（Hz）、Noneの場合は適用しない
            voice_boost_db: 音声帯域のブースト量（dB）
            apply_deesser_flag: ディエッサーを適用するかどうか
        
        Returns:
            成功時True、失敗時False
        """
        try:
            self._print("=" * 70)
            self._print("フルEQチェーン処理を開始")
            self._print("=" * 70)
            
            temp_dir = Path(input_file).parent / "temp_eq_processing"
            temp_dir.mkdir(exist_ok=True)
            
            current_file = input_file
            step = 1
            
            # 1. ハイパスフィルタ
            if highpass_cutoff > 0:
                temp_file = str(temp_dir / f"step{step}_highpass.wav")
                if not self.apply_highpass_filter(current_file, temp_file, highpass_cutoff):
                    return False
                current_file = temp_file
                step += 1
            
            # 2. ローパスフィルタ（オプション）
            if lowpass_cutoff is not None:
                temp_file = str(temp_dir / f"step{step}_lowpass.wav")
                if not self.apply_lowpass_filter(current_file, temp_file, lowpass_cutoff):
                    return False
                current_file = temp_file
                step += 1
            
            # 3. 音声帯域EQブースト
            if voice_boost_db > 0:
                temp_file = str(temp_dir / f"step{step}_eq_boost.wav")
                if not self.apply_voice_band_eq(current_file, temp_file, voice_boost_db):
                    return False
                current_file = temp_file
                step += 1
            
            # 4. ディエッサー（オプション）
            if apply_deesser_flag:
                temp_file = str(temp_dir / f"step{step}_deesser.wav")
                if not self.apply_deesser(current_file, temp_file):
                    return False
                current_file = temp_file
                step += 1
            
            # 最終出力にコピー
            y, sr = librosa.load(current_file, sr=None, mono=False)
            sf.write(output_file, y.T if y.ndim > 1 else y, sr)
            
            # 一時ファイル削除
            import shutil
            shutil.rmtree(temp_dir)
            
            self._print("=" * 70)
            self._print("✅ フルEQチェーン処理完了")
            self._print("=" * 70)
            return True
            
        except Exception as e:
            self._print(f"エラー: {e}")
            return False

    def batch_eq_process_directory(
        self,
        input_dir: str,
        output_dir: str,
        highpass_cutoff: float = 100.0,
        lowpass_cutoff: Optional[float] = None,
        voice_boost_db: float = 2.0,
        apply_deesser_flag: bool = False
    ) -> None:
        """
        ディレクトリ内の全音声ファイルにEQ処理を適用
        
        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            highpass_cutoff: ハイパスフィルタのカットオフ周波数（Hz）
            lowpass_cutoff: ローパスフィルタのカットオフ周波数（Hz）
            voice_boost_db: 音声帯域のブースト量（dB）
            apply_deesser_flag: ディエッサーを適用するかどうか
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
            output_file = output_path / f"{audio_file.stem}_eq_processed.wav"
            
            self.apply_full_eq_chain(
                str(audio_file),
                str(output_file),
                highpass_cutoff=highpass_cutoff,
                lowpass_cutoff=lowpass_cutoff,
                voice_boost_db=voice_boost_db,
                apply_deesser_flag=apply_deesser_flag
            )
        
        self._print(f"\n✅ 全ての処理が完了しました: {output_dir}")



def main():
    """メイン処理"""
    input_file = "downloads/audio/【粗品】最近のSNSニュース斬った【1人賛否】.wav"
    output_file = "downloads/audio/【粗品】最近のSNSニュース斬った【1人賛否】_eq_processed.wav"
    
    # EQプロセッサーのインスタンスを作成
    processor = AudioEQProcessor(verbose=True)
    
    # 個別処理の例
    print("=" * 70)
    print("【個別EQ処理の例】")
    print("=" * 70)
    
    # ハイパスフィルタのみ
    # processor.apply_highpass_filter(input_file, output_file, cutoff_freq=100.0)
    
    # フルチェーン処理
    if os.path.exists(input_file):
        processor.apply_full_eq_chain(
            input_file,
            output_file,
            highpass_cutoff=100.0,      # 100Hz以下カット
            lowpass_cutoff=16000.0,     # 16kHz以上カット
            voice_boost_db=2.5,         # 音声帯域を+2.5dBブースト
            apply_deesser_flag=True     # ディエッサー適用
        )
    else:
        print(f"ファイルが見つかりません: {input_file}")
    
    # バッチ処理の例
    # processor.batch_eq_process_directory(
    #     input_dir="downloads/audio",
    #     output_dir="downloads/audio_eq_processed",
    #     highpass_cutoff=100.0,
    #     voice_boost_db=2.0,
    #     apply_deesser_flag=False
    # )


if __name__ == "__main__":
    main()
