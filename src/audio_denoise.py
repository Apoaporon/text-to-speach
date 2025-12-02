"""
音声ファイルのノイズ除去処理スクリプト
"""
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy import signal


class AudioDenoiser:
    """音声ノイズ除去クラス"""

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

    def remove_white_noise(
        self,
        input_file: str,
        output_file: str,
        noise_reduce_strength: float = 0.5
    ) -> bool:
        """
        ホワイトノイズ・ヒスノイズをスペクトル減算法で除去する

        Args:
            input_file: 入力ファイルパス
            output_file: 出力ファイルパス
            noise_reduce_strength: ノイズ除去の強さ（0.0〜1.0）デフォルト: 0.5

        Returns:
            成功時True
        """
        try:
            self._print(f"ホワイトノイズ除去中: {Path(input_file).name}")
            self._print(f"  強度: {noise_reduce_strength}")

            # 音声データを読み込み
            y, sr = librosa.load(input_file, sr=None, mono=False)

            # ステレオの場合は各チャンネルを処理
            if y.ndim == 1:
                y_denoised = self._spectral_subtraction(y, sr, noise_reduce_strength)
            else:
                y_denoised = np.array([
                    self._spectral_subtraction(y[ch], sr, noise_reduce_strength)
                    for ch in range(y.shape[0])
                ])

            # 保存
            sf.write(output_file, y_denoised.T if y.ndim > 1 else y_denoised, sr)

            self._print(f"✅ 完了: {output_file}")
            return True

        except (OSError, ValueError, RuntimeError) as e:
            self._print(f"エラー: {e}")
            return False

    def _spectral_subtraction(self, y: np.ndarray, sr: int | float, strength: float) -> np.ndarray:
        """
        スペクトル減算法によるノイズ除去（内部関数）

        Args:
            y: 音声信号
            sr: サンプリングレート
            strength: ノイズ除去の強さ

        Returns:
            ノイズ除去後の信号
        """
        # STFTでスペクトログラムを取得
        d_stft = librosa.stft(y)
        magnitude = np.abs(d_stft)
        phase = np.angle(d_stft)

        # ノイズプロファイルを推定（最初の0.5秒をノイズと仮定）
        noise_frames = int(0.5 * sr / 512)  # 0.5秒分のフレーム数
        noise_profile = np.median(magnitude[:, :noise_frames], axis=1, keepdims=True)

        # スペクトル減算
        magnitude_denoised = magnitude - (noise_profile * strength)
        magnitude_denoised = np.maximum(magnitude_denoised, 0.01 * magnitude)  # 最小値を設定

        # 位相を戻してISTFT
        d_stft_denoised = magnitude_denoised * np.exp(1j * phase)
        y_denoised = librosa.istft(d_stft_denoised, length=len(y))

        return y_denoised

    def remove_hum_noise(
        self,
        input_file: str,
        output_file: str,
        hum_frequency: int = 60,
        harmonics: int = 5
    ) -> bool:
        """
        ハムノイズ（50/60Hz等のブーン音）をノッチフィルタで除去する

        Args:
            input_file: 入力ファイルパス
            output_file: 出力ファイルパス
            hum_frequency: ハムの基本周波数（50 or 60 Hz）デフォルト: 60
            harmonics: 除去する高調波の数 デフォルト: 5

        Returns:
            成功時True
        """
        try:
            self._print(f"ハムノイズ除去中: {Path(input_file).name}")
            self._print(f"  基本周波数: {hum_frequency} Hz")
            self._print(f"  高調波数: {harmonics}")

            # 音声データを読み込み
            y, sr = librosa.load(input_file, sr=None, mono=False)

            # ノッチフィルタを適用
            if y.ndim == 1:
                y_filtered = self._apply_notch_filters(y, sr, hum_frequency, harmonics)
            else:
                y_filtered = np.array([
                    self._apply_notch_filters(y[ch], sr, hum_frequency, harmonics)
                    for ch in range(y.shape[0])
                ])

            # 保存
            sf.write(output_file, y_filtered.T if y.ndim > 1 else y_filtered, sr)

            self._print(f"✅ 完了: {output_file}")
            return True

        except (OSError, ValueError, RuntimeError) as e:
            self._print(f"エラー: {e}")
            return False

    def _apply_notch_filters(
        self,
        y: np.ndarray,
        sr: int | float,
        hum_freq: int,
        harmonics: int
    ) -> np.ndarray:
        """
        複数のノッチフィルタを適用（内部関数）

        Args:
            y: 音声信号
            sr: サンプリングレート
            hum_freq: ハムの基本周波数
            harmonics: 高調波の数

        Returns:
            フィルタ適用後の信号
        """
        y_filtered = y.copy()

        # 基本周波数と高調波にノッチフィルタを適用
        for i in range(1, harmonics + 1):
            freq = hum_freq * i
            if freq < sr / 2:  # ナイキスト周波数以下のみ
                # ノッチフィルタ設計（Q=30で鋭いノッチ）
                b, a = signal.iirnotch(freq, Q=30, fs=sr)
                y_filtered = signal.filtfilt(b, a, y_filtered)

        return y_filtered

    def remove_click_noise(
        self,
        input_file: str,
        output_file: str,
        threshold: float = 3.0
    ) -> bool:
        """
        ポップノイズ・クリック音を除去する（デクリック処理）

        Args:
            input_file: 入力ファイルパス
            output_file: 出力ファイルパス
            threshold: クリック検出の閾値（標準偏差の倍数）デフォルト: 3.0

        Returns:
            成功時True
        """
        try:
            self._print(f"クリック・ポップノイズ除去中: {Path(input_file).name}")
            self._print(f"  閾値: {threshold} σ")

            # 音声データを読み込み
            y, sr = librosa.load(input_file, sr=None, mono=False)

            # デクリック処理
            if y.ndim == 1:
                y_declicked = self._declick(y, threshold)
            else:
                y_declicked = np.array([
                    self._declick(y[ch], threshold)
                    for ch in range(y.shape[0])
                ])

            # 保存
            sf.write(output_file, y_declicked.T if y.ndim > 1 else y_declicked, sr)

            self._print(f"✅ 完了: {output_file}")
            return True

        except (OSError, ValueError, RuntimeError) as e:
            self._print(f"エラー: {e}")
            return False

    def _declick(self, y: np.ndarray, threshold: float) -> np.ndarray:
        """
        クリックノイズ除去（内部関数）

        Args:
            y: 音声信号
            threshold: 閾値

        Returns:
            デクリック後の信号
        """
        # 微分で急激な変化を検出
        diff = np.diff(y, prepend=y[0])

        # 閾値を超える変化をクリックとして検出
        std_diff = np.std(diff)
        clicks = np.abs(diff) > (threshold * std_diff)

        # クリック位置を補間
        y_declicked = y.copy()
        click_indices = np.where(clicks)[0]

        for idx in click_indices:
            # 前後5サンプルの平均で補間
            start = max(0, idx - 5)
            end = min(len(y), idx + 6)

            # クリック位置を除いた平均
            valid_samples = np.concatenate([y[start:idx], y[idx+1:end]])
            if len(valid_samples) > 0:
                y_declicked[idx] = np.mean(valid_samples)

        return y_declicked

    def reduce_background_noise(
        self,
        input_file: str,
        output_file: str,
        strength: float = 0.3
    ) -> bool:
        """
        環境音などの背景ノイズを軽減する

        Args:
            input_file: 入力ファイルパス
            output_file: 出力ファイルパス
            strength: ノイズ抑制の強さ（0.0〜1.0）デフォルト: 0.3（やり過ぎ注意）

        Returns:
            成功時True
        """
        try:
            self._print(f"背景ノイズ抑制中: {Path(input_file).name}")
            self._print(f"  強度: {strength}（注意: 強すぎるとロボ声になります）")

            # 音声データを読み込み
            y, sr = librosa.load(input_file, sr=None, mono=False)

            # ノイズ抑制処理
            if y.ndim == 1:
                y_reduced = self._wiener_filter(y, sr, strength)
            else:
                y_reduced = np.array([
                    self._wiener_filter(y[ch], sr, strength)
                    for ch in range(y.shape[0])
                ])

            # 保存
            sf.write(output_file, y_reduced.T if y.ndim > 1 else y_reduced, sr)

            self._print(f"✅ 完了: {output_file}")
            return True

        except (OSError, ValueError, RuntimeError) as e:
            self._print(f"エラー: {e}")
            return False

    def _wiener_filter(self, y: np.ndarray, sr: int | float, strength: float) -> np.ndarray:
        """
        ウィーナーフィルタによる背景ノイズ抑制（内部関数）

        Args:
            y: 音声信号
            sr: サンプリングレート
            strength: 抑制の強さ

        Returns:
            ノイズ抑制後の信号
        """
        # STFT
        d_stft = librosa.stft(y)
        magnitude = np.abs(d_stft)
        phase = np.angle(d_stft)

        # パワースペクトル
        power = magnitude ** 2

        # ノイズパワーを推定（最初の0.5秒）
        noise_frames = int(0.5 * sr / 512)
        noise_power = np.median(power[:, :noise_frames], axis=1, keepdims=True)

        # ウィーナーフィルタのゲイン
        gain = np.maximum(1 - (strength * noise_power / (power + 1e-10)), 0.1)

        # フィルタ適用
        magnitude_filtered = magnitude * gain

        # ISTFT
        d_stft_filtered = magnitude_filtered * np.exp(1j * phase)
        y_filtered = librosa.istft(d_stft_filtered, length=len(y))

        return y_filtered

    def batch_denoise_directory(
        self,
        input_dir: str,
        output_dir: str,
        noise_type: str = "white",
        **kwargs
    ) -> None:
        """
        ディレクトリ内の全ファイルを一括ノイズ除去

        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            noise_type: ノイズタイプ（"white", "hum", "click", "background"）
            **kwargs: 各ノイズ除去関数への追加パラメータ
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 対象ファイルを取得
        audio_files = (
            list(input_path.glob("*.wav")) +
            list(input_path.glob("*.mp3")) +
            list(input_path.glob("*.m4a"))
        )

        self._print(f"\n{len(audio_files)}個のファイルを処理します")
        self._print(f"ノイズタイプ: {noise_type}\n")

        # ノイズ除去関数を選択
        noise_funcs = {
            "white": self.remove_white_noise,
            "hum": self.remove_hum_noise,
            "click": self.remove_click_noise,
            "background": self.reduce_background_noise
        }

        denoise_func = noise_funcs.get(noise_type)
        if not denoise_func:
            self._print(f"エラー: 不明なノイズタイプ '{noise_type}'")
            return

        for i, audio_file in enumerate(audio_files, 1):
            self._print(f"[{i}/{len(audio_files)}]")
            output_file = output_path / audio_file.name
            denoise_func(str(audio_file), str(output_file), **kwargs)
            self._print("")


def main():
    """メイン処理"""
    audio_file = "downloads/audio/【粗品】最近のSNSニュース斬った【1人賛否】.wav"

    if not os.path.exists(audio_file):
        print(f"ファイルが見つかりません: {audio_file}")
        return

    print("=" * 70)
    print("ノイズ除去処理")
    print("=" * 70)

    # デノイザーのインスタンスを作成
    denoiser = AudioDenoiser(verbose=True)

    # 各種ノイズ除去の例（コメント解除して使用）

    # 1. ホワイトノイズ除去
    denoiser.remove_white_noise(
        audio_file,
        "denoised_white.wav",
        noise_reduce_strength=0.5
    )

    # 2. ハムノイズ除去（60Hz）
    denoiser.remove_hum_noise(
        audio_file,
        "denoised_hum.wav",
        hum_frequency=60,
        harmonics=5
    )

    # 3. クリック・ポップノイズ除去
    denoiser.remove_click_noise(
        audio_file,
        "denoised_click.wav",
        threshold=3.0
    )

    # 4. 背景ノイズ抑制
    denoiser.reduce_background_noise(
        audio_file,
        "denoised_background.wav",
        strength=0.3
    )

    # バッチ処理例
    # denoiser.batch_denoise_directory(
    #     input_dir="wav_file",
    #     output_dir="denoised_output",
    #     noise_type="white",
    #     noise_reduce_strength=0.5
    # )

    print("\n処理を実行するには、main()内のコメントを解除してください")


if __name__ == "__main__":
    main()
