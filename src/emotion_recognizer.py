"""
音声感情認識（Speech Emotion Recognition）モジュール

日本語音声に対応したXLS-Rベースの多言語モデルを使用して、
音声ファイルから感情を認識します。
"""
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor


class EmotionRecognizer:
    """音声感情認識クラス"""

    # 日本語音声感情認識モデル（Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition）
    DEFAULT_MODEL = "Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition"
    
    # JTESデータセットの感情ラベル
    EMOTION_LABELS = {
        0: "angry",    # 怒り
        1: "happy",    # 喜び
        2: "neutral",  # 中立
        3: "sad",      # 悲しみ
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        初期化

        Args:
            model_name: 使用するモデル名（省略時はDEFAULT_MODELを使用）
            device: 使用するデバイス（'cuda', 'cpu', Noneで自動検出）
            verbose: 処理状況を出力するかどうか
        """
        self.verbose = verbose
        self.model_name = model_name or self.DEFAULT_MODEL
        
        # デバイスの設定（GPU自動検出）
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self._print(f"🔧 音声感情認識モデルを初期化中...")
        self._print(f"   モデル: {self.model_name}")
        self._print(f"   デバイス: {self.device}")
        
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self._print(f"   GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # モデルとfeature extractorの読み込み
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_name
            )
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.to(self.device)
            self.model.eval()  # 評価モード
            
            self._print(f"✅ モデルの読み込み完了")
            
        except Exception as e:
            self._print(f"❌ モデルの読み込みに失敗: {e}")
            raise

    def _print(self, message: str) -> None:
        """verboseがTrueの場合のみ出力"""
        if self.verbose:
            print(message)

    def _load_and_preprocess_audio(
        self, 
        audio_file: str
    ) -> Tuple[torch.Tensor, int]:
        """
        音声ファイルを読み込み、前処理を実行

        Args:
            audio_file: 音声ファイルのパス

        Returns:
            (waveform, sample_rate)のタプル
        """
        # soundfileで音声読み込み（torchaudio互換性問題を回避）
        waveform, sample_rate = sf.read(audio_file, dtype='float32')
        
        # numpy配列をtensorに変換
        waveform = torch.from_numpy(waveform)
        
        # モノラルでない場合は変換
        if waveform.ndim == 1:
            # モノラル: (samples,) -> (1, samples)
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            # ステレオ: (samples, channels) -> (channels, samples)
            waveform = waveform.T
            # モノラルに変換
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 16kHzにリサンプリング（必要な場合）
        if sample_rate != 16000:
            # scipy.signalでリサンプリング
            from scipy import signal
            num_samples = int(waveform.shape[1] * 16000 / sample_rate)
            waveform_np = waveform.squeeze().numpy()
            waveform_resampled = signal.resample(waveform_np, num_samples)
            waveform = torch.from_numpy(waveform_resampled).unsqueeze(0).float()
            sample_rate = 16000
        
        return waveform, sample_rate

    def recognize_emotion(
        self, 
        audio_file: str
    ) -> Dict[str, any]:
        """
        音声ファイルから感情を認識

        Args:
            audio_file: 音声ファイルのパス

        Returns:
            感情認識結果の辞書
            {
                'emotions': {'angry': 0.1, 'happy': 0.3, 'neutral': 0.5, 'sad': 0.1},
                'dominant_emotion': 'neutral',
                'confidence': 0.5,
                'processing_time': 0.123,
                'error': None
            }
        """
        start_time = time.time()
        
        try:
            # 音声の読み込みと前処理
            waveform, sample_rate = self._load_and_preprocess_audio(audio_file)
            
            # numpy配列に変換（feature extractorの入力形式）
            audio_array = waveform.squeeze().numpy()
            
            # Feature extraction
            inputs = self.feature_extractor(
                audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            # デバイスに転送
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # 推論実行
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # ソフトマックスで確率に変換
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            probabilities = probabilities.cpu().numpy()[0]
            
            # 感情ごとのスコア
            emotions = {
                self.EMOTION_LABELS[i]: float(probabilities[i])
                for i in range(len(probabilities))
            }
            
            # 最も確率の高い感情
            dominant_idx = np.argmax(probabilities)
            dominant_emotion = self.EMOTION_LABELS[dominant_idx]
            confidence = float(probabilities[dominant_idx])
            
            processing_time = time.time() - start_time
            
            return {
                'emotions': emotions,
                'dominant_emotion': dominant_emotion,
                'confidence': confidence,
                'processing_time': processing_time,
                'error': None
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._print(f"❌ 感情認識エラー: {e}")
            
            return {
                'emotions': {},
                'dominant_emotion': None,
                'confidence': 0.0,
                'processing_time': processing_time,
                'error': str(e)
            }

    def recognize_emotion_batch(
        self, 
        audio_files: list[str],
        show_progress: bool = True
    ) -> list[Dict[str, any]]:
        """
        複数の音声ファイルから感情を一括認識

        Args:
            audio_files: 音声ファイルパスのリスト
            show_progress: プログレスバーを表示するかどうか

        Returns:
            感情認識結果のリスト
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(audio_files, desc="感情認識処理中")
            except ImportError:
                self._print("⚠️  tqdmがインストールされていません。プログレスバーなしで実行します。")
                iterator = audio_files
        else:
            iterator = audio_files
        
        for audio_file in iterator:
            result = self.recognize_emotion(audio_file)
            result['filename'] = Path(audio_file).name
            result['filepath'] = str(audio_file)
            results.append(result)
        
        return results

    def get_device_info(self) -> Dict[str, str]:
        """
        使用中のデバイス情報を取得

        Returns:
            デバイス情報の辞書
        """
        info = {
            'device': self.device,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_total'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
            info['gpu_memory_allocated'] = f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}GB"
        
        return info


def main():
    """テスト用のメイン関数"""
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python emotion_recognizer.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not Path(audio_file).exists():
        print(f"❌ ファイルが見つかりません: {audio_file}")
        sys.exit(1)
    
    # 感情認識の実行
    recognizer = EmotionRecognizer(verbose=True)
    
    print(f"\n{'=' * 80}")
    print(f"音声ファイル: {audio_file}")
    print(f"{'=' * 80}\n")
    
    result = recognizer.recognize_emotion(audio_file)
    
    if result['error'] is None:
        print(f"\n✅ 感情認識結果:")
        print(f"   支配的感情: {result['dominant_emotion']} (信頼度: {result['confidence']:.2%})")
        print(f"\n   感情スコア:")
        for emotion, score in sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True):
            print(f"     {emotion:10s}: {score:.2%} {'█' * int(score * 50)}")
        print(f"\n   処理時間: {result['processing_time']:.3f}秒")
    else:
        print(f"\n❌ エラー: {result['error']}")
    
    # デバイス情報の表示
    print(f"\n{'=' * 80}")
    print("デバイス情報:")
    print(f"{'=' * 80}")
    device_info = recognizer.get_device_info()
    for key, value in device_info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
