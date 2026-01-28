"""テスト用の簡単なスクリプト"""
from src.emotion_recognizer import EmotionRecognizer
from pathlib import Path
import sys

# 指定されたディレクトリから最初のwavファイルを取得
audio_dir = Path(r"C:\Users\usago\python\audio-data\zundamon\ROHAN4600_zumndamon_normal_synchronized_wav")
audio_files = list(audio_dir.glob("*.wav"))

if not audio_files:
    print(f"❌ WAVファイルが見つかりません: {audio_dir}")
    sys.exit(1)

# 最初の3ファイルをテスト
test_files = audio_files[:3]
print(f"📁 テストディレクトリ: {audio_dir}")
print(f"📊 総ファイル数: {len(audio_files)}個")
print(f"🎯 テストファイル数: {len(test_files)}個\n")

# CPUモードで実行（RTX-5080サポート待ち）
recognizer = EmotionRecognizer(device='cpu', verbose=True)

for i, audio_file in enumerate(test_files, 1):
    print(f"\n{'=' * 80}")
    print(f"テスト [{i}/{len(test_files)}]: {audio_file.name}")
    print(f"{'=' * 80}")
    
    result = recognizer.recognize_emotion(str(audio_file))

    if result['error'] is None:
        print(f"\n✅ 感情認識結果:")
        print(f"   支配的感情: {result['dominant_emotion']} (信頼度: {result['confidence']:.2%})")
        print(f"\n   感情スコア:")
        for emotion, score in sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True):
            print(f"     {emotion:10s}: {score:.2%}")
        print(f"\n   処理時間: {result['processing_time']:.3f}秒")
    else:
        print(f"\n❌ エラー: {result['error']}")
