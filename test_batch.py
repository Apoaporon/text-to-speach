"""バッチ処理の小規模テスト"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from analyze_emotion_batch import analyze_directory

# 指定ディレクトリの最初の10ファイルのみを処理するテスト
input_dir = r"C:\Users\usago\python\audio-data\zundamon\ROHAN4600_zumndamon_normal_synchronized_wav"
output_dir = "test_output"

# パターンで最初の数ファイルのみ取得（テスト用）
print("🧪 バッチ処理テスト開始\n")
analyze_directory(
    input_dir=input_dir,
    output_dir=output_dir,
    pattern="ROHAN4600_000[1-9].wav",  # 最初の9ファイルのみ
    recursive=False,
    model_name=None,
    device="cpu",
    output_format="both"
)
