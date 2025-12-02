# 音声処理パイプライン

音声ファイルを自動分析し、必要な前処理だけを実行するスマートなパイプラインです。

## 🚀 クイックスタート

```python
from src.audio_pipeline import AudioProcessingPipeline

# パイプラインを作成
pipeline = AudioProcessingPipeline(verbose=True)

# 自動処理を実行（分析→判定→処理）
pipeline.execute_pipeline("input.wav", "output.wav")
```

たったこれだけで、以下の処理が自動的に実行されます:
- 📊 音声の分析（レベル、ノイズ検出）
- 🎯 必要な処理の自動判定
- ⚙️ 最適な順序で前処理を実行

## ✨ 主な機能

### 自動判定される処理

- ✅ **無音トリミング**: 先頭・末尾の無音を自動削除
- 🔇 **ノイズ除去**: ホワイトノイズ、ハムノイズ、クリックノイズ、背景ノイズを検出して除去
- 🎚️ **EQ調整**: ハイパスフィルタ、ボイスバンドEQ、ディエッサーを自動適用
- 📈 **レベル補正**: クリッピング修復、ピーク/ラウドネスノーマライゼーション
- ⏱️ **無音圧縮**: 長い無音区間を短縮してデータを効率化
- ✂️ **🆕 無音区間で分割**: オプションで最終処理として音声を分割（`--split`オプション）

### 判定ロジックの例

| 条件 | 実行される処理 |
|------|--------------|
| クリック音が10個以上検出 | クリックノイズ除去 |
| ホワイトノイズレベル > -60dB | ホワイトノイズ除去 |
| ハムノイズ（50/60Hz）検出 | ハムノイズ除去 |
| クリッピング率 > 0.01% | デクリップ処理 |
| ピークレベル < -6dB | ピークノーマライズ |
| ラウドネスが-16LUFS±3dBの範囲外 | ラウドネスノーマライズ |

## 📖 使用例

### 1. 基本的な使い方

```python
from src.audio_pipeline import AudioProcessingPipeline

pipeline = AudioProcessingPipeline(verbose=True)
pipeline.execute_pipeline("input.wav", "output.wav")
```

### 2. 分析結果を確認してから処理

```python
# 先に分析だけ実行
analysis = pipeline.analyze_and_plan("input.wav")

# 処理計画を表示
print("\n実行予定の処理:")
for i, (name, func_name, params) in enumerate(analysis['processing_plan'], 1):
    print(f"{i}. {name}")

# 処理を実行
pipeline.execute_pipeline("input.wav", "output.wav", analysis)
```

### 3. ディレクトリ一括処理

```python
# ディレクトリ内の全WAVファイルを一括処理
results = pipeline.process_directory(
    input_dir="audio/input",
    output_dir="audio/output",
    file_pattern="*.wav"
)

# 結果表示
for filename, success in results.items():
    status = "✅" if success else "❌"
    print(f"{status} {filename}")
```

### 4. コマンドラインから実行

```bash
# 単一ファイル処理
python src/audio_pipeline.py input.wav output.wav

# 出力パスを省略（自動的に audio/output/ に保存）
python src/audio_pipeline.py input.wav

# ディレクトリ一括処理
python src/audio_pipeline.py input_dir/ output_dir/ --batch

# 🆕 無音区間で分割して出力（audio/output/split_segments/ に保存）
python src/audio_pipeline.py input.wav --split

# 🆕 分割パラメータを調整
python src/audio_pipeline.py input.wav --split --split-thresh -35.0 --min-voice 1.0

# 静かに実行（詳細出力なし）
python src/audio_pipeline.py input.wav output.wav --quiet
```

## 📂 プロジェクト構成

```
text-to-speech/
├── src/
│   ├── audio_pipeline.py          ⭐ 自動処理パイプライン（メイン）
│   ├── analyze_audio.py           📊 音声分析
│   ├── audio_denoise.py           🔇 ノイズ除去
│   ├── audio_eq_processor.py      🎚️ EQ調整
│   ├── audio_level_processor.py   📈 レベル処理
│   ├── audio_silence_processor.py ⏱️ 無音処理
│   └── download_youtube.py        📥 YouTube音声ダウンロード
├── pipeline_example.py            📝 使用例スクリプト
└── 対応メモ/
    └── src_modules_overview.md    📖 詳細ドキュメント
```

## 🔧 処理の流れ

```
入力ファイル
    ↓
┌─────────────────────┐
│ 1. 音声分析         │
│  - ファイル情報取得 │
│  - レベル測定       │
│  - ノイズ検出       │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ 2. 処理計画の作成   │
│  - 必要な処理を判定 │
│  - 最適な順序を決定 │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ 3. 前処理の実行     │
│  - 無音トリミング   │
│  - ノイズ除去       │
│  - EQ調整           │
│  - レベル補正       │
│  - 無音圧縮         │
└─────────────────────┘
    ↓
出力ファイル
```

## 🎓 より詳しい情報

各モジュールの詳細や、個別の処理について知りたい場合は、以下のドキュメントを参照してください:

📖 **[src_modules_overview.md](対応メモ/src_modules_overview.md)** - 全モジュールの詳細ドキュメント

## 🤝 個別モジュールの使い方

パイプラインを使わず、個別のモジュールを使うこともできます:

```python
from src.audio_denoise import AudioDenoiser
from src.audio_eq_processor import AudioEQProcessor

# ノイズ除去だけを実行
denoiser = AudioDenoiser(verbose=True)
denoiser.remove_white_noise("input.wav", "denoised.wav")

# EQ処理だけを実行
eq = AudioEQProcessor(verbose=True)
eq.apply_voice_band_eq("denoised.wav", "output.wav")
```

## 💡 Tips

- 処理前にバックアップを取ることを推奨します
- 初めて使う場合は`verbose=True`で実行して処理内容を確認
- バッチ処理は時間がかかるため、まず1ファイルで試すことを推奨
- 閾値が合わない場合は、個別モジュールを使って手動調整も可能

## 📊 対応フォーマット

- **入力**: WAV, MP3, M4A, FLAC等（librosa対応フォーマット）
- **出力**: WAV（16bit/24bit）

## 📝 ライセンス

このプロジェクトは個人用途での使用を想定しています。

---

**作成日**: 2025年12月2日
