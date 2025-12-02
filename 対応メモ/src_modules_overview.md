# src配下のモジュール概要

このディレクトリには、音声処理に関する6つの専門モジュールが含まれています。

---

## 🎯 src/モジュール概要（各ファイルの役割）

| ファイル名 | 役割 | 主な機能 |
|-----------|------|---------|
| **analyze_audio.py** | 音声分析・可視化 | ファイル情報取得、波形表示、スペクトログラム、特徴量抽出 |
| **audio_denoise.py** | ノイズ除去 | ホワイトノイズ、ハムノイズ、クリック音、背景ノイズの除去 |
| **audio_eq_processor.py** | 周波数調整（EQ） | ハイパス/ローパスフィルタ、EQブースト、ディエッサー |
| **audio_level_processor.py** | レベル処理 | レベル分析、ピーク/ラウドネスノーマライズ、デクリップ |
| **audio_silence_processor.py** | 無音処理 | トリミング、無音圧縮、VAD分割、無音削除 |
| **download_youtube.py** | YouTubeダウンロード | 動画から音声抽出、プレイリスト対応、形式変換 |
| **🆕 audio_pipeline.py** | **自動処理パイプライン** | **分析→判定→処理を自動実行** |

### 📊 処理フロー例

#### 🔹 手動処理フロー（各モジュールを個別に使用）
```
YouTube動画 
  ↓ (download_youtube.py)
音声ファイル
  ↓ (analyze_audio.py) → 分析・確認
  ↓ (audio_denoise.py) → ノイズ除去
  ↓ (audio_eq_processor.py) → 周波数調整
  ↓ (audio_level_processor.py) → 音量統一
  ↓ (audio_silence_processor.py) → 無音処理
完成した音声ファイル
```

#### 🔹 自動処理フロー（**audio_pipeline.py**を使用）
```
音声ファイル
  ↓
[audio_pipeline.py]
  │
  ├─ 1. 分析（analyze_audio.py）
  │   ├─ ファイル情報取得
  │   ├─ レベル測定
  │   └─ ノイズ検出
  │
  ├─ 2. 処理計画の自動生成
  │   └─ 分析結果に基づき必要な処理を判定
  │
  └─ 3. 自動処理実行
      ├─ 無音トリミング（必要に応じて）
      ├─ ノイズ除去（必要に応じて）
      ├─ EQ調整（必要に応じて）
      ├─ レベル補正（必要に応じて）
      └─ 無音圧縮（必要に応じて）
  ↓
完成した音声ファイル
```

**💡 パイプラインの利点**
- 分析結果に基づいて自動的に最適な処理を選択
- 処理の実行順序も自動で最適化
- 不要な処理はスキップして効率的に実行
- バッチ処理で複数ファイルを一括処理可能

### 🔑 共通の特徴

- **クラスベース設計**: 4つのプロセッサー（Denoiser, EQProcessor, LevelProcessor, SilenceProcessor）
- **verbose制御**: 処理状況の出力をON/OFF可能
- **バッチ処理対応**: ディレクトリ単位での一括処理機能
- **ステレオ/モノラル対応**: 自動判別して適切に処理
- **エラーハンドリング**: 安全な例外処理とユーザーフレンドリーなメッセージ

---

# 各モジュールの詳細

以下、各ファイルの詳細な機能説明です。

---

## 📂 モジュール一覧

### 1. `analyze_audio.py`
**音声ファイルの情報取得・分析**

#### 主な機能
- 音声ファイルの詳細情報取得
  - ファイルサイズ、長さ、サンプリングレート
  - チャンネル数、ビット深度、フォーマット
  - ピーク振幅、RMSレベル
- 音声の可視化
  - 波形プロット
  - スペクトログラム表示
  - メルスペクトログラム表示
- 音声特徴量の抽出
  - MFCC（メル周波数ケプストラム係数）
  - クロマ特徴量
  - スペクトルコントラスト
- ディレクトリ一括分析

#### 主な関数
- `analyze_audio_file()`: 音声ファイルの詳細情報を取得
- `visualize_waveform()`: 波形を可視化
- `visualize_spectrogram()`: スペクトログラムを表示
- `visualize_mel_spectrogram()`: メルスペクトログラムを表示
- `extract_audio_features()`: 音声特徴量を抽出
- `analyze_directory()`: ディレクトリ内の全ファイルを分析

---

### 2. `audio_denoise.py`
**音声ノイズ除去処理**

#### クラス: `AudioDenoiser`

#### 主な機能
- **ホワイトノイズ・ヒスノイズ除去**
  - スペクトル減算法による除去
  - 除去強度の調整可能
- **ハムノイズ除去**
  - 50Hz/60Hzのブーン音を除去
  - ノッチフィルタによる高調波除去
- **ポップノイズ・クリック音除去**
  - デクリック処理
  - 閾値ベースの検出と補間
- **環境音などの背景ノイズ軽減**
  - ウィーナーフィルタによる抑制
- **一括処理機能**
  - ディレクトリ内の全ファイルをバッチ処理

#### 主なメソッド
- `remove_white_noise()`: ホワイトノイズ除去
- `remove_hum_noise()`: ハムノイズ除去
- `remove_click_noise()`: クリックノイズ除去
- `reduce_background_noise()`: 背景ノイズ抑制
- `batch_denoise_directory()`: ディレクトリ一括処理

#### 使用例
```python
denoiser = AudioDenoiser(verbose=True)
denoiser.remove_white_noise("input.wav", "output.wav", noise_reduce_strength=0.5)
```

---

### 3. `audio_eq_processor.py`
**音声の周波数特性調整（EQ処理）**

#### クラス: `AudioEQProcessor`

#### 主な機能
- **ハイパスフィルタ**
  - 低域カット（100Hz以下など）
  - バターワースフィルタ使用
- **ローパスフィルタ**
  - 高域カット（16kHz以上など）
- **音声帯域EQブースト**
  - ピーキングフィルタによる特定周波数帯域の強調
  - 中心周波数・バンド幅・ブースト量の調整可能
- **ディエッサー**
  - シビランス（サ行・シ音）の抑制
  - 6kHz-10kHz帯域の圧縮処理
- **フルEQチェーン**
  - 複数のEQ処理を一括適用
- **一括処理機能**

#### 主なメソッド
- `apply_highpass_filter()`: ハイパスフィルタ適用
- `apply_lowpass_filter()`: ローパスフィルタ適用
- `apply_voice_band_eq()`: 音声帯域EQブースト
- `apply_deesser()`: ディエッサー適用
- `apply_full_eq_chain()`: フルEQチェーン処理
- `batch_eq_process_directory()`: ディレクトリ一括処理

#### 使用例
```python
processor = AudioEQProcessor(verbose=True)
processor.apply_full_eq_chain(
    "input.wav", 
    "output.wav",
    highpass_cutoff=100.0,
    lowpass_cutoff=16000.0,
    voice_boost_db=2.5,
    apply_deesser_flag=True
)
```

---

### 4. `audio_level_processor.py`
**音声レベル（音量・クリッピング）の分析・修正**

#### クラス: `AudioLevelProcessor`

#### 主な機能
- **レベル分析**
  - ピークレベル（dBFS）測定
  - RMSレベル測定
  - ラウドネス（LUFS）推定
  - クレストファクター計算
  - クリッピング検出
  - ダイナミックレンジ測定
  - ヘッドルーム計算
- **ピークノーマライズ**
  - 目標ピークレベルへの調整（-1dBFS等）
- **ラウドネスノーマライズ**
  - 目標ラウドネスへの調整（-20 LUFS等）
  - クリッピング防止機能付き
- **デクリップ処理**
  - クリッピング部分の修正（簡易版）
- **一括ノーマライズ**
  - ディレクトリ内の全ファイルを統一レベルに調整

#### 主なメソッド
- `analyze_audio_levels()`: レベル情報を詳細分析
- `print_level_analysis()`: 分析結果を整形表示
- `normalize_peak()`: ピークノーマライズ
- `normalize_loudness()`: ラウドネスノーマライズ
- `declip_audio()`: デクリップ処理
- `batch_normalize_directory()`: ディレクトリ一括ノーマライズ

#### 使用例
```python
processor = AudioLevelProcessor(verbose=True)
info = processor.analyze_audio_levels("audio.wav")
processor.print_level_analysis(info, "audio.wav")
processor.normalize_peak("input.wav", "output.wav", target_db=-1.0)
```

---

### 5. `audio_silence_processor.py`
**音声の無音・不要区間処理**

#### クラス: `AudioSilenceProcessor`

#### 主な機能
- **先頭・末尾のトリミング**
  - 無音部分の自動カット
  - 閾値ベースの検出
- **長い無音の圧縮**
  - 指定時間以上の無音を短縮
  - 編集時の時間短縮に有効
- **VAD（Voice Activity Detection）分割**
  - 音声区間ごとに自動分割
  - 個別ファイルとして保存
- **全無音区間の削除**
  - 音声部分のみを結合
- **一括処理機能**
  - 複数ファイルの同時処理

#### 主なメソッド
- `trim_silence()`: 先頭・末尾のトリミング
- `compress_long_silence()`: 長い無音の圧縮
- `split_by_vad()`: VAD分割
- `remove_all_silence()`: 全無音削除
- `batch_silence_process_directory()`: ディレクトリ一括処理

#### 使用例
```python
processor = AudioSilenceProcessor(verbose=True)
processor.trim_silence("input.wav", "output.wav", silence_thresh_db=-40.0)
files = processor.split_by_vad(
    "long_audio.wav",
    "output_dir",
    silence_thresh_db=-40.0,
    min_voice_duration=0.5
)
```

---

### 6. `download_youtube.py`
**YouTube動画のダウンロード**

#### 主な機能
- **YouTube動画から音声抽出**
  - yt-dlpを使用した高品質ダウンロード
  - 複数形式対応（WAV、MP3、M4A等）
  - 音質選択可能
- **プレイリスト対応**
  - プレイリスト全体のダウンロード
  - 個別ファイルとして保存
- **複数URL対応**
  - リストからの一括ダウンロード
- **動画情報取得**
  - タイトル、長さ、フォーマット情報の取得

#### 主な関数
- `download_youtube_audio()`: 単一動画の音声ダウンロード
- `download_playlist()`: プレイリストのダウンロード
- `download_multiple_urls()`: 複数URLの一括ダウンロード
- `get_video_info()`: 動画情報の取得

#### 使用例
```python
# 単一動画のダウンロード
file_path = download_youtube_audio(
    "https://www.youtube.com/watch?v=xxxxx",
    output_dir="downloads/audio",
    audio_format="wav",
    quality="best"
)

# プレイリストのダウンロード
files = download_playlist(
    "https://www.youtube.com/playlist?list=xxxxx",
    output_dir="downloads/playlist"
)
```

---

## 🔧 依存ライブラリ

各モジュールで使用している主な外部ライブラリ：

- **librosa**: 音声信号処理・分析
- **numpy**: 数値計算
- **soundfile**: 音声ファイルの読み書き
- **scipy**: 信号処理（フィルタ設計等）
- **matplotlib**: 可視化（analyze_audio.pyのみ）
- **yt-dlp**: YouTube動画ダウンロード（download_youtube.pyのみ）

---

## 📝 共通の設計思想

### クラスベース設計（ノイズ除去・EQ・レベル・無音処理）
- `verbose`パラメータで出力制御
- 内部メソッド `_print()` による統一された出力管理
- 各処理を独立したメソッドとして提供
- バッチ処理機能を標準搭載

### エラーハンドリング
- try-exceptブロックによる安全な処理
- エラー時はFalseまたはNoneを返却
- ユーザーフレンドリーなエラーメッセージ

### 柔軟性
- デフォルト値を提供しつつ、細かいパラメータ調整が可能
- ステレオ/モノラルの自動判別と適切な処理
- 複数のファイル形式に対応

---

## 🚀 推奨される処理フロー

### 手動処理の場合

1. **ダウンロード** (`download_youtube.py`)
   - YouTube等から素材を入手

2. **分析** (`analyze_audio.py`)
   - 音声の状態を確認

3. **ノイズ除去** (`audio_denoise.py`)
   - 不要なノイズを除去

4. **EQ調整** (`audio_eq_processor.py`)
   - 周波数特性を整える

5. **レベル調整** (`audio_level_processor.py`)
   - 音量を統一

6. **無音処理** (`audio_silence_processor.py`)
   - 不要な無音を削除・調整

### ⭐ 自動処理の場合（推奨）

**`audio_pipeline.py`を使用** すれば、上記の処理を自動化できます！

---

# 🆕 7. audio_pipeline.py - 自動処理パイプライン

## 概要

音声ファイルを分析し、その結果に基づいて必要な前処理だけを自動的に実行するパイプラインです。

### 🎯 主な機能

- **自動分析**: ファイル情報、レベル、ノイズを自動検出
- **自動判定**: 分析結果から必要な処理を自動選択
- **自動実行**: 選択された処理を最適な順序で実行
- **バッチ処理**: ディレクトリ単位での一括処理

## クラス: AudioProcessingPipeline

### 主要メソッド

#### `analyze_and_plan(input_file: str)`
音声ファイルを分析し、処理計画を作成

**判定ロジック:**
- **無音トリミング**: 常に実行（データ整形のため）
- **クリックノイズ除去**: クリック音が10個以上検出された場合
- **ハムノイズ除去**: 50Hz/60Hzのピークが周囲より20dB以上大きい場合
- **ホワイトノイズ除去**: ノイズレベルが-60dB以上の場合
  - ノイズレベルに応じて強度を自動調整（0.3〜0.7）
- **背景ノイズ除去**: 背景ノイズレベルが-50dB以上の場合
- **ハイパスフィルタ**: サンプリングレート≥44.1kHzまたは背景ノイズがある場合
- **ボイスバンドEQ**: 常に実行（音声明瞭化のため）
- **ディエッサー**: サンプリングレート≥44.1kHzの場合
- **デクリップ**: クリッピング率が0.01%以上の場合
- **ピークノーマライズ**: ピークレベルが-6dB未満の場合
- **ラウドネスノーマライズ**: ラウドネスが-16LUFS±3dBの範囲外の場合
- **無音圧縮**: 常に実行（データ効率化のため）

#### `execute_pipeline(input_file: str, output_file: str, analysis_result: Optional[Dict] = None)`
パイプラインを実行

**処理の流れ:**
1. 分析が未実施の場合は自動分析
2. 処理計画を表示
3. 一時ファイルを使用して順次処理
4. 最終結果を出力ファイルに保存
5. 一時ファイルを自動削除

#### `process_directory(input_dir: str, output_dir: str, file_pattern: str = "*.wav")`
ディレクトリ内の全ファイルを一括処理

## 使用例

### 基本的な使い方

```python
from src.audio_pipeline import AudioProcessingPipeline

# パイプラインを作成
pipeline = AudioProcessingPipeline(verbose=True)

# 単一ファイルを自動処理
pipeline.execute_pipeline("input.wav", "output.wav")
```

### 分析結果を確認してから処理

```python
# 分析のみ実行
analysis = pipeline.analyze_and_plan("input.wav")

# 処理計画を確認
for name, func_name, params in analysis['processing_plan']:
    print(f"- {name}: {params}")

# 処理を実行
pipeline.execute_pipeline("input.wav", "output.wav", analysis)
```

### ディレクトリ一括処理

```python
# WAVファイルを一括処理
results = pipeline.process_directory(
    input_dir="audio/wav_file",
    output_dir="audio/output",
    file_pattern="*.wav"
)

# 結果を確認
for filename, success in results.items():
    print(f"{'✅' if success else '❌'} {filename}")
```

### コマンドラインから実行

```bash
# 単一ファイル処理
python src/audio_pipeline.py input.wav output.wav

# ディレクトリ一括処理
python src/audio_pipeline.py input_dir/ output_dir/ --batch

# 詳細出力を抑制
python src/audio_pipeline.py input.wav output.wav --quiet

# ファイルパターンを指定
python src/audio_pipeline.py input_dir/ output_dir/ --batch --pattern "*.mp3"
```

## パラメータ

### コンストラクタ
- `verbose` (bool): 処理状況の出力（デフォルト: True）

### execute_pipeline
- `input_file` (str): 入力ファイルパス
- `output_file` (str): 出力ファイルパス
- `analysis_result` (Optional[Dict]): 事前分析結果（Noneの場合は自動分析）

### process_directory
- `input_dir` (str): 入力ディレクトリ
- `output_dir` (str): 出力ディレクトリ
- `file_pattern` (str): ファイルパターン（デフォルト: "*.wav"）

## 処理順序

パイプラインは以下の順序で処理を実行します:

1. **無音トリミング** - 先頭・末尾の無音削除
2. **クリックノイズ除去** - パルス性ノイズの除去
3. **ハムノイズ除去** - 電源ハムの除去
4. **ホワイトノイズ除去** - 定常的なノイズの除去
5. **背景ノイズ除去** - 環境ノイズの除去
6. **ハイパスフィルタ** - 低域ノイズのカット
7. **ボイスバンドEQ** - 音声帯域の強調
8. **ディエッサー** - 歯擦音の抑制
9. **デクリップ** - クリッピングの修復
10. **ノーマライゼーション** - 音量の統一
11. **無音圧縮** - 長い無音の短縮

この順序は音声処理の一般的なベストプラクティスに基づいています。

## 注意事項

- 処理には時間がかかる場合があります（特にバッチ処理）
- 一時ファイルは自動的に削除されますが、エラー時に残る場合があります
- 分析の閾値は標準的な値ですが、素材によっては調整が必要な場合があります
- 処理前に必ずバックアップを取ることを推奨します

## サンプルスクリプト

詳細な使用例は `pipeline_example.py` を参照してください。

---

## 💡 Tips

- 各処理クラスは `verbose=False` で静かに実行可能
- バッチ処理機能を使えば大量のファイルを効率的に処理
- **audio_pipeline.pyを使えば、分析→判定→処理を全自動化できる**
- 処理の順序を工夫することで、より高品質な結果が得られる
- 設定値は素材に応じて調整が必要（特に閾値系パラメータ）

---

**最終更新日**: 2025年12月2日
