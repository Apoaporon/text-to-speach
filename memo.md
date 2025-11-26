# プロジェクト対応まとめ

## 1. 仮想環境の構築・有効化
- プロジェクト直下にPython仮想環境を作成
- 有効化コマンド（zsh）:
  ```zsh
  source .venv/bin/activate
  ```

## 2. コーディング規約・警告対応
- flake8: 1行200文字まで許容（.flake8）
- mypy: 型警告対象外モジュール設定（mypy.ini）
  - parler_tts, soundfile, rubyinserter, transformers
- Pylint: 1行300文字まで許容（pyproject.toml）
- 定数名の命名規則対応
  - device → DEVICE
  - description → DESCRIPTION

---

# sample.py スクリプト概要

- 日本語Parler TTSモデルを使い、テキストから音声（WAV）を生成するサンプル
- 主な処理：
  1. 必要なライブラリ・モデル・トークナイザーをインポート
  2. GPUが使える場合はCUDA、なければCPUを利用
  3. モデルとトークナイザーをロード
  4. 音声化したい日本語テキスト（prompt）と話者説明（DESCRIPTION）を用意
  5. add_ruby関数で日本語テキストにルビ（ふりがな）を付与
  6. トークナイザーでテキストをID化し、モデルに入力
  7. モデルで音声データを生成
  8. 生成した音声をWAVファイルとして保存

- 生成ファイル：
  - parler_tts_japanese_out.wav（日本語音声）

---

# 実行手順

1. 仮想環境を有効化
   ```zsh
   source .venv/bin/activate
   ```
2. 必要なパッケージをインストール（例）
   ```zsh
   pip install torch parler-tts rubyinserter transformers soundfile
   ```
3. スクリプトを実行
   ```zsh
   python sample.py
   ```
4. `parler_tts_japanese_out.wav` が生成される

---

ご不明点や追加事項があれば追記してください。

https://huggingface.co/parler-tts

これが触っているモデル
https://huggingface.co/parler-tts/parler-tts-mini-v1

上記の学習の元ネタ
https://www.text-description-to-speech.com/



こちらを再学習させて以下の日本語用の軽量モデルを作成している
2121-8/japanese-parler-tts-mini


上記のPaler-TTSモデルは自然言語的に学習しているデータの中から自然言語的に指定された声を出す
| 項目         | 旧来TTS         | Parler-TTS            |
| ---------- | ------------- | --------------------- |
| 声質の指定方法    | speaker_id 固定 | 自然言語説明（柔軟）            |
| 話者のバリエーション | 学習済みIDのみ      | 任意に「文で」指定可能           |
| ランダム性      | なし（ID固定）      | 説明を変えれば無限に変化          |
| 表現の自由度     | 低い            | 高い（話し方・録音環境・トーンを文で指示） |

description: "A gentle Japanese female VTuber with a soft and warm voice."
↓
BERT系エンコーダでベクトル化（768次元）
↓
text_tokenizer(prompt) と結合
↓
Transformerデコーダが音声波形（またはコード）を生成


その他
https://github.com/SWivid/F5-TTS
参照音声を渡すことで声真似ができるらしいが、商用利用はNGとのこと

プロンプト制御方式
https://huggingface.co/2121-8/canary-tts-0.5b


https://zenn.dev/kun432/scraps/127eaa5cb72fca
LLMベースのTTSモデル？


Zero-shot TTS = Voice Cloningの話っぽい

Spark TTS
https://note.com/easy_ai_opt/n/n3933f54aa43c
中国のではあるところが若干心配。