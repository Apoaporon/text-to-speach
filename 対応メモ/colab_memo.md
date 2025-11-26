# Style-Bert-VITS2 Google Colab セットアップメモ

## 基本設定

```python
# 元となる音声ファイル（wav形式）を入れるディレクトリ
input_dir = "/content/drive/MyDrive/Style-Bert-VITS2/inputs"

# モデル名（話者名）を入力
model_name = "your_model_name"

# こういうふうに書き起こして欲しいという例文（句読点の入れ方・笑い方や固有名詞等）
initial_prompt = "こんにちは。元気、ですかー？ふふっ、私は……ちゃんと元気だよ！"
```

## スクリプト実行

```bash
!python slice.py -i {input_dir} --model_name {model_name}
!python transcribe.py --model_name {model_name} --initial_prompt {initial_prompt} --use_hf_whisper
```

### エラーが出る場合の対処法

上記のコード実行でエラーが出る場合は、以下のように書き換えてください：

```bash
# 変更前
!python transcribe.py --model_name {model_name} --initial_prompt {initial_prompt} --use_hf_whisper

# 変更後
!python transcribe.py --model_name {model_name} --initial_prompt {initial_prompt} --use_hf_whisper --hf_repo_id 'openai/whisper-large-v3'
```

## slice.py の修正

### 1. torchaudioのインポートを追加

ファイル冒頭に以下を追加：

```python
import torchaudio  # ←これ追加
```

### 2. torchaudioのモンキーパッチを追加

インポート部分の後に以下を追加：

```python
# ==== ここから追加：torchaudio のモンキーパッチ ====
# 新しい torchaudio には list_audio_backends が無いのでダミー実装を差し込む
if not hasattr(torchaudio, "list_audio_backends"):
    def _dummy_list_audio_backends():
        # silero-vad 側は「何らかのバックエンド名のリスト」が返ってくれば満足なので
        # 実在しそうな名前をテキトウに返しておく
        return ["sox_io"]

    torchaudio.list_audio_backends = _dummy_list_audio_backends
# ==== ここまで追加 ====
```

### 3. get_stamps メソッドを書き換え

```python
def get_stamps(
    vad_model: Any,
    utils: Any,
    audio_file: Path,
    min_silence_dur_ms: int = 700,
    min_sec: float = 2,
    max_sec: float = 12,
):
    """
    min_silence_dur_ms: int (ミリ秒):
        このミリ秒数以上を無音だと判断する。
    （中略）
    """

    (get_speech_timestamps, *_) = utils
    target_sr = 16000  # Silero VAD 推奨サンプリングレート

    min_ms = int(min_sec * 1000)

    # --- ここから自前ロードに差し替え ---
    wav, sr = torchaudio.load(str(audio_file))  # [channels, time]

    # モノラル化（複数チャンネルの場合）
    if wav.dim() > 1:
        wav = wav.mean(dim=0, keepdim=False)

    # サンプリングレートが違う場合はリサンプリング
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    # --- ここまで ---

    # Silero VAD は 1D Tensor を想定
    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=sr,
        min_silence_duration_ms=min_silence_dur_ms,
        min_speech_duration_ms=min_ms,
        max_speech_duration_s=max_sec,
    )

    return speech_timestamps
```