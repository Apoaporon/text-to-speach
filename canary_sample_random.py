"""Generate random canary TTS audio sample."""
# このスクリプトは、Canary TTSモデルを使って、指定した話者説明と日本語テキストから音声（WAV）を生成します。
import torch
import torchaudio
from canary_tts.xcodec2.modeling_xcodec2 import XCodec2Model
from rubyinserter import add_ruby
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("2121-8/canary-tts-0.5b")
model = AutoModelForCausalLM.from_pretrained("2121-8/canary-tts-0.5b", device_map="auto", torch_dtype=torch.bfloat16)
codec = XCodec2Model.from_pretrained("HKUSTAudio/xcodec2")

DESCRIPTION = (
    "An elderly male voice with a gentle and slightly trembling tone, "
    "speaking slowly and warmly in a calm and relaxed manner. "
    "The overall atmosphere feels nostalgic and kind, "
    "as if he is telling a story to his grandchildren in a quiet room."
)

PROMPT = 'こんにちは。お元気ですか？'  # 音声化する日本語テキスト

PROMPT = add_ruby(PROMPT)  # 日本語テキストにルビ（ふりがな）を付与
# モデルへの入力をchat形式で作成
chat = [
    {"role": "system", "content": DESCRIPTION},  # システム（話者説明）
    {"role": "user", "content": PROMPT}         # ユーザー（発話テキスト）
]
tokenized_input = tokenizer.apply_chat_template(
    chat,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
).to(model.device)  # chat形式をトークン化

with torch.no_grad():
    # 音声生成モデルで音声トークンを生成
    output = model.generate(
        tokenized_input,
        max_new_tokens=1024,         # 生成するトークン数の最大値
        top_p=0.95,                 # nucleus samplingの確率
        temperature=0.7,            # 生成の多様性
        repetition_penalty=1.05,    # 繰り返し抑制
    )[0]

audio_tokens = output[len(tokenized_input[0]):]  # 生成された音声トークン部分のみ抽出
output_audios = codec.decode_code(audio_tokens.unsqueeze(0).unsqueeze(0).cpu())  # コーデックで音声トークンをWAVデータにデコード
torchaudio.save("sample001.wav", src=output_audios[0].cpu(), sample_rate=16000)  # WAVファイルとして保存
