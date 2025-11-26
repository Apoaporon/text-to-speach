"""指定の声で音声合成を行うサンプルコード。"""
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from rubyinserter import add_ruby
from transformers import AutoTokenizer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL = ParlerTTSForConditionalGeneration.from_pretrained("2121-8/japanese-parler-tts-mini").to(DEVICE)
prompt_tokenizer = AutoTokenizer.from_pretrained("2121-8/japanese-parler-tts-mini", subfolder="prompt_tokenizer")
description_tokenizer = AutoTokenizer.from_pretrained("2121-8/japanese-parler-tts-mini", subfolder="description_tokenizer")

PROMPT = "こんにちは、今日はどのようにお過ごしですか？"
# 話者の特徴を記載するとその音声で合成される
DESCRIPTION = (
    "A cheerful and gentle Japanese female VTuber with a soothing and kind tone, "
    "speaking in a warm and friendly manner that feels comforting and cute."
)
PROMPT = add_ruby(PROMPT)
input_ids = description_tokenizer(DESCRIPTION, return_tensors="pt").input_ids.to(DEVICE)
prompt_input_ids = prompt_tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)

generation = MODEL.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_japanese_setting_output.wav", audio_arr, MODEL.config.sampling_rate)
