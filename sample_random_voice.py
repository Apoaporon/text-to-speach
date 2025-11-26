"""Sample code for Japanese Parler TTS model inference."""
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from rubyinserter import add_ruby
from transformers import AutoTokenizer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("2121-8/japanese-parler-tts-mini").to(DEVICE)
prompt_tokenizer = AutoTokenizer.from_pretrained("2121-8/japanese-parler-tts-mini", subfolder="prompt_tokenizer")
description_tokenizer = AutoTokenizer.from_pretrained("2121-8/japanese-parler-tts-mini", subfolder="description_tokenizer")

PROMPT = "私の名前はヒトデマンです。よろしくお願いすると思いましたか？びっくりですね。"
DESCRIPTION = "A female speaker with a slightly high-pitched voice delivers her words at a moderate speed with a quite monotone tone in a confined environment, resulting in a quite clear audio recording."


PROMPT = add_ruby(PROMPT)
input_ids = description_tokenizer(DESCRIPTION, return_tensors="pt").input_ids.to(DEVICE)
prompt_input_ids = prompt_tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_japanese_out_001.wav", audio_arr, model.config.sampling_rate)
