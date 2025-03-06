import io
import re
import tempfile
from pathlib import Path
from typing import Optional
import os
from pathlib import Path
from cached_path import cached_path

import numpy as np
import soundfile as sf

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT

model = "F5-TTS-small" 
### keep your ref audio and text here ###

ref_audio = "/home/cmi_10101/Documents/voice/Hindi/my_dataset/data/2892452d-2d04-4a70-ad1f-b0409f4ece0a_1_002.wav"
ref_text = '''हाँ भैया अभी दसेहरी आम में दसेहरी रखें हैं  ओर केले भी हैं हमारे पास अभी मतलब बताइये आपको कित्ते के चाहिए 
'''
##### end #####


####### keep your Desired generated text #####

gen_text = '''  
योजना अभी निकाली है बच्चों के लिए जिसका नाम है सीखो कमाओ योजना जिसके अंतर्गत हम ट्रैनिंग के साथ साथ पैसे भी कमा सकते हैं और ट्रैनिंग के बाद अच्छी जॉब भी
'''
####### end ########

output_dir = "gen_audios"
output_file = "generated_audio.wav"
wav_path = Path(output_dir) / output_file

##### keep checkpoints path here ###
""" 
keep your checkpoints in ckpts directory 
and vocab.txt file in data dir
"""

ckpt_file = "/home/cmi_10101/Documents/voice/F5-TTS/ckpts/model_10000.pt"
vocab_file = "/home/cmi_10101/Documents/voice/F5-TTS/data/vocab.txt"

#### end ####

remove_silence = True
speed = 1.0 
nfe_step = 32
vocoder_name = "vocos"
indic=True
mel_spec_type = vocoder_name 

# Hardcoded local path for the vocoder (adjust as needed)
if vocoder_name == "vocos":
    vocoder_local_path = "../ckpts/vocos-mel-24khz"
elif vocoder_name == "bigvgan":
    print("No default F5-TTS-small ckpt available for bigvgan yet")
    exit(1)

# Load vocoder (using the hardcoded local path and setting is_local to False or True as required)
vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=False, local_path=vocoder_local_path)


# For F5-TTS-small, set the model class and configuration.
model_cls = DiT
model_cfg = dict(dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, conv_layers=4)

print(f"Using model: {model} ...")
ema_model = load_model(model_cls, model_cfg, ckpt_file, mel_spec_type=mel_spec_type, vocab_file=vocab_file)


# -------------
# MAIN PROCESSING FUNCTION
# -------------
def main_process(ref_audio, ref_text, text_gen, model_obj, mel_spec_type, remove_silence, speed):
    # Use a single voice (named "main") with hardcoded reference values.
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    voices = {"main": main_voice}

    # Preprocess the reference audio and text for the main voice.
    for voice in voices:
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"], voices[voice]["ref_text"]
        )
        print("Voice:", voice)
        print("Processed Ref Audio:", voices[voice]["ref_audio"])
        print("Processed Ref Text:", voices[voice]["ref_text"])

    generated_audio_segments = []
    # Optionally, if you include voice tags in gen_text (like [main]), the regex below will parse them.
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, text_gen)
    reg2 = r"\[(\w+)\]"
    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        voice = match.group(1) if match else "main"
        if voice not in voices:
            print(f"Voice '{voice}' not found, using 'main'.")
            voice = "main"
        # Remove voice tag if present and strip whitespace.
        text_clean = re.sub(reg2, "", text).strip()
        print(f"Generating audio for voice: {voice} with text: {text_clean}")
        audio, final_sample_rate, _ = infer_process(
            voices[voice]["ref_audio"],
            voices[voice]["ref_text"],
            text_clean,
            model_obj,
            vocoder,
            mel_spec_type=mel_spec_type,
            speed=speed,
            nfe_step=nfe_step,
            indic=indic
        )
        generated_audio_segments.append(audio)

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Write the audio file. (Note: sf.write accepts a file path as its first argument.)
        sf.write(str(wav_path), final_wave, final_sample_rate)
        # Optionally, remove silence from the generated file.
        if remove_silence:
            remove_silence_for_generated_wav(str(wav_path))
        print(f"Generated audio written to {wav_path}")

def main():
    main_process(ref_audio, ref_text, gen_text, ema_model, mel_spec_type, remove_silence, speed)

if __name__ == "__main__":
    main()