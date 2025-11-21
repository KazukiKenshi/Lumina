import torch
import numpy as np
import torchaudio
from bark import generate_audio, preload_models
import tempfile
import time

# Preload small models for faster inference
preload_models(text_use_small=True, coarse_use_small=True, fine_use_small=True)

def stream_bark_tts(text, pause=1.5):
    """
    Stream short chunks of Bark-generated speech.
    """
    phrases = text.replace("...", ".").split(". ")
    for phrase in phrases:
        if not phrase.strip():
            continue
        print(f"[INFO] Generating Bark audio for: {phrase}")
        audio_array = generate_audio(phrase.strip())
        tmp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        torchaudio.save(tmp_path, torch.tensor(audio_array), 24000)
        yield tmp_path
        time.sleep(pause)  # simulate streaming latency
