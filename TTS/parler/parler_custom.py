import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# Load the lightweight Parler model
print("[INIT] Loading Parler-TTS Mini...")
MODEL_NAME = "parler-tts/parler_tts_mini_v1"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()

def synthesize_speech(text, style="neutral", pitch_shift=0):
    """
    Generate speech with optional style and pitch shift.
    """
    if not text.strip():
        raise ValueError("Text input is empty.")

    # Adjust emotional tone slightly (prompt-based conditioning)
    if style == "happy":
        text = f"(cheerful, bright tone) {text}"
    elif style == "sad":
        text = f"(soft, melancholic tone) {text}"
    elif style == "angry":
        text = f"(firm, serious tone) {text}"
    elif style == "cute":
        text = f"(anime high-pitched cute tone) {text}"

    inputs = processor(text=text, return_tensors="pt")

    with torch.no_grad():
        output_audio = model.generate(**inputs)
    
    # Convert to numpy and adjust pitch if needed
    audio = output_audio.cpu().numpy().flatten()

    # Optional pitch shift (simple resample-based)
    if pitch_shift != 0:
        import librosa
        audio = librosa.effects.pitch_shift(audio, sr=24000, n_steps=pitch_shift)

    # Save to temporary file
    import tempfile
    tmp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(tmp_path, audio, 24000)
    return tmp_path
