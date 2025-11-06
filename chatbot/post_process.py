import re
import random
from typing import Dict, Any, Optional, TypedDict, List

EMOTION_TO_AVATAR = {
    "happy": {"expression": "soft_smile", "animation": "gentle_nod"},
    "sad": {"expression": "concerned_soft", "animation": "slow_breath"},
    "empathetic": {"expression": "warm_smile", "animation": "head_tilt"},
    "concerned": {"expression": "concerned_eyebrows", "animation": "slow_nod"},
    "neutral": {"expression": "neutral", "animation": "idle_breathe"}
}

def estimate_speech_duration(text: str, wpm: int = 140) -> float:
    """Estimate speech duration based on average speaking speed."""
    words = len(re.findall(r'\w+', text))
    return round(words / (wpm / 60), 2)  # seconds
import re

def analyze_emotion(text: str) -> str:
    """Lightweight emotion classification based on keywords."""
    text_l = text.lower()

    if any(w in text_l for w in ["sad", "sorry", "difficult", "hard", "hopeless", "pain"]):
        return "sad"
    if any(w in text_l for w in ["glad", "happy", "great", "relieved", "improve", "better"]):
        return "happy"
    if any(w in text_l for w in ["understand", "hear you", "support", "okay"]):
        return "empathetic"
    if any(w in text_l for w in ["worried", "concerned", "stress", "anxious"]):
        return "concerned"
    return "neutral"

def compact_response(text: str, max_sentences: int = 3) -> str:
    """Trim or summarize model responses for TTS and avatar timing."""
    sentences = text.split(". ")
    if len(sentences) <= max_sentences:
        return text.strip()

    short_text = ". ".join(sentences[:max_sentences]) + "."
    return short_text.strip()


def format_for_avatar(state: Dict[str, Any]) -> Dict[str, Any]:
    """Format chatbot output for Suno Bark TTS and 3D avatar sync."""

    text = compact_response(state["response"])
    emotion = analyze_emotion(text)
    meta = EMOTION_TO_AVATAR.get(emotion, EMOTION_TO_AVATAR["neutral"])

    # Add Bark-style emotional cues
    tone_tags = {
        "happy": "(gentle happy tone)",
        "sad": "(sad tone)",
        "empathetic": "(soft reassuring tone)",
        "concerned": "(concerned tone)",
        "neutral": "(calm neutral tone)"
    }

    # Add expressive pauses and ellipses for realism
    text = re.sub(r"([.?!])(\s|$)", r"\1... ", text)  # turn . into ...
    text = text.strip().replace("..", ".")  # clean over-ellipses

    # Insert emotion tag at start
    bark_script = f"{tone_tags.get(emotion, '(neutral tone)')} {text}"

    # Estimate duration
    duration = estimate_speech_duration(text)

    return {
        "speech_script": bark_script.strip(),
        "expression": meta["expression"],
        "animation": meta["animation"],
        "emotion": emotion,
        "duration": duration
    }
