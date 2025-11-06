"""
mode_classifier.py
------------------
Lightweight local intent classifier for mental health chatbot.
Replaces the Mistral-based mode selector with a fast embedding-based NLP classifier.

Modes:
    - diagnosis   → user describing symptoms, feelings, or disorders
    - counselling → user asking for emotional support or coping help
    - wellness    → user discussing general mental health, lifestyle, prevention

Usage:
    from mode_classifier import classify_mode
    mode, confidence = classify_mode(user_text)
"""

from sentence_transformers import SentenceTransformer, util
import torch

# Load once (fast and small)
_model = SentenceTransformer("all-MiniLM-L6-v2")

# Example phrases per category — add or tune based on your data
_EXAMPLES = {
    "diagnosis": [
        "I feel sad and empty",
        "I have anxiety and panic attacks",
        "I can't sleep at night",
        "I think I might be depressed",
        "I'm losing interest in things",
        "I feel worthless and tired",
        "I can't focus and feel numb"
    ],
    "counselling": [
        "How can I cope with this?",
        "I need help managing my emotions",
        "What should I do when I panic?",
        "How do I handle stress?",
        "Give me advice to feel better",
        "I want to talk about my problems",
        "How do I calm myself down?"
    ],
    "wellness": [
        "How do I stay mentally healthy?",
        "Give me tips for better sleep",
        "How to focus better at work",
        "Daily self care routine",
        "How to prevent burnout",
        "Improving concentration and motivation",
        "Habits for positive mindset"
    ],
}

# Pre-compute example embeddings
with torch.no_grad():
    _EXAMPLE_EMBS = {k: _model.encode(v, convert_to_tensor=True, normalize_embeddings=True)
                     for k, v in _EXAMPLES.items()}



def classify_mode(user_text: str, return_confidence: bool = True):
    """
    Classify the user's intent (mode) locally.
    Returns (mode, confidence) where confidence is cosine similarity [0–1].
    """

    if not user_text or not user_text.strip():
        return "wellness", 0.0

    query_emb = _model.encode(user_text, convert_to_tensor=True, normalize_embeddings=True)

    scores = {}
    for label, embs in _EXAMPLE_EMBS.items():
        sim = util.cos_sim(query_emb, embs).max().item()
        scores[label] = sim

    mode = max(scores, key=scores.get)
    confidence = round(scores[mode], 3)

    if return_confidence:
        return mode, confidence
    return mode


def classify_mode_with_context(user_text: str, history_text: str = ""):
    """
    Optional: consider recent conversation context.
    Concatenates recent user turns to refine classification.
    """

    combined = f"{history_text} {user_text}".strip()
    return classify_mode(combined)
