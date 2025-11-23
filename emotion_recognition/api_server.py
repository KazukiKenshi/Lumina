import io
import os
import base64
from typing import Dict
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from model import MiniXceptionResNet

app = Flask(__name__)
CORS(app)

EMOTION_LABELS = ["Happiness", "Neutral", "Sadness"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = MiniXceptionResNet(num_classes=3).to(DEVICE)
model_path = "best_emotion_model.pth"
model_loaded = False
try:
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    model_loaded = True
except FileNotFoundError:
    print(f"Warning: {model_path} not found. Predictions will be unavailable.")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_from_image(img: Image.Image) -> Dict[str, float]:
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    return {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model_loaded})

@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503
    # Accept either multipart file or JSON base64
    if 'file' in request.files:
        file = request.files['file']
        try:
            img = Image.open(file.stream).convert("RGB")
        except Exception:
            return jsonify({"error": "Invalid image file"}), 400
    else:
        try:
            data = request.get_json(force=True)
            b64 = data.get('image_base64')
            if not b64:
                return jsonify({"error": "image_base64 field required"}), 400
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            return jsonify({"error": "Invalid JSON or base64 image"}), 400
    probs = predict_from_image(img)
    # Determine top emotion
    pred_label = max(probs.items(), key=lambda kv: kv[1])[0]
    print(f"{pred_label} {probs}")
    return jsonify({"emotion": pred_label, "probabilities": probs})

if __name__ == "__main__":
    # Simple dev server
    app.run(host="0.0.0.0", port=os.getenv("EMOTION_API_PORT") or 8001)
