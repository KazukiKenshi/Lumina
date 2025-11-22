"""Realtime facial expression detection using MiniXceptionResNet and webcam.

Requirements (install if missing):
    pip install opencv-python torch torchvision

Optional: For better face localization place a Haar cascade file in the same folder
or use OpenCV's default: cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

Exit the window with 'q' or ESC.
"""

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import MiniXceptionResNet
from collections import deque

emotion_labels = [
    "Happiness",
    "Neutral",
    "Sadness"
]

# Transformation consistent with training
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_model(model_path="best_emotion_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniXceptionResNet(num_classes=3).to(device)
    # Warm-up forward to instantiate any lazy layers (e.g. dynamic fc1)
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 64, 64, device=device)
        _ = model(dummy)
    state = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading state_dict: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading state_dict (ignored): {unexpected}")
    model.eval()
    return model, device

def init_face_detector():
    # Try default Haar cascade shipped with OpenCV
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("[WARN] Haar cascade not found; falling back to full-frame inference.")
        return None
    return face_cascade

def select_face(frame, face_cascade):
    if face_cascade is None:
        return frame  # use entire frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return frame
    # Choose largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # Expand a little margin
    pad = int(0.15 * w)
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, frame.shape[1])
    y1 = min(y + h + pad, frame.shape[0])
    return frame[y0:y1, x0:x1]

def smooth_probabilities(prob, buffer, max_len=8):
    buffer.append(prob)
    # Average smoothing
    avg = torch.stack(list(buffer), dim=0).mean(dim=0)
    return avg

def main():
    model, device = load_model()
    face_cascade = init_face_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    prob_buffer = deque(maxlen=8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    print("[INFO] Press 'q' or ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed.")
            break

        face_region = select_face(frame, face_cascade)
        # Convert face_region to PIL format via OpenCV -> RGB
        rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        try:
            from PIL import Image
            pil_img = Image.fromarray(rgb)
        except Exception as e:
            print("[ERROR] PIL conversion failed:", e)
            break

        tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            prob = F.softmax(logits, dim=1).squeeze(0)
        smooth_prob = smooth_probabilities(prob, prob_buffer)
        pred_idx = int(torch.argmax(smooth_prob).item())
        pred_label = emotion_labels[pred_idx] if pred_idx < len(emotion_labels) else "Unknown"
        confidence = float(smooth_prob[pred_idx].item())

        # Overlay
        # Show only the most likely emotion label (no percentage)
        text = f"{pred_label}"
        cv2.putText(frame, text, (10, 30), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw face rectangle if detection is active
        if face_cascade is not None and face_region is not frame:
            # Re-run detection to get coordinates for display (optimisation omitted for clarity)
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_full = face_cascade.detectMultiScale(gray_full, 1.1, 5, minSize=(60, 60))
            if len(faces_full):
                x, y, w, h = max(faces_full, key=lambda f: f[2]*f[3])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Realtime Emotion', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()