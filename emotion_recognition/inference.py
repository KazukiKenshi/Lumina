# inference.py
import torch
from PIL import Image
from torchvision import transforms
from model import MiniXception

emotion_labels = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happiness",
    "Sadness",
    "Anger",
    "Neutral"
]

def predict(image_path, model_path="emotion_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    model = MiniXception(num_classes=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(img)
        pred = torch.argmax(logits, dim=1).item()

    return emotion_labels[pred]


if __name__ == "__main__":
    print(predict("test_face.jpg"))
