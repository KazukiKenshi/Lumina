import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MiniXceptionResNet


def evaluate(model_path: str, data_dir: str, batch_size: int = 64, num_workers: int = 2, per_class: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = MiniXceptionResNet(num_classes=7).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    total = 0
    correct = 0

    per_class_total = None
    per_class_correct = None

    if per_class:
        n_classes = len(dataset.classes)
        per_class_total = [0 for _ in range(n_classes)]
        per_class_correct = [0 for _ in range(n_classes)]

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            if per_class:
                for p, t in zip(preds.tolist(), labels.tolist()):
                    per_class_total[t] += 1
                    if p == t:
                        per_class_correct[t] += 1

    acc = 100.0 * correct / max(total, 1)
    print(f"Overall accuracy: {acc:.2f}% ({correct}/{total})")

    if per_class:
        print("Per-class accuracy:")
        for idx, name in enumerate(dataset.classes):
            t = per_class_total[idx]
            c = per_class_correct[idx]
            a = 100.0 * c / max(t, 1)
            print(f"  {name:>10s}: {a:.2f}% ({c}/{t})")


def main():
    parser = argparse.ArgumentParser(description="Validate MiniXceptionResNet on test dataset")
    parser.add_argument("--model-path", default="emotion_model.pth", help="Path to trained model .pth file")

    default_test_dir = os.path.join(os.path.dirname(__file__), "DATASET", "test")
    parser.add_argument("--data-dir", default=default_test_dir, help="Path to test dataset root (ImageFolder)")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--per-class", action="store_true", help="Show per-class accuracy breakdown")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        per_class=args.per_class,
    )


if __name__ == "__main__":
    main()
