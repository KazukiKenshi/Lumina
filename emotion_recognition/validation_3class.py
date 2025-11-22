import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import MiniXceptionResNet
from sklearn.metrics import f1_score, classification_report

# Map original indices to 3-class indices: Happiness->0, Neutral->1, Sadness->2
selected_map = {3: 0, 6: 1, 4: 2}
selected_names = ["Happiness", "Neutral", "Sadness"]

def filter_dataset(ds):
    imgs = []
    targets = []
    for path, label in ds.samples:
        if label in selected_map:
            imgs.append(path)
            targets.append(selected_map[label])
    ds.samples = list(zip(imgs, targets))
    ds.targets = targets
    ds.classes = selected_names
    ds.class_to_idx = {c: i for i, c in enumerate(ds.classes)}
    return ds

def evaluate(model_path, data_dir, batch_size=64, num_workers=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_test = datasets.ImageFolder(data_dir, transform=transform)
    test_dataset = filter_dataset(full_test)
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    num_classes = 3
    model = MiniXceptionResNet(num_classes=num_classes).to(device)

    state = torch.load(model_path, map_location=device)
    fc_out_weight_key = 'model.fc_out.weight'
    fc_out_bias_key = 'model.fc_out.bias'
    if fc_out_weight_key in state and state[fc_out_weight_key].shape != model.model.fc_out.weight.shape:
        print(f"Skipping loading {fc_out_weight_key} due to shape mismatch.")
        del state[fc_out_weight_key]
    if fc_out_bias_key in state and state[fc_out_bias_key].shape != model.model.fc_out.bias.shape:
        print(f"Skipping loading {fc_out_bias_key} due to shape mismatch.")
        del state[fc_out_bias_key]
    model.load_state_dict(state, strict=False)
    model.eval()

    total = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100.0 * correct / max(total, 1)
    print(f"Overall accuracy: {acc:.2f}% ({correct}/{total})")

    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    print(f"Macro F1 score: {f1_macro:.4f}")
    print("Per-class F1 scores:")
    for idx, name in enumerate(selected_names):
        print(f"  {name:>10s}: {f1_per_class[idx]:.4f}")
    print("Classification report:")
    print(classification_report(all_labels, all_preds, digits=3, target_names=selected_names))

def main():
    parser = argparse.ArgumentParser(description="Validate 3-class MiniXceptionResNet on test dataset")
    parser.add_argument("--model-path", default="best_emotion_model.pth", help="Path to trained model .pth file")
    default_test_dir = os.path.join(os.path.dirname(__file__), "DATASET", "test")
    parser.add_argument("--data-dir", default=default_test_dir, help="Path to test dataset root (ImageFolder)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

if __name__ == "__main__":
    main()