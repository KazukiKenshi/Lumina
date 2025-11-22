# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import Counter
try:
    from sklearn.metrics import f1_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from model import MiniXceptionResNet


def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_train = datasets.ImageFolder("DATASET/train", transform=transform)
    full_test = datasets.ImageFolder("DATASET/test", transform=transform)

    # Original index order assumed: 0:Surprise 1:Fear 2:Disgust 3:Happiness 4:Sadness 5:Anger 6:Neutral
    # Map selected to 3-class indices: Happiness->0, Neutral->1, Sadness->2
    selected_map = {3: 0, 6: 1, 4: 2}

    def filter_dataset(ds):
        imgs = []
        targets = []
        for path, label in ds.samples:
            if label in selected_map:
                imgs.append(path)
                targets.append(selected_map[label])
        ds.samples = list(zip(imgs, targets))
        ds.targets = targets
        ds.classes = ["Happiness", "Neutral", "Sadness"]
        ds.class_to_idx = {c: i for i, c in enumerate(ds.classes)}
        return ds

    train_dataset = filter_dataset(full_train)
    test_dataset = filter_dataset(full_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model = MiniXceptionResNet(num_classes=3).to(device)

    # Compute class weights (inverse frequency) for focal loss alpha
    counts = Counter(train_dataset.targets)  # indices: 0:Happiness 1:Neutral 2:Sadness
    inv_freq = {cls: 1.0 / count for cls, count in counts.items() if count > 0}
    min_w = min(inv_freq.values())
    weights = [inv_freq[i] / min_w for i in range(len(train_dataset.classes))]
    alpha = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"Class counts: {counts} -> Alpha (normalized inverse freq): {weights}")

    # Focal Loss definition
    class FocalLoss(nn.Module):
        def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
            super().__init__()
            self.alpha = alpha  # tensor of shape [C]
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, logits, targets):
            log_probs = F.log_softmax(logits, dim=1)          # [B, C]
            probs = log_probs.exp()                           # pt
            # Gather log_pt for the true class
            targets = targets.view(-1, 1)
            log_pt = log_probs.gather(1, targets).squeeze(1)  # [B]
            pt = log_pt.exp()                                 # [B]
            # Base CE loss: -log_pt
            ce_loss = -log_pt                                 # [B]
            # Focal term
            focal_term = (1 - pt) ** self.gamma               # [B]
            # Alpha weighting per example
            if self.alpha is not None:
                alpha_weight = self.alpha.gather(0, targets.squeeze(1))  # [B]
                loss = alpha_weight * focal_term * ce_loss
            else:
                loss = focal_term * ce_loss
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            return loss

    criterion = FocalLoss(alpha=alpha, gamma=2.0, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100
    best_acc = 0.0
    best_f1 = 0.0
    # Early stopping parameters (macro F1 based)
    patience = 5          # epochs to wait for significant improvement
    min_delta = 0.01      # required improvement in macro F1
    epochs_without_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        print(f"\nEpoch {epoch+1}/{epochs}")
        train_bar = tqdm(train_loader, desc="Training", leave=False)

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix({"loss": loss.item()})

        avg_loss = train_loss / len(train_loader)
        print(f"Average Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        val_loss_total = 0.0
        all_preds = []
        all_labels = []

        test_bar = tqdm(test_loader, desc="Testing", leave=False)
        with torch.no_grad():
            for images, labels in test_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss_val = criterion(outputs, labels)
                val_loss_total += loss_val.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        acc = 100 * correct / total
        val_loss_avg = val_loss_total / max(len(test_loader), 1)
        if SKLEARN_AVAILABLE and len(set(all_labels)) > 1:
            macro_f1 = f1_score(all_labels, all_preds, average='macro')
        else:
            macro_f1 = acc / 100.0  # fallback approximation
        print(f"Validation Loss: {val_loss_avg:.4f} | Accuracy: {acc:.2f}% | Macro F1: {macro_f1:.4f}")
        if SKLEARN_AVAILABLE:
            print("Per-class report (Happiness / Neutral / Sadness):\n" + classification_report(all_labels, all_preds, digits=3, target_names=train_dataset.classes))

        # Save best model so far (macro F1 primary criterion)
        if macro_f1 > best_f1 + min_delta:
            best_f1 = macro_f1
            best_acc = acc if acc > best_acc else best_acc
            torch.save(model.state_dict(), "best_emotion_model.pth")
            epochs_without_improve = 0
            print(f"New best model saved: best_emotion_model.pth (Macro F1: {best_f1:.4f}, Acc: {acc:.2f}%)")
        else:
            epochs_without_improve += 1
            print(f"No significant F1 improvement (> {min_delta}). Stagnation epochs: {epochs_without_improve}/{patience}. Best F1: {best_f1:.4f}")

        # Early stopping check
        if epochs_without_improve >= patience:
            print(f"Early stopping triggered after {patience} stagnant epochs. Best Macro F1: {best_f1:.4f}, Best Acc: {best_acc:.2f}%")
            break


if __name__ == "__main__":
    train_model()
