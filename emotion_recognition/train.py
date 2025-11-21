# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import MiniXception


def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder("DATASET/train", transform=transform)
    test_dataset = datasets.ImageFolder("DATASET/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model = MiniXception(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 25

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

        test_bar = tqdm(test_loader, desc="Testing", leave=False)
        with torch.no_grad():
            for images, labels in test_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Test Accuracy: {acc:.2f}%")

        # Save checkpoint
        torch.save(model.state_dict(), "emotion_model.pth")
        print("Saved checkpoint: emotion_model.pth")


if __name__ == "__main__":
    train_model()
