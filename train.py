import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

from model import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Config
DATA_DIR = "dataset"
BATCH_SIZE = 16
EPOCHS = 3
LR = 5e-5

# Transforms (updated augmentation)
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)

# Dataset
train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
val_dataset = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_transform)

print("Classes:", train_dataset.classes)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
model = get_model(num_classes=len(train_dataset.classes)).to(DEVICE)

# Load pre-trained weights
model.load_state_dict(torch.load("model.pth"))
print("Loaded previous trained model for fine-tuning")

# layer freezing
for param in model.parameters():
    param.requires_grad = False

# unfereezing
for param in model.head.parameters():
    param.requires_grad = True

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Calculate average loss
    avg_loss = total_loss / len(train_loader)

    # Validation phase
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total if total > 0 else 0

    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")

# Save model - learned weights, bias vals, all params model trained blah blah (avoids training each time iff already...)
torch.save(model.state_dict(), "model.pth")

print("Training complete. Model saved.")
