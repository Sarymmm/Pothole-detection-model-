import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ── Transforms ────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Load full dataset from dataset/ folder ────────────
# ImageFolder expects subfolders as class names:
# dataset/
#   normal/   → class 0
#   pothole/  → class 1

full_dataset = datasets.ImageFolder("dataset", transform=train_transforms)

print("Classes found:", full_dataset.classes)
print("Total images :", len(full_dataset))

# ── Split 80% train / 20% val ─────────────────────────
val_size   = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size

train_data, val_data = random_split(full_dataset, [train_size, val_size])

# Apply separate val transforms to val split
val_data.dataset = datasets.ImageFolder("dataset", transform=val_transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False, num_workers=0)

print(f"Train: {train_size} images | Val: {val_size} images\n")

# ── Model (ResNet18 pretrained) ────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512, 2)
model = model.to(device)

# ── Training ──────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

EPOCHS = 10
best_acc = 0.0

for epoch in range(EPOCHS):
    # Train
    model.train()
    total_loss = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % 20 == 0:
            print(f"  Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    # Validate
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

    acc = 100 * correct / total
    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.3f} | Val Accuracy: {acc:.1f}%")

    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "training/model.pth")
        print(f"  >> Best model saved (acc: {acc:.1f}%)\n")

print(f"\nTraining complete. Best accuracy: {best_acc:.1f}%")
print("Model saved to training/model.pth")