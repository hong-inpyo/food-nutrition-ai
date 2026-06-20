import torch
import torch.nn as nn
import timm

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# =====================
# CONFIG
# =====================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4

print("Device:", DEVICE)

# =====================
# TRANSFORM
# =====================

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =====================
# DATASET
# =====================

train_ds = datasets.Food101(
    root="./data",
    split="train",
    download=True,
    transform=train_tf
)

test_ds = datasets.Food101(
    root="./data",
    split="test",
    download=True,
    transform=val_tf
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# =====================
# MODEL
# =====================

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=101
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)

best_acc = 0

# =====================
# TRAIN
# =====================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    pbar = tqdm(train_loader)

    for images, labels in pbar:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        pbar.set_description(
            f"Epoch {epoch+1}/{EPOCHS}"
        )

        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    # =====================
    # VALIDATION
    # =====================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            preds = outputs.argmax(1)

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

    acc = 100 * correct / total

    print(
        f"\nEpoch {epoch+1}"
        f" | Accuracy = {acc:.2f}%"
    )

    if acc > best_acc:

        best_acc = acc

        torch.save(
            model.state_dict(),
            "best_swin_food101.pth"
        )

        print(
            f"Best Model Saved "
            f"({best_acc:.2f}%)"
        )

print(
    f"\nBest Accuracy: "
    f"{best_acc:.2f}%"
)