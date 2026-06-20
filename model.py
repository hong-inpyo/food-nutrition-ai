

from datasets import load_dataset
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import torch.nn as nn

# Food101 로드
dataset = load_dataset("food101")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

class FoodDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item["image"].convert("RGB")
        image = transform(image)

        label = item["label"]

        return image, label

train_dataset = FoodDataset(dataset["train"])
test_dataset = FoodDataset(dataset["validation"])

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=101
)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

for epoch in range(5):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1} Loss {total_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    "swin_food101.pth"
)

print("학습 완료")