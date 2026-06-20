import json
import torch
import timm

from PIL import Image
from torchvision import transforms

# -------------------------
# 클래스 로드
# -------------------------

with open("foodseg_classes.json", "r") as f:
    classes = json.load(f)

# -------------------------
# 모델
# -------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=103
)

checkpoint = torch.load(
    "best_swin_foodseg103.pth",
    map_location=device
)

# 저장 형식 대응
if "model" in checkpoint:
    model.load_state_dict(
        checkpoint["model"]
    )
else:
    model.load_state_dict(
        checkpoint
    )

model.to(device)
model.eval()

# -------------------------
# transform
# -------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# -------------------------
# predict
# -------------------------

def predict_food(image_path):

    img = Image.open(
        image_path
    ).convert("RGB")

    x = transform(img).unsqueeze(0)

    x = x.to(device)

    with torch.no_grad():

        out = model(x)

        prob = torch.softmax(
            out,
            dim=1
        )

        score, pred = prob.max(1)

    return (
        classes[pred.item()],
        score.item()
    )