import sys
import cv2
import torch

sys.path.append("./Depth-Anything-V2")

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device :", DEVICE)

model_configs = {
    'vits': {
        'encoder': 'vits',
        'features': 64,
        'out_channels': [48, 96, 192, 384]
    }
}

# 모델 생성
model = DepthAnythingV2(**model_configs['vits'])

# 가중치 로드
checkpoint = torch.load(
    "models/depth_anything_v2_vits.pth",
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

print("Model Loaded!")

# 이미지 읽기
img_path = r"C:\Users\hjp36\OneDrive\바탕 화면\KakaoTalk_20260601_191726293.jpg"

img = cv2.imread(img_path)

print(img.shape)

# 깊이 추론
depth = model.infer_image(img)

print(depth.shape)
print(depth.min(), depth.max())