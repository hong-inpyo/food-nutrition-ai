import sys
import torch

sys.path.append("./Depth-Anything-V2")

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device :", DEVICE)

model_configs = {
    'vits': {
        'encoder': 'vits',
        'features': 64,
        'out_channels': [48,96,192,384]
    }
}

model = DepthAnythingV2(**model_configs['vits'])

checkpoint = torch.load(
    "models/depth_anything_v2_vits.pth",
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

print("Depth Anything Loaded Successfully!")