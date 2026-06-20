import os
import urllib.request

os.makedirs("models", exist_ok=True)

url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
save_path = "models/depth_anything_v2_vits.pth"

print("다운로드 시작...")

urllib.request.urlretrieve(url, save_path)

print("다운로드 완료!")
print(save_path)