# extract_food_regions.py

from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO("best.pt")

img_path = r"C:\Users\hjp36\OneDrive\바탕 화면\KakaoTalk_20260601_191726293.jpg"

img = cv2.imread(img_path)

results = model(img)

r = results[0]

os.makedirs("food_crops", exist_ok=True)

food_infos = []

if r.masks is not None:

    for idx, mask in enumerate(r.masks.data):

        # --------------------
        # 클래스명
        # --------------------
        cls_id = int(r.boxes.cls[idx])

        food_name = model.names[cls_id]

        # --------------------
        # 마스크
        # --------------------
        mask_np = mask.cpu().numpy()

        mask_np = cv2.resize(
            mask_np,
            (img.shape[1], img.shape[0])
        )

        binary_mask = (mask_np > 0.5).astype(np.uint8)

        # --------------------
        # 면적
        # --------------------
        pixel_area = int(binary_mask.sum())

        # --------------------
        # 음식만 남기기
        # --------------------
        food = img.copy()

        food[binary_mask == 0] = 0

        save_path = f"food_crops/food_{idx}.png"

        cv2.imwrite(
            save_path,
            food
        )

        food_infos.append({
            "name": food_name,
            "area": pixel_area,
            "path": save_path
        })

# --------------------
# 결과 출력
# --------------------

print("\n검출 결과")

for food in food_infos:

    print(
        f"{food['name']:15s}"
        f" area={food['area']}"
    )

print(type(r.masks))
print(r.masks.data.shape)