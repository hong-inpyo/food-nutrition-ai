import os
import cv2
import numpy as np
from tqdm import tqdm

ROOT = "datasets/foodseg103/FoodSeg103/Images"

IMG_TRAIN = os.path.join(ROOT, "img_dir/train")
MASK_TRAIN = os.path.join(ROOT, "ann_dir/train")

OUT_IMG = "datasets/foodseg_yolo/images/train"
OUT_LABEL = "datasets/foodseg_yolo/labels/train"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LABEL, exist_ok=True)


def mask_to_yolo(mask_path, txt_path):

    mask = cv2.imread(mask_path, 0)

    H, W = mask.shape

    lines = []

    for cls in np.unique(mask):

        if cls == 0:
            continue

        binary = np.uint8(mask == cls)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:

            if len(contour) < 3:
                continue

            contour = contour.squeeze()

            if contour.ndim != 2:
                continue

            points = []

            for x, y in contour:

                points.extend([
                    x / W,
                    y / H
                ])

            line = f"{cls-1} " + " ".join(
                f"{p:.6f}" for p in points
            )

            lines.append(line)

    with open(txt_path, "w") as f:

        f.write("\n".join(lines))


images = os.listdir(IMG_TRAIN)

for img_name in tqdm(images):

    base = os.path.splitext(img_name)[0]

    img_path = os.path.join(
        IMG_TRAIN,
        img_name
    )

    mask_path = os.path.join(
        MASK_TRAIN,
        base + ".png"
    )

    if not os.path.exists(mask_path):
        continue

    txt_path = os.path.join(
        OUT_LABEL,
        base + ".txt"
    )

    mask_to_yolo(
        mask_path,
        txt_path
    )

    cv2.imwrite(
        os.path.join(
            OUT_IMG,
            img_name
        ),
        cv2.imread(img_path)
    )

print("변환 완료")