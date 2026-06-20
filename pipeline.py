import sys
import cv2
import torch
import numpy as np
from ultralytics import YOLO

def draw_result(img, box, text):

    x1,y1,x2,y2 = box

    cv2.rectangle(
        img,
        (x1,y1),
        (x2,y2),
        (0,255,0),
        2
    )

    cv2.putText(
        img,
        text,
        (x1,y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )
# ==========================================
# Depth Anything V2
# ==========================================

sys.path.append("./Depth-Anything-V2")

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device :", DEVICE)

model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384]
    }
}

depth_model = DepthAnythingV2(**model_configs["vits"])

depth_model.load_state_dict(
    torch.load(
        "models/depth_anything_v2_vits.pth",
        map_location=DEVICE
    )
)

depth_model = depth_model.to(DEVICE)
depth_model.eval()

print("Depth Model Loaded")


# ==========================================
# YOLO
# ==========================================

yolo = YOLO("best.pt")


# ==========================================
# Hyper Parameter
# ==========================================

MIN_AREA = 3000

SCALE = 0.00005


# ==========================================
# Density
# g/cm³
# ==========================================

DENSITY = {

    "rice":0.70,

    "pork":0.95,

    "sausage":0.92,

    "garlic":0.60,

    "cabbage":0.35,

    "onion":0.50,

    "sauce":1.05,

    "carrot":0.64,

    "pepper":0.45,

    "potato":0.75,

    "soup":1.0
}


# ==========================================
# Shape Factor
# ==========================================

SHAPE = {

    "rice":0.70,

    "pork":0.80,

    "sausage":0.75,

    "garlic":0.45,

    "cabbage":0.35,

    "onion":0.50,

    "sauce":1.0,

    "carrot":0.65,

    "pepper":0.55,

    "potato":0.70,

    "soup":0.70
}


# ==========================================
# Nutrition DB (100g)
# ==========================================

NUTRITION = {

    "rice":{
        "kcal":130,
        "protein":2.7,
        "fat":0.3,
        "carb":28
    },

    "pork":{
        "kcal":242,
        "protein":27,
        "fat":14,
        "carb":0
    },

    "sausage":{
        "kcal":301,
        "protein":12,
        "fat":27,
        "carb":2
    },

    "garlic":{
        "kcal":149,
        "protein":6.4,
        "fat":0.5,
        "carb":33
    },

    "cabbage":{
        "kcal":25,
        "protein":1.3,
        "fat":0.1,
        "carb":6
    },

    "onion":{
        "kcal":40,
        "protein":1.1,
        "fat":0.1,
        "carb":9
    },

    "sauce":{
        "kcal":120,
        "protein":1,
        "fat":2,
        "carb":25
    },

    "carrot":{
        "kcal":41,
        "protein":0.9,
        "fat":0.2,
        "carb":10
    },

    "pepper":{
        "kcal":31,
        "protein":1,
        "fat":0.3,
        "carb":6
    },

    "potato":{
        "kcal":77,
        "protein":2,
        "fat":0.1,
        "carb":17
    },

    "soup":{
        "kcal":35,
        "protein":1.5,
        "fat":1,
        "carb":5
    }
}


# ==========================================
# Main Function
# ==========================================

def predict_food_nutrition(img_path):

    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Cannot read image")


    results = yolo(img)

    r = results[0]

    if r.masks is None:
        return []


    depth = depth_model.infer_image(img)


    merged = {}


    for idx, mask in enumerate(r.masks.data):

        box = r.boxes.xyxy[idx].cpu().numpy()

        x1, y1, x2, y2 = map(int, box)


        cls_id = int(r.boxes.cls[idx])

        food_name = yolo.names[cls_id].strip()


        mask_np = mask.cpu().numpy()

        mask_np = cv2.resize(

            mask_np,

            (img.shape[1], img.shape[0])

        )


        binary_mask = (mask_np > 0.5).astype(np.uint8)


        area = int(binary_mask.sum())


        if area < MIN_AREA:

            continue


        food_depth = depth[binary_mask > 0]


        if len(food_depth) == 0:

            continue


        median_depth = float(

            np.median(food_depth)

        )


        shape_factor = SHAPE.get(

            food_name,

            0.7

        )


        volume = (

            area *

            median_depth *

            shape_factor

        )


        density = DENSITY.get(

            food_name,

            0.7

        )


        weight = (

            volume *

            density *

            SCALE

        )


        nutrition = NUTRITION.get(

            food_name,

            {}

        )


        factor = weight / 100


        kcal = nutrition.get("kcal",0) * factor

        protein = nutrition.get("protein",0) * factor

        fat = nutrition.get("fat",0) * factor

        carb = nutrition.get("carb",0) * factor


        # ==========================
        # Draw Bounding Box
        # ==========================

        text = f"{food_name} {round(weight,1)}g"


        draw_result(

            img,

            (x1,y1,x2,y2),

            text

        )


        # ==========================
        # Merge Same Food
        # ==========================

        if food_name not in merged:


            merged[food_name] = {


                "food": food_name,

                "area": area,

                "median_depth": median_depth,

                "volume": volume,

                "weight_g": weight,

                "kcal": kcal,

                "protein_g": protein,

                "fat_g": fat,

                "carb_g": carb

            }


        else:


            merged[food_name]["area"] += area

            merged[food_name]["volume"] += volume

            merged[food_name]["weight_g"] += weight

            merged[food_name]["kcal"] += kcal

            merged[food_name]["protein_g"] += protein

            merged[food_name]["fat_g"] += fat

            merged[food_name]["carb_g"] += carb



    # ==========================
    # Final Result
    # ==========================

    result = []


    for food in merged.values():


        food["median_depth"] = round(

            food["median_depth"],

            3

        )


        food["volume"] = round(

            food["volume"],

            2

        )


        food["weight_g"] = round(

            food["weight_g"],

            1

        )


        food["kcal"] = round(

            food["kcal"],

            1

        )


        food["protein_g"] = round(

            food["protein_g"],

            1

        )


        food["fat_g"] = round(

            food["fat_g"],

            1

        )


        food["carb_g"] = round(

            food["carb_g"],

            1

        )


        result.append(food)



    # ==========================
    # Save Result Image
    # ==========================

    cv2.imwrite(

        "result.jpg",

        img

    )


    print("Saved : result.jpg")


    return result