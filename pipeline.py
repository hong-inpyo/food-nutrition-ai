import sys
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import pandas as pd
from llm_feedback import generate_feedback


def load_correction():

    df = pd.read_csv("food_calibration.csv")

    correction = {}

    foods = df["food"].unique()

    for food in foods:

        sub = df[df["food"]==food]

        real_mean = sub["real_weight"].mean()

        pred_mean = sub["pred_weight"].mean()

        correction[food] = round(
            real_mean / pred_mean,
            3
        )

    return correction


AUTO_CORRECTION = load_correction()





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

yolo = YOLO("old_best.pt")


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

"rice":1.70,
"fried_rice":0.95,

"ramen_noodle":0.55,
"udon_noodle":0.55,
"pasta_spaghetti":0.65,
"noodles":0.60,

"bread":0.30,
"sandwich":0.45,
"pizza":0.55,
"hamburger":0.65,
"french_fries":0.45,

"chicken":1.0,
"beef_steak":1.00,
"pork":1.0,
"sausage":0.92,
"fried_chicken":0.85,

"sushi":0.80,
"fish":1.00,
"shrimp":1.05,
"crab":0.95,

"salad":0.25,

"potato":0.75,
"cabbage":0.35,
"broccoli":0.40,
"mushroom":0.50,
"corn":0.72,
"tomato":0.94,
"eggplant":0.60,

"tofu":0.90,
"egg":1.03,

"soup":1.0,
"curry":1.0,
"stew":1.0,

"cake":0.55,
"ice_cream":0.52,
"cookie":0.60,
"doughnut":0.45,
"chocolate":1.10,
"pudding":1.0,
"waffle_pancake":0.50,

"apple":0.80,
"banana":0.94,
"strawberry":0.60,
"orange":0.85,
"watermelon":0.96,

"coffee":1.0,
"juice":1.0,

"dumpling":0.80,
"popcorn":0.10,
"mochi":1.0
}

# ==========================================
# Shape Factor
# ==========================================

SHAPE = {

"rice":0.70,
"fried_rice":0.75,

"ramen_noodle":0.55,
"udon_noodle":0.55,
"pasta_spaghetti":0.60,
"noodles":0.60,

"bread":0.50,
"sandwich":0.65,
"pizza":0.25,
"hamburger":0.55,
"french_fries":0.45,

"chicken":0.70,
"beef_steak":0.70,
"pork":0.80,
"sausage":0.75,
"fried_chicken":0.75,

"sushi":0.40,
"fish":0.65,
"shrimp":0.55,
"crab":0.60,

"salad":0.25,

"potato":0.70,
"cabbage":0.35,
"broccoli":0.40,
"mushroom":0.50,
"corn":0.65,
"tomato":0.65,
"eggplant":0.60,

"tofu":0.80,
"egg":0.75,

"soup":0.70,
"curry":0.70,
"stew":0.70,

"cake":0.60,
"ice_cream":0.50,
"cookie":0.35,
"doughnut":0.45,
"chocolate":0.50,
"pudding":0.80,
"waffle_pancake":0.60,

"apple":0.70,
"banana":0.65,
"strawberry":0.50,
"orange":0.70,
"watermelon":0.80,

"coffee":1.0,
"juice":1.0,

"dumpling":0.65,
"popcorn":0.20,
"mochi":0.70
}

# ==========================================
# 보정
# ==========================================

CORRECTION = {

"rice":2.0,
"fried_rice":1.7,

"ramen_noodle":1.5,
"udon_noodle":1.5,
"pasta_spaghetti":1.4,
"noodles":1.5,

"bread":1.0,
"sandwich":1.1,
"pizza":1.2,
"hamburger":1.1,
"french_fries":1.2,

"chicken":1.2,
"beef_steak":1.1,
"pork":1.2,
"sausage":1.0,
"fried_chicken":1.3,

"sushi":1.2,
"fish":1.1,
"shrimp":0.8,
"crab":0.9,

"salad":1.5,

"potato":1.1,
"cabbage":1.2,
"broccoli":1.2,
"mushroom":1.0,
"corn":1.0,
"tomato":1.0,
"eggplant":1.0,

"tofu":1.0,
"egg":1.0,

"soup":2.0,
"curry":1.7,
"stew":1.8,

"cake":1.0,
"ice_cream":1.0,
"cookie":1.0,
"doughnut":1.0,
"chocolate":1.0,
"pudding":1.0,
"waffle_pancake":1.0,

"apple":1.0,
"banana":1.0,
"strawberry":1.0,
"orange":1.0,
"watermelon":1.0,

"coffee":1.0,
"juice":1.0,

"dumpling":1.1,
"popcorn":1.0,
"mochi":1.0
}

# ==========================================
# Nutrition DB (100g)
# ==========================================

NUTRITION = {

"rice":{"kcal":130,"protein":2.7,"fat":0.3,"carb":28},
"fried_rice":{"kcal":180,"protein":4.0,"fat":6.0,"carb":28},
"ramen_noodle":{"kcal":436,"protein":10,"fat":17,"carb":60},
"udon_noodle":{"kcal":127,"protein":3.5,"fat":0.5,"carb":25},
"pasta_spaghetti":{"kcal":158,"protein":5.8,"fat":0.9,"carb":31},
"noodles":{"kcal":138,"protein":4.5,"fat":1.5,"carb":27},

"bread":{"kcal":265,"protein":9,"fat":3.2,"carb":49},
"sandwich":{"kcal":250,"protein":11,"fat":8,"carb":33},
"pizza":{"kcal":266,"protein":11,"fat":10,"carb":33},
"hamburger":{"kcal":295,"protein":17,"fat":14,"carb":30},
"french_fries":{"kcal":312,"protein":3.4,"fat":15,"carb":41},

"chicken":{"kcal":239,"protein":27,"fat":14,"carb":0},
"beef_steak":{"kcal":271,"protein":25,"fat":19,"carb":0},
"pork":{"kcal":242,"protein":27,"fat":14,"carb":0},
"sausage":{"kcal":301,"protein":12,"fat":27,"carb":2},
"fried_chicken":{"kcal":320,"protein":20,"fat":22,"carb":11},

"sushi":{"kcal":143,"protein":6,"fat":0.5,"carb":30},
"fish":{"kcal":206,"protein":22,"fat":12,"carb":0},
"shrimp":{"kcal":99,"protein":24,"fat":0.3,"carb":0},
"crab":{"kcal":97,"protein":19,"fat":1.5,"carb":0},

"salad":{"kcal":25,"protein":1.5,"fat":0.2,"carb":4},
"potato":{"kcal":77,"protein":2,"fat":0.1,"carb":17},
"cabbage":{"kcal":25,"protein":1.3,"fat":0.1,"carb":6},
"broccoli":{"kcal":34,"protein":2.8,"fat":0.4,"carb":7},
"mushroom":{"kcal":22,"protein":3.1,"fat":0.3,"carb":3},
"corn":{"kcal":96,"protein":3.4,"fat":1.5,"carb":21},
"tomato":{"kcal":18,"protein":0.9,"fat":0.2,"carb":3.9},
"eggplant":{"kcal":25,"protein":1,"fat":0.2,"carb":6},

"tofu":{"kcal":76,"protein":8,"fat":4.8,"carb":1.9},
"egg":{"kcal":155,"protein":13,"fat":11,"carb":1},

"soup":{"kcal":40,"protein":2,"fat":1.5,"carb":5},
"curry":{"kcal":120,"protein":3,"fat":6,"carb":13},
"stew":{"kcal":90,"protein":6,"fat":4,"carb":7},

"cake":{"kcal":257,"protein":3.6,"fat":10,"carb":38},
"ice_cream":{"kcal":207,"protein":3.5,"fat":11,"carb":24},
"cookie":{"kcal":502,"protein":6,"fat":24,"carb":64},
"doughnut":{"kcal":452,"protein":4.9,"fat":25,"carb":51},
"chocolate":{"kcal":546,"protein":4.9,"fat":31,"carb":61},
"pudding":{"kcal":120,"protein":3,"fat":3,"carb":20},
"waffle_pancake":{"kcal":291,"protein":8,"fat":10,"carb":42},

"apple":{"kcal":52,"protein":0.3,"fat":0.2,"carb":14},
"banana":{"kcal":89,"protein":1.1,"fat":0.3,"carb":23},
"strawberry":{"kcal":32,"protein":0.7,"fat":0.3,"carb":7.7},
"orange":{"kcal":47,"protein":0.9,"fat":0.1,"carb":12},
"watermelon":{"kcal":30,"protein":0.6,"fat":0.2,"carb":8},

"coffee":{"kcal":2,"protein":0,"fat":0,"carb":0},
"juice":{"kcal":45,"protein":0.5,"fat":0.1,"carb":11},

"dumpling":{"kcal":220,"protein":8,"fat":7,"carb":30},
"popcorn":{"kcal":375,"protein":11,"fat":4,"carb":74},
"mochi":{"kcal":235,"protein":4,"fat":0.5,"carb":50}

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

        return {

            "foods": [],

            "summary": {

                "kcal":0,

                "protein":0,

                "fat":0,

                "carb":0

            },

            "feedback":"음식을 인식하지 못했습니다."

        }


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


        correction = AUTO_CORRECTION.get(
            food_name,
            1.0
        )

        weight = (

            volume *

            density *

            SCALE *

            correction

        )
        if food_name=="rice":
            weight=max(weight,180)
        
        if food_name=="soup":
            weight=max(weight,70)

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

        food["median_depth"] = round(food["median_depth"],3)
        food["volume"] = round(food["volume"],2)

        food["weight_g"] = round(food["weight_g"],1)

        food["kcal"] = round(food["kcal"],1)

        food["protein_g"] = round(food["protein_g"],1)

        food["fat_g"] = round(food["fat_g"],1)

        food["carb_g"] = round(food["carb_g"],1)

        result.append(food)


    # ==========================
    # Nutrition Summary
    # ==========================

    total_kcal = sum(x["kcal"] for x in result)

    total_protein = sum(x["protein_g"] for x in result)

    total_fat = sum(x["fat_g"] for x in result)

    total_carb = sum(x["carb_g"] for x in result)

    food_names = [x["food"] for x in result]


    # ==========================
    # AI Feedback
    # ==========================

    feedback = generate_feedback(

        total_kcal,

        total_protein,

        total_fat,

        total_carb,

        food_names

    )

    # ==========================
    # Save Result Image
    # ==========================

    cv2.imwrite(

        "result.jpg",

        img

    )


    print("Saved : result.jpg")


    return {

        "foods": result,

        "summary": {

            "kcal": round(total_kcal,1),

            "protein": round(total_protein,1),

            "fat": round(total_fat,1),

            "carb": round(total_carb,1)

        },

        "feedback": feedback

    }