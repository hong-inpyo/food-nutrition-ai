
from fastapi import FastAPI, UploadFile, File
import shutil
import os

from pipeline import predict_food_nutrition


app = FastAPI(
    title="Food Nutrition API"
)


UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


@app.get("/")
def home():

    return {

        "message":"Food Nutrition API Running"

    }



@app.post("/predict")
async def predict(

    file: UploadFile = File(...)

):

    save_path = os.path.join(

        UPLOAD_DIR,

        file.filename

    )


    with open(

        save_path,

        "wb"

    ) as buffer:

        shutil.copyfileobj(

            file.file,

            buffer

        )


    result = predict_food_nutrition(

        save_path

    )


    result = predict_food_nutrition(save_path)

    result["result_image"] = "result.jpg"

    return result