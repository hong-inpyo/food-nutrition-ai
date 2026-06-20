import gradio as gr

from pipeline import predict_food_nutrition


def predict(img):

    import tempfile
    import cv2

    tmp = tempfile.NamedTemporaryFile(

        suffix=".jpg",

        delete=False

    )

    cv2.imwrite(tmp.name, img[:,:,::-1])


    result = predict_food_nutrition(

        tmp.name

    )


    text = ""


    for food in result:

        text += f"""
음식 : {food['food']}

무게 : {food['weight_g']} g

칼로리 : {food['kcal']} kcal

단백질 : {food['protein_g']} g

지방 : {food['fat_g']} g

탄수화물 : {food['carb_g']} g

----------------------

"""


    return text



demo = gr.Interface(

    fn=predict,

    inputs=gr.Image(type="numpy"),

    outputs="text",

    title="Food Nutrition Estimator"

)



if __name__=="__main__":

    demo.launch()