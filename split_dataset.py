import streamlit as st
import requests
import pandas as pd
from PIL import Image


API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(

    page_title="AI Food Nutrition",

    page_icon="🍱",

    layout="wide"

)


# ==========================================
# Language
# ==========================================

lang = st.selectbox(

    "Language / 언어",

    ["한국어", "English"]

)


if lang == "한국어":

    TXT = {

        "title":"🍱 AI 음식 영양 분석 서비스",

        "desc":"음식 사진을 업로드하면 음식 종류, 무게, 칼로리를 분석합니다.",

        "upload":"음식 사진 업로드",

        "origin":"원본 이미지",

        "result":"분석 결과 이미지",

        "analyze":"🔍 영양 분석 시작",

        "loading":"AI 분석중...",

        "food_count":"음식 개수",

        "weight":"총 무게",

        "kcal":"총 칼로리",

        "nutrition":"영양 정보",

        "json":"JSON 결과 보기"

    }

else:

    TXT = {

        "title":"🍱 AI Food Nutrition Analysis",

        "desc":"Upload a food image and analyze food type, weight and calories.",

        "upload":"Upload Food Image",

        "origin":"Original Image",

        "result":"Prediction Result",

        "analyze":"🔍 Analyze",

        "loading":"Analyzing...",

        "food_count":"Food Count",

        "weight":"Total Weight",

        "kcal":"Total Calories",

        "nutrition":"Nutrition Information",

        "json":"View JSON"

    }



# ==========================================
# Title
# ==========================================

st.title(

    TXT["title"]

)

st.write(

    TXT["desc"]

)


# ==========================================
# Upload
# ==========================================

uploaded_file = st.file_uploader(

    TXT["upload"],

    type=["jpg","jpeg","png"]

)


# ==========================================
# Main
# ==========================================

if uploaded_file:


    col1,col2 = st.columns(2)


    with col1:

        st.subheader(

            TXT["origin"]

        )


        image = Image.open(

            uploaded_file

        )


        st.image(

            image,

            width=500

        )



    if st.button(

        TXT["analyze"]

    ):


        files = {

            "file": (

                uploaded_file.name,

                uploaded_file.getvalue(),

                uploaded_file.type

            )

        }



        with st.spinner(

            TXT["loading"]

        ):


            response = requests.post(

                API_URL,

                files=files

            )



        result = response.json()


        foods = result["foods"]



        with col2:


            st.subheader(

                TXT["result"]

            )


            st.image(

                "result.jpg",

                width=500

            )



        st.divider()



        total_weight = sum(

            x["weight_g"]

            for x in foods

        )


        total_kcal = sum(

            x["kcal"]

            for x in foods

        )



        c1,c2,c3 = st.columns(3)


        c1.metric(

            TXT["food_count"],

            len(foods)

        )


        c2.metric(

            TXT["weight"],

            f"{total_weight:.1f} g"

        )


        c3.metric(

            TXT["kcal"],

            f"{total_kcal:.1f} kcal"

        )



        st.divider()



        table = []


        for food in foods:


            table.append({

                "Food":food["food"],

                "Weight(g)":food["weight_g"],

                "Kcal":food["kcal"],

                "Protein(g)":food["protein_g"],

                "Fat(g)":food["fat_g"],

                "Carb(g)":food["carb_g"]

            })



        df = pd.DataFrame(

            table

        )



        st.subheader(

            TXT["nutrition"]

        )


        st.dataframe(

            df,

            use_container_width=True

        )



        with st.expander(

            TXT["json"]

        ):


            st.json(

                result

            )