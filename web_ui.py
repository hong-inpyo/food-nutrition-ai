import streamlit as st
import requests
import json

# 페이지 설정
st.set_page_config(
    page_title="AI 맞춤형 식단 피드백 서비스",
    page_icon="🥗",
    layout="centered"
)

# 제목
st.title("🥗 AI 기반 맞춤형 식단 피드백 서비스")
st.write("음식 사진을 업로드하고 건강 정보를 입력하면 맞춤 피드백을 제공합니다.")

# 사이드바
st.sidebar.header("👤 사용자 정보")

age = st.sidebar.number_input(
    "나이",
    min_value=1,
    max_value=120,
    value=25
)

height = st.sidebar.number_input(
    "키(cm)",
    min_value=100.0,
    max_value=250.0,
    value=175.0
)

weight = st.sidebar.number_input(
    "체중(kg)",
    min_value=20.0,
    max_value=300.0,
    value=70.0
)

disease = st.sidebar.selectbox(
    "보유 질환",
    ["없음", "당뇨", "고혈압"]
)

goal = st.sidebar.selectbox(
    "관리 목표",
    ["건강유지", "다이어트", "벌크업"]
)

# 이미지 업로드
st.subheader("📸 음식 사진 업로드")

uploaded_file = st.file_uploader(
    "음식 사진을 선택하세요",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="업로드한 음식 이미지",
        use_container_width=True
    )

    if st.button("🚀 식단 분석 시작"):

        user_info = {
            "age": age,
            "height": height,
            "weight": weight,
            "disease": disease,
            "goal": goal
        }

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        data = {
            "user_info": json.dumps(user_info)
        }

        with st.spinner("AI가 분석 중입니다..."):

            try:

                response = requests.post(
                    "http://127.0.0.1:8000/analyze-food",
                    files=files,
                    data=data,
                    timeout=30
                )

                if response.status_code == 200:

                    result = response.json()

                    st.success("분석 완료!")

                    st.header("📊 분석 결과")

                    st.json(result)

                else:

                    st.error(
                        f"서버 오류 발생 (상태코드: {response.status_code})"
                    )

                    st.text(response.text)

            except Exception as e:

                st.error("백엔드 서버 연결 실패")

                st.exception(e)