import streamlit as st
import requests
import pandas as pd
from PIL import Image

# API_URL = "http://backend:8000/predict"
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="AI Food Nutrition",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ====================================
# Custom CSS
# ====================================

st.markdown("""
<style>
    /* 전체 배경 */
    .stApp {
        background: linear-gradient(180deg, #FFF9F2 0%, #FFFFFF 100%);
    }

    /* 메인 타이틀 영역 */
    .hero-box {
        background: linear-gradient(135deg, #FF8C42 0%, #FF6B6B 100%);
        padding: 2.2rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(255, 107, 107, 0.25);
    }
    .hero-title {
        color: white;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
    }
    .hero-desc {
        color: rgba(255,255,255,0.92);
        font-size: 1rem;
        margin-top: 0.4rem;
    }

    /* 섹션 헤더 (카드 배경 없이, 구분 텍스트용) */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #3A2E2A;
        margin: 1.4rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* 섹션을 감싸는 컨테이너 (st.container 기반) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: white;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 1.2rem;
        border: 1px solid #F2E8DD !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 1.6rem 1.8rem;
    }

    /* 원형 게이지 카드 */
    .gauge-row {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-around;
        gap: 1rem;
        margin-top: 0.5rem;
    }
    .gauge-card {
        flex: 1 1 140px;
        max-width: 180px;
        display: flex;
        flex-direction: column;
        align-items: center;
        background: #FFFBF6;
        border: 1px solid #F2E8DD;
        border-radius: 16px;
        padding: 1rem 0.6rem;
    }
    .gauge-label {
        font-weight: 700;
        font-size: 0.95rem;
        color: #3A2E2A;
        margin-top: 0.4rem;
    }
    .gauge-sub {
        font-size: 0.78rem;
        color: #9A8474;
        text-align: center;
        margin-top: 0.2rem;
        line-height: 1.3;
    }

    /* AI 피드백 - 진단 헤더 */
    .diag-banner {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        background: linear-gradient(135deg, #FFF3E8 0%, #FFE9DC 100%);
        border: 1px solid #FFD9B8;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .diag-banner-icon {
        font-size: 2rem;
        line-height: 1;
    }
    .diag-banner-title {
        font-weight: 800;
        font-size: 1.05rem;
        color: #3A2E2A;
        margin: 0;
    }
    .diag-banner-sub {
        font-size: 0.85rem;
        color: #8A6D5C;
        margin: 0.15rem 0 0 0;
    }

    /* AI 피드백 - 개별 카드 */
    .fb-card {
        display: flex;
        gap: 0.8rem;
        background: #FFFFFF;
        border: 1px solid #F0E6DA;
        border-left: 5px solid var(--fb-color, #FF8C42);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
    }
    .fb-icon {
        font-size: 1.4rem;
        line-height: 1.3;
        flex-shrink: 0;
    }
    .fb-body {
        flex: 1;
    }
    .fb-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #3A2E2A;
        margin: 0 0 0.25rem 0;
    }
    .fb-desc {
        font-size: 0.85rem;
        color: #6B5A4F;
        line-height: 1.5;
        margin: 0;
    }
    .fb-tag {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        margin-left: 0.4rem;
        vertical-align: middle;
    }

    /* 메트릭 카드 */
    div[data-testid="stMetric"] {
        background: #FFF6ED;
        border: 1px solid #FFE1C2;
        border-radius: 14px;
        padding: 0.9rem 0.8rem 0.6rem 0.8rem;
        text-align: center;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #8A6D5C;
    }
    div[data-testid="stMetricValue"] {
        color: #FF6B35;
        font-weight: 800;
    }

    /* 버튼 */
    .stButton > button {
        background: linear-gradient(135deg, #FF8C42 0%, #FF6B6B 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        font-size: 1.05rem;
        box-shadow: 0 4px 14px rgba(255, 107, 107, 0.3);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(255, 107, 107, 0.4);
    }

    /* progress bar 색상 */
    div[data-testid="stProgress"] div[role="progressbar"] > div {
        background: linear-gradient(90deg, #FF8C42, #FF6B6B);
    }

    /* 구분선 여백 줄이기 */
    hr {
        margin: 1.2rem 0;
    }

    /* 이미지 라운드 처리 */
    div[data-testid="stImage"] img {
        border-radius: 14px;
    }

    /* expander 스타일 */
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #F2E8DD;
    }
</style>
""", unsafe_allow_html=True)


# ====================================
# Language
# ====================================

top_col1, top_col2 = st.columns([4, 1])

with top_col2:
    lang = st.selectbox(
        "Language / 언어",
        ["한국어", "English"],
        label_visibility="collapsed",
    )


if lang == "한국어":

    TXT = {
        "title": "🍱 AI 음식 영양 분석 서비스",
        "desc": "음식 사진을 업로드하면 음식 종류와 영양정보를 분석해 드려요.",
        "user_info": "사용자 정보",
        "upload": "음식 사진 업로드",
        "upload_help": "JPG, JPEG, PNG 형식을 지원합니다",
        "origin": "📷 원본 이미지",
        "result": "🔍 분석 결과 이미지",
        "analyze": "🔍 영양 분석 시작",
        "loading": "AI가 음식을 분석하고 있어요...",
        "summary": "📊 영양 요약",
        "food_count": "음식 개수",
        "weight": "총 무게",
        "kcal": "총 칼로리",
        "protein": "총 단백질",
        "fat": "총 지방",
        "carb": "총 탄수화물",
        "nutrition": "🧾 상세 영양 정보",
        "feedback": "🤖 AI 맞춤 피드백",
        "diag_title": "오늘 식사 진단 리포트",
        "meal_ratio": "🍽 한 끼 권장 칼로리 대비",
        "age": "나이",
        "gender": "성별",
        "activity": "활동량",
        "male": "남성",
        "female": "여성",
        "low": "낮음",
        "normal": "보통",
        "high": "높음",
        "upload_prompt": "👆 분석할 음식 사진을 업로드해주세요",
        "json": "📦 원본 JSON 데이터 보기",
    }

else:

    TXT = {
        "title": "🍱 AI Food Nutrition",
        "desc": "Upload a food photo and get instant nutrition analysis.",
        "user_info": "User Information",
        "upload": "Upload Image",
        "upload_help": "Supports JPG, JPEG, PNG formats",
        "origin": "📷 Original Image",
        "result": "🔍 Prediction Result",
        "analyze": "🔍 Analyze",
        "loading": "AI is analyzing your food...",
        "summary": "📊 Nutrition Summary",
        "food_count": "Food Count",
        "weight": "Total Weight",
        "kcal": "Calories",
        "protein": "Protein",
        "fat": "Fat",
        "carb": "Carbohydrate",
        "nutrition": "🧾 Nutrition Details",
        "feedback": "🤖 AI Feedback",
        "diag_title": "Today's Meal Diagnosis",
        "meal_ratio": "🍽 Meal Ratio vs Recommended",
        "age": "Age",
        "gender": "Gender",
        "activity": "Activity",
        "male": "Male",
        "female": "Female",
        "low": "Low",
        "normal": "Normal",
        "high": "High",
        "upload_prompt": "👆 Upload a food photo to get started",
        "json": "📦 View raw JSON data",
    }


# ====================================
# Hero / Title
# ====================================

st.markdown(
    f'<div class="hero-box"><p class="hero-title">{TXT["title"]}</p>'
    f'<p class="hero-desc">{TXT["desc"]}</p></div>',
    unsafe_allow_html=True,
)


# ====================================
# User Info
# ====================================

with st.container(border=True):
    st.markdown(f'<div class="section-header">👤 {TXT["user_info"]}</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input(
            TXT["age"],
            min_value=1,
            max_value=100,
            value=23,
        )

    with c2:
        gender = st.selectbox(
            TXT["gender"],
            [TXT["male"], TXT["female"]],
        )

    with c3:
        activity = st.selectbox(
            TXT["activity"],
            [TXT["low"], TXT["normal"], TXT["high"]],
        )


# ====================================
# Upload
# ====================================

with st.container(border=True):
    st.markdown(f'<div class="section-header">📤 {TXT["upload"]}</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        TXT["upload"],
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help=TXT["upload_help"],
    )

    if not uploaded_file:
        st.info(TXT["upload_prompt"])


if uploaded_file:

    image = Image.open(uploaded_file)

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f'<div class="section-header">{TXT["origin"]}</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        analyze_clicked = st.button(TXT["analyze"], use_container_width=True)

    if analyze_clicked:

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type,
            )
        }

        with st.spinner(TXT["loading"]):
            response = requests.post(API_URL, files=files)

        result = response.json()
        foods = result["foods"]
        summary = result["summary"]
        feedback = result["feedback"]
        # ======================
        # Result image (in same card area)
        # ======================

        with st.container(border=True):
            rcol1, rcol2 = st.columns(2)
            with rcol2:
                st.markdown(f'<div class="section-header">{TXT["result"]}</div>', unsafe_allow_html=True)
                st.image("result.jpg", use_container_width=True)
            with rcol1:
                st.markdown(f'<div class="section-header">{TXT["origin"]}</div>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                
        with st.container(border=True):

            st.markdown(
                '<div class="section-header">🧠 AI 영양 코치</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div style="
                background:#FFF9F2;
                border-left:5px solid #FF8C42;
                padding:20px;
                border-radius:15px;
                font-size:16px;
                line-height:1.8;
                color:#333;
                ">
                {feedback}
                </div>
                """,
                unsafe_allow_html=True
            )
        total_weight = sum(x["weight_g"] for x in foods)
        total_kcal = sum(x["kcal"] for x in foods)
        total_protein = sum(x["protein_g"] for x in foods)
        total_fat = sum(x["fat_g"] for x in foods)
        total_carb = sum(x["carb_g"] for x in foods)

        # ======================
        # Recommend (calorie & macro targets per meal)
        # ======================

        if gender == TXT["male"]:
            recommend = 2200
        else:
            recommend = 1800

        if activity == TXT["low"]:
            recommend -= 200
        elif activity == TXT["high"]:
            recommend += 300

        meal_recommend = recommend / 3

        # 3대 영양소 비율: 탄수화물 50% / 단백질 20% / 지방 30% (칼로리 기준)
        # 1g: 탄수화물 4kcal, 단백질 4kcal, 지방 9kcal
        protein_target_g = (meal_recommend * 0.20) / 4
        fat_target_g = (meal_recommend * 0.30) / 9
        carb_target_g = (meal_recommend * 0.50) / 4

        kcal_ratio = min(total_kcal / meal_recommend, 1.0)
        protein_ratio = min(total_protein / protein_target_g, 1.0) if protein_target_g else 0
        fat_ratio = min(total_fat / fat_target_g, 1.0) if fat_target_g else 0
        carb_ratio = min(total_carb / carb_target_g, 1.0) if carb_target_g else 0

        # ======================
        # AI 맞춤 피드백 (의사 진단 + 트레이너 코칭 스타일)
        # ======================

        # --- 부족/충분/과다 판정을 위한 비율 (게이지와 동일 기준) ---
        deficits = {
            "kcal": total_kcal - meal_recommend,
            "protein": total_protein - protein_target_g,
            "fat": total_fat - fat_target_g,
            "carb": total_carb - carb_target_g,
        }

        # 음식 추천 풀: (이름, 1회 제공량 g, 단백질g, 지방g, 탄수화물g, kcal)
        if lang == "한국어":
            FOOD_BANK = {
                "protein": [
                    ("닭가슴살 100g", 100, 23, 2, 0),
                    ("삶은 계란 1개", 50, 6, 5, 0.5),
                    ("그릭요거트 1컵", 150, 15, 4, 6),
                    ("두부 1/2모", 150, 12, 6, 3),
                ],
                "fat_low": [
                    ("아몬드 한 줌(15g)", 15, 3, 9, 3),
                    ("아보카도 1/2개", 100, 2, 15, 9),
                    ("올리브오일 1큰술", 14, 0, 14, 0),
                ],
                "carb": [
                    ("바나나 1개", 120, 1, 0, 27),
                    ("현미밥 1/2공기", 100, 3, 1, 38),
                    ("고구마 1개(중)", 130, 2, 0, 30),
                ],
            }
        else:
            FOOD_BANK = {
                "protein": [
                    ("100g grilled chicken breast", 100, 23, 2, 0),
                    ("1 boiled egg", 50, 6, 5, 0.5),
                    ("1 cup Greek yogurt", 150, 15, 4, 6),
                    ("1/2 block of tofu", 150, 12, 6, 3),
                ],
                "fat_low": [
                    ("a handful of almonds (15g)", 15, 3, 9, 3),
                    ("1/2 avocado", 100, 2, 15, 9),
                    ("1 tbsp olive oil", 14, 0, 14, 0),
                ],
                "carb": [
                    ("1 banana", 120, 1, 0, 27),
                    ("1/2 bowl brown rice", 100, 3, 1, 38),
                    ("1 medium sweet potato", 130, 2, 0, 30),
                ],
            }

        def pick_food_suggestion(category, gap_g):
            """gap_g(부족량)을 채우기 위한 음식 1~2개를 자연스러운 문장으로 추천."""
            options = FOOD_BANK[category]
            picks = []
            remaining = gap_g
            for name, serving, p, f, c in sorted(options, key=lambda x: -x[1 if category != "fat_low" else 3]):
                key_val = {"protein": p, "fat_low": f, "carb": c}[category]
                if key_val <= 0:
                    continue
                picks.append(name)
                remaining -= key_val
                if remaining <= 0 or len(picks) >= 2:
                    break
            if not picks:
                picks = [options[0][0]]
            return (" 또는 " if lang == "한국어" else " or ").join(picks)

        # --- 진단 카드 생성 ---
        # 각 카드: (severity, icon, color, title, desc)
        # severity: "good" / "watch" / "alert" — 배너 전체 톤 결정에 사용
        diag_cards = []

        # 1) 칼로리 총평 (의사의 1차 진단)
        kcal_gap = deficits["kcal"]
        if total_kcal < meal_recommend * 0.6:
            diag_cards.append((
                "alert", "🩺", "#E0524C",
                "식사량이 많이 부족해요" if lang == "한국어" else "Meal size is significantly low",
                (
                    f"권장 칼로리의 {kcal_ratio*100:.0f}%만 섭취했어요. 이렇게 부족한 식사가 반복되면 "
                    f"기초대사가 떨어지고 다음 식사에 과식할 위험이 커져요. "
                    f"지금 {abs(kcal_gap):.0f}kcal 정도를 더 채워주는 게 좋아요."
                    if lang == "한국어" else
                    f"You're at only {kcal_ratio*100:.0f}% of your target calories for this meal. "
                    f"Repeated under-eating can slow your metabolism and increase the risk of overeating later. "
                    f"Try adding about {abs(kcal_gap):.0f} kcal."
                ),
            ))
        elif total_kcal < meal_recommend * 0.85:
            diag_cards.append((
                "watch", "🩺", "#FF8C42",
                "식사량이 살짝 부족해요" if lang == "한국어" else "Meal size is a bit low",
                (
                    f"권장량보다 약 {abs(kcal_gap):.0f}kcal 적게 드셨어요. 가벼운 간식 하나로 보충하면 충분해요."
                    if lang == "한국어" else
                    f"You're about {abs(kcal_gap):.0f} kcal under target. A light snack would close the gap."
                ),
            ))
        elif total_kcal <= meal_recommend * 1.15:
            diag_cards.append((
                "good", "✅", "#4CAF50",
                "칼로리 섭취가 균형적이에요" if lang == "한국어" else "Calorie intake is well balanced",
                (
                    "이번 식사는 활동량과 나이를 고려한 권장 칼로리에 잘 맞춰져 있어요. 지금 패턴을 유지해보세요."
                    if lang == "한국어" else
                    "This meal lines up well with your recommended intake for your age and activity level. Keep this pattern up."
                ),
            ))
        else:
            over_pct = (total_kcal / meal_recommend - 1) * 100
            diag_cards.append((
                "alert" if over_pct > 40 else "watch", "🩺", "#E0524C" if over_pct > 40 else "#FF8C42",
                "칼로리가 다소 높아요" if lang == "한국어" else "Calorie intake is on the high side",
                (
                    f"권장량보다 {over_pct:.0f}% 더 섭취했어요. 다음 식사에서는 양을 조금 줄이거나, "
                    f"오늘 가벼운 산책이나 운동으로 균형을 맞춰보는 걸 추천해요."
                    if lang == "한국어" else
                    f"You're {over_pct:.0f}% over your target. Consider a lighter next meal, "
                    f"or balance it out with a walk or light exercise today."
                ),
            ))

        # 2) 단백질 진단 (트레이너 코칭)
        if protein_ratio >= 1.0:
            diag_cards.append((
                "good", "💪", "#4CAF50",
                "단백질 섭취 우수해요" if lang == "한국어" else "Excellent protein intake",
                (
                    f"단백질 {total_protein:.1f}g으로 목표치를 채웠어요. 근육 회복과 유지에 좋은 식사였어요."
                    if lang == "한국어" else
                    f"You hit your protein target with {total_protein:.1f}g. Great for muscle recovery and maintenance."
                ),
            ))
        else:
            gap = protein_target_g - total_protein
            suggestion = pick_food_suggestion("protein", gap)
            severity = "alert" if protein_ratio < 0.5 else "watch"
            diag_cards.append((
                severity, "💪", "#E0524C" if severity == "alert" else "#FF8C42",
                "단백질이 부족해요" if lang == "한국어" else "Protein is running low",
                (
                    f"목표보다 단백질이 {gap:.0f}g 부족해요. 근손실을 막고 회복을 도우려면 "
                    f"{suggestion} 같은 메뉴를 추가해보세요."
                    if lang == "한국어" else
                    f"You're short by about {gap:.0f}g of protein. To support recovery and prevent muscle loss, "
                    f"consider adding {suggestion}."
                ),
            ))

        # 3) 지방 진단
        if total_fat > fat_target_g * 1.3:
            over_g = total_fat - fat_target_g
            diag_cards.append((
                "watch", "🫧", "#FF8C42",
                "지방 섭취가 다소 높아요" if lang == "한국어" else "Fat intake is a bit high",
                (
                    f"목표보다 지방이 {over_g:.0f}g 많아요. 튀김류나 기름진 소스를 줄이고, "
                    f"다음 식사는 담백한 구이나 찜 요리로 균형을 맞춰보세요."
                    if lang == "한국어" else
                    f"You're about {over_g:.0f}g over your fat target. Try cutting back on fried foods or rich sauces, "
                    f"and balance your next meal with grilled or steamed options."
                ),
            ))
        elif total_fat < fat_target_g * 0.4:
            gap = fat_target_g - total_fat
            suggestion = pick_food_suggestion("fat_low", gap)
            diag_cards.append((
                "watch", "🫧", "#5B8DEF",
                "지방이 너무 적어요" if lang == "한국어" else "Fat intake is quite low",
                (
                    f"지방이 부족하면 지용성 비타민 흡수와 호르몬 균형에 영향을 줄 수 있어요. "
                    f"{suggestion} 같은 건강한 지방을 조금 추가해보세요."
                    if lang == "한국어" else
                    f"Very low fat intake can affect fat-soluble vitamin absorption and hormone balance. "
                    f"Try adding a healthy fat source like {suggestion}."
                ),
            ))

        # 4) 탄수화물 진단
        if total_carb > carb_target_g * 1.3:
            over_g = total_carb - carb_target_g
            diag_cards.append((
                "watch", "🍞", "#FF8C42",
                "탄수화물이 다소 많아요" if lang == "한국어" else "Carb intake is on the high side",
                (
                    f"목표보다 탄수화물이 {over_g:.0f}g 많아요. 식후 졸림이나 혈당 스파이크를 줄이려면 "
                    f"다음 식사에서 밥이나 면 양을 조금 줄이고 채소나 단백질 비중을 높여보세요."
                    if lang == "한국어" else
                    f"You're about {over_g:.0f}g over your carb target. To reduce post-meal fatigue or blood sugar spikes, "
                    f"try cutting back on rice or noodles next time and adding more vegetables or protein."
                ),
            ))
        elif total_carb < carb_target_g * 0.4:
            gap = carb_target_g - total_carb
            suggestion = pick_food_suggestion("carb", gap)
            diag_cards.append((
                "watch", "🍞", "#5B8DEF",
                "탄수화물이 부족해요" if lang == "한국어" else "Carb intake is quite low",
                (
                    f"탄수화물이 부족하면 운동 중 에너지가 빨리 떨어질 수 있어요. "
                    f"{suggestion} 정도를 추가해서 에너지를 보충해보세요."
                    if lang == "한국어" else
                    f"Low carb intake can cause your energy to dip quickly during activity. "
                    f"Add something like {suggestion} to top up your energy."
                ),
            ))

        # --- 종합 배너 톤 결정 ---
        n_alert = sum(1 for c in diag_cards if c[0] == "alert")
        n_watch = sum(1 for c in diag_cards if c[0] == "watch")
        if n_alert > 0:
            banner_icon, banner_title, banner_sub = (
                "🧑‍⚕️",
                "오늘 식사, 보완이 필요해요" if lang == "한국어" else "This meal needs some adjustments",
                f"음식 {len(foods)}가지 분석 결과, 중점적으로 챙겨야 할 항목이 있어요." if lang == "한국어"
                else f"Based on {len(foods)} food items, there are a few key areas to address.",
            )
        elif n_watch > 0:
            banner_icon, banner_title, banner_sub = (
                "🧑‍⚕️",
                "전반적으로 양호하지만 다듬을 부분이 있어요" if lang == "한국어" else "Overall good, with a few tweaks suggested",
                f"음식 {len(foods)}가지를 분석했어요. 아래 코칭을 참고해보세요." if lang == "한국어"
                else f"Analyzed {len(foods)} food items — here's some coaching based on your meal.",
            )
        else:
            banner_icon, banner_title, banner_sub = (
                "🧑‍⚕️",
                "균형 잡힌 훌륭한 식사예요" if lang == "한국어" else "A well-balanced, great meal",
                f"음식 {len(foods)}가지 모두 영양 균형이 잘 맞아요. 이 패턴을 유지해보세요." if lang == "한국어"
                else f"All {len(foods)} food items show good nutritional balance. Keep this pattern going.",
            )

        severity_tag = {
            "alert": ("#FDEAEA", "#E0524C", "주의" if lang == "한국어" else "ALERT"),
            "watch": ("#FFF1E0", "#FF8C42", "체크" if lang == "한국어" else "CHECK"),
            "good": ("#E9F7EC", "#4CAF50", "양호" if lang == "한국어" else "GOOD"),
        }

        def render_fb_card(severity, icon, color, title, desc):
            bg, fg, label = severity_tag[severity]
            tag_html = (
                f'<span class="fb-tag" style="background:{bg};color:{fg};">{label}</span>'
            )
            return (
                f'<div class="fb-card" style="--fb-color:{color};">'
                f'<div class="fb-icon">{icon}</div>'
                f'<div class="fb-body">'
                f'<p class="fb-title">{title}{tag_html}</p>'
                f'<p class="fb-desc">{desc}</p>'
                f'</div></div>'
            )

        with st.container(border=True):
            st.markdown(f'<div class="section-header">{TXT["feedback"]}</div>', unsafe_allow_html=True)

            banner_html = (
                '<div class="diag-banner">'
                f'<div class="diag-banner-icon">{banner_icon}</div>'
                '<div>'
                f'<p class="diag-banner-title">{banner_title}</p>'
                f'<p class="diag-banner-sub">{banner_sub}</p>'
                '</div></div>'
            )
            st.markdown(banner_html, unsafe_allow_html=True)

            cards_html = "".join(render_fb_card(*c) for c in diag_cards)
            st.markdown(cards_html, unsafe_allow_html=True)

        # ======================
        # Summary
        # ======================

        with st.container(border=True):
            st.markdown(f'<div class="section-header">{TXT["summary"]}</div>', unsafe_allow_html=True)

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric(TXT["food_count"], len(foods))
            m2.metric(TXT["weight"], f"{total_weight:.1f} g")
            m3.metric(TXT["kcal"], f"{total_kcal:.1f}")
            m4.metric(TXT["protein"], f"{total_protein:.1f} g")
            m5.metric(TXT["fat"], f"{total_fat:.1f} g")
            m6.metric(TXT["carb"], f"{total_carb:.1f} g")

        # ======================
        # Gauge Cards (원형 게이지)
        # ======================

        def gauge_svg(percent, color, label, value_text, sub_text):
            pct = max(0, min(percent, 1.0))
            radius = 54
            circumference = 2 * 3.14159265 * radius
            offset = circumference * (1 - pct)
            return (
                '<div class="gauge-card">'
                f'<svg width="140" height="140" viewBox="0 0 140 140">'
                f'<circle cx="70" cy="70" r="{radius}" fill="none" stroke="#F1E9E0" stroke-width="14" />'
                f'<circle cx="70" cy="70" r="{radius}" fill="none" stroke="{color}" stroke-width="14" '
                f'stroke-linecap="round" stroke-dasharray="{circumference:.1f}" '
                f'stroke-dashoffset="{offset:.1f}" transform="rotate(-90 70 70)" />'
                f'<text x="70" y="76" text-anchor="middle" font-size="26" font-weight="800" fill="#3A2E2A">'
                f'{int(round(pct * 100))}%</text>'
                '</svg>'
                f'<div class="gauge-label">{label}</div>'
                f'<div class="gauge-sub">{value_text}<br>{sub_text}</div>'
                '</div>'
            )

        gauges_html = (
            '<div class="gauge-row">'
            + gauge_svg(kcal_ratio, "#FF6B35", TXT["kcal"], f"{total_kcal:.0f} kcal", f"/ {meal_recommend:.0f} kcal")
            + gauge_svg(protein_ratio, "#4CAF50", TXT["protein"], f"{total_protein:.1f} g", f"/ {protein_target_g:.0f} g")
            + gauge_svg(fat_ratio, "#FF8C42", TXT["fat"], f"{total_fat:.1f} g", f"/ {fat_target_g:.0f} g")
            + gauge_svg(carb_ratio, "#5B8DEF", TXT["carb"], f"{total_carb:.1f} g", f"/ {carb_target_g:.0f} g")
            + '</div>'
        )

        with st.container(border=True):
            st.markdown(f'<div class="section-header">{TXT["meal_ratio"]}</div>', unsafe_allow_html=True)
            st.markdown(gauges_html, unsafe_allow_html=True)

        # ======================
        # Table
        # ======================

        table = []
        for food in foods:
            table.append({
                "Food": food["food"],
                "Weight(g)": food["weight_g"],
                "Kcal": food["kcal"],
                "Protein": food["protein_g"],
                "Fat": food["fat_g"],
                "Carb": food["carb_g"],
            })

        df = pd.DataFrame(table)

        with st.container(border=True):
            st.markdown(f'<div class="section-header">{TXT["nutrition"]}</div>', unsafe_allow_html=True)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

            st.bar_chart(
                df.set_index("Food")[["Kcal", "Protein", "Fat", "Carb"]],
                use_container_width=True,
            )

        with st.expander(TXT["json"]):
            st.json(result)