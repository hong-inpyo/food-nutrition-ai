# app.py
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import json
from PIL import Image
import io

app = FastAPI(title="AI 맞춤형 영양 분석 서비스 API")

# 간이 영양 데이터베이스 (기획서의 영양 DB 데이터 샘플)
# 실제 구현 시에는 별도의 json 파일이나 SQL db로 확장 가능합니다.
NUTRITION_DB = {
    "미소국": {"calories": 35, "carbs": 4, "protein": 2, "fat": 1},
    "연어회": {"calories": 160, "carbs": 0, "protein": 20, "fat": 8},
    "비프까스": {"calories": 320, "carbs": 22, "protein": 18, "fat": 19},
    "양배추 샐러드": {"calories": 15, "carbs": 3, "protein": 1, "fat": 0},
    "와사비": {"calories": 5, "carbs": 1, "protein": 0, "fat": 0}
}

# 분석 요청 시 함께 받을 사용자 프로필 데이터 구조 정의
class UserProfile(BaseModel):
    age: int
    height: float
    weight: float
    disease: str  # 당뇨, 없음 등
    goal: str     # 다이어트, 벌크업, 건강유지 등

@app.post("/analyze-food")
async def analyze_food(
    file: UploadFile = File(...),
    user_info: str = Form(...)  # JSON 문자열 형태로 전달받음
):
    # 1. 이미지 파일 읽기 (기획서 상의 전처리 단계의 시작)
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # [데이터 전처리 & AI 모델 추론 단계]
    # 실제 환경에서는 이곳에 YOLO11-seg, Swin Transformer, Depth Anything V2 모델을 로드하여 
    # 예측된 음식 클래스명과 중량(부피)을 추출하게 됩니다.
    # 여기서는 샘플 흐름을 보여주기 위해 가상의 인식 결과("연어회", "미so국")를 사용합니다.
    detected_foods = ["연어회", "미소국"] 
    
    # 2. 사용자 정보 파싱
    user_data = json.loads(user_info)
    
    # 3. 영양 분석 엔진 로직 수행 (기획서 10페이지 알고리즘 구현)
    total_calories = 0
    total_carbs = 0
    total_protein = 0
    total_fat = 0
    
    for food in detected_foods:
        if food in NUTRITION_DB:
            total_calories += NUTRITION_DB[food]["calories"]
            total_carbs += NUTRITION_DB[food]["carbs"]
            total_protein += NUTRITION_DB[food]["protein"]
            total_fat += NUTRITION_DB[food]["fat"]
            
    # 맞춤형 피드백 생성 로직
    feedback_list = []
    
    # 질환 맞춤형 피드백
    if user_data.get("disease") == "당뇨":
        feedback_list.append("당뇨 환자용: 혈당 관리를 위해 탄수화물 섭취량이 적절한지 체크했습니다. 정제 탄수화물 섭취에 유의하세요.")
    
    # 목표 맞춤형 피드백
    if user_data.get("goal") == "벌크업":
        if total_protein < 30:
            feedback_list.append("목표(벌크업) 대비 단백질 섭취량이 다소 부족합니다. 단백질 위주의 음식을 다음 식단에 추가하세요.")
    elif user_data.get("goal") == "다이어트":
        feedback_list.append("목표(다이어트): 칼로리 연소를 돕는 섬유질 채소 위주 식단을 지속해 주세요.")
        
    # 종합 피드백 통합
    final_feedback = " ".join(feedback_list) if feedback_list else "영양 균형이 우수한 식단입니다. 건강 유지를 위해 규칙적인 식습관을 이어가세요."
    
    return {
        "detected_foods": detected_foods,
        "nutrition_summary": {
            "calories": total_calories,
            "carbs": total_carbs,
            "protein": total_protein,
            "fat": total_fat
        },
        "health_score": 85,  # 가상의 건강 점수 계산 결과
        "custom_feedback": final_feedback,
        "recommendation": "닭가슴살 샐러드" if user_data.get("goal") == "벌크업" else "현미밥 중심 정식"
    }