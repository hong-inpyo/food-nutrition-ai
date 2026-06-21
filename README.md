# 🍱 AI Food Nutrition Coach

음식 사진 한 장을 업로드하면

* 음식 종류 인식 (YOLO Segmentation)
* 음식 부피 및 무게 추정 (Depth Estimation)
* 영양성분 계산 (칼로리, 탄수화물, 단백질, 지방)
* AI 영양사(Qwen LLM) 피드백 생성

을 수행하는 AI 영양 분석 서비스입니다.

---

## 📌 Features

### 1. Food Segmentation

YOLO Segmentation 모델을 이용하여 음식 영역을 분할합니다.

* 음식 종류 인식
* 음식 Mask 추출
* Bounding Box 생성

---

### 2. Depth Estimation

Depth Anything 모델을 사용하여 음식의 깊이를 추정합니다.

이를 이용하여

* 음식 부피(Volume)
* 음식 무게(Weight)

를 계산합니다.

---

### 3. Nutrition Estimation

음식별 영양 데이터베이스를 이용하여

* Calories (kcal)
* Protein (g)
* Fat (g)
* Carbohydrate (g)

를 계산합니다.

---

### 4. AI Nutrition Feedback

Qwen2.5-1.5B-Instruct를 사용하여

* 식단 평가
* 장점
* 부족한 점
* 건강 조언

을 자연스러운 한국어로 생성합니다.

---

## 🏗 Model Pipeline

```text
Input Image
      │
      ▼
YOLO Segmentation
      │
      ▼
Depth Estimation
      │
      ▼
Volume Estimation
      │
      ▼
Weight Estimation
      │
      ▼
Nutrition Calculation
      │
      ▼
Rule-based Analysis
      │
      ▼
Qwen LLM Feedback
      │
      ▼
Final Result
```

---

## 📂 Project Structure

```text
food-nutrition-service

├── fastapi_app.py
├── streamlit_app.py
├── pipeline.py
├── llm_feedback.py

├── Dockerfile
├── docker-compose.yml
├── requirements.txt

├── uploads/

├── result.jpg

└── README.md
```

---

## 🚀 Run with Docker

### Build

```bash
docker-compose build
```

### Run

```bash
docker-compose up
```

---

## 🚀 Run FastAPI

```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

API:

```text
POST /predict
```

---

## 🚀 Run Streamlit

```bash
streamlit run streamlit_app.py
```

---

## Example Output

### Input

Food Image

↓

### Output

Detected Foods

* Rice
* Kimchi
* Soup

Nutrition Summary

* Calories : 630 kcal
* Protein : 28 g
* Fat : 18 g
* Carbohydrate : 89 g

AI Nutrition Feedback

* 균형 잡힌 식사입니다.
* 단백질 섭취가 충분합니다.
* 채소 섭취를 조금 늘리면 더욱 좋습니다.

---

## Tech Stack

* Python
* FastAPI
* Streamlit
* PyTorch
* YOLO Segmentation
* Depth Anything
* HuggingFace Transformers
* Qwen2.5-1.5B-Instruct
* Docker

---

## Author

Hong Inpyo

Deep Learning based Food Nutrition Analysis Service
