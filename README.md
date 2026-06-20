# 🍱 AI Food Nutrition Analysis

AI-based food nutrition analysis web service.

Upload a food image and the system automatically:

* Detect foods (YOLO11-seg)
* Classify foods (Swin Transformer)
* Estimate depth (Depth Anything V2)
* Estimate food weight
* Calculate nutrition facts
* Generate personalized feedback

---

## Features

* Food Segmentation
* Food Classification
* Depth Estimation
* Weight Estimation
* Nutrition Analysis
* Korean / English UI
* Personalized Feedback

---

## Tech Stack

### Frontend

* Streamlit

### Backend

* FastAPI

### AI Models

* YOLO11-seg
* Swin Transformer
* Depth Anything V2

### Framework

* PyTorch

---

## Run

### Backend

uvicorn fastapi_app:app --reload

### Frontend

streamlit run streamlit_app.py

---

## Demo Flow

Food Image

↓

Food Detection

↓

Depth Estimation

↓

Weight Estimation

↓

Nutrition Analysis

↓

Personalized Feedback
