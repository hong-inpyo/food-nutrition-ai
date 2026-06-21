from transformers import pipeline
import torch

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)


def generate_feedback(
        total_kcal,
        protein,
        fat,
        carb,
        foods
):

    # =====================
    # Rule Based
    # =====================

    if total_kcal < 500:
        evaluation = "전체 섭취량이 다소 적은 식사입니다."
    elif total_kcal < 800:
        evaluation = "균형 잡힌 건강한 식사입니다."
    else:
        evaluation = "칼로리가 다소 높은 식사입니다."


    strengths = []

    if protein >= 25:
        strengths.append("단백질 섭취가 충분합니다.")

    if fat <= 30:
        strengths.append("지방 섭취가 적절합니다.")

    if carb <= 100:
        strengths.append("탄수화물 섭취가 균형적입니다.")


    weaknesses = []

    if protein < 20:
        weaknesses.append("단백질 섭취가 부족합니다.")

    if fat > 35:
        weaknesses.append("지방 섭취량이 다소 높습니다.")

    if total_kcal > 900:
        weaknesses.append("총 칼로리가 높은 편입니다.")


    advice = []

    if protein < 20:
        advice.append("계란이나 닭가슴살 같은 단백질 식품을 추가해보세요.")

    if fat > 35:
        advice.append("튀김이나 기름진 음식은 줄이는 것이 좋습니다.")

    if total_kcal > 900:
        advice.append("채소 위주의 식단으로 조절해보세요.")

    if len(advice) == 0:
        advice.append("현재 식습관을 유지하시면 좋겠습니다.")


    # 비어있는 경우 대비
    if len(strengths) == 0:
        strengths.append("영양소 섭취가 전반적으로 무난합니다.")

    if len(weaknesses) == 0:
        weaknesses.append("특별히 부족한 부분은 보이지 않습니다.")


    # =====================
    # Qwen은 문장만 다듬기
    # =====================

    prompt = f"""
당신은 대한민국 전문 영양사입니다.

아래 내용을 자연스럽고 친절한 한국어로 정리하세요.

식단 평가:
{evaluation}

장점:
{", ".join(strengths)}

부족한 점:
{", ".join(weaknesses)}

건강 조언:
{", ".join(advice)}

조건

- 반드시 한국어
- 음식과 영양 이야기만
- 새로운 사실을 만들지 말 것
- 출력 예시를 쓰지 말 것
- 설명은 짧고 자연스럽게
- 각 항목은 반드시 한 문장만 작성

답변:
"""


    outputs = generator(
        prompt,
        max_new_tokens=120,
        do_sample=False,
        repetition_penalty=1.15,
        return_full_text=False
    )


    text = outputs[0]["generated_text"].strip()

    # 깨진 문자 제거
    text = text.replace("�", "")

    # 출력 예시 제거
    remove_words = [
        "출력 예시",
        "예시:",
        "Example:",
        "답변 예시"
    ]

    for w in remove_words:
        if w in text:
            text = text.split(w)[0]


    return text