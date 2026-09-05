import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"


def _extract_json_text(response) -> str:
    raw_text = (response.choices[0].message.content or "").strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`").replace("json\n", "", 1)
    return raw_text


def generate_questions(resume_text: str, target_role: str, num_questions: int = 5) -> list:
    prompt = f"""You are an experienced interviewer preparing to interview a candidate \
for the role of "{target_role or 'a role matching their resume'}".

Based on the resume below, generate {num_questions} interview questions:
- Mix of behavioral (STAR-style) and role-relevant questions
- Reference specific experience from the resume where relevant
- Order from warm-up to more challenging

Return ONLY a JSON object with this exact structure, no markdown fences, no preamble:
{{
  "questions": ["<question 1>", "<question 2>", ...]
}}

Resume:
---
{resume_text}
---
"""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=800,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = _extract_json_text(response)
    parsed = json.loads(raw_text)
    # Support either a bare list or the {"questions": [...]} wrapper above.
    if isinstance(parsed, dict) and "questions" in parsed:
        return parsed["questions"]
    return parsed


def evaluate_answer(question: str, answer: str) -> dict:
    prompt = f"""You are a supportive but honest interview coach. The candidate was asked:

Question: "{question}"

Their answer: "{answer}"

Evaluate this answer and return ONLY a JSON object (no markdown fences) with:
{{
  "score": <integer 1-10>,
  "strengths": ["<what worked>"],
  "improvements": ["<specific, actionable tip>"],
  "used_star_method": <true/false>,
  "encouragement": "<one warm, genuine sentence of encouragement>"
}}
"""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=500,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = _extract_json_text(response)
    return json.loads(raw_text)


def generate_final_summary(qa_pairs: list) -> dict:
    transcript = "\n\n".join(
        f"Q: {pair['question']}\nA: {pair['answer']}\nScore: {pair['feedback']['score']}/10"
        for pair in qa_pairs
    )

    prompt = f"""Here is a full mock interview transcript with per-answer scores:

{transcript}

Return ONLY a JSON object (no markdown fences) summarizing overall performance:
{{
  "average_score": <float>,
  "overall_feedback": "<3-4 sentence summary of how they did>",
  "biggest_strength": "<one sentence>",
  "biggest_growth_area": "<one sentence>",
  "confidence_note": "<one encouraging, honest sentence about their readiness>"
}}
"""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=500,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = _extract_json_text(response)
    return json.loads(raw_text)
