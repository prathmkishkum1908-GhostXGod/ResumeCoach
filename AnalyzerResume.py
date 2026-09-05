import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"

ANALYSIS_PROMPT_TEMPLATE = """You are an expert resume reviewer and career coach with experience \
in technical recruiting and ATS (Applicant Tracking System) optimization.

Analyze the following resume text and return ONLY a valid JSON object (no markdown \
fences, no preamble) with this exact structure:

{{
  "overall_score": <integer 0-100>,
  "summary": "<2-3 sentence overall impression>",
  "strengths": ["<strength 1>", "<strength 2>", ...],
  "weaknesses": [
    {{"issue": "<what's wrong>", "why_it_matters": "<impact on recruiter/ATS>", "fix": "<concrete suggestion>"}}
  ],
  "grammar_and_formatting_issues": ["<issue 1>", "<issue 2>", ...],
  "missing_sections_or_keywords": ["<e.g. no quantified achievements>", ...],
  "ats_compatibility_notes": "<short note on ATS friendliness>",
  "top_3_priority_fixes": ["<fix 1>", "<fix 2>", "<fix 3>"]
}}

Be specific and actionable. Reference actual content from the resume where possible. \
If a target role is provided, tailor the feedback to that role.

Target role (may be empty): {target_role}

Resume text:
---
{resume_text}
---

Return ONLY the JSON object.
"""


def analyze_resume(resume_text: str, target_role: str = "") -> dict:
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to a .env file in the project "
            "root (GROQ_API_KEY) or set it as an environment variable."
        )

    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        target_role=target_role or "Not specified",
        resume_text=resume_text,
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2000,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.choices[0].message.content or ""

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse AI response as JSON: {e}\nRaw: {raw_text}")
