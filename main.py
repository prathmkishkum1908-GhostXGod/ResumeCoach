from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from ResumeChecker import extract_resume_text
from AnalyzerResume import analyze_resume
from InterviewCoach import generate_questions, evaluate_answer, generate_final_summary

app = FastAPI(title="ResumeCoach")
class InterviewStartRequest(BaseModel):
    resume_text: str
    target_role: Optional[str] = ""
    num_questions: int = 5

class EvaluateAnswerRequest(BaseModel):
    question: str
    answer: str

class QAPair(BaseModel):
    question: str
    answer: str
    feedback: dict

class SummaryRequest(BaseModel):
    quesans_pairs: List[QAPair]

@app.post("/api/analyze-resume")
async def analyze_resume_endpoint(
    file: UploadFile = File(...),
    target_role: str = Form(""),
):
    if not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(400, "Please upload a PDF or DOCX file.")

    file_bytes = await file.read()

    try:
        resume_text = extract_resume_text(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))

    try:
        analysis = analyze_resume(resume_text, target_role)
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")

    return {"resume_text": resume_text, "analysis": analysis}


@app.post("/api/interview/start")
async def start_interview(req: InterviewStartRequest):
    try:
        questions = generate_questions(req.resume_text, req.target_role, req.num_questions)
    except Exception as e:
        raise HTTPException(500, f"Could not generate questions: {e}")
    return {"questions": questions}


@app.post("/api/interview/evaluate")
async def evaluate_interview_answer(req: EvaluateAnswerRequest):
    try:
        feedback = evaluate_answer(req.question, req.answer)
    except Exception as e:
        raise HTTPException(500, f"Could not evaluate answer: {e}")
    return {"feedback": feedback}


@app.post("/api/interview/summary")
async def interview_summary(req: SummaryRequest):
    try:
        quesans_pairs = [pair.model_dump() for pair in req.quesans_pairs]
        summary = generate_final_summary(quesans_pairs)
    except Exception as e:
        raise HTTPException(500, f"Could not generate summary: {e}")
    return {"summary": summary}

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")
