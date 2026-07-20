import os
import uvicorn
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from slm_core.model_loader import generate
from slm_core.candidate_evaluator import CandidateEvaluator

app = FastAPI(
    title="Rez_SLM Local AI Engine API",
    description="Local REST API Server serving Gemma 4 QLoRA Fine-Tuned Candidate Evaluator & SLM Engine.",
    version="1.0.0"
)

# Enable CORS for external web/desktop apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

evaluator = CandidateEvaluator(model_generate_fn=generate)

# Secret API Key to prevent unauthorized access from untrusted processes
API_KEY_SECRET = os.getenv("REZ_SLM_API_KEY", "rez_slm_secure_local_key_2026")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized API Key")
    return x_api_key

# Request Schemas
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 256

class CandidateItem(BaseModel):
    name: str
    resume: str

class EvaluateCandidateRequest(BaseModel):
    job_description: str
    candidates: List[CandidateItem]

@app.get("/health")
def health_check():
    return {"status": "online", "engine": "Rez_SLM Gemma 4 RTX 4070 SUPER"}

@app.post("/v1/generate")
def api_generate(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
    try:
        response = generate(req.prompt, max_tokens=req.max_tokens)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/evaluate_candidates")
def api_evaluate(req: EvaluateCandidateRequest, api_key: str = Depends(verify_api_key)):
    try:
        candidate_list = [{"name": c.name, "resume": c.resume} for c in req.candidates]
        ranked_results = evaluator.rank_candidates(req.job_description, candidate_list)
        return {"job_description": req.job_description, "results": ranked_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("[Rez_SLM API Engine] Starting Local API Server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
