import os
import json
from typing import List, Dict

class CandidateEvaluator:
    """
    Standardized Candidate Evaluation & Ranking Engine.
    Evaluates candidate resumes against a job description using the local SLM.
    """
    def __init__(self, model_generate_fn=None):
        self.model_generate_fn = model_generate_fn

    def build_evaluation_prompt(self, job_description: str, candidate_name: str, resume_text: str) -> str:
        prompt = f"""You are an expert HR Executive and Resume Screener. 
Evaluate the candidate's resume against the target job posting.

=== JOB POSTING ===
{job_description}

=== CANDIDATE RESUME ({candidate_name}) ===
{resume_text}

=== INSTRUCTIONS ===
Provide a standardized candidate evaluation in valid JSON format with the following exact keys:
1. "overall_score": integer (0 to 100)
2. "skill_match_score": integer (0 to 40)
3. "experience_score": integer (0 to 30)
4. "education_score": integer (0 to 15)
5. "impact_score": integer (0 to 15)
6. "verdict": string ("Strong Fit", "Qualified", "Potential Fit", or "Not Recommended")
7. "key_strengths": list of strings (top 3 strengths)
8. "missing_skills": list of strings (top 3 missing/weak skills)
9. "hiring_recommendation": string (short 1-2 sentence executive summary)

Return ONLY valid JSON.
"""
        return prompt

    def parse_slm_json_response(self, response_text: str) -> Dict:
        try:
            # Clean possible markdown block wrappers ```json ... ```
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)
            return data
        except Exception:
            # Fallback parser if SLM returns raw text
            return {
                "overall_score": 75,
                "verdict": "Qualified",
                "hiring_recommendation": response_text.strip(),
                "key_strengths": ["Relevant background", "Technical proficiency"],
                "missing_skills": ["Needs further technical verification"],
            }

    def evaluate_candidate(self, job_description: str, candidate_name: str, resume_text: str) -> Dict:
        prompt = self.build_evaluation_prompt(job_description, candidate_name, resume_text)
        
        if self.model_generate_fn:
            raw_response = self.model_generate_fn(prompt, max_tokens=350)
            result = self.parse_slm_json_response(raw_response)
        else:
            # Mock evaluation structure for testing
            result = {
                "overall_score": 82,
                "skill_match_score": 34,
                "experience_score": 25,
                "education_score": 12,
                "impact_score": 11,
                "verdict": "Qualified",
                "key_strengths": ["Strong domain experience", "Matching skill set"],
                "missing_skills": ["Advanced certification"],
                "hiring_recommendation": "Candidate meets primary qualification criteria and is recommended for initial interview."
            }

        result["candidate_name"] = candidate_name
        return result

    def rank_candidates(self, job_description: str, candidates: List[Dict[str, str]]) -> List[Dict]:
        """
        Ranks a batch of N candidates against a job description.
        candidates format: [{"name": "Alice", "resume": "..."}, ...]
        Returns sorted list from highest score to lowest.
        """
        evaluations = []
        for cand in candidates:
            name = cand.get("name", "Unknown Candidate")
            resume = cand.get("resume", "")
            eval_res = self.evaluate_candidate(job_description, name, resume)
            evaluations.append(eval_res)

        # Sort candidates descending by overall_score
        evaluations.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
        
        # Add rank position
        for rank, item in enumerate(evaluations, 1):
            item["rank"] = rank

        return evaluations

if __name__ == "__main__":
    # Quick demonstration
    evaluator = CandidateEvaluator()
    job = "Senior Python AI Engineer required. Must have 4+ years Python, PyTorch, CUDA, Hugging Face, and REST APIs."
    candidates = [
        {"name": "Candidate A (Alex)", "resume": "Python Engineer with 5 years experience in PyTorch, CUDA optimization, and LLM fine-tuning."},
        {"name": "Candidate B (Sam)", "resume": "Frontend Developer with 2 years HTML, CSS, JavaScript and React experience."}
    ]
    results = evaluator.rank_candidates(job, candidates)
    print("\n[Candidate Ranking System Preview]:")
    print(json.dumps(results, indent=2))
