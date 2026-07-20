import os
import sqlite3
import json
from typing import List, Dict, Optional

class DataBank:
    """
    Dual Memory Data Bank for Job Postings and Candidate Resumes.
    Operates in tandem with Rez_SLM engine.
    """
    def __init__(self, db_path: str = "data/databank.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Job Postings Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS job_postings (
                    job_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    company TEXT,
                    description TEXT NOT NULL,
                    required_skills TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 2. Candidate Resumes Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    job_id TEXT,
                    name TEXT NOT NULL,
                    email TEXT,
                    resume_text TEXT NOT NULL,
                    extracted_skills TEXT,
                    match_score INTEGER DEFAULT 0,
                    verdict TEXT DEFAULT 'Pending',
                    evaluation_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES job_postings (job_id)
                )
            """)
            conn.commit()

    # --- Job Description Operations ---
    def add_job_posting(self, job_id: str, title: str, description: str, company: str = "Internal", required_skills: List[str] = None) -> bool:
        skills_str = json.dumps(required_skills or [])
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO job_postings (job_id, title, company, description, required_skills)
                VALUES (?, ?, ?, ?, ?)
            """, (job_id, title, company, description, skills_str))
            conn.commit()
        return True

    def get_job_posting(self, job_id: str) -> Optional[Dict]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM job_postings WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["required_skills"] = json.loads(d.get("required_skills") or "[]")
                return d
        return None

    # --- Candidate Resume Operations ---
    def add_candidate(self, candidate_id: str, job_id: str, name: str, resume_text: str, email: str = "") -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO candidates (candidate_id, job_id, name, email, resume_text)
                VALUES (?, ?, ?, ?, ?)
            """, (candidate_id, job_id, name, email, resume_text))
            conn.commit()
        return True

    def get_candidates_for_job(self, job_id: str) -> List[Dict]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM candidates WHERE job_id = ? ORDER BY match_score DESC", (job_id,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                if d.get("evaluation_json"):
                    d["evaluation_json"] = json.loads(d["evaluation_json"])
                results.append(d)
            return results

    def save_evaluation_results(self, candidate_id: str, score: int, verdict: str, eval_data: Dict):
        eval_str = json.dumps(eval_data)
        skills_str = json.dumps(eval_data.get("key_strengths", []))
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE candidates 
                SET match_score = ?, verdict = ?, evaluation_json = ?, extracted_skills = ?
                WHERE candidate_id = ?
            """, (score, verdict, eval_str, skills_str, candidate_id))
            conn.commit()

if __name__ == "__main__":
    db = DataBank()
    db.add_job_posting("JOB101", "Senior AI Engineer", "Looking for Python PyTorch expert", "Tech Corp", ["Python", "PyTorch"])
    db.add_candidate("CAND001", "JOB101", "Alex Smith", "Experienced PyTorch AI developer")
    print("\n[Data Bank Initialized Successfully!]")
    print("Job Posting:", db.get_job_posting("JOB101"))
    print("Candidates:", db.get_candidates_for_job("JOB101"))
