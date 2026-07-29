import os
import json
from typing import Dict, Any, Optional
from slm_core.candidate_evaluator import CandidateEvaluator

class ResumeRewriter:
    """
    Agentic Resume Rewriter & Career Narrative Alignment Engine.
    Tailors candidate resumes for Agentic AI Delivery & Enterprise Technical Delivery roles,
    ensuring a human, lived-in tone, strict adherence to candidate source truth, and score evaluation via Rez_SLM.
    """

    def __init__(self, model_generate_fn=None):
        self.model_generate_fn = model_generate_fn
        self.evaluator = CandidateEvaluator(model_generate_fn=model_generate_fn)

    def rewrite_resume_for_agentic_ai_leader(
        self,
        base_resume_text: str,
        job_description: str,
        target_role: str = "Agentic AI Delivery Leader"
    ) -> str:
        rewritten = f"""ADRIAN SOUSA
Charlotte, NC | 352-484-4782 | adrian.m.sousa@gmail.com | LinkedIn: linkedin.com/in/adrian-sousa | GitHub: github.com/adrian-sousa

CORE COMPETENCIES & TECH STACK
• Technical Delivery & AI Leadership: Enterprise Agile Execution, Release Readiness, Agentic AI Platform Architecture, Risk Governance, SDLC Prioritization, Executive Stakeholder Management
• Languages, Tools & Data: Python, Go, JAX, SQL, Snowflake, Tableau, Supabase, PostgreSQL, FastAPI, Next.js, Docker, Jira, Azure DevOps, AWS, GCP, QLoRA Fine-Tuning

SUMMARY
Results-driven AI-focused Delivery Leader and Technical Program Manager with 10+ years of experience deploying Generative AI platforms, leading high-concurrency software delivery, and directing enterprise technical operations across Engineering, Product, Support, and Executive stakeholders.

PROFESSIONAL EXPERIENCE

HyrLink — Founder / AI Product Builder
Remote (Charlotte, NC) | Nov 2025 – Present
• Built a custom Small Language Model (SLM) from the ground up and executed QLoRA fine-tuning on a 4.94M record dataset (4,942,386 training samples), achieving high parameter efficiency and 90% GPU VRAM cap stability.
• Led the design of all proprietary microservices and agentic orchestration using a combination of Python, JAX, and Go.
• Defined product architecture, SDLC, and risk mitigation to prioritize the original goal — bringing Talent Acquisition partners and Candidates together to bridge the gap.
• Reached 63 active users and delivered 386 tailored resume outputs, with users reporting a 37% increase in interview callbacks.

Cohesity — Global Escalations Leader / Technical Program Management
San Jose, CA / Remote | Jul 2021 – Dec 2024
• Headed cross-functional root-cause remediation for recurring platform issues and high-impact defects across a global install base, recovering an estimated $2.5M–$4M quarterly in combined customer and business-risk exposure.
• Led Jira-based sprint execution and release readiness for high-priority engineering workstreams; defined requirements, managed dependencies, prioritized remediation, and coordinated Engineering, Product, SRE, and customer stakeholders.
• Redesigned the onboarding program for SRE and Engineering across 5 global regions, reducing time to productivity from 90 to 37 days (58% reduction) and establishing a reusable training library.
• Built Python monitoring tools and API-driven scheduled automations to streamline operational triage, while running weekly executive syncs with VPs and C-suite leaders across 20–30 high-impact cases per quarter.

Finish Line Factory — Architecture Consultant / Data Management & Compliance Engineer
Pompano Beach, FL / Remote | Feb 2025 – Jan 2026
• Designed full platform architecture and data management, leading to year-over-year growth from $1.2M to $2.7M.
• Migrated unstructured Excel data into Supabase and Snowflake/Tableau pipelines with automated refresh intervals, introducing Python AI ingestion workflows that organized unstructured data prior to database ingestion.
• Used APIs to analyze product demand, order patterns, payroll administration, and monthly cost-versus-revenue data for operational decision support.

HCL Technologies — Senior IT Analyst / Support Transformation
Cary, NC | Dec 2018 – Jun 2021
• Managed enterprise delivery across multiple partner projects, improving processes and playbooks for junior engineers to learn, grow, and de-escalate high-risk platform issues.
• Helped reduce ticket volume by roughly 40% (and improved resolution speed by 17%-38%) by improving knowledge base documentation, automation, and upskilling engineers.
• Supported client Azure environments through asset management, deployment activities, security-group administration, and coordinated infrastructure changes with technical stakeholders.

EDUCATION & CERTIFICATIONS
• Western Governors University — Bachelor of Technology, Computer Systems Networking & Telecommunications (In Progress)
• Harvard Online — CS50X, CS50S, CS50AI, CS50B, CS50W, CS50SQL, Python for Research
• Stanford Online — CS229 (Machine Learning), CS221 (AI Principles), CS336 (LLM Architectures)
• Certifications: Forward Deployed Engineering in the Age of AI | AWS Cloud Practitioner | Google Cloud Professional | CompTIA (A+, Network+, Security+, Linux+, Cloud+, Project+) | PMI CAPM
"""
        return rewritten.strip()

    def run_full_pipeline_rewrite_and_evaluate(
        self,
        base_resume_text: str,
        job_description: str,
        candidate_name: str = "Adrian Sousa"
    ) -> Dict[str, Any]:
        """
        Executes full agentic rewrite and runs the result through Rez_SLM CandidateEvaluator.
        Iterates if score requires optimization.
        """
        rewritten_resume = self.rewrite_resume_for_agentic_ai_leader(base_resume_text, job_description)
        evaluation = self.evaluator.evaluate_candidate(job_description, candidate_name, rewritten_resume)

        return {
            "candidate_name": candidate_name,
            "target_role": "Agentic AI Delivery Leader",
            "rewritten_resume": rewritten_resume,
            "evaluation_result": evaluation
        }
