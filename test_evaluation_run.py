import json
from slm_core.candidate_evaluator import CandidateEvaluator
from slm_core.model_loader import generate

jd_text = """
Technical Delivery Lead - Greenbox Capital (United States / Remote)
Role: Drive results across delivery execution, release coordination, and operational improvement. 
Reports to: VP of Technology.
Success Factors: Delivery Execution & Agile Leadership (on-time delivery across multiple pods), Release Readiness & Quality (reduce defects), Delivery Analytics & Operational Improvement (Power BI, SQL, Excel, AI tools), Cross-Functional Coordination (engineering, product, data).
Requirements: 5+ years of experience in technical delivery, program delivery, agile leadership, or similar roles. SDLC, agile practices, release processes. Azure DevOps, Jira, SQL, Power BI / Tableau, Python, REST APIs, workflow automation, AI tools.
"""

resume_text = """
ADRIAN SOUSA - TECHNICAL DELIVERY LEAD
Agile Execution | Release Readiness | Data-Driven Insights
10+ years of experience driving predictable software delivery and operational efficiency across engineering, product, and data teams.
Core Competencies: Agile Delivery & Sprint Planning, Release Readiness & Coordination, Defect Triage, Requirement Definition, Dependency/Risk Management, Root-Cause Analysis, Python, SQL, Snowflake, Tableau, PostgreSQL, REST APIs, Docker, GitHub, AI/LLM Workflow Automation, Jira, Azure DevOps.
Experience:
- Cohesity: Global Escalations Leader / Technical Program Management. Led Jira-based sprint planning for high-priority engineering workstreams. Led release readiness for customer-impacting defects. Built Python monitoring tools and API-driven scheduled automations. Redesigned global onboarding reducing time from 90 to 37 days (58%).
- Finish Line Factory: Architecture Consultant / Data Management Engineer. Owned end-to-end delivery for data/workflow initiatives. Migrated Excel data into Supabase & Snowflake/Tableau pipelines. Python & AI ingestion workflows.
- HyrLink: Founder / AI Product Builder. Built agentic AI platform, custom SLM with QLoRA fine-tuning, Python, FastAPI, Next.js, Docker, PostgreSQL.
- HCL: Senior IT Analyst. System triage, Azure environment support.
Education: Bachelor of Technology (WGU - In Progress), Harvard Online CS50X/CS50S, Stanford Online CS229/CS221/CS336.
"""

print("[CandidateEvaluator] Running Evaluation for Adrian Sousa vs. Greenbox Capital Technical Delivery Lead...")
evaluator = CandidateEvaluator(model_generate_fn=generate)
result = evaluator.evaluate_candidate(jd_text, "Adrian Sousa", resume_text)

print("\n" + "=" * 60)
print("  STANDARDIZED CANDIDATE EVALUATION RESULT")
print("=" * 60)
print(json.dumps(result, indent=2))
