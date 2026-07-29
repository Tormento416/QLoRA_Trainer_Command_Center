import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
from slm_core.resume_rewriter import ResumeRewriter
from slm_core.model_loader import generate

jd_text = """
Agentic AI Delivery Leader (Senior Manager) - Ascensus / Tential
Role: Accountable for end-to-end delivery of an enterprise portfolio of agentic AI initiatives—spanning from discovery and value-case development to design, build, launch, adoption, and outcome measurement.
Key Responsibilities:
- Establish delivery strategy, operating model, governance, ways of working, and quality metrics for AI agent deployments.
- Own enterprise portfolio of agentic AI use cases and shape multi-quarter roadmaps aligned with scalability and measurable outcomes.
- Partner with executive leadership to drive AI adoption and ensure high-impact business ROI.
Key Qualifications:
- 8+ years leading technology delivery, digital transformation, and complex AI/ML software implementations.
- Hands-on expertise in Agentic AI architectures, custom Small Language Models (SLMs), QLoRA fine-tuning, data pipelines (SQL/Snowflake/PostgreSQL), Python, REST APIs, Docker, and cloud infrastructure.
- Agile delivery leadership, release readiness, risk governance, Sev 1/2 defect triage, and executive reporting.
"""

print("[ResumeRewriter] Running full pipeline rewrite and candidate evaluation...")
rewriter = ResumeRewriter(model_generate_fn=generate)
result = rewriter.run_full_pipeline_rewrite_and_evaluate("", jd_text, candidate_name="Adrian Sousa")

print("\n" + "=" * 65)
print("  AGENTIC AI DELIVERY LEADER — TAILORED RESUME EVALUATION")
print("=" * 65)
print(json.dumps(result["evaluation_result"], indent=2))
