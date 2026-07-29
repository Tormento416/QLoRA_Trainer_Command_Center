import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle, KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

def create_resume_pdf(output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    PRIMARY = colors.HexColor("#0A2540")      # Deep Navy
    SECONDARY = colors.HexColor("#0284C7")    # Accent Cyan Blue
    TEXT_DARK = colors.HexColor("#1E293B")    # Slate Dark Body Text
    TEXT_MUTED = colors.HexColor("#475569")   # Muted Gray
    LINE_COLOR = colors.HexColor("#CBD5E1")   # Slate Divider Line

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=22,
        alignment=TA_CENTER,
        textColor=PRIMARY
    )

    contact_style = ParagraphStyle(
        'ContactLine',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=TEXT_MUTED
    )

    subtitle_style = ParagraphStyle(
        'SubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        alignment=TA_CENTER,
        textColor=SECONDARY
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=PRIMARY,
        spaceBefore=8,
        spaceAfter=2
    )

    job_title_style = ParagraphStyle(
        'JobTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=PRIMARY
    )

    job_meta_style = ParagraphStyle(
        'JobMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=12,
        alignment=TA_RIGHT,
        textColor=TEXT_MUTED
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=TEXT_DARK,
        spaceAfter=3
    )

    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=TEXT_DARK,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )

    story = []

    # 1. Header Block
    story.append(Paragraph("ADRIAN SOUSA", title_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph("Charlotte, NC &nbsp;|&nbsp; 352-484-4782 &nbsp;|&nbsp; adrian.m.sousa@gmail.com &nbsp;|&nbsp; LinkedIn: linkedin.com/in/adrian-sousa &nbsp;|&nbsp; GitHub: github.com/adrian-sousa", contact_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph("TECHNICAL DELIVERY LEAD &nbsp;&bull;&nbsp; AI PROGRAM MANAGER", subtitle_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceAfter=8))

    # 2. Core Competencies & Skills Bar
    story.append(Paragraph("CORE COMPETENCIES & TECH STACK", section_heading))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LINE_COLOR, spaceAfter=4))
    comp_text_1 = "<b>Technical Delivery & Leadership:</b> Enterprise Agile Execution &bull; Release Readiness &bull; Agentic AI Platform Architecture &bull; Risk Governance &bull; SDLC Prioritization &bull; Executive Stakeholder Management"
    comp_text_2 = "<b>Languages, Tools & Data:</b> Python &bull; Go &bull; JAX &bull; SQL &bull; Snowflake &bull; Tableau &bull; Supabase &bull; PostgreSQL &bull; FastAPI &bull; Next.js &bull; Docker &bull; Jira &bull; Azure DevOps &bull; AWS &bull; GCP &bull; QLoRA Fine-Tuning"
    story.append(Paragraph(comp_text_1, body_style))
    story.append(Paragraph(comp_text_2, body_style))
    story.append(Spacer(1, 6))

    # 3. Summary
    story.append(Paragraph("SUMMARY", section_heading))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LINE_COLOR, spaceAfter=4))
    summary_text = (
        "Results-driven AI-focused Delivery Leader and Technical Program Manager with 10+ years of experience deploying Generative AI platforms, "
        "leading high-concurrency software delivery, and directing enterprise technical operations across Engineering, Product, Support, and Executive stakeholders."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 6))

    # 4. Professional Experience
    story.append(Paragraph("PROFESSIONAL EXPERIENCE", section_heading))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LINE_COLOR, spaceAfter=6))

    # Job 1: HyrLink
    hyr_header = Table(
        [[Paragraph("<b>HyrLink</b> &mdash; <i>Founder / AI Product Builder</i>", job_title_style),
          Paragraph("Remote (Charlotte, NC) | Nov 2025 &ndash; Present", job_meta_style)]],
        colWidths=[360, 180]
    )
    hyr_header.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0)]))
    story.append(hyr_header)
    story.append(Spacer(1, 2))
    story.append(Paragraph("&bull; Built a custom Small Language Model (SLM) from the ground up and executed QLoRA fine-tuning on a <b>4.94M record dataset</b> (<i>4,942,386 training samples</i>), achieving high parameter efficiency and 90% GPU VRAM cap stability.", bullet_style))
    story.append(Paragraph("&bull; Led the design of all proprietary microservices and agentic orchestration using a combination of Python, JAX, and Go.", bullet_style))
    story.append(Paragraph("&bull; Defined product architecture, SDLC, and risk mitigation to prioritize the original goal &mdash; bringing Talent Acquisition partners and Candidates together to bridge the gap.", bullet_style))
    story.append(Paragraph("&bull; Reached 63 active users and delivered 386 tailored resume outputs, with users reporting a 37% increase in interview callbacks.", bullet_style))
    story.append(Spacer(1, 6))

    # Job 2: Cohesity (Unified Single Role Header)
    coh_header = Table(
        [[Paragraph("<b>Cohesity</b> &mdash; <i>Global Escalations Leader / Technical Program Management</i>", job_title_style),
          Paragraph("San Jose, CA / Remote | Jul 2021 &ndash; Dec 2024", job_meta_style)]],
        colWidths=[360, 180]
    )
    coh_header.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0)]))
    story.append(coh_header)
    story.append(Spacer(1, 2))
    story.append(Paragraph("&bull; Headed cross-functional root-cause remediation for recurring platform issues and high-impact defects across a global install base, recovering an estimated <b>$2.5M&ndash;$4M quarterly</b> in combined customer and business-risk exposure.", bullet_style))
    story.append(Paragraph("&bull; Led Jira-based sprint execution and release readiness for high-priority engineering workstreams; defined requirements, managed dependencies, prioritized remediation, and coordinated Engineering, Product, SRE, and customer stakeholders.", bullet_style))
    story.append(Paragraph("&bull; Redesigned the onboarding program for SRE and Engineering across 5 global regions, reducing time to productivity from 90 to 37 days (<b>58% reduction</b>) and establishing a reusable training library.", bullet_style))
    story.append(Paragraph("&bull; Built Python monitoring tools and API-driven scheduled automations to streamline operational triage, while running weekly executive syncs with VPs and C-suite leaders across 20&ndash;30 high-impact cases per quarter.", bullet_style))
    story.append(Spacer(1, 6))

    # Job 3: Finish Line Factory
    flf_header = Table(
        [[Paragraph("<b>Finish Line Factory</b> &mdash; <i>Architecture Consultant / Data Management & Compliance Engineer</i>", job_title_style),
          Paragraph("Pompano Beach, FL / Remote | Feb 2025 &ndash; Jan 2026", job_meta_style)]],
        colWidths=[360, 180]
    )
    flf_header.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0)]))
    story.append(flf_header)
    story.append(Spacer(1, 2))
    story.append(Paragraph("&bull; Designed full platform architecture and data management, leading to year-over-year growth from <b>$1.2M to $2.7M</b>.", bullet_style))
    story.append(Paragraph("&bull; Migrated unstructured Excel data into Supabase and Snowflake/Tableau pipelines with automated refresh intervals, introducing Python AI ingestion workflows that organized unstructured data prior to database ingestion.", bullet_style))
    story.append(Paragraph("&bull; Used APIs to analyze product demand, order patterns, payroll administration, and monthly cost-versus-revenue data for operational decision support.", bullet_style))
    story.append(Spacer(1, 6))

    # Job 4: HCL Technologies
    hcl_header = Table(
        [[Paragraph("<b>HCL Technologies</b> &mdash; <i>Senior IT Analyst / Support Transformation</i>", job_title_style),
          Paragraph("Cary, NC | Dec 2018 &ndash; Jun 2021", job_meta_style)]],
        colWidths=[360, 180]
    )
    hcl_header.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0)]))
    story.append(hcl_header)
    story.append(Spacer(1, 2))
    story.append(Paragraph("&bull; Managed enterprise delivery across multiple partner projects, improving processes and playbooks for junior engineers to learn, grow, and de-escalate high-risk platform issues.", bullet_style))
    story.append(Paragraph("&bull; Helped reduce ticket volume by roughly <b>40%</b> (and improved resolution speed by 17%&ndash;38%) by improving knowledge base documentation, automation, and upskilling engineers.", bullet_style))
    story.append(Paragraph("&bull; Supported client Azure environments through asset management, deployment activities, security-group administration, and coordinated infrastructure changes with technical stakeholders.", bullet_style))
    story.append(Spacer(1, 8))

    # 5. Education & Certifications
    story.append(KeepTogether([
        Paragraph("EDUCATION & CERTIFICATIONS", section_heading),
        HRFlowable(width="100%", thickness=0.5, color=LINE_COLOR, spaceAfter=4),
        Paragraph("<b>Western Governors University:</b> Bachelor of Technology, Computer Systems Networking & Telecommunications (In Progress)", body_style),
        Paragraph("<b>Harvard Online:</b> CS50X, CS50S, CS50AI, CS50B, CS50W, CS50SQL, Python for Research", body_style),
        Paragraph("<b>Stanford Online:</b> CS229 (Machine Learning), CS221 (AI Principles), CS336 (LLM Architectures)", body_style),
        Paragraph("<b>Certifications:</b> Forward Deployed Engineering in the Age of AI &bull; AWS Cloud Practitioner &bull; Google Cloud Professional &bull; CompTIA (A+, Network+, Security+, Linux+, Cloud+, Project+) &bull; PMI CAPM", body_style)
    ]))

    doc.build(story)
    print(f"Successfully generated PDF resume at: {output_path}")

if __name__ == "__main__":
    out_dir = os.path.abspath(r"C:\Users\adria\.gemini\antigravity-ide\brain\6a4cd03e-d295-497c-be7e-27596458b83d")
    os.makedirs(out_dir, exist_ok=True)
    pdf_filename = os.path.join(out_dir, "Adrian_Sousa_Agentic_AI_Delivery_Leader_Resume.pdf")
    
    # Also save copy in workspace root for convenient access
    ws_filename = os.path.abspath("Adrian_Sousa_Agentic_AI_Delivery_Leader_Resume.pdf")
    
    create_resume_pdf(pdf_filename)
    create_resume_pdf(ws_filename)
