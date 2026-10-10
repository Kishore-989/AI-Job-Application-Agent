import os
from typing import TypedDict, List, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


class JobRequirements(BaseModel):
    role_title: str = Field(description="Likely job title")
    required_skills: List[str] = Field(description="Skills explicitly required")
    preferred_skills: List[str] = Field(description="Skills described as preferred or a plus")
    responsibilities: List[str] = Field(description="Main job responsibilities")
    qualifications: List[str] = Field(description="Education, experience, or eligibility requirements")


class ResumeEvidence(BaseModel):
    candidate_summary: str
    skills_found: List[str]
    projects_and_evidence: List[str]
    strengths: List[str]
    unclear_or_missing_evidence: List[str]


class MatchReport(BaseModel):
    matched_skills: List[str]
    missing_or_unproven_skills: List[str]
    transferable_strengths: List[str]
    suggested_learning_priorities: List[str]
    match_score: int = Field(ge=0, le=100, description="Heuristic overlap score, not probability of getting hired")
    score_explanation: str


class ApplicationDraft(BaseModel):
    tailored_summary: str
    resume_bullet_suggestions: List[str]
    cover_letter: str
    verification_checklist: List[str]


class ReviewResult(BaseModel):
    approved_for_user_review: bool
    issues_found: List[str]
    revised_cover_letter: str


class AgentState(TypedDict, total=False):
    resume: str
    job_description: str
    job_requirements: Dict[str, Any]
    resume_evidence: Dict[str, Any]
    match_report: Dict[str, Any]
    application_draft: Dict[str, Any]
    review: Dict[str, Any]
    error: str


def get_model():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")
    return ChatOpenAI(model=MODEL_NAME, temperature=0)


def analyze_job(state: AgentState):
    model = get_model().with_structured_output(JobRequirements)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a careful recruiting analyst. Extract only information supported by the job description. Do not infer requirements that are not stated."),
        ("human", "JOB DESCRIPTION:\n{job_description}")
    ])
    result = (prompt | model).invoke({"job_description": state["job_description"]})
    return {"job_requirements": result.model_dump()}


def analyze_resume(state: AgentState):
    model = get_model().with_structured_output(ResumeEvidence)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a resume evidence analyst. Only report skills and experience supported by the supplied resume. Do not invent achievements, years of experience, or credentials. Put unclear items under unclear_or_missing_evidence."),
        ("human", "RESUME:\n{resume}")
    ])
    result = (prompt | model).invoke({"resume": state["resume"]})
    return {"resume_evidence": result.model_dump()}


def compare_match(state: AgentState):
    model = get_model().with_structured_output(MatchReport)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Compare the resume evidence with the job requirements. Treat a skill as matched only when the resume gives reasonable evidence. Missing means not evidenced, not necessarily absent. Give a transparent heuristic score, not a probability of hiring. Do not reward invented qualifications."),
        ("human", "JOB REQUIREMENTS:\n{job_requirements}\n\nRESUME EVIDENCE:\n{resume_evidence}")
    ])
    result = (prompt | model).invoke({
        "job_requirements": str(state["job_requirements"]),
        "resume_evidence": str(state["resume_evidence"])
    })
    return {"match_report": result.model_dump()}


def draft_application(state: AgentState):
    model = get_model().with_structured_output(ApplicationDraft)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a truthful application-writing assistant. Tailor wording to the job description using only evidence in the resume. Never fabricate tools, projects, metrics, work history, degrees, or certifications. If a bullet is only a suggestion, clearly phrase it as something the user should use only if true. Keep the cover letter concise and professional."),
        ("human", "JOB DESCRIPTION:\n{job_description}\n\nRESUME:\n{resume}\n\nMATCH REPORT:\n{match_report}\n\nCreate a tailored summary, truthful bullet suggestions, cover letter, and verification checklist.")
    ])
    result = (prompt | model).invoke({
        "job_description": state["job_description"],
        "resume": state["resume"],
        "match_report": str(state["match_report"])
    })
    return {"application_draft": result.model_dump()}


def review_draft(state: AgentState):
    model = get_model().with_structured_output(ReviewResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Review the draft for unsupported claims, invented metrics, exaggerated experience, and mismatches with the job. Remove claims that are not supported by the resume. This is a draft for human review, not automatic submission."),
        ("human", "RESUME:\n{resume}\n\nDRAFT:\n{application_draft}")
    ])
    result = (prompt | model).invoke({
        "resume": state["resume"],
        "application_draft": str(state["application_draft"])
    })
    return {"review": result.model_dump()}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("analyze_job", analyze_job)
    graph.add_node("analyze_resume", analyze_resume)
    graph.add_node("compare_match", compare_match)
    graph.add_node("draft_application", draft_application)
    graph.add_node("review_draft", review_draft)

    graph.add_edge(START, "analyze_job")
    graph.add_edge("analyze_job", "analyze_resume")
    graph.add_edge("analyze_resume", "compare_match")
    graph.add_edge("compare_match", "draft_application")
    graph.add_edge("draft_application", "review_draft")
    graph.add_edge("review_draft", END)

    return graph.compile()


def run_agent(resume: str, job_description: str):
    if not resume.strip():
        raise ValueError("Please enter your resume text.")
    if not job_description.strip():
        raise ValueError("Please paste the job description.")
    app = build_graph()
    return app.invoke({
        "resume": resume.strip(),
        "job_description": job_description.strip()
    })
