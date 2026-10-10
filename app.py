import streamlit as st
from agent import run_agent

st.set_page_config(page_title="AI Job Application Agent", page_icon="🧭", layout="wide")
st.title("🧭 AI Job Application Agent")
st.caption("Analyze a job, compare resume evidence, and draft application materials—with you in control.")

with st.sidebar:
    st.header("Workflow")
    st.markdown("1. Analyze job description\n2. Analyze resume evidence\n3. Compare skills\n4. Draft application\n5. Review for unsupported claims")
    st.warning("Generated content can be wrong. Verify every claim before using it.")

left, right = st.columns(2)
with left:
    resume = st.text_area(
        "Resume text",
        height=330,
        placeholder="Paste your resume text here..."
    )
with right:
    job_description = st.text_area(
        "Job description",
        height=330,
        placeholder="Paste the complete job description here..."
    )

if st.button("Analyze and generate drafts", type="primary", use_container_width=True):
    if not resume.strip() or not job_description.strip():
        st.error("Please provide both your resume and the job description.")
    else:
        with st.spinner("Running the LangGraph workflow..."):
            try:
                result = run_agent(resume, job_description)
                st.session_state["agent_result"] = result
            except Exception as exc:
                st.error(f"Could not complete the workflow: {exc}")
                st.info("Check your .env API key, installed dependencies, and internet connection.")

result = st.session_state.get("agent_result")
if result:
    st.divider()
    st.header("1. Job requirements")
    st.json(result.get("job_requirements", {}))

    st.header("2. Resume evidence")
    st.json(result.get("resume_evidence", {}))

    st.header("3. Match report")
    report = result.get("match_report", {})
    st.metric("Heuristic match score", f"{report.get('match_score', 0)}/100")
    st.caption(report.get("score_explanation", "This is not a probability of getting hired."))
    st.json(report)

    st.header("4. Tailored application draft")
    draft = result.get("application_draft", {})
    st.subheader("Professional summary")
    st.write(draft.get("tailored_summary", ""))
    st.subheader("Resume bullet suggestions")
    for item in draft.get("resume_bullet_suggestions", []):
        st.markdown(f"- {item}")
    st.subheader("Cover letter")
    st.text_area("Edit and copy your cover letter", value=draft.get("cover_letter", ""), height=260)
    st.subheader("Before using this draft")
    for item in draft.get("verification_checklist", []):
        st.markdown(f"- [ ] {item}")

    st.header("5. Quality review")
    review = result.get("review", {})
    st.write("Ready for your review:", review.get("approved_for_user_review", False))
    if review.get("issues_found"):
        st.write("Issues to verify:")
        for issue in review["issues_found"]:
            st.markdown(f"- {issue}")
    st.subheader("Reviewed cover letter")
    st.text_area("Reviewed version", value=review.get("revised_cover_letter", ""), height=260)

    st.download_button(
        "Download results as JSON",
        data=__import__("json").dumps(result, indent=2, ensure_ascii=False),
        file_name="job_application_analysis.json",
        mime="application/json"
    )
