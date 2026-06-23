from pypdf import PdfReader
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="AI Resume Job Matcher", page_icon="📄", layout="wide")

st.title("AI Resume & Job Description Matcher")
st.write("Upload your resume PDF and get job match scores, missing skills, and recommendations.")


@st.cache_data
def load_job_data():
    df = pd.read_excel("data/jobs_skill_matrix_updated_v41.xlsx")

    basic_cols = ["company", "role", "seniority"]
    skill_cols = [col for col in df.columns if col not in basic_cols]

    df["skills_list"] = df[skill_cols].apply(
        lambda row: [skill for skill in skill_cols if row[skill] == 1],
        axis=1
    )

    df["job_text"] = (
        df["role"].fillna("") + " " +
        df["seniority"].fillna("") + " " +
        df["skills_list"].apply(lambda x: " ".join(x))
    )

    return df, skill_cols


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


df, skill_cols = load_job_data()
model = load_model()
job_embeddings = model.encode(df["job_text"].tolist())

uploaded_file = st.file_uploader("Upload your resume PDF", type=["pdf"])

if uploaded_file:
    st.success("Resume uploaded successfully!")

    reader = PdfReader(uploaded_file)
    resume_text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            resume_text += page_text

    st.subheader("Extracted Resume Text")
    st.write(resume_text[:1000])

    resume_embedding = model.encode(resume_text)

    similarities = cosine_similarity([resume_embedding], job_embeddings)

    df["match_score"] = similarities[0]

    top_matches = df[["company", "role", "seniority", "match_score"]].copy()
    top_matches["match_percent"] = (top_matches["match_score"] * 100).round(2)
    top_matches = top_matches.sort_values("match_score", ascending=False)

    st.subheader("Top Matching Jobs")
    st.dataframe(top_matches.head(10))

    best_match_index = top_matches.index[0]
    best_job = df.loc[best_match_index]

    resume_text_lower = resume_text.lower()

    resume_skills = [
        skill for skill in skill_cols
        if skill.replace("_", " ") in resume_text_lower
    ]

    best_job_skills = best_job["skills_list"]

    matched_skills = [
        skill for skill in best_job_skills
        if skill in resume_skills
    ]

    missing_skills = [
        skill for skill in best_job_skills
        if skill not in resume_skills
    ]

    st.subheader("Matched Skills")
    for skill in matched_skills:
        st.success(skill)

    st.subheader("Missing Skills")
    for skill in missing_skills[:10]:
        st.error(skill)
