import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2

# Load BERT model once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="AI Resume Ranker", layout="wide")
st.title("📄 AI Resume Ranker / Job Matching System")
st.write("Upload resumes and paste a job description. The system will rank resumes by relevance.")

# Job Description input
job_desc = st.text_area("📝 Paste Job Description", height=150)

# Resume upload
uploaded_files = st.file_uploader("📂 Upload Resumes (PDF only)", accept_multiple_files=True, type=["pdf"])

# Button to start ranking
if st.button("🚀 Rank Resumes"):
    if not job_desc or not uploaded_files:
        st.error("⚠️ Please provide both a job description and at least one resume.")
    else:
        # Encode JD
        jd_embedding = model.encode(job_desc, convert_to_tensor=True)

        scores = []
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            res_embedding = model.encode(text, convert_to_tensor=True)
            sim = util.cos_sim(jd_embedding, res_embedding).item()
            scores.append((file.name, sim))

        # Sort resumes
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)

        # Show results
        st.subheader("🏆 Resume Ranking")
        for idx, (file, score) in enumerate(ranked, start=1):
            st.write(f"**{idx}. {file}** → Match Score: `{score:.2f}`")
