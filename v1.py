import streamlit as st
from summarizer import summarize_legal_text
import pdfplumber

st.set_page_config(page_title="Legal Document Summarizer", layout="wide")
st.title("📄 Legal Document Summarizer")
st.write("Paste your legal document below or upload a `.txt` or `.pdf` file. Get a concise summary in seconds!")

# Text input
text_input = st.text_area("Paste Legal Document Here", height=300)

# File upload
uploaded_file = st.file_uploader("Or upload a legal document (.txt or .pdf)", type=["txt", "pdf"])

document_text = ""

if uploaded_file:
    if uploaded_file.name.endswith(".txt"):
        document_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            document_text = "\n".join([page.extract_text() or "" for page in pdf.pages])

elif text_input:
    document_text = text_input

# Show raw text
if document_text:
    st.subheader("📘 Extracted Document Content:")
    st.text_area("Document", value=document_text[:2000], height=200)

    if st.button("🧠 Generate Summary"):
        with st.spinner("Summarizing..."):
            summary = summarize_legal_text(document_text)
        st.success("✅ Summary Generated!")
        st.subheader("📝 Summary:")
        st.write(summary)
