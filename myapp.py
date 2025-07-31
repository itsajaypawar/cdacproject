import streamlit as st
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return tokenizer, model

tokenizer, model = load_model()

# Summarization function
def summarize_text(text, max_input_length=1024, max_output_length=150):
    inputs = tokenizer.batch_encode_plus(
        [text],
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt"
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        length_penalty=2.0,
        max_length=max_output_length,
        min_length=20,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit UI
st.title("📄 Legal Document Summarizer (Indian Constitution)")
st.write("Upload a CSV file with legal text (e.g., Constitution articles) and get concise summaries.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "Information" not in df.columns:
        st.error("CSV must contain an 'Information' column with legal text.")
    else:
        st.write("📘 Preview of Uploaded Data:")
        st.dataframe(df.head())

        if st.button("🔍 Generate Summaries"):
            with st.spinner("Summarizing..."):
                df["Summary"] = df["Information"].apply(summarize_text)

            st.success("✅ Summarization complete!")
            st.dataframe(df[["Article", "Name", "Summary"]])

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Summarized CSV", data=csv, file_name="legal_summaries.csv", mime="text/csv")
