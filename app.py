
import sys
from pathlib import Path

import streamlit as st
import joblib

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
MODEL_PATH = PROJECT_DIR / "models" / "invoice_receipt_model.joblib"

sys.path.insert(0, str(SRC_DIR))

from extract_text import extract_text_from_pdf

st.set_page_config(
    page_title="PDF Document Classifier",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF Document Classification")
st.write(
    "Upload a PDF to extract its text and classify it "
    "as an Invoice or Receipt."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:
    try:
        pdf_bytes = uploaded_file.getvalue()

        with st.spinner("Extracting text from PDF..."):
            extracted_text, extraction_methods = (
                extract_text_from_pdf(pdf_bytes)
            )

        st.subheader("Extracted Text")
        st.text_area(
            "Text extracted from your PDF",
            extracted_text,
            height=250
        )

        st.caption(
            "Extraction methods: " + ", ".join(extraction_methods)
        )

        if st.button("Classify Document"):
            model = load_model()
            prediction = model.predict([extracted_text])[0]

            st.subheader("ML Model Prediction")
            st.success(f"Predicted category: {prediction}")

            text_lower = extracted_text.lower()
            bill_keywords = [
                "electricity bill",
                "utility bill",
                "statement of account",
                "amount due",
                "billing period",
                "meter reading",
                "previous balance",
                "current charges",
                "kwh",
                "due date"
            ]

            matched_keywords = [
                keyword for keyword in bill_keywords
                if keyword in text_lower
            ]

            st.subheader("Additional Document Check")

            if len(matched_keywords) >= 2:
                st.warning(
                    "Possible Bill — needs verification. "
                    "This is a rule-based check, not an ML prediction."
                )
                st.caption(
                    "Matching terms: " + ", ".join(matched_keywords)
                )
            else:
                st.info(
                    "No strong Bill keyword signal detected. "
                    "The ML model currently supports Invoice and Receipt only."
                )

    except Exception as error:
        st.error(
            "Unable to process this PDF. Please upload a valid, "
            "readable PDF and try again."
        )
        st.caption(f"Technical details: {error}")

