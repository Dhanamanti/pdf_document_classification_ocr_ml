# pdf_document_classification_ocr_ml
PDF document classification using OCR, TF-IDF, Logistic Regression, and Streamlit, with preliminary Bill keyword screening.
# PDF Document Classification Using OCR and Machine Learning

## Project Overview

This project extracts text from PDF documents and uses Machine Learning to classify documents as **Invoice** or **Receipt**.

It uses PyMuPDF for text extraction from text-based PDFs and Tesseract OCR as a fallback for scanned PDFs. A Streamlit web application allows users to upload PDFs, view extracted text, and obtain a prediction.

A separate keyword-based check flags possible bills for verification. Bill classification is not yet supported by the trained ML model.

## Features

- Upload PDF documents through a Streamlit interface.
- Extract text from selectable-text PDFs.
- Use OCR when a page contains insufficient selectable text.
- Classify documents as Invoice or Receipt.
- Display extracted text and extraction methods.
- Flag possible bills using a preliminary keyword-based check.
- Handle unreadable or invalid PDFs with an error message.

## Technologies Used

- Python
- PyMuPDF
- Tesseract OCR
- Pytesseract and Pillow
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit
- Joblib

## Machine Learning Workflow

1. Collect and extract document text.
2. Prepare and inspect the text dataset.
3. Split the labeled data into training and testing sets.
4. Convert text into numerical features using TF-IDF.
5. Train a Logistic Regression classifier.
6. Evaluate the model on the held-out test split.
7. Save the trained pipeline for use in the Streamlit application.

## Preliminary Evaluation

The current Invoice-versus-Receipt model achieved **100% accuracy** on its held-out test split.

- Training samples: 614
- Test samples: 154
- Invoice test samples: 13
- Receipt test samples: 141

This result is preliminary and should not be interpreted as proof of generalization to unseen sources. Further testing on independent documents and checks for source or template leakage are needed.

## Current Limitations

- The trained ML model predicts only Invoice or Receipt.
- Bill detection is a keyword-based screening step, not a trained three-class classifier.
- The available Bill examples are insufficient for reliable supervised training.
- Some candidate Invoice labels still require manual verification.
- OCR accuracy can decrease for blurry, rotated, or low-quality scans.
- Independent testing across different document sources is still needed.

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Tesseract OCR must also be installed as a system application. On Debian or Ubuntu, use:

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

## Run the Application

From the project root directory, run:

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit in your browser.

## Project Structure

```text
PDF_Document_Classification_Project/
├── app.py
├── requirements.txt
├── src/
│   └── extract_text.py
├── models/
│   └── invoice_receipt_model.joblib
├── dataset/
│   ├── raw/
│   └── processed/
└── reports/
```

The repository structure may be updated as additional training scripts, evaluation reports, and documentation are finalized.

## Dataset and Permissions

Dataset sources and usage permissions should be documented before redistribution. Do not upload confidential documents or redistribute datasets unless their licenses permit it. See the project’s dataset source documentation for source-specific details.

## Future Improvements

- Collect and verify more Invoice, Bill, and Receipt documents.
- Train and evaluate a genuine three-class classifier.
- Test OCR on scanned PDFs with varying image quality.
- Evaluate performance on independent document sources.
- Add confidence-aware or inconclusive prediction handling.
- Expand the evaluation report with class-wise precision, recall, F1-score, and a confusion matrix.

## Project Status

The Invoice-versus-Receipt prototype and basic PDF upload interface are implemented. Bill classification and broader independent validation remain future improvements.
