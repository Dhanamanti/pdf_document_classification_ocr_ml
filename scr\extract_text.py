
import io
import pymupdf
import pytesseract

from PIL import Image


def extract_text_from_pdf(pdf_bytes):
    """
    Extract selectable PDF text first.
    If a page has no selectable text, use Tesseract OCR.
    Returns extracted text and extraction details.
    """
    all_text = []
    methods = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text().strip()

            if len(text) >= 20:
                methods.append(f"Page {page_number}: PyMuPDF")
            else:
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False
                )

                image = Image.open(
                    io.BytesIO(pix.tobytes("png"))
                )

                text = pytesseract.image_to_string(
                    image,
                    lang="eng"
                ).strip()

                methods.append(f"Page {page_number}: Tesseract OCR")

            all_text.append(text)

    extracted_text = "\n\n".join(all_text).strip()

    if not extracted_text:
        raise ValueError(
            "No text could be extracted. Try a clearer PDF."
        )

    return extracted_text, methods

