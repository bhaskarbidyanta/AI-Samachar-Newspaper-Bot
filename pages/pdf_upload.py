import PyPDF2
import datetime
import streamlit as st
from db import pdfs_collection  # Import MongoDB collection
from pdf2image import convert_from_path
import pytesseract
import pdfplumber
import os
from PIL import Image
import io

st.title("Upload PDFs")

# ✅ Check if user is logged in and an admin
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please log in as an admin.")
    st.stop()

if st.session_state.get("user_role") != "admin":
    st.error("Access Denied! Only admins can upload PDFs.")
    st.stop()

def extract_text_pypdf2(file):
    try:
        reader = PyPDF2.PdfReader(file)
        extracted_text = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text.append(page_text)
        return "\n".join(extracted_text)
    
    except Exception as e:
        st.error(f"Error extracting text with PyPDF2: {str(e)}")
        return ""
    
def extract_text_ocr(file):
    try:
        extracted_text = []

        with pdfplumber.open(file) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                # Extract images from each page
                for image in page.images:
                    # Extract image bytes
                    image_obj = pdf.pages[page_number - 1].to_image()
                    pil_image = image_obj.original_image

                    # Convert to RGB (required for pytesseract)
                    pil_image = pil_image.convert("RGB")

                    # Extract text from the image
                    text = pytesseract.image_to_string(pil_image)

                    if text.strip():
                        extracted_text.append(f"Page {page_number}:\n{text}")
        
        if extracted_text:
            return "\n".join(extracted_text)
        else:
            return ""  # Return empty string if no text found

    except Exception as e:
        st.error(f"Error extracting text with OCR: {str(e)}")
        return ""

# ✅ File uploader
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        # First try PyPDF2 extraction
        text = extract_text_pypdf2(file)

        # If PyPDF2 fails, try OCR extraction
        if not text.strip():
            st.warning(f"No text found with PyPDF2 in {file.name}. Trying OCR extraction...")
            text = extract_text_ocr(file)

        # ✅ Skip insertion if no text was extracted
        if not text.strip():
            st.warning(f"Warning: No text extracted from {file.name}. Skipping upload.")
            continue

        # ✅ Check for duplicate filenames before inserting
        existing_file = pdfs_collection.find_one({"filename": file.name})
        if existing_file:
            st.warning(f"Warning: A file with the name '{file.name}' already exists. Skipping upload.")
            continue

        # ✅ Insert into MongoDB
        try:
            pdf_data = {
                "filename": file.name,
                "content": text,
                "uploaded_at": datetime.datetime.utcnow()
            }
            result=pdfs_collection.insert_one(pdf_data)
            st.success(f"Uploaded: {file.name} (ID: {result.inserted_id})")
        except Exception as e:
            st.error(f"Failed to upload {file.name}: {str(e)}")

    #st.success(f"Uploaded: {file.name} (ID: {result.inserted_id})")

    
# Logout Button
if st.button("Logout"):
    st.session_state.clear()
    st.switch_page("mainapp.py")