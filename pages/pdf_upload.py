import PyPDF2
import datetime
import streamlit as st
from db import pdfs_collection  # Import MongoDB collection
from pdf2image import convert_from_path
import pytesseract
import tempfile
import os

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
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        
        images = convert_from_path(temp_file_path, dpi=300)
        extracted_text = []

        for image in images:
            text = pytesseract.image_to_string(image)
            if text.strip():
                extracted_text.append(text)
            
        os.remove(temp_file_path)  # Clean up the temporary file
        return "\n".join(extracted_text)
    
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