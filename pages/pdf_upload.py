import PyPDF2
import datetime
import streamlit as st
from db import pdfs_collection  # Import MongoDB collection

st.title("Upload PDFs")

# ✅ Check if user is logged in and an admin
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please log in as an admin.")
    st.stop()

if st.session_state.get("user_role") != "admin":
    st.error("Access Denied! Only admins can upload PDFs.")
    st.stop()

# ✅ File uploader
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        reader = PyPDF2.PdfReader(file)
        extracted_text = []

        # ✅ Extract text from each page safely
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:  # Only add non-empty text
                extracted_text.append(page_text)

        text = "\n".join(extracted_text)

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
        pdf_data = {
            "filename": file.name,
            "content": text,
            "uploaded_at": datetime.datetime.utcnow()
        }
        result=pdfs_collection.insert_one(pdf_data)

    st.success(f"Uploaded: {file.name} (ID: {result.inserted_id})")

    
# Logout Button
if st.button("Logout"):
    st.session_state.clear()
    st.switch_page("mainapp.py")