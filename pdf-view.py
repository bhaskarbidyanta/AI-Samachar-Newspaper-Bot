#E:\bhaskar\users\Project_ML\Multi-file-chatbot\news\2025-05-29\downloaded_pdfs_nc\NCpage_3.pdf
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import io # To handle binary data

st.title("PDF Viewer in Streamlit")

# uploaded_file = st.file_uploader("Upload a PDF", type=("pdf"))

# if uploaded_file is not None:
#     # Read the uploaded file as bytes
#     pdf_bytes = uploaded_file.getvalue()

#     # Display the PDF using the component
#     pdf_viewer(input=pdf_bytes, width=700) # You can adjust width/height
#     st.success("PDF displayed successfully!")
# else:
#     st.info("Please upload a PDF file to view it.")

# You can also display a PDF from a local file path (not recommended for cloud deployment)
# For local testing:
try:
    with open("E:/bhaskar/users/Project_ML/Multi-file-chatbot/news/2025-05-29/downloaded_pdfs_nc/NCpage_3.pdf", "rb") as f:
        local_pdf_bytes = f.read()
    st.subheader("Local PDF (for testing)")
    pdf_viewer(input=local_pdf_bytes, width=700)
except FileNotFoundError:
    st.warning("my_document.pdf not found for local display.")