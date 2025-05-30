import streamlit as st
import datetime
import os
from streamlit_pdf_viewer import pdf_viewer as show_pdf
import io
#from pages.news import download_pdfs_from_site

def main():
    import os
    # User inputs
    selected_date = st.date_input("Select Date", datetime.date.today())
    paper_type = st.selectbox("Select Paper Type", ["Main Paper", "NC Paper"])
    page_number = st.number_input("Select Page Number", min_value=1, max_value=12, value=1, step=1)

    # Format date and file path
    formatted_date = selected_date.strftime("%Y-%m-%d")

    if paper_type == "Main Paper":
        pdf_file = f"news/{formatted_date}/downloaded_pdfs/Mpage_{page_number}.pdf"
    else:
        pdf_file = f"news/{formatted_date}/downloaded_pdfs_nc/NCpage_{page_number}.pdf"

    if os.path.exists(pdf_file):
        with open(pdf_file, "rb") as f:
            local_pdf_bytes = f.read()
            show_pdf(input=local_pdf_bytes, width=700)
    else:
        st.warning("PDF not found for selected date.")

if __name__ == "__main__":
    main()