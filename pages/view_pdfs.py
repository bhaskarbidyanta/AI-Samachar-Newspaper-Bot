import streamlit as st
import datetime
import os
from streamlit_pdf_viewer import pdf_viewer as show_pdf
import io
def main():
    import os
    selected_date = st.date_input("Select a date", datetime.date.today())
    formatted_date = selected_date.strftime("%Y-%m-%d")

    
    pdf_file = f"news/{formatted_date}/downloaded_pdfs_nc/NCpage_3.pdf"

    if os.path.exists(pdf_file):
        with open(pdf_file, "rb") as f:
            local_pdf_bytes = f.read()
            show_pdf(input=local_pdf_bytes, width=700)
    else:
        st.warning("PDF not found for selected date.")

if __name__ == "__main__":
    main()