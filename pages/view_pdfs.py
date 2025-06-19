import streamlit as st
import datetime
import os
from streamlit_pdf_viewer import pdf_viewer as show_pdf
import io
from utils import logout, navbar,page_buttons
#from pages.news import download_pdfs_from_site

#def main():
import os



# User inputs
selected_date = st.date_input("Select Date", datetime.date.today())
paper_type = st.selectbox("Select Paper Type", ["Main Paper", "NC Paper"])
page_number = st.number_input("Select Page Number", min_value=1, max_value=12, value=1, step=1)

# Format date and file path
formatted_date = selected_date.strftime("%Y-%m-%d")

if paper_type == "Main Paper":
    paper_code = "Mpage"
    pdf_file = f"news/{formatted_date}/downloaded_pdfs/Mpage_{page_number}.pdf"
else:
    paper_code = "NCpage"
    pdf_file = f"news/{formatted_date}/downloaded_pdfs_nc/NCpage_{page_number}.pdf"

if os.path.exists(pdf_file):
    with open(pdf_file, "rb") as f:
        local_pdf_bytes = f.read()
        show_pdf(input=local_pdf_bytes, width=700)
else:
    st.warning("PDF not found for selected date.")

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def show_online_pdf(pdf_url):
    st.sidebar.subheader("📄 Click below to view the PDF:")
    st.sidebar.markdown(f'[📄 View PDF Page {page_number}]({pdf_url})', unsafe_allow_html=True)


year = selected_date.year
month = str(selected_date.month).zfill(2)
day = str(selected_date.day).zfill(2)
base_url = "https://www.ehitavada.com/encyc/6"
# paper_code = "Mpage" if paper_type == "Main Paper" else "NCpage"

# # Construct full PDF URL

pdf_url = f"{base_url}/{year}/{month}/{day}/{paper_code}_{page_number}.pdf"

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# year = selected_date.year
# month = str(selected_date.month).zfill(2)
# day = str(selected_date.day).zfill(2)
# base_url = "https://www.ehitavada.com/encyc/6"
# paper_code = "Mpage" if paper_type == "Main Paper" else "NCpage"

# # Construct full PDF URL
# page_number = st.sidebar.number_input("📄 Page Number", min_value=1, max_value=12, step=1)
# pdf_url = f"{base_url}/{year}/{month}/{day}/{paper_code}_{page_number}.pdf"

st.write(f"📄 Viewing: {pdf_url}")
show_online_pdf(pdf_url)

# st.write(f"📄 Viewing: {pdf_url}")
# show_online_pdf(pdf_url)
if st.sidebar.button("Logout"):
    logout()

page_buttons()
