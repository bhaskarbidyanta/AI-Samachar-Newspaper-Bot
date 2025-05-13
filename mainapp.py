import streamlit as st
from urllib.parse import urlparse, parse_qs
from components import show_navbar, show_footer
import pages.news as news
import pages.chatbot as chatbot
import pages.pdf_upload as pdf_upload
import pages.Login_Signup as login_signup


st.set_page_config(page_title="AI Samachar", layout="wide")

# your page content here
#st.title("Welcome to AI Samachar - Home Page")

selected_page = show_navbar()

if selected_page == "Home":
    st.title("Welcome to AI Samachar - Home Page")
elif selected_page == "News":
    news.main()
elif selected_page == "Chatbot":
    chatbot.main()
elif selected_page == "Pdf Upload":
    pdf_upload.main()
elif selected_page == "Login-Signup":
    login_signup.main()
else:
    st.error("Page not found.")

show_footer()