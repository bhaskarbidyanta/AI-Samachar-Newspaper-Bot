import streamlit as st
from urllib.parse import urlparse, parse_qs
from components import show_navbar, show_footer
#import pages.news as news
#import pages.chatbot as chatbot
#import pages.pdf_upload as pdf_upload
#import pages.Login_Signup as login_signup
#import pages.view_pdfs as view_pdfs
#import pages.home as home
from streamlit_option_menu import option_menu
from utils import navbar

st.set_page_config(page_title="AI Samachar", layout="wide")

# your page content here
#st.title("Welcome to AI Samachar - Home Page")

#if "selected_page" not in st.session_state:
#    st.session_state.selected_page = None  # default page

# Show navbar and update only if changed
#selected_page = show_navbar()



st.title("Welcome to AI Samachar 🗞️")
st.markdown("""
## AI-powered PDF & News Chatbot
Use the sidebar to navigate through the app:
- 📥 Upload PDFs
- 🤖 Chat with your files
- 📰 Read news
- 🔐 Login/Signup for personalized experience
""")



show_footer()