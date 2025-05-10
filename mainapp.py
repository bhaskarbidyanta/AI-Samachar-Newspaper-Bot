import streamlit as st
from urllib.parse import urlparse, parse_qs
from components import show_header, show_footer

st.set_page_config(page_title="AI Samachar", layout="wide")
show_header()

# your page content here
st.title("Welcome to AI Samachar - Home Page")

def jump_to_page(page_name):
    st.switch_page(f"pages/{page_name}.py")

col1, col2 = st.columns([1, 1])
col3, col4 = st.columns([1, 1])
with col1:
    st.button("News", on_click=jump_to_page("news"), args=("news",))
with col2:
    st.button("Chatbot", on_click=jump_to_page, args=("chatbot",))
with col3:
    st.button("Upload News", on_click=jump_to_page, args=("pdf_upload",))  
with col4:
    st.button("Login/Signup", on_click=jump_to_page, args=("Login_Signup",))
show_footer()