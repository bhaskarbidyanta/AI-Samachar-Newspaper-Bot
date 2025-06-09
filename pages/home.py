import streamlit as st
# import pages.news as news
# import pages.chatbot as chatbot
# import pages.pdf_upload as pdf_upload
# import pages.Login_Signup as login_signup
# import pages.view_pdfs as view_pdfs
# import pages.home as home
from streamlit_option_menu import option_menu

selected_page = option_menu(
    menu_title="AI Samachar",
    options=["Home", "News", "Chatbot", "Pdf Upload", "Login-Signup", "View NewsPaper"],
    icons=["house", "newspaper", "chat", "file-upload", "person", "eye"],
    menu_icon="cast",
    default_index=0,
    styles={
        "container": {"padding": "5!important"},
        "icon": {"color": "#ffffff"},
        "nav-link": {"font-size": "16px"},
        "nav-link-selected": {"background-color": "#004080"},
    },
    orientation="horizontal",
)

st.write("Hello")
if selected_page == "Home":
    st.switch_page("pages/home.py")
elif selected_page == "News":
    st.switch_page("pages/news.py")
elif selected_page == "Chatbot":
    st.switch_page("pages/chatbot.py")
elif selected_page == "Pdf Upload":
    st.switch_page("pages/pdf_upload.py")
elif selected_page == "Login-Signup":
    st.switch_page("pages/Login_Signup.py")
elif selected_page == "View NewsPaper":
    st.switch_page("pages/view_pdfs.py")
else:
    st.error("Page not found.")