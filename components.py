# components.py

import streamlit as st
from streamlit_community_navigation_bar import st_navbar

def show_navbar():
    pages = ["Home","News","Chatbot","Pdf Upload","Login-Signup","GitHub","View NewsPaper"]
    url = {
        "GitHub":"https://github.com/bhaskarbidyanta/AI-Samachar-Newspaper-Bot"
    }
    styles = {
        "nav": {"background-color": "#004080"},
        "div": {"max-width": "36rem"},
        "span": {
            "border-radius": "0.5rem",
            "color": "white",
            "margin": "0 0.25rem",
            "padding": "0.5rem 0.75rem",
        },
        "active": {"background-color": "rgba(255, 255, 255, 0.25)"},
        "hover": {"background-color": "rgba(255, 255, 255, 0.35)"},
    }
    selected = st_navbar(
        pages,
        urls=url,
        styles=styles,
    )
    return selected

def show_footer():
    st.markdown("""
        <div class="custom-footer">
            © 2025 AI Samachar | Built with ❤️ using Streamlit
        </div>
    """, unsafe_allow_html=True)


