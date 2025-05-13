# components.py

import streamlit as st
from streamlit_community_navigation_bar import st_navbar

# def inject_custom_css():
#     st.markdown("""
#         <style>
#             .custom-header {
#                 background-color: #004080;
#                 color: white;
#                 padding: 10px 20px;
#                 font-size: 20px;
#                 font-weight: bold;
#                 display: flex;
#                 justify-content: space-between;
#                 align-items: center;
#             }
#             .custom-header a {
#                 color: white;
#                 text-decoration: none;
#                 margin: 0 10px;
#             }
#             .custom-header a:hover {
#                 text-decoration: underline;
#             }
#             .custom-footer {
#                 background-color: #f1f1f1;
#                 text-align: center;
#                 padding: 10px;
#                 position: fixed;
#                 bottom: 0;
#                 left: 0;
#                 right: 0;
#                 font-size: 14px;
#                 color: #666;
#             }
#         </style>
#     """, unsafe_allow_html=True)

# def show_header():
#     inject_custom_css()
#     st.markdown("""
#         <style>
#             .custom-header {
#                 background-color: #004080;
#                 color: white;
#                 padding: 12px 24px;
#                 display: flex;
#                 justify-content: space-between;
#                 align-items: center;
#                 border-bottom: 2px solid #ccc;
#             }

#             .custom-header a {
#                 color: white;
#                 text-decoration: none;
#                 margin: 0 12px;
#                 font-weight: bold;
#                 transition: color 0.2s ease;
#             }

#             .custom-header a:hover {
#                 color: #ffcc00;
#                 text-decoration: underline;
#             }
#         </style>
#         <div class="custom-header">
#             <div>📰 AI Samachar</div>
#             <div>
#                 <a href="/news" target="_self">News</a>
#                 <a href="/chatbot" target="_self">Chatbot</a>
#                 <a href="/pdf_upload" target="_self">Upload News</a>
#                 <a href="/Login_Signup" target="_self">Login/Signup</a>
#             </div>
#         </div>
#     """, unsafe_allow_html=True)

def show_navbar():
    pages = ["Home","News","Chatbot","Pdf Upload","Login-Signup","GitHub"]
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


