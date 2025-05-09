# components.py

import streamlit as st

def inject_custom_css():
    st.markdown("""
        <style>
            .custom-header {
                background-color: #004080;
                color: white;
                padding: 10px 20px;
                font-size: 20px;
                font-weight: bold;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .custom-header a {
                color: white;
                text-decoration: none;
                margin: 0 10px;
            }
            .custom-header a:hover {
                text-decoration: underline;
            }
            .custom-footer {
                background-color: #f1f1f1;
                text-align: center;
                padding: 10px;
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                font-size: 14px;
                color: #666;
            }
        </style>
    """, unsafe_allow_html=True)

def show_header():
    inject_custom_css()
    st.markdown("""
        <div class="custom-header">
            <div>📰 AI Samachar</div>
            <div>
                <a href="/news">News</a>
                <a href="/chatbot">Chatbot</a>
                <a href="/pdf_upload">Upload News</a>
                <a href="/Login_Signup">Login/Signup</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

def show_footer():
    st.markdown("""
        <div class="custom-footer">
            © 2025 AI Samachar | Built with ❤️ using Streamlit
        </div>
    """, unsafe_allow_html=True)
