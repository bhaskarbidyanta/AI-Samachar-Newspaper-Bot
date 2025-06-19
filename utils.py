#utils.py
import datetime
import streamlit as st
import smtplib
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

def generate_otp():

    return str(random.randint(100000, 999999))

def send_otp_email(receiver_email, otp):
    load_dotenv()
    sender_email = st.secrets["SENDER_EMAIL"]  # Replace with your sender email
    sender_password = st.secrets["SENDER_PASSWORD"]  # Use App Password if 2FA is enabled

    subject = "AI Samachar Password Reset OTP"
    body = f"Your OTP for resetting your password is: {otp}"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print("Error sending email:", e)
        return False
    
def logout():
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.otp_sent = False
    st.session_state.otp_value = None
    st.session_state.otp_email = None
    st.session_state.otp_verified = False
    st.success("Logged out successfully!")
    #st.clear_session()    
    st.switch_page("pages/Login_Signup.py")

def navbar():
    selected_page = option_menu(
        menu_title="AI Samachar",
        options=["HomePage","Home", "News", "Chatbot", "Pdf Upload", "Login-Signup", "View NewsPaper"],
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
    return selected_page

def page_buttons():
    # Custom CSS for uniform button width and spacing
    st.markdown("""
        <style>
            .nav-button {
                width: 130px;
                margin: 0 auto;
                display: flex;
                justify-content: center;
                
            }
        </style>
    """, unsafe_allow_html=True)

    # Create six evenly spaced columns
    cols = st.columns(6)

    buttons = [
        ("Home", "mainapp.py"),
        ("News", "pages/News.py"),
        ("Chatbot", "pages/chatbot.py"),
        ("Upload PDF", "pages/pdf_upload.py"),
        ("Login", "pages/Login_Signup.py"),
        ("View NewsPapers", "pages/view_pdfs.py")
    ]

    for i, (label, target) in enumerate(buttons):
        with cols[i]:
            if st.button(label, key=f"navbtn_{label}", help=f"Go to {label}", type="secondary"):
                st.session_state.page = label
                st.switch_page(target)

    # Add GitHub icon-style button centered
    # st.markdown("<br>", unsafe_allow_html=True)
    # st.markdown(
    #     '<div style="text-align:center;">'
    #     '<a href="https://github.com/bhaskarbidyanta/AI-Samachar-Newspaper-Bot" target="_blank">'
    #     '<button class="nav-button">🌐 Project GitHub</button>'
    #     '</a>'
    #     '</div>',
    #     unsafe_allow_html=True
    #)


def show_github_button():
    if st.button("🔗 GitHub"):
        st.components.v1.html(
            """<script>
            window.open("https://github.com/bhaskarbidyanta/AI-Samachar-Newspaper-Bot", "_blank");
            </script>""",
            height=0
        )


def show_footer():
    year = datetime.datetime.now().year

    st.markdown("""
        <style>
        .fixed-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f0d0ce;
            color: #050000;
            text-align: center;
            padding: 10px 0;
            font-size: 0.85em;
            z-index: 9999;
            border-top: 1px solid #333;
        }

        /* Avoid footer overlapping page content */
        .stApp {
            padding-bottom: 50px; /* height of footer */
        }

        a.footer-link {
            color: black;
            text-decoration: none;
            margin: 0 8px;
        }

        a.footer-link:hover {
            color: #fff;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="fixed-footer">
            © {year} AI Samachar · Built by 
            <a href="https://github.com/bhaskarbidyanta" target="_blank" class="footer-link">Bhaskar Bidyanta</a> |
            <a href="https://github.com/bhaskarbidyanta/AI-Samachar-Newspaper-Bot" target="_blank" class="footer-link">GitHub</a> |
            <a href="https://www.linkedin.com/in/bhaskarbidyanta/" target="_blank" class="footer-link">LinkedIn</a>
        </div>
    """, unsafe_allow_html=True)


                