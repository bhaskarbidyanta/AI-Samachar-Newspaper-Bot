#utils.py
import smtplib
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
from dotenv import load_dotenv

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