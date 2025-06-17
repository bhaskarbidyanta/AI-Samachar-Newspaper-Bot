import streamlit as st
import pymongo
from db import users_collection
from components import show_navbar, show_footer
import bcrypt
from utils import send_otp_email,generate_otp
#import pages.pdf_upload as pdf_upload
#import pages.chatbot as chatbot

#def main():
#show_navbar()

# Initialize session state variables if not set
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "otp_sent" not in st.session_state:
    st.session_state.otp_sent = False
if "otp_value" not in st.session_state:
    st.session_state.otp_value = None
if "otp_email" not in st.session_state:
    st.session_state.otp_email = None
if "otp_verified" not in st.session_state:
    st.session_state.otp_verified = False

# Logout button

# Handle automatic redirection after login
if st.session_state.logged_in:
    if st.session_state.user_role == "admin":
        #pdf_upload.main()
        #st.session_state.selected_page = "Pdf Upload"
        st.switch_page("pages/pdf_upload.py")
    else:
        #st.session_state.selected_page = "View NewsPaper"
        #chatbot.main()
        st.switch_page("pages/view_pdfs.py")
    #return  
# Streamlit UI for login/signup
st.title("AI Samachar")

menu = st.sidebar.selectbox("Select an option", ["Login", "Sign Up","Forgot Password"])

if menu == "Login":
    st.subheader("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = users_collection.find_one({"email": email})
        if user and bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
            st.session_state.logged_in = True
            st.session_state.user_role = user["role"]
            st.success("Login successful! Redirecting...")
            #st.rerun()
            if user['role'] == 'admin':
                #st.session_state.selected_page = "Pdf Upload"
                st.switch_page("pages/pdf_upload.py")
            #      pdf_upload.main()
            else:
                #st.session_state.selected_page = "View NewsPaper"
                st.switch_page("pages/view_pdfs.py")
            # #     chatbot.main()
            #return
        else:
            st.error("Invalid email or password!")


elif menu == "Sign Up":
    st.subheader("Sign Up")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    #role = st.selectbox("Role", ["user", "admin"])
    if st.button("Sign Up"):
        if users_collection.find_one({"email": email}):
            st.error("Email already registered!")
        else:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            users_collection.insert_one({"email": email, "password": hashed_password.decode('utf-8'), "role": "user"})  # or role

            st.success("Account created! Please login.")
    #This is a test line

elif menu == "Forgot Password":
    st.subheader("Reset Password")
    

    if not st.session_state.otp_sent:
        email = st.text_input("Enter your registered email")
        if st.button("Send OTP"):
            user = users_collection.find_one({"email":email})
            if user:
                otp = generate_otp()
                if send_otp_email(email, otp):
                    st.session_state.otp_sent = True
                    st.session_state.otp_value = otp
                    st.session_state.otp_email = email
                    st.success("OTP sent to your email!")
                else:
                    st.error("Failed to send OTP.Try again.")
            else:
                st.error("No account found with this email.")
    elif not st.session_state.otp_verified:
        otp_input = st.text_input("Enter the OTP sent to your email")
        if st.button("Verify OTP"):
            if otp_input == st.session_state.otp_value:
                st.session_state.otp_verified = True
                st.success("OTP verified! You can now reset your password.")
            else:
                st.error("Invlid OTP! Please try again.")
    else:
        new_password = st.text_input("Enter new password", type="password")
        confirm_password = st.text_input("Confirm new password", type="password")
        if st.button("Reset Password"):
            if new_password == confirm_password:
                hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
                users_collection.update_one(
                    {"email": st.session_state.otp_email},
                    {"$set": {"password": hashed.decode('utf-8')}}
                )
                st.success("Password reset successful! You can now login.")
                st.session_state.otp_sent = False
                st.session_state.otp_value = None
                st.session_state.otp_email = None
                st.session_state.otp_verified = False
            else:
                st.error("Passwords do not match! Please try again.")

  # Redirect to login page
    


  #show_footer()
