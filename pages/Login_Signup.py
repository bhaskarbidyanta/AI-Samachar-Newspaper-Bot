import streamlit as st
import pymongo
from db import users_collection
from components import show_navbar, show_footer
import bcrypt

def main():
    #show_navbar()

    # Initialize session state variables if not set
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_role" not in st.session_state:
        st.session_state.user_role = None
    # Logout button

    # Handle automatic redirection after login
    if st.session_state.logged_in:
        if st.session_state.user_role == "admin":
            st.switch_page("pages/pdf_upload.py")
        else:
            st.switch_page("pages/chatbot.py")

    # Streamlit UI for login/signup
    st.title("Multi-PDF Chatbot")

    menu = st.sidebar.selectbox("Select an option", ["Login", "Sign Up"])

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
                st.rerun()
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

    #show_footer()