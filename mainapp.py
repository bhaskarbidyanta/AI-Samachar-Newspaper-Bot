import streamlit as st
from urllib.parse import urlparse, parse_qs
from components import show_navbar, show_footer
import pages.news as news
import pages.chatbot as chatbot
import pages.pdf_upload as pdf_upload
import pages.Login_Signup as login_signup
import pages.view_pdfs as view_pdfs


st.set_page_config(page_title="AI Samachar", layout="wide")

# your page content here
#st.title("Welcome to AI Samachar - Home Page")

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"  # default page

# Show navbar and update only if changed
selected_page = show_navbar()


if selected_page == "Home":
    st.title("Welcome to AI Samachar - Home Page")
    st.header("📚 Project Info & API Overview")

    st.write(
        """
        This application is a multi-language conversational chatbot built using **Streamlit** and **Gemini Pro (Google Generative AI)**. It allows users to upload PDFs, extract their contents, ask questions, and receive answers in **English**, **Hindi**, or **Marathi**.

        ### 🧠 Core Features
        - PDF upload and text extraction via `PyPDF2`
        - Multilingual support with automatic language detection and translation
        - Question-answering using `ConversationalRetrievalChain` and `FAISS` for semantic search
        - Sentiment analysis of responses using `TextBlob`
        - Audio playback of bot responses using `gTTS`
        - Session-based chat history

        ### 🔄 Tech Stack
        - `Streamlit` for frontend UI
        - `PyPDF2` for PDF reading
        - `Google Generative AI` via `langchain` for chat-based Q&A
        - `langdetect` for language detection
        - `gTTS` for text-to-speech
        - `TextBlob` for sentiment analysis
        - `FAISS` for fast vector-based search
        - `MongoDB` (optional) for backend storage (if enabled)

        ### 🧾 How It Works
        1. Upload one or more PDFs using the UI.
        2. The app extracts text from each page and splits it into chunks.
        3. Chunks are embedded using `GoogleGenerativeAIEmbeddings` and stored in FAISS.
        4. You can ask questions based on the uploaded content.
        5. The response is translated into your preferred language (English, Hindi, Marathi).
        6. You can analyze the sentiment and even play the response as audio.

        ### 🗣️ Languages Supported
        - English (`en`)
        - Hindi (`hi`)
        - Marathi (`mr`)

        ### 🔈 Audio Generation
        The last response can be converted to audio using `gTTS` and played directly in-browser. This helps visually impaired users or those who prefer listening.

        ### 🧪 Optional Add-ons
        - Support for image-based PDFs via `pytesseract` (OCR)
        - User authentication and MongoDB-backed storage

        ---
        **Note**: All processing is done live in memory for Streamlit Cloud compatibility. No sensitive data is stored.
        """
    )
    st.sidebar.write("### Steps For Use")
    st.sidebar.write(
        """
        1. **News Bot**: Use the "News" page to access your Newspaper Bot.
        2. **Select Type of Newspaper**: Navigate to the "Chatbot" page to ask questions based on the uploaded content.
        3. **Select Date**: Check the "News" page for the latest updates.
        4. **Click on Load Downloaded PDFs**: Use the "Login-Signup" page for user authentication (if enabled).
        5. **Select Language**: Select the language of your choice (English, Hindi, Marathi) for the bot's responses.
        6. **Ask Questions**: Type your questions or select options in the chat interface and receive answers in your selected language.
        7. **Audio Playback**: Click the audio button to listen to the bot's response.
        8. **Analyze Sentiment**: Analyze the sentiment of the bot's response using the built-in sentiment analysis feature.
        9. **Detect Bias**: Use the bias detection feature to identify any potential biases in the bot's responses.
        10. **View PDFs**: View newspaper articles in PDF format.
        """
    )

elif selected_page == "News":
    news.main()
elif selected_page == "Chatbot":
    chatbot.main()
elif selected_page == "Pdf Upload":
    pdf_upload.main()
elif selected_page == "Login-Signup":
    login_signup.main()
elif selected_page == "View NewsPaper":
    view_pdfs.main()
else:
    st.error("Page not found.")

show_footer()