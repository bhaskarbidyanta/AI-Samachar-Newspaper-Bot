import os
import streamlit as st
import speech_recognition as sr
from pymongo import MongoClient
from db import pdfs_collection
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from textblob import TextBlob
import emoji
import requests
import datetime
import PyPDF2
from components import show_navbar, show_footer
from utils import navbar, logout , page_buttons
# Load API key
load_dotenv()
google_api_key = st.secrets["GEMINI_API_KEY"]

# Ensure chat_history exists in session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize the conversational chain if it's not in session_state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None


if not google_api_key:
    st.error("❌ Error: Google API Key is missing. Set 'GEMINI_API_KEY' in your .env file.")
    st.stop()
# Model Configuration UI
st.sidebar.header("🛠️ Chatbot Settings")

# Model Selection
EMBEDDING_MODEL = "models/embedding-001"
model_options = ["gemini-1.5-pro", "gemini-1.5-flash"]
selected_model = st.sidebar.selectbox("🔍 Select Model:", model_options, index=0)

# Response Style (Mode Selection)
#response_modes = {
#    "🎭 Creative": {"temperature": 0.8, "max_output_tokens": 800},
#    "📜 Simple": {"temperature": 0.2, "max_output_tokens": 500},
#    "🧠 Intelligent": {"temperature": 0.5, "max_output_tokens": 1000},
#}
#selected_mode = st.sidebar.radio("🎨 Choose Response Style:", list(response_modes.keys()))

# Custom Temperature & Token Limit Control
#temperature = st.sidebar.slider("🔥 Temperature (Randomness)", 0.0, 1.0, response_modes[selected_mode]["temperature"])
#max_tokens = st.sidebar.slider("📝 Max Tokens", 100, 2000, response_modes[selected_mode]["max_output_tokens"])

st.title("📄 Multi-PDF Chatbot with Voice Commands")

# Fetch PDFs from MongoDB
pdfs = list(pdfs_collection.find({}, {"_id": 1, "filename": 1, "content": 1}))

if not pdfs:
    st.error("❌ No PDFs found in MongoDB. Please upload a PDF first.")
    st.stop()

pdf_list = {str(doc["_id"]): doc.get("filename", "Unnamed PDF") for doc in pdfs}
selected_pdf_ids = st.multiselect("📂 Choose PDFs", list(pdf_list.keys()), format_func=lambda x: pdf_list[x])

if st.button("📥 Load PDFs"):
    selected_pdfs = [doc for doc in pdfs if str(doc["_id"]) in selected_pdf_ids]
    
    if not selected_pdfs:
        st.error("❌ No valid PDFs selected.")
        st.stop()
    
    all_texts = []
    for pdf in selected_pdfs:
        if "content" in pdf and pdf["content"].strip():
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_text(pdf["content"])
            all_texts.extend(chunks)
    
    if not all_texts:
        st.error("❌ Error: Selected PDFs have empty or missing content.")
        st.stop()
    
    # Store text chunks in FAISS
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=google_api_key)
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_texts(all_texts, embeddings)
    
    # Setup conversational retrieval chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatGoogleGenerativeAI(model=selected_model, google_api_key=google_api_key),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    st.success("✅ PDFs loaded! You can now ask questions.")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.qa_chain = qa_chain  # Store chain in session
#Voice recognition function
def get_voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        try:
            audio = recognizer.listen(source,timeout=10)
            command = recognizer.recognize_google(audio)
            return command
        except sr.UnknownValueError:
            st.warning("Could not understand the voice input.")
        except sr.RequestError:
            st.warning("Speech recognition service is unavailable.")

# Chatbot interface
if "qa_chain" in st.session_state:
    col1,col2 = st.columns([3,1])

    with col1:
        user_input = st.text_input("💬 Ask a question about the PDFs:")

    with col2:
        if st.button("🎤 Use Voice"):
            voice_query = get_voice_command()
            if voice_query:
                user_input = voice_query
                st.text(f"You said: {voice_query}")
    
    options = {
        "📌 Summarize": "Summarize the content in a 10-15 sentences.",
        "Compare": "Compare the content in the selected pdfs and share difference between them in 5-10 sentences.",
        "Relevance": "Check the relevance of the content in the selected pdfs.",
    }
    
    selected_options = st.multiselect("📢 Choose topics to get updates:", list(options.keys()))

    if st.button("Get answer!"):
        if selected_options:
            for option in selected_options:
                query = options[option]
                response = st.session_state.qa_chain.run(query)
                st.session_state.chat_history.append((option, response))
        #st.write(f"**{option}:**", response)

        if user_input:
            response = st.session_state.qa_chain.run(user_input)
            st.session_state.chat_history.append((user_input, response))
    
    # Display chat history
    for question, answer in st.session_state.chat_history:
        st.write(f"**You:** {question}")
        st.write(f"**Bot:** {answer}")

    # Button for sentiment analysis
    if st.button("📊 Analyze Sentiment"):
        if st.session_state.chat_history:
            #sentiments = []
            #for _, answer in st.session_state.chat_history:
            #    sentiment_score = TextBlob(answer).sentiment.polarity
            #    sentiments.append(sentiment_score)
            latest_response = st.session_state.chat_history[-1][1]
            sentiment_score = TextBlob(latest_response).sentiment.polarity
            
            if sentiment_score > 0.1:
                sentiment_label = "😊 Positive"
            elif sentiment_score < -0.1:
                sentiment_label = "😟 Negative"
            else:
                sentiment_label = "😐 Neutral"
            
            st.subheader(f"🧠 Sentiment of Latest Response: {sentiment_label} ({sentiment_score:.2f})")
        else:
            st.warning("⚠️ No news updates found! Try fetching news first.")

if st.button("Logout"):
    st.session_state.clear()
    st.switch_page("mainapp.py")



def load_and_process_pdfs(download_dir, google_api_key, embedding_model, selected_model):
    """This method will load PDFs from the specified directory, extract text, and process them."""
    all_texts = []
    
    # Loop through each PDF file in the downloaded directory
    for pdf_file in os.listdir(download_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(download_dir, pdf_file)
            
            # Read the PDF and extract its content
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                extracted_text = []
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text.append(page_text)
                
                text = "\n".join(extracted_text)
                
                if text.strip():  # Only add if the PDF has content
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = text_splitter.split_text(text)
                    all_texts.extend(chunks)
                else:
                    st.warning(f"Warning: No text extracted from {pdf_file}. Skipping.")
    
    if not all_texts:
        st.error("❌ Error: No valid content found in the downloaded PDFs.")
        return None
    
    # Store the text chunks in FAISS
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=google_api_key)
    vectorstore = FAISS.from_texts(all_texts, embeddings)
    
    # Set up the conversational retrieval chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatGoogleGenerativeAI(model=selected_model, google_api_key=google_api_key),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    st.session_state.qa_chain = qa_chain  # Store in session
    st.success("✅ PDFs processed successfully! You can now ask questions.")

    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.qa_chain = qa_chain  # Store the QA chain in session

    return qa_chain

page_buttons()

#show_footer()
