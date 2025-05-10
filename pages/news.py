import os
import streamlit as st
import speech_recognition as sr
from pymongo import MongoClient
from db import pdfs_collection
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from textblob import TextBlob
import emoji
import requests
import datetime
import PyPDF2
from textblob import TextBlob
import numpy as np
#import re
from transformers import pipeline
from components import inject_custom_css, show_header,show_footer
from pathlib import Path

show_header()

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

st.title("Newspaper PDF Chatbot")

paper_type = st.radio("Select Paper Type",["Main Paper","Nagpur CityLine"])

selected_date = st.date_input("Select Date",datetime.datetime.now())

def download_pdfs_from_site(base_url, paper_code, selected_date):
    # Get the current date to create folders
    formatted_date = selected_date.strftime("%Y-%m-%d")
    download_dir_main = os.path.join("news", formatted_date, "downloaded_pdfs")
    download_dir_nc = os.path.join("news", formatted_date, "downloaded_pdfs_nc")
    year = selected_date.year
    month = str(selected_date.month).zfill(2)
    day = str(selected_date.day).zfill(2)
    
    # Create the folders if they don't exist
    os.makedirs(download_dir_main, exist_ok=True)
    os.makedirs(download_dir_nc, exist_ok=True)
    
    page_number = 1  # Start from page 1
    
    # Set page limits
    if paper_code == "Mpage":
        max_pages = 12
    else:
        max_pages = 8

    while page_number <= max_pages:
        # Construct the URL for the current page
        url = f"{base_url}/{year}/{month}/{day}/{paper_code}_{page_number}.pdf"
        response = requests.get(url)
        
        # If the URL returns a 404 error, stop downloading
        if response.status_code == 404:
            st.warning(f"Stopped downloading. {url} returned a 404 error.")
            break
        
        # Save the PDF to the appropriate directory
        if paper_code == "Mpage":
            pdf_filename_main = os.path.join(download_dir_main, f"{paper_code}_{page_number}.pdf")
            with open(pdf_filename_main, 'wb') as file:
                file.write(response.content)
            #st.success(f"Downloaded: {pdf_filename_main}")
        else:
            pdf_filename_nc = os.path.join(download_dir_nc, f"{paper_code}_{page_number}.pdf")
            with open(pdf_filename_nc, 'wb') as file:
                file.write(response.content)
            #st.success(f"Downloaded: {pdf_filename_nc}")
        
        page_number += 1  # Move to the next page
        if page_number == max_pages:
            st.success(f"Downloaded all {max_pages} pages of {paper_code}.")
            break



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

#sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Political Bias Keywords (Extendable)
left_bias_words = [
    "social justice", "climate change", "income inequality", "progressive",
    "universal healthcare", "gun control", "racial equity", "minimum wage","Congress",
    "Aam Aadmi Party", "AAP", "Indian National Congress",
]

right_bias_words = [
    "tax cuts", "border security", "traditional values", "free market",
    "law and order", "second amendment", "private enterprise", "fiscal responsibility","BJP","Bharatiya Janata Party",
]

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define Bias Labels
bias_labels = ["Left-Leaning", "Right-Leaning", "Neutral"]

def detect_bias(text):
    """Detect bias using Zero-Shot Classification"""
    result = classifier(text, bias_labels)
    scores = {label: score for label, score in zip(result['labels'], result['scores'])}
    detected_label = result['labels'][0]  # Label with highest score
    return detected_label, scores


# Usage in your Streamlit code
if st.button("📥 Load Downloaded PDFs"):
    base_url = "https://www.ehitavada.com/encyc/6"
    paper_code = "Mpage" if paper_type == "Main Paper" else "NCpage"
    try:
        # Call the function to download PDFs starting from mpage_1
        download_pdfs_from_site(base_url,paper_code,selected_date)
    except Exception as e:
        st.error(f"An error occurred: {e}")

    formatted_date = selected_date.strftime("%Y-%m-%d")
    
    if paper_type == "Main Paper":
        download_dir = os.path.join("news", formatted_date, "downloaded_pdfs")
    else:
        download_dir = os.path.join("news", formatted_date, "downloaded_pdfs_nc")
    
    if os.path.exists(download_dir) and os.listdir(download_dir):
        qa_chain = load_and_process_pdfs(download_dir, google_api_key, EMBEDDING_MODEL, selected_model)
        
        if qa_chain:
            st.session_state.qa_chain = qa_chain
    else:
        st.error(f"❌ No PDFs found in {download_dir}. Please download PDFs first.")


    
options = {
    "All News": "Get all the news headlines mentioned in these pdfs.",
    "📌 Summarize": "Summarize all the news in these PDFS .",
    "⚽ Sports News": "Give me the sports news in these pdf.",
    "🌍 International News": "Provide me with the mentioned international news in these pdf.",
    "🇮🇳 National News": "Show me the latest national news in India mentioned in these pdfs.",
    "🏙️ City News": "What are the latest updates in my city mentioned in these pdfs?",
    "💼 Jobs": "List all job openings mentioned in these pdfs.",
    "🚔 Crime News": "Provide recent crime news updates provided in these pdfs.",
    "🏞️ Weather News":"Today's weather updates in these newspaper pdfs.",
}
    
selected_options = st.multiselect("📢 Choose topics to get updates:", list(options.keys()))

if st.button("Get answer!") and selected_options:
    for option in selected_options:
        query = options[option]
        response = st.session_state.qa_chain.run(query)
        st.session_state.chat_history.append((option, response))
        #st.write(f"**{option}:**", response)

#if user_input:
#    response = st.session_state.qa_chain.run(user_input)
#    st.session_state.chat_history.append((user_input, response))
    
# Display chat history
for question, answer in st.session_state.chat_history:
    st.write(f"**You:** {question}")
    st.write(f"**Bot:** {answer}")

from googletrans import Translator

translator = Translator()
language = st.sidebar.selectbox("🌐 Select Language:", ["English", "Hindi", "Marathi"])
def translate_text(text, target_language):
    if target_language == "Hindi":
        translated = translator.translate(text, dest='hi')
    elif target_language == "Marathi":
        translated = translator.translate(text, dest='mr')
    else:
        return text  # Return original text if language is English
    return translated.text

col1, col2, col3 = st.columns([1, 1, 2])

# Button for sentiment analysis
with col1:
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
with col2:
    if st.button("📊 Analyze Bias"):
        if st.session_state.qa_chain:
            retrieved_docs = st.session_state.qa_chain.retriever.get_relevant_documents("politics")

            if retrieved_docs:
                for index, doc in enumerate(retrieved_docs, 1):
                    article_text = doc.page_content
                    st.subheader(f"📰 Bias Analysis of Document {index}")
                
                    # Display a snippet of the document for context
                    st.write(f"**Analyzing Text (Snippet):** {article_text[:500]}...")
                
                    # Analyze Bias
                    detected_label, scores = detect_bias(article_text[:1024])  # Limit to 1024 tokens
                    st.write({
                        "Detected Bias": detected_label,
                        "Scores": scores
                    })
            else:
                st.warning("⚠️ No relevant documents found for bias analysis.")
        else:
            st.warning("⚠️ QA Chain not initialized.")
with col3:
    if st.button("Get answer in Hindi or Marathi!") and selected_options:
        for option in selected_options:
            query = options[option]
            response = st.session_state.qa_chain.run(query)
            translated_response = translate_text(response, language)
            st.session_state.chat_history.append((option, translated_response))
            st.write(f"**{option}:**", translated_response)


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def show_online_pdf(pdf_url):
    st.subheader("📄 Click below to view the PDF:")
    st.markdown(f'[📄 View PDF]({pdf_url})', unsafe_allow_html=True)


year = selected_date.year
month = str(selected_date.month).zfill(2)
day = str(selected_date.day).zfill(2)
base_url = "https://www.ehitavada.com/encyc/6"
paper_code = "Mpage" if paper_type == "Main Paper" else "NCpage"

# Construct full PDF URL
page_number = st.number_input("📄 Page Number", min_value=1, max_value=12, step=1)
pdf_url = f"{base_url}/{year}/{month}/{day}/{paper_code}_{page_number}.pdf"

st.write(f"📄 Viewing: {pdf_url}")
show_online_pdf(pdf_url)

if st.button("Logout"):
    st.session_state.clear()
    st.switch_page("mainapp.py")

show_footer()