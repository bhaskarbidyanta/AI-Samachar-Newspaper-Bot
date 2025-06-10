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
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from textblob import TextBlob
import emoji
import requests
import datetime
import PyPDF2
from textblob import TextBlob
import numpy as np
#import re
#from transformers import pipeline
from components import show_navbar,show_footer
from pathlib import Path
from gtts import gTTS
import base64
import io
from langdetect import detect
from streamlit_option_menu import option_menu
import os
#show_navbar()

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
#selected_model = st.sidebar.selectbox("🔍 Select Model:", model_options, index=0)
selected_model = "gemini-1.5-pro"  # Default model

st.title("Newspaper PDF Chatbot")

paper_type = st.radio("Select Paper Type",["Main Paper","Nagpur CityLine"])

selected_date = st.date_input("Select Date",datetime.datetime.now())
# with st.sidebar:
#     selected = option_menu(
#         menu_title=None,
#         options=["📥 Select Paper", "🗞️ Select Date", "🤖 Select Language"],
#         icons=["cloud-download", "newspaper", "chat"],
#         default_index=0,
#         orientation="horizontal",
#         styles={
#             "container": {"padding": "0!important"},
#             "icon": {"color": "black", "font-size": "20px"},
#             "nav-link": {"font-size": "16px", "--hover-color": "#51fcff"},
#             "nav-link-selected": {"background-color": "#007bff"},
#         }
#     )
# if selected == "📥 Select Paper":
#     paper_type = st.radio("Select Paper Type", ["Main Paper", "Nagpur CityLine"], index=0)
# elif selected == "🗞️ Select Date":
#     selected_date = st.date_input("Select Date", datetime.datetime.now())
# elif selected == "🤖 Select Language":
#     language = st.selectbox("🌐 Select Language:", ["English", "Hindi", "Marathi"], index=0)

TOPIC_KEYWORDS = {
    "Sports": ["cricket", "football", "match", "tournament", "goal", "medal", "olympics"],
    "International": ["un", "russia", "china", "us", "global", "international","europe","uk"],
    "National": ["india", "pm modi", "lok sabha", "national","rajya sabha"],
    "City": ["nagpur", "bhopal", "local", "municipal", "district","NMC"],
    "Jobs": ["recruitment", "vacancy", "hiring", "walk-in", "job", "career"],
    "Crime": ["arrested", "murder", "police", "theft", "robbery", "fir","supreme","court","police","crime","accident"],
    "Weather": ["rain", "forecast", "temperature", "imd", "weather","rainfall","monsoon","aqi","sunrise","sunset"],
    "Business": ["stock", "market", "share", "business", "sensex", "economy", "rbi"],
    "Politics": ["government","election", "bjp", "congress", "mp", "mla", "cabinet"],
    "Editorial": ["editorial", "opinion", "column", "author"],
    "War": ["war", "conflict", "border", "attack", "terror"],
    "Health": ["health", "hospital", "covid", "doctor", "disease", "vaccine"],
    "Education": ["exam", "results", "admission", "cbse", "school", "university"],
    "Entertainment": ["film", "movie", "actor", "actress", "box office", "bollywood"]
}

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False


def categorize_text(text):
    text = text.lower()
    categories = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(k in text for k in keywords):
            categories.append(topic)
    return categories

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
        
            # Move to the next page
        if page_number == max_pages:
            st.success(f"Downloaded all {max_pages} pages of {paper_code}.")
            break

        page_number += 1



def process_pdf_by_page(
    pdf_path,
    page_num,
    google_api_key,
    embedding_model,
    selected_model
):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        page = reader.pages[page_num]
        text = page.extract_text()

        lines = text.split("\n")
        headlines = [line.strip() for line in lines if len(line.strip()) < 100 and (line.strip().isupper() or line.strip().istitle())]
        article_lines = [line.strip() for line in lines if line.strip() not in headlines]

        # Combine article and split
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = headlines + splitter.split_text("\n".join(article_lines))

        # Tag chunks with category
        tagged_chunks = []
        for chunk in chunks:
            categories = categorize_text(chunk)
            if categories:
                chunk = f"[{' | '.join(categories)}]\n{chunk}"  # Attach tags visibly
            tagged_chunks.append(chunk)

        # Store in vectorstore
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=google_api_key)
        vectorstore = FAISS.from_texts(tagged_chunks, embeddings)

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatGoogleGenerativeAI(model=selected_model, google_api_key=google_api_key),
            retriever=vectorstore.as_retriever(),
            memory=memory
        )

        if "categorized_chunks" not in st.session_state:
            st.session_state.categorized_chunks = {}

        category_map = {}
        for chunk in tagged_chunks:
            label_start = chunk.find("[")
            label_end = chunk.find("]")
            if label_start != -1 and label_end != -1:
                labels = chunk[label_start + 1:label_end].split(" | ")
                content = chunk[label_end + 1:].strip()
                for label in labels:
                    category_map.setdefault(label.strip(), []).append(content)
        st.session_state.categorized_chunks[os.path.basename(pdf_path)] = category_map

        # Summarize chunks
        summarizer = ChatGoogleGenerativeAI(model=selected_model, google_api_key=google_api_key)
        summary_prompt = PromptTemplate.from_template("""
        Summarize the following news chunk in one bullet point:
        "{chunk}"
        """)

        grouped_summaries = []
        for chunk in tagged_chunks:
            if len(chunk) > 100:
                try:
                    summary = summarizer.predict(summary_prompt.format(chunk=chunk)).strip()
                    categories = categorize_text(chunk)
                    for category in categories:
                        grouped_summaries.setdefault(category,[]).append(f"- {summary}")
                    #summaries.append(f"- {summary.strip()}")
                except:
                    pass

        pdf_filename = os.path.basename(pdf_path)
        if "summaries_grouped" not in st.session_state:
            st.session_state.summaries_grouped = {}
        st.session_state.summaries_grouped[pdf_filename] = grouped_summaries

        # Show news summaries
        #if summaries:
        #    st.markdown("### 📰 Important News Highlights")
        #    for s in summaries:
        #        st.markdown(s)
        
        return qa_chain
    st.success(f"✅ Processed page {page_num + 1} of {pdf_path}")
    

#sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Political Bias Keywords (Extendable)
# left_bias_words = [
#     "social justice", "climate change", "income inequality", "progressive",
#     "universal healthcare", "gun control", "racial equity", "minimum wage","Congress",
#     "Aam Aadmi Party", "AAP", "Indian National Congress",
# ]

# right_bias_words = [
#     "tax cuts", "border security", "traditional values", "free market",
#     "law and order", "second amendment", "private enterprise", "fiscal responsibility","BJP","Bharatiya Janata Party",
# ]

# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# # Define Bias Labels
# bias_labels = ["Left-Leaning", "Right-Leaning", "Neutral"]

# def detect_bias(text):
#     """Detect bias using Zero-Shot Classification"""
#     result = classifier(text, bias_labels)
#     scores = {label: score for label, score in zip(result['labels'], result['scores'])}
#     detected_label = result['labels'][0]  # Label with highest score
#     return detected_label, scores


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
    download_dir = os.path.join("news", formatted_date, "downloaded_pdfs" if paper_code == "Mpage" else "downloaded_pdfs_nc")
    num_pages = 12 if paper_code == "Mpage" else 8

    st.session_state.qa_chains = {}

    for i in range(1, num_pages + 1):
        pdf_path = os.path.join(download_dir, f"{paper_code}_{i}.pdf")
        if os.path.exists(pdf_path):
            qa_chain = process_pdf_by_page(
                pdf_path=pdf_path,
                page_num=0,
                google_api_key=google_api_key,
                embedding_model=EMBEDDING_MODEL,
                selected_model=selected_model
            )
            if qa_chain:
                st.session_state.qa_chains[i] = qa_chain
    
    st.session_state.pdf_loaded = True

from googletrans import Translator

translator = Translator()

# Sidebar language selector
language = st.sidebar.selectbox("🌐 Select Language:", ["English", "Hindi", "Marathi"])

def translate_text(text, target_language):
    if target_language == "Hindi":
        return translator.translate(text, dest='hi').text
    elif target_language == "Marathi":
        return translator.translate(text, dest='mr').text
    else:
        return text  # English - no translation


#user_input = st.chat_input("💬 Ask a question about the PDFs:")
# Function to render message bubbles
def render_message(message, sender="user"):
    if sender == "user":
        alignment = "right"
        bg_color = "#DCF8C6"  # WhatsApp greenish for user
        label = "🙋 You"
    else:
        alignment = "left"
        bg_color = "#E6E6FA"  # Light purple for bot
        label = "🤖 Bot"

    st.markdown(
        f"""
        <div style='text-align: {alignment}; margin: 10px 0;'>
            <div style='display: inline-block; background-color: {bg_color}; 
                        padding: 10px 15px; border-radius: 10px; max-width: 80%;'>
                <strong>{label}</strong><br>{message}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# 💬 Display chat history

if "selected_option" not in st.session_state:
    st.session_state.selected_option = ""

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "News"

if "text_input" not in st.session_state:
    st.session_state.text_input = ""

if "submit_triggered" not in st.session_state:
    st.session_state.submit_triggered = False

if "dropdown_value" not in st.session_state:
    st.session_state.dropdown_value = ""

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "temp_option" not in st.session_state:
    st.session_state.temp_option = ""

if "send_triggered" not in st.session_state:
    st.session_state.send_triggered = False

options = {
    "🗞️ All News Headlines": "List all the news headlines from these PDFs, separated by category if possible.",
    "🧠 Full Summary": "List all news stories covered across all pages of the PDFs.",
    
    "⚽ Sports News": "Extract and list all sports-related news from these PDFs.",
    "🌍 International News": "Summarize international news events mentioned in these PDFs.",
    "🇮🇳 National News": "List and summarize national-level news relevant to India from these PDFs.",
    "🏙️ City/Local News": "Show all local or city-specific news stories found in these PDFs.",
    
    "💼 Job Listings": "List all job-related information and employment opportunities found in the newspaper PDFs.",
    "🚔 Crime Reports": "List all recent crime-related news articles mentioned in the PDFs.",
    "🏞️ Weather Updates": "Extract all weather-related news, forecasts, and alerts from these PDFs.",
    
    "💰 Business News": "List and provide all business and financial news covered in the newspaper PDFs.",
    "📰 Political News": "List the major political updates, parties, and leaders mentioned in these PDFs.",
    "🗳️ Election Coverage": "List all the latest election updates, results, or campaign stories mentioned in the PDFs.",
    
    "🧾 Editorial Section": "Summarize the editorial articles and opinion pieces found in these PDFs.",
    "⚔️ War or Conflict News": "Extract and list any war-related or conflict-specific news mentioned in the PDFs.",
    
    "🏥 Health News": "List all health-related updates or medical news in these PDFs.",
    "🎓 Education": "List all news or announcements related to education, schools, or exams in the PDFs.",
    "🎭 Entertainment": "List all the entertainment or celebrity-related news from the newspaper.",
}

# --- UI ---
# Input area
# Input section
# with st.container():
    
#     with st.sidebar.form(key="chat_input_form", clear_on_submit=True):
#         col1, col2 = st.columns([6, 1])
#         with col1:
#             temp_option = st.selectbox(
#                 "📌 Quick Prompt",
#                 [""] + list(options.keys()),
#                 key="selectbox", # Use the session_state key directly
#                 #label_visibility="collapsed" # Hide the default label for cleaner look
#             )

#         with col2:
#             send = st.form_submit_button("Send")

#     st.markdown("</div>", unsafe_allow_html=True)  # Close the fixed input div

# temp_input = st.chat_input("💬 Or type your message")

# # Processing logic
# query = None

# if temp_input:
#         query = temp_input.strip()
# elif send:
#     if temp_option:
#         query = options[temp_option]


if "qa_chains" in st.session_state and st.session_state.qa_chains:
    selected_page = st.sidebar.selectbox("Select Page to Ask Questions From:", list(st.session_state.qa_chains.keys()))
# Optional selectbox prompts
    options = {
        "📋 Everything Important": "Extract and list all major articles, headlines, and key updates from this page. Include political, business, sports, health, job, and city news. Ensure no important content is skipped. Return the results as a clean, well-structured bullet list.",
        "🧠 Full Summary": "List all major news articles from this page clearly and concisely in bullet points, limiting to the top 20 items.",
        "🗞️ All Headlines": "List all from this page, preserving formatting and spacing. Aim for 15 to 25 entries.",
        "🔎 Important Headlines Only": "List all the major, category-worthy headlines from this page. Prioritize political, national, international, and sports news. Limit to 20 key points.",
        "⚽ Sports": "Listall all sports news from this page including teams, matches, scores, and events. Use up to 20 lines.",
        "🌍 International": "List all international stories, global leaders, countries, and incidents reported on this page in up to 25 detailed bullet points.",
        "📰 Politics": "List all political headlines and key political developments from this page. Include leaders, decisions, or rallies. Use up to 20 concise bullet points.",
        "💼 Jobs": "List all job openings, recruitment drives, walk-in interviews or job-related news mentioned. Aim for 25 specific job items if available.",
        "🎓 Education": "List all updates related to exams, results, school/university news, and government schemes for students. Include 20 informative lines.",
        "🏥 Health": "List all all health-related news including diseases, vaccination drives, hospital reports, or health tips. Use up to 20 clear points.",
        "📉 Business": "List all market updates, company announcements, and economic news in up to 25 crisp points with sector or stock details.",
        "🚨 Crime": "List all detailed crime reports including FIRs, police actions, thefts, or judicial developments. Summarize in 25 lines.",
        "🌦️ Weather": "List all weather forecasts, rainfall alerts, temperature readings in Nagpur and in Vidarbha region in 20 to 25 well-structured lines."
    }
    selected_prompt = st.sidebar.selectbox("📌 Quick Prompt", [""] + list(options.keys()))

    question = st.chat_input("💬 Ask something about this page") or options.get(selected_prompt)

    paper_code = "Mpage" if paper_type == "Main Paper" else "NCpage"
        

    if question:
        # Optional: Use raw summaries instead of retrieval-based QA for summary-type prompts
        summary_triggers = ["summary", "summarize", "everything important", "all headlines", "important news", "key updates"]
        is_summary_request = any(trigger in question.lower() for trigger in summary_triggers)
        
        pdf_filename = f"{paper_code}_{selected_page}.pdf"  # or however you name it
        summaries_by_cat = st.session_state.get("summaries_grouped", {}).get(pdf_filename)

        if is_summary_request and summaries_by_cat:
            summary_text = ""
            
            # Filter by category if prompt matches
            for category, bullet_points in summaries_by_cat.items():
                if category.lower() in question.lower() or question.lower() in category.lower() or "everything" in question.lower() or "summary" in question.lower():
                    summary_text += f"## {category}\n" + "\n".join(bullet_points) + "\n\n"

            # If nothing matched specifically, show all
            if not summary_text.strip():
                for category, bullet_points in summaries_by_cat.items():
                    summary_text += f"## {category}\n" + "\n".join(bullet_points) + "\n\n"

            translated_response = translate_text(summary_text.strip(), language)
            st.session_state.chat_history.append((question, translated_response))

            
        else:
            qa = st.session_state.qa_chains[selected_page]
            response = qa.run(question)
            translated_response = translate_text(response, language)
            st.session_state.chat_history.append((question, translated_response))
else:
    if st.session_state.get("pdf_loaded") == True:
        st.warning("⚠️ PDFs were loaded but no QA chains were created. Check file content.")
    else:
        st.info("📄Click the button above to load and process the PDFs first.")

#for question, answer in st.session_state.chat_history:
#    render_message(question, sender="user")
#    render_message(answer, sender="bot")

chat_history_container = st.container()

with chat_history_container:
    for user_msg, bot_msg in st.session_state.chat_history:
        render_message(user_msg, sender="user")
        render_message(bot_msg, sender="bot")
        # with st.chat_message("user"):
        #     st.markdown(user_msg)
        # with st.chat_message("assistant"):
        #     st.markdown(bot_msg)

        # # "Reset" inputs (can’t clear selectbox/text_input forcibly, but this will visually reset on rerender)
        # st.session_state.temp_input = ""
        # st.session_state.temp_option = ""
        # st.rerun()
    # # Quick prompt buttons shown above chat_input
# with st.chat_message("user"):
#     selected_option = st.selectbox(
#         "📢 Choose a quick prompt or type your own below 👇",
#         [""] + list(options.keys())
#     )

# if selected_option and st.session_state.selected_option != selected_option:
#     st.session_state.selected_option = selected_option

# # Then normal chat input
# user_input = st.chat_input("💬 Or ask something else:")

# if st.session_state.selected_option and not user_input:
#     query = options[selected_option]
#     response = st.session_state.qa_chain.run(query)
#     translated_response = translate_text(response, language)
#     st.session_state.chat_history.append((st.session_state.selected_option, translated_response))
#     st.session_state.selected_option = ""  # Reset selected option
#     st.rerun()  # To immediately reflect in chat

# elif user_input:
#     response = st.session_state.qa_chain.run(user_input)
#     translated_response = translate_text(response, language)
#     st.session_state.chat_history.append((user_input, translated_response))        
#     st.rerun()  # To immediately reflect in chat

# with st.chat_message("user"):
#     selected_option = st.selectbox("📢 Choose a prompt (or ignore and type your own below):", [""] + list(options.keys()))

# # Step 2: Convert selected option into prefilled query
# prefill_query = options[selected_option] if selected_option else ""

# # Step 3: Let user type/edit their query (pre-filled if dropdown was used)
# user_query = st.chat_input("💬 Ask a question about the PDFs:", value=prefill_query)

# # Step 4: Submit only when they press Enter / Submit (no automatic processing on dropdown)
# if user_query:
#     response = st.session_state.qa_chain.run(user_query)
#     translated_response = translate_text(response, language)
#     st.session_state.chat_history.append((user_query, translated_response))


# if selected_option and not user_input:
#     #for option in selected_options:
#     query = options[selected_option]
#     response = st.session_state.qa_chain.run(query)
#     translated_response = translate_text(response, language)
#     st.session_state.chat_history.append((selected_option, translated_response))

# Process custom user query
# if user_query:
#     response = st.session_state.qa_chain.run(user_input)
#     translated_response = translate_text(response, language)
#     st.session_state.chat_history.append((user_input, translated_response))

# Add a button to clear chat history
if st.sidebar.button("🔊 Get Audio of Last Bot Response"):
    if st.session_state.chat_history:
        last_response = st.session_state.chat_history[-1][1]  # just the bot reply

        # Detect language
        lang = detect(last_response)
        lang_map = {'en': 'en', 'hi': 'hi', 'mr': 'mr'}

        if lang not in lang_map:
            st.warning(f"⚠️ Detected language '{lang}' not supported for audio.")
        else:
            try:
                tts = gTTS(text=last_response, lang=lang_map[lang])
                # Save to memory (BytesIO)
                audio_bytes = io.BytesIO()
                tts.write_to_fp(audio_bytes)
                audio_bytes.seek(0)

                # Base64 encode for streamlit audio
                b64_audio = base64.b64encode(audio_bytes.read()).decode()
                audio_html = f"""
                <audio controls autoplay>
                    <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
                st.success(f"🎧 Playing response in {lang.upper()}")
            except Exception as e:
                st.error(f"❌ Audio generation failed: {e}")
    else:
        st.warning("⚠️ No response available yet.")

if st.sidebar.button("📊 Analyze Sentiment"):
    if st.session_state.chat_history:
        latest_response = st.session_state.chat_history[-1][1]
        
        # Detect and translate
        try:
            detected = translator.detect(latest_response)
            if detected.lang != 'en':
                translated = translator.translate(latest_response, src=detected.lang, dest='en')
                text_for_analysis = translated.text
            else:
                text_for_analysis = latest_response

            sentiment_score = TextBlob(text_for_analysis).sentiment.polarity
            
            if sentiment_score > 0.1:
                sentiment_label = "😊 Positive"
            elif sentiment_score < -0.1:
                sentiment_label = "😟 Negative"
            else:
                sentiment_label = "😐 Neutral"
            
            st.subheader(f"🧠 Sentiment of Latest Response: {sentiment_label} ({sentiment_score:.2f})")

        except Exception as e:
            st.error(f"Translation or sentiment error: {e}")

    else:
        st.warning("⚠️ No news updates found! Try fetching news first.")
    

# if st.sidebar.button("📊 Analyze Bias"):
#         if st.session_state.qa_chain:
#             retrieved_docs = st.session_state.qa_chain.retriever.get_relevant_documents("politics")

#             if retrieved_docs:
#                 for index, doc in enumerate(retrieved_docs, 1):
#                     article_text = doc.page_content
#                     st.subheader(f"📰 Bias Analysis of Document {index}")
                
#                     # Display a snippet of the document for context
#                     st.write(f"**Analyzing Text (Snippet):** {article_text[:500]}...")
                
#                     # Analyze Bias
#                     detected_label, scores = detect_bias(article_text[:1024])  # Limit to 1024 tokens
#                     st.write({
#                         "Detected Bias": detected_label,
#                         "Scores": scores
#                     })
#             else:
#                 st.warning("⚠️ No relevant documents found for bias analysis.")
#         else:
#             st.warning("⚠️ QA Chain not initialized.")


# if st.sidebar.button("Logout"):
#     st.session_state.clear()
#     st.switch_page("mainapp.py")

#show_footer()
