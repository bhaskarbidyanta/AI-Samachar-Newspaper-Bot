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
from datetime import datetime
import PyPDF2
from textblob import TextBlob
import numpy as np
#import re
#from transformers import pipeline
from components import show_navbar
from pathlib import Path
from gtts import gTTS
import base64
import io
from langdetect import detect
from streamlit_option_menu import option_menu
import os
from db import summary_collection
from utils import show_footer, page_buttons, logout
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

selected_date = st.date_input("Select Date",datetime.now())
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


if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False


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
    max_pages = 12 if paper_code == "Mpage" else 8

    for page_number in range(1, max_pages + 1):
        url = f"{base_url}/{year}/{month}/{day}/{paper_code}_{page_number}.pdf"
        response = requests.get(url)

        if response.status_code == 404:
            st.warning(f"⛔ {url} returned a 404. Stopped downloading at page {page_number}.")
            break

        download_dir = download_dir_main if paper_code == "Mpage" else download_dir_nc
        pdf_filename = os.path.join(download_dir, f"{paper_code}_{page_number}.pdf")

        with open(pdf_filename, 'wb') as file:
            file.write(response.content)

        st.success(f"✅ Downloaded: {paper_code}_{page_number}.pdf")

    st.success(f"✅ Completed download of {paper_code} up to page {page_number}")



CATEGORIES = [
    "Politics", "Business", "Crime", "Sports", "Weather", "Jobs", "Health","Medical","Hospital", "Education", "International","India",
    "Nagpur", "Entertainment", "Editorial", "War/Conflict", "Local News", "Opinion", "Technology", "Environment","Vidarbha","Maharashtra",
    "Uncategorized", "Other","Headline"
]

def process_pdf_by_page(pdf_path, page_num, google_api_key, embedding_model, selected_model):
    import re
    pdf_filename = os.path.basename(pdf_path)

    if "summaries_grouped" not in st.session_state:
        st.session_state.summaries_grouped = {}
    if "categorized_chunks" not in st.session_state:
        st.session_state.categorized_chunks = {}
    if "headlines_grouped" not in st.session_state:
        st.session_state.headlines_grouped = {}

    existing = summary_collection.find_one({"pdf": pdf_filename,"date": datetime.now().strftime("%Y-%m-%d")})
    if existing:
        #st.session_state.summaries_grouped[pdf_filename] = existing["summaries_grouped"]
        st.session_state.categorized_chunks[pdf_filename] = existing.get("categorized_chunks", {})
        st.session_state.headlines_grouped[pdf_filename] = existing.get("headlines_grouped", {})
        st.success(f"✅ Loaded cached summaries/headlines for {pdf_filename}")
        #st.success(f"✅ Loaded cached summaries for {pdf_filename}")
        return st.session_state.qa_chains.get(pdf_filename)

    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        try:
            page = reader.pages[page_num]
            text = page.extract_text()
        except Exception:
            text = ""
    
    if not text or len(text.strip()) < 30:
        st.warning("⚠️ No readable text found on this page (likely an ad or scanned image). Skipping.")
        return None

    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 20 and not line.strip().isupper()]
    clean_text = "\n".join(lines)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(clean_text)

    summarizer = ChatGoogleGenerativeAI(model=selected_model, google_api_key=google_api_key)

    categorizer_prompt = PromptTemplate.from_template("""
    Categorize the following news chunk into one of the following categories:
    {categories}

    News:
    {chunk}

    Category:
    """)

    CATEGORIES = [
        "Politics", "Business", "Crime", "Sports", "Weather", "Jobs", "Health","Medical","Hospital", "Education", "International","India",
        "Nagpur", "Entertainment", "Editorial", "War/Conflict", "Local News", "Opinion", "Technology", "Environment","Vidarbha","Maharashtra",
        "Uncategorized", "Other","Headline"
    ]
    # summary_prompt = PromptTemplate.from_template("""
    # Summarize the following news article in a clear, concise bullet point:
    # {chunk}

    # Summary:
    # """)

    category_map = {}
    summaries_grouped = {}
    headlines_grouped = {}
    tagged_chunks = []

    BATCH_SIZE = 2
    batched_chunks = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]

    for chunk_group in batched_chunks:
        chunk_group = [chunk for chunk in chunk_group if len(chunk.split()) >= 10]
        if not chunk_group:
            continue

        group_text = ""
        for idx, chunk in enumerate(chunk_group, 1):
            group_text += f"[{idx}] {chunk.strip()}\n\n"

        try:
            # ✅ Step 1: Categorize individually
            group_categories = []
            for chunk in chunk_group:
                category = summarizer.predict(
                    categorizer_prompt.format(chunk=chunk, categories=", ".join(CATEGORIES))
                ).strip()
                if category not in CATEGORIES:
                    category = "Uncategorized"
                group_categories.append(category)
                category_map.setdefault(category, []).append(chunk)
                tagged_chunks.append(f"[{category}]\n{chunk}")

            # ✅ Step 2: Summarize entire group at once
            headline_prompt = f"""
            You are a headline generator. For each numbered news section below, generate a short, clear headline.

            News Sections:
            {group_text}

            Instructions:
            - Return exactly {len(chunk_group)} headlines.
            - Prefix each headline with the section number in [brackets] (e.g., [1], [2], etc.).
            - Do not skip or merge any.
            - Headlines should be concise and informative.

            Output:
            """
            headline_output = summarizer.predict(headline_prompt).strip()
            headlines = {}
            for line in headline_output.splitlines():
                match = re.match(r"\[(\d+)]\s*(.+)", line.strip())
                if match:
                    idx = int(match.group(1)) - 1
                    headlines[idx] = match.group(2).strip()

            # ✅ Step 3: Distribute summaries back by order
            for i in range(min(len(headlines), len(group_categories))):
                hl = headlines.get(i, chunk_group[i][:80] + "...")
                cat = group_categories[i]
                headlines_grouped.setdefault(cat, []).append(hl)

        except Exception as e:
            st.warning(f"⚠️ Skipping batch due to error: {e}")
            continue

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=google_api_key)
    # ✅ SAFETY CHECK: Avoid IndexError if no valid content found
    if not tagged_chunks:
        st.warning("⚠️ No valid tagged chunks found for embedding. Skipping QA chain setup.")
        return None
    vectorstore = FAISS.from_texts(tagged_chunks, embeddings)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatGoogleGenerativeAI(model=selected_model, google_api_key=google_api_key),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    st.session_state.summaries_grouped[pdf_filename] = summaries_grouped
    st.session_state.categorized_chunks[pdf_filename] = category_map
    st.session_state.headlines_grouped[pdf_filename] = headlines_grouped

    if "qa_chains" not in st.session_state:
        st.session_state.qa_chains = {}
    st.session_state.qa_chains[pdf_filename] = qa_chain

    now = datetime.now().strftime("%Y-%m-%d")
    summary_collection.update_one(
        {"pdf": pdf_filename,"date": now},
        {"$set": {
            "pdf": pdf_filename,
            "date": now,
            #"summaries_grouped": summaries_grouped,
            "categorized_chunks": category_map,
            "headlines_grouped": headlines_grouped,
        }},
        upsert=True
    )

    st.success(f"✅ Page processed, summaries and QA ready for {pdf_filename}")
    return qa_chain
    

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

            pdf_filename = os.path.basename(pdf_path)
            if "headlines_grouped" in st.session_state and pdf_filename in st.session_state.headlines_grouped:
                st.markdown(f"## 📰 Headlines from {pdf_filename}")
                

                headlines_grouped = st.session_state.headlines_grouped[pdf_filename]

                for category, headlines in headlines_grouped.items():
                    if not headlines:
                        continue

                    unique_headlines = list(dict.fromkeys([
                        hl.strip().replace("Headlines:", "").strip() for hl in headlines if hl.strip()
                    ]))

                    if unique_headlines:
                        st.markdown(f"### {category}")
                        for idx, hl in enumerate(unique_headlines, 1):
                            st.markdown(f"{idx}. {hl}")

            if qa_chain:
                st.session_state.qa_chains[f"{paper_code}_{i}"] = qa_chain

    st.session_state.pdf_loaded = True


from googletrans import Translator

translator = Translator()

# Sidebar language selector
language = st.sidebar.selectbox("🌐 Select Language:", ["English", "Hindi", "Marathi"])

def translate_text(text, target_language):
    if not text:
        return "❌ No summary available to translate."

    try:
        if target_language == "Hindi":
            return translator.translate(text, dest='hi').text
        elif target_language == "Marathi":
            return translator.translate(text, dest='mr').text
        else:
            return text  # English - no translation
    except Exception as e:
        return f"❌ Translation failed: {str(e)}"



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
                        padding: 10px 15px; border-radius: 10px; max-width: 80%;color: black;'>
                <strong>{label}</strong><br>{message}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# 💬 Display chat history

if "selected_option" not in st.session_state:
    st.session_state.selected_option = ""

if "prompt_query" not in st.session_state:
    st.session_state.prompt_query = None

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


if st.session_state.get("headlines_grouped"):
    if "selected_option" not in st.session_state:
        st.session_state.selected_option = ""

    if "prompt_query" not in st.session_state:
        st.session_state.prompt_query = None

    if "run_prompt" not in st.session_state:
        st.session_state.run_prompt = False

# Optional selectbox prompts
    CATEGORIES1 = [
        "Politics", "Business", "Crime", "Sports", "Weather", "Jobs", "Health","Medical","Hospital", "Education", "International","National",
        "Nagpur City", "Entertainment", "Editorial", "War/Conflict", "Local News", "Opinion", "Technology", "Environment","Vidarbha News","Maharashtra News",
        "Uncategorized", "Other"
    ]
    # ✅ Get only pages that have QA chains initialized
    # --- Gather all summaries across all pages ---
    headlines_grouped = st.session_state.get("headlines_grouped", {})
    headlines_by_cat_all_pages = {}

    for page_headlines in headlines_grouped.values():
        for category, bullets in page_headlines.items():
            if bullets:
                normalized = category.strip().title()
                headlines_by_cat_all_pages.setdefault(normalized, []).extend(bullets)

    headlines_grouped = st.session_state.get("headlines_grouped",{})
    # --- Filter CATEGORIES based on available summaries ---
    available_categories = [""] + [cat for cat in CATEGORIES if cat in headlines_by_cat_all_pages] + ["🧠 Full Summary"]+['All Headlines']

    # --- QA chains (optional) ---
    qa_chains = st.session_state.get("qa_chains", {})

    # --- Sidebar Selectbox ---
    selected_option = st.selectbox("📌 Quick Prompt", available_categories, key="selected_option")

    # --- Prompt trigger ---
    if st.button("▶️ Run Prompt"):
        st.session_state.prompt_query = selected_option  # Use selected category directly as prompt
        st.session_state.run_prompt = True

    # --- Main logic (No free chat_input, only selected options) ---
    question = st.session_state.prompt_query if st.session_state.run_prompt else None

    paper_code = "Mpage" if paper_type == "Main Paper" else "NCpage"
        
    #pdf_filename = os.path.basename(f"{paper_code}_{selected_page}.pdf")
    if question == "All Headlines":
        full_text = ""
        for pdf_filename, page_headlines in st.session_state.get("headlines_grouped", {}).items():
            full_text += f"## 📰 Headlines from {pdf_filename}\n"
            for category, headlines in page_headlines.items():
                if not headlines:
                    continue
                cleaned = list(dict.fromkeys([
                    hl.strip().replace("Headlines:", "").strip()
                    for hl in headlines if hl.strip()
                ]))
                if cleaned:
                    full_text += f"### {category}\n"
                    for i, hl in enumerate(cleaned, 1):
                        full_text += f"{i}. {hl}\n"
            full_text += "\n"

        st.session_state.chat_history.append((question, full_text.strip()))
        st.session_state.run_prompt = False
        st.session_state.prompt_query = None    

    elif question:
        headlines_by_cat = {}
        for page_headlines in st.session_state.get("headlines_grouped", {}).values():
            for category, bullets in page_headlines.items():
                headlines_by_cat.setdefault(category.lower(), []).extend(bullets)  # ✅ FIXED

        headlines_text = ""
        match_found = False
        if question in ["🧠 Full Summary","Headlines"]:
            for category, bullets in headlines_by_cat.items():
                if bullets:
                    headlines_text += f"### {category.title()}\n" + "\n".join(bullets) + "\n\n"
            match_found = True
        else:
            matched_cat = None
            for cat in CATEGORIES:
                if cat.lower() in question.lower():
                    matched_cat = cat.lower()
                    matched_cat_display = cat  # Store original for title display
                    break

            is_general_summary = any(kw in question.lower() for kw in ["summary", "summarize", "everything", "important"])
            is_headlines_only = "headlines" in question.lower()

            if matched_cat or is_general_summary:
                if matched_cat and headlines_by_cat.get(matched_cat):
                    headlines_text += f"### {matched_cat_display}\n" + "\n".join(headlines_by_cat[matched_cat]) + "\n\n"
                    match_found = True

                elif matched_cat:
                    fallback = headlines_by_cat.get("uncategorized", [])
                    if fallback:
                        headlines_text += f"### Uncategorized\n" + "\n".join(fallback) + "\n\n"
                        match_found = True

                elif is_general_summary:
                    for category, bullets in headlines_by_cat.items():
                        if bullets:
                            headlines_text += f"### {category.title()}\n" + "\n".join(bullets) + "\n\n"
                            match_found = True

        if headlines_text and headlines_text.strip():
            cleaned_text = headlines_text.strip()
            try:
                # translated_response = translate_text(summary_text.strip(), language)
                # #st.markdown(translated_response)
                # st.session_state.chat_history.append((question, translated_response))
                if language != "English":
                    translated_response = translate_text(cleaned_text, language)
                else:
                    translated_response = cleaned_text
                st.session_state.chat_history.append((question, translated_response))
                st.session_state.run_prompt = False
                st.session_state.prompt_query = None
            except Exception as e:
                st.warning(f"❌ Translation failed: {e}")
                st.session_state.chat_history.append((question, cleaned_text))
        else:
            st.warning("⚠️ No summary was generated for this page to translate.")

        if not match_found:
            qa_chains = st.session_state.get("qa_chains", {})
            if qa_chains:
                first_qa = next(iter(qa_chains.values()))
                try:
                    response = first_qa.run(question)
                    translated = translate_text(response, language)
                    st.session_state.chat_history.append((question, translated))
                except Exception as e:
                    st.error(f"QA failed: {e}")
            else:
                st.warning("❌ No QA chains available to process your query.")


        # if not match_found:
        #     if "qa_chains" in st.session_state and st.session_state.qa_chains:
        #         selected_page = st.sidebar.selectbox("📄 Select Page to Ask From:", list(st.session_state.qa_chains.keys()))
                
        #         if not question and selected_option:
        #             question = options[selected_option]

        #         if question:
        #             qa_chain = st.session_state.qa_chains.get(selected_page)
        #             if qa_chain:
        #                 response = qa_chain.run(question)
        #                 translated_response = translate_text(response, language)
        #                 st.session_state.chat_history.append((question, translated_response))
        #             else:
        #                 st.error(f"❌ No QA chain available for {selected_page}.")    


else:
    if st.session_state.get("pdf_loaded"):
        st.warning("⚠️ PDFs were loaded but no summaries available. Check processing.")
    else:
        st.info("📄 Click the button above to load and process the PDFs first.")

#for question, answer in st.session_state.chat_history:
#    render_message(question, sender="user")
#    render_message(answer, sender="bot")

chat_history_container = st.container()

with chat_history_container:
    for user_msg, bot_msg in st.session_state.chat_history:
        render_message(user_msg, sender="user")
        render_message(bot_msg, sender="bot")

if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history.clear()
    st.rerun()

page_buttons()
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

if st.sidebar.button("Logout"):
    logout()

def scroll_to_top_button():
    st.markdown("""
        <style>
        .scroll-to-top-btn {
            position: fixed;
            bottom: 100px; /* Lifted above footer */
            right: 30px;
            z-index: 9999;
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 12px;
            cursor: pointer;
            font-size: 16px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
        }
        .scroll-to-top-btn:hover {
            background-color: #45a049;
        }
        </style>

        <button class="scroll-to-top-btn" onclick="window.scrollTo({top: 0, behavior: 'smooth'});">🔝 Top</button>
        """, unsafe_allow_html=True)
scroll_to_top_button()

show_footer()
