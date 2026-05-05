import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
import streamlit.components.v1 as components
import uuid
import re
import urllib.parse
import unicodedata
from google.cloud import storage
import io
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# --- 1. Page Config & Floating UI CSS ---
st.set_page_config(layout="wide", page_title="TechM GuruCool Prototype")

st.markdown("""
    <style>
    /* Main Dashboard Background */
    .stApp {
        background-color: #f4f7f9;
    }
    
    /* Floating Chat Container Styling */
    .floating-chat {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 450px;
        background: white;
        border-radius: 15px;
        box-shadow: 0px 10px 25px rgba(0,0,0,0.1);
        z-index: 1000;
        border: 1px solid #e0e0e0;
        padding: 10px;
    }
    
    /* Header styling for the Bot */
    .bot-header {
        background: #0078d4;
        color: white;
        padding: 10px;
        border-radius: 10px 10px 0 0;
        font-weight: bold;
        margin: -10px -10px 10px -10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. NLTK & GCS Initialization ---
@st.cache_resource
def setup_resources():
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

setup_resources()

@st.cache_resource
def get_gcs_client():
    try:
        import json
        credentials_info = json.loads(st.secrets["gcs"]["credentials"])
        return storage.Client.from_service_account_info(credentials_info)
    except Exception as e:
        st.error(f"GCS Error: {e}")
        return None

gcs_client = get_gcs_client()
GCS_BUCKET_NAME = st.secrets["gcs"]["bucket_name"]

# --- 3. Data Loading Utilities ---
def read_csv_from_gcs(file_name_gcs, default_cols=None):
    if gcs_client is None: return pd.DataFrame(columns=default_cols or [])
    bucket = gcs_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(file_name_gcs)
    if not blob.exists(): return pd.DataFrame(columns=default_cols or [])
    try:
        csv_content = blob.download_as_text(encoding='utf-8')
        return pd.read_csv(io.StringIO(csv_content))
    except:
        return pd.DataFrame(columns=default_cols or [])

def write_csv_to_gcs(df, file_name_gcs):
    if gcs_client is None: return
    bucket = gcs_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(file_name_gcs)
    blob.upload_from_string(df.to_csv(index=False), 'text/csv')

# --- 4. NLP & FAQ Engine ---
lemmatizer = nltk.stem.WordNetLemmatizer()
def tfidf_preprocess(text):
    if pd.isna(text): return ""
    text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

class FAQBot:
    def __init__(self, df):
        self.df = df
        if df.empty: return
        self.vectorizer = TfidfVectorizer(preprocessor=tfidf_preprocess, ngram_range=(1, 3))
        self.vectors = self.vectorizer.fit_transform(df['Question'].astype(str))

    def search(self, query, top_n=5):
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.vectors).flatten()
        indices = sims.argsort()[-top_n:][::-1]
        return [{"idx": i, "score": sims[i], "q": self.df.iloc[i]['Question']} for i in indices if sims[i] > 0.1]

# --- 5. Session State ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'messages' not in st.session_state: st.session_state.messages = []
if 'match_details' not in st.session_state: st.session_state.match_details = None

# --- 6. PROTOTYPE LOGIN (Email Only) ---
if not st.session_state.authenticated:
    st.title("🚀 GuruCool Prototype Login")
    email_input = st.text_input("Enter your Email to enter:")
    if st.button("Enter Portal"):
        if "@" in email_input:
            st.session_state.authenticated = True
            st.session_state.user_email = email_input
            st.rerun()
        else:
            st.error("Please enter a valid email address.")
    st.stop()

# --- 7. Load Data ---
df_qa = read_csv_from_gcs("data.csv")
bot = FAQBot(df_qa)

# --- 8. Sidebar Navigation ---
with st.sidebar:
    st.image("https://via.placeholder.com/150?text=GuruCool", width=100)
    st.title("Navigation")
    choice = st.radio("Go to", ["Dashboard", "Product View", "Statistics"])
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

# --- 9. Dashboard Logic ---
if choice == "Dashboard":
    st.title("🗺️ Maps Knowledge Portal")
    
    # Hero Section
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.video("https://www.youtube.com/watch?v=hA_-MkU0Nfw")
    
    st.markdown("---")
    st.subheader("🚀 Quick Access Domains")
    d1, d2, d3 = st.columns(3)
    with d1: 
        st.info("### 🛣️ Linear")
        st.caption("Line mapping & boundaries")
    with d2:
        st.success("### 🔷 Polygon")
        st.caption("Area mapping & geometry")
    with d3:
        st.warning("### 🚦 Signals")
        st.caption("Traffic signal configs")

# --- 10. Floating Chatbot Logic ---
# This container renders the bot as a persistent UI element
with st.container():
    st.markdown('<div class="floating-chat">', unsafe_allow_html=True)
    st.markdown('<div class="bot-header">🪐 GuruCool AI Support</div>', unsafe_allow_html=True)
    
    # Scrollable chat area
    chat_box = st.container(height=300)
    for m in st.session_state.messages:
        with chat_box.chat_message(m["role"]):
            st.markdown(m["content"])
            
    # Chat Input
    if prompt := st.chat_input("Ask me about mapping rules..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # NLP Search
        results = bot.search(prompt)
        
        with chat_box.chat_message("assistant"):
            if results:
                if results[0]['score'] > 0.8: # Confidence threshold for direct answer
                    best = df_qa.iloc[results[0]['idx']]
                    ans = f"**Found exact match:** {best['Question']}\n\n{best['Answer']}"
                    st.markdown(ans)
                    st.session_state.match_details = best.to_dict()
                else:
                    ans = "I found a few matches. Did you mean one of these?"
                    st.markdown(ans)
                    for r in results:
                        if st.button(f"🔗 {r['q']}", key=f"btn_{r['idx']}"):
                            # Update details on click
                            st.session_state.match_details = df_qa.iloc[r['idx']].to_dict()
                            st.rerun()
            else:
                ans = "I'm sorry, I couldn't find an exact match. Try rephrasing your question."
                st.markdown(ans)
        
        st.session_state.messages.append({"role": "assistant", "content": ans})

    if st.button("Clear Chat", type="minor"):
        st.session_state.messages = []
        st.session_state.match_details = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- 11. Statistics & Product View Placeholders ---
if choice == "Product View":
    st.title("📚 Product Catalog")
    st.dataframe(df_qa[['Product', 'Category', 'Question']].dropna())

if choice == "Statistics":
    st.title("📊 Usage Analytics")
    st.metric("Logged In User", st.session_state.user_email)
    st.info("GCS Logging is active. Every query is currently being recorded to chat_logs.csv.")
