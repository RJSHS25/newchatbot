import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
import uuid
import os
import io
import nltk
import string
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. Page Config & Floating UI CSS ---
st.set_page_config(layout="wide", page_title="TechM GuruCool Prototype")

st.markdown("""
    <style>
    .stApp { background-color: #f4f7f9; }
    
    /* Floating Chat Container */
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

# --- 2. NLTK Setup ---
@st.cache_resource
def setup_nltk():
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

setup_nltk()

# --- 3. Local Data Loading (Replaced GCS) ---
@st.cache_data
def load_local_data():
    # Load Main Q&A Data
    if os.path.exists("data.csv"):
        df = pd.read_csv("data.csv")
    else:
        # Create dummy data if file is missing for the prototype
        df = pd.DataFrame({
            'Question': ['How to map linear boundaries?', 'What is polygon mapping?'],
            'Answer': ['Use the linear tool to trace boundaries.', 'Polygon mapping covers area geometry.'],
            'Category': ['Linear', 'Polygon'],
            'Product': ['Maps', 'Maps']
        })
    return df

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
    st.title("🚀 GuruCool Prototype")
    st.subheader("Login with Email (No Password Required)")
    email_input = st.text_input("Email Address:")
    if st.button("Enter Portal"):
        if "@" in email_input:
            st.session_state.authenticated = True
            st.session_state.user_email = email_input
            st.rerun()
        else:
            st.error("Please enter a valid email.")
    st.stop()

# --- 7. Execution ---
df_qa = load_local_data()
bot = FAQBot(df_qa)

# --- 8. Sidebar ---
with st.sidebar:
    st.title("Navigation")
    choice = st.radio("Go to", ["Dashboard", "Product View"])
    st.markdown("---")
    st.write(f"Logged in: **{st.session_state.user_email}**")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

# --- 9. Main Dashboard ---
if choice == "Dashboard":
    st.title("🗺️ Maps Knowledge Portal")
    st.markdown("### Welcome to the GuruCool Prototype")
    
    col1, col2, col3 = st.columns(3)
    with col1: st.info("#### 🛣️ Linear\nMapping line assets.")
    with col2: st.success("#### 🔷 Polygon\nMapping area assets.")
    with col3: st.warning("#### 🚦 Signals\nTraffic configurations.")

# --- 10. Floating Chatbot ---
st.markdown('<div class="floating-chat">', unsafe_allow_html=True)
st.markdown('<div class="bot-header">🪐 GuruCool Chatbot</div>', unsafe_allow_html=True)

chat_box = st.container(height=300)
for m in st.session_state.messages:
    with chat_box.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    results = bot.search(prompt)
    
    with chat_box.chat_message("assistant"):
        if results:
            if results[0]['score'] > 0.7:
                best = df_qa.iloc[results[0]['idx']]
                ans = f"**Answer:** {best['Answer']}"
                st.markdown(ans)
            else:
                ans = "I found these similar topics. Click to view:"
                st.markdown(ans)
                for r in results:
                    if st.button(f"🔗 {r['q']}", key=f"btn_{r['idx']}"):
                        best = df_qa.iloc[r['idx']]
                        st.session_state.messages.append({"role": "assistant", "content": f"**Answer for {r['q']}:** {best['Answer']}"})
                        st.rerun()
        else:
            ans = "No match found. Try typing 'Linear boundaries'."
            st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# --- 11. Product View ---
if choice == "Product View":
    st.title("📚 Product Information")
    st.dataframe(df_qa)
