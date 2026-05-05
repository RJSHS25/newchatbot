import streamlit as st
import pandas as pd
import os
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. Page Config & CSS ---
st.set_page_config(layout="wide", page_title="TechM GuruCool Prototype")

st.markdown("""
    <style>
    .stApp { background-color: #f4f7f9; }
    .floating-chat {
        position: fixed; bottom: 20px; right: 20px; width: 450px;
        background: white; border-radius: 15px;
        box-shadow: 0px 10px 25px rgba(0,0,0,0.1);
        z-index: 1000; border: 1px solid #e0e0e0; padding: 10px;
    }
    .bot-header {
        background: #0078d4; color: white; padding: 10px;
        border-radius: 10px 10px 0 0; font-weight: bold; margin: -10px -10px 10px -10px;
    }
    .chat-container { height: 400px; overflow-y: auto; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. NLP Setup ---
@st.cache_resource
def setup_nltk():
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

setup_nltk()
lemmatizer = nltk.stem.WordNetLemmatizer()

def tfidf_preprocess(text):
    if pd.isna(text): return ""
    text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

# --- 3. FAQ Bot Engine ---
class FAQBot:
    def __init__(self, df):
        self.df = df
        if df.empty:
            self.vectorizer = None
            return
        self.vectorizer = TfidfVectorizer(preprocessor=tfidf_preprocess, ngram_range=(1, 3))
        self.vectors = self.vectorizer.fit_transform(df['Question'].astype(str))

    def search(self, query, top_n=3):
        if self.df.empty or self.vectorizer is None: return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.vectors).flatten()
        indices = sims.argsort()[-top_n:][::-1]
        
        results = []
        for i in indices:
            if sims[i] > 0.1:
                results.append({
                    "idx": i, 
                    "score": sims[i], 
                    "q": self.df.iloc[i]['Question'], 
                    "a": self.df.iloc[i]['Answer']
                })
        return results

# --- 4. Load Data ---
@st.cache_data
def load_data():
    if os.path.exists("data.csv"):
        return pd.read_csv("data.csv")
    return pd.DataFrame({
        'Question': ['How to reset password?', 'Where is the office?'], 
        'Answer': ['Go to settings > security.', 'It is in Hyderabad.']
    })

df_qa = load_data()
bot = FAQBot(df_qa)

# --- 5. Session State ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'temp_results' not in st.session_state:
    st.session_state.temp_results = []

# --- 6. Prototype Login ---
if not st.session_state.authenticated:
    st.title("🚀 GuruCool Prototype")
    email = st.text_input("Enter Email to start:")
    if st.button("Login"):
        if "@" in email:
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.rerun()
    st.stop()

# --- 7. Sidebar & Main UI ---
with st.sidebar:
    st.title("Settings")
    if st.button("Clear History"):
        st.session_state.messages = []
        st.session_state.temp_results = []
        st.rerun()

st.title("🗺️ Maps Knowledge Portal")
st.info(f"Welcome, {st.session_state.user_email}")

# --- 8. Floating Chatbot Logic ---
# Wrap everything in a container to maintain visual structure
st.markdown('<div class="floating-chat">', unsafe_allow_html=True)
st.markdown('<div class="bot-header">🪐 GuruCool AI Support</div>', unsafe_allow_html=True)

chat_box = st.container(height=380)

# A. Render Chat History
with chat_box:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # B. Render Suggestion Buttons (if any exist in state)
    if st.session_state.temp_results:
        with st.chat_message("assistant"):
            st.write("I found these related topics. Click one to view the answer:")
            for r in st.session_state.temp_results:
                if st.button(f"👉 {r['q']}", key=f"btn_{r['idx']}"):
                    # Update History with the answer
                    final_answer = f"**{r['q']}**\n\n{r['a']}"
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    # Clear suggestions and refresh
                    st.session_state.temp_results = []
                    st.rerun()

# C. Chat Input Processing
if prompt := st.chat_input("Ask about savings, mapping, etc..."):
    # Clear suggestions if user types a new question
    st.session_state.temp_results = []
    
    # 1. Add User Message to History
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Perform Search
    results = bot.search(prompt)
    
    # 3. Handle Logic
    if results:
        # If very high match, give answer directly
        if results[0]['score'] > 0.8:
            answer_text = f"**{results[0]['q']}**\n\n{results[0]['a']}"
            st.session_state.messages.append({"role": "assistant", "content": answer_text})
        else:
            # Otherwise, store suggestions to be rendered in the next cycle
            options_msg = "I found a few matches. Which one are you looking for?"
            st.session_state.messages.append({"role": "assistant", "content": options_msg})
            st.session_state.temp_results = results
    else:
        st.session_state.messages.append({"role": "assistant", "content": "I couldn't find an answer for that. Try rephrasing your question."})
    
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
