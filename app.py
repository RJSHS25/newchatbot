import streamlit as st
import pandas as pd
import os
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. Page Config & Styles ---
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
    </style>
    """, unsafe_allow_html=True)

# --- 2. NLP & Context Setup ---
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

# --- 3. FAQ Bot Engine with Memory ---
class FAQBot:
    def __init__(self, df):
        self.df = df
        if df.empty:
            self.vectorizer = None
            return
        self.vectorizer = TfidfVectorizer(preprocessor=tfidf_preprocess, ngram_range=(1, 3))
        self.vectors = self.vectorizer.fit_transform(df['Question'].astype(str))

    def search(self, query, context_q="", top_n=3):
        if self.df.empty or self.vectorizer is None: return []
        
        # CONVERSATIONAL LOGIC: Merge current query with previous context if it's short
        # (e.g., "How to open one?" + "Savings Account")
        combined_query = f"{context_q} {query}" if context_q else query
        
        query_vec = self.vectorizer.transform([combined_query])
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
    return pd.DataFrame({'Question': ['Example?'], 'Answer': ['Example Answer.']})

df_qa = load_data()
bot = FAQBot(df_qa)

# --- 5. Session State (The "Brain") ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_topic' not in st.session_state:
    st.session_state.last_topic = ""
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# --- 6. Login ---
if not st.session_state.authenticated:
    st.title("🚀 GuruCool Prototype")
    email = st.text_input("Enter Email:")
    if st.button("Login"):
        if "@" in email:
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.rerun()
    st.stop()

# --- 7. Main Dashboard UI ---
st.title("🗺️ Maps Knowledge Portal")
st.info(f"Conversational Mode Active | User: {st.session_state.user_email}")

# --- 8. Floating Chatbot ---
st.markdown('<div class="floating-chat">', unsafe_allow_html=True)
st.markdown('<div class="bot-header">🪐 GuruCool AI Support</div>', unsafe_allow_html=True)

chat_box = st.container(height=380)

# Display Chat History
for m in st.session_state.messages:
    with chat_box.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat Input Processing
if prompt := st.chat_input("Ask me a question..."):
    # Show User Input
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # SEARCH WITH CONTEXT
    results = bot.search(prompt, context_q=st.session_state.last_topic)
    
    if results:
        # High Confidence Match
        if results[0]['score'] > 0.55:
            res = results[0]
            answer_text = f"**{res['q']}**\n\n{res['a']}"
            st.session_state.messages.append({"role": "assistant", "content": answer_text})
            # UPDATE MEMORY: Remember this topic for follow-up
            st.session_state.last_topic = res['q']
        else:
            # Low Confidence: Show options
            st.session_state.messages.append({"role": "assistant", "content": "I'm not 100% sure, but are you asking about one of these?"})
            st.session_state.temp_results = results
    else:
        st.session_state.messages.append({"role": "assistant", "content": "I couldn't find a match. Could you try rephrasing?"})
        st.session_state.last_topic = "" # Clear memory if search fails
    
    st.rerun()

# Handle Button Clicks (Options)
if 'temp_results' in st.session_state and st.session_state.temp_results:
    with chat_box.chat_message("assistant"):
        for r in st.session_state.temp_results:
            if st.button(f"👉 {r['q']}", key=f"btn_{r['idx']}"):
                final_answer = f"**{r['q']}**\n\n{r['a']}"
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                # UPDATE MEMORY: Remember this selection
                st.session_state.last_topic = r['q']
                st.session_state.temp_results = []
                st.rerun()

if st.button("Clear History"):
    st.session_state.messages = []
    st.session_state.last_topic = ""
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
