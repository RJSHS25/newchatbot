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
    </style>
    """, unsafe_allow_html=True)

# --- 2. NLP Setup ---
@st.cache_resource
def setup_nltk():
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        return nltk.stem.WordNetLemmatizer()
    except:
        return None

lemmatizer = setup_nltk()

def tfidf_preprocess(text):
    if pd.isna(text) or text == "": return ""
    # Remove punctuation and lower case
    text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
    # Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

# --- 3. FAQ Bot Engine ---
class FAQBot:
    def __init__(self, df):
        self.df = df
        if df.empty:
            self.vectorizer = None
            self.vectors = None
            return
        
        # We use a broader ngram range to capture phrases better
        self.vectorizer = TfidfVectorizer(preprocessor=tfidf_preprocess, ngram_range=(1, 2))
        # Ensure we are fitting on the actual Question column
        self.vectors = self.vectorizer.fit_transform(self.df['Question'].astype(str))

    def search(self, query, context="", top_n=3):
        if self.vectorizer is None or self.vectors is None: return []
        
        # Combine current query with context for conversational memory
        full_query = f"{context} {query}".strip()
        query_vec = self.vectorizer.transform([full_query])
        
        # Calculate similarities
        sims = cosine_similarity(query_vec, self.vectors).flatten()
        
        # Get indices of top matches
        indices = sims.argsort()[-top_n:][::-1]
        
        results = []
        for i in indices:
            score = sims[i]
            if score > 0.15: # Confidence threshold to prevent "same answer every time"
                results.append({
                    "idx": i, 
                    "score": score, 
                    "q": self.df.iloc[i]['Question'], 
                    "a": self.df.iloc[i]['Answer']
                })
        return results

# --- 4. Data Loading ---
@st.cache_data
def load_data():
    if os.path.exists("data.csv"):
        df = pd.read_csv("data.csv")
        # Critical: Clean the data to remove empty rows that confuse the NLP
        df = df.dropna(subset=['Question', 'Answer'])
        return df
    return pd.DataFrame(columns=['Question', 'Answer'])

df_qa = load_data()
bot = FAQBot(df_qa)

# --- 5. Session State ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_q' not in st.session_state:
    st.session_state.last_q = ""
if 'auth' not in st.session_state:
    st.session_state.auth = False

# --- 6. Simplified Login ---
if not st.session_state.auth:
    st.title("🚀 GuruCool Prototype")
    email = st.text_input("Enter Email:")
    if st.button("Login"):
        if "@" in email:
            st.session_state.auth = True
            st.session_state.user_email = email
            st.rerun()
    st.stop()

# --- 7. UI Dashboard ---
st.title("🗺️ Maps Knowledge Portal")

# --- 8. Floating Chatbot ---
st.markdown('<div class="floating-chat">', unsafe_allow_html=True)
st.markdown('<div class="bot-header">🪐 GuruCool AI Support</div>', unsafe_allow_html=True)

chat_box = st.container(height=380)

for m in st.session_state.messages:
    with chat_box.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Analyze with context
    results = bot.search(prompt, context=st.session_state.last_q)
    
    with chat_box.chat_message("assistant"):
        if results:
            # If the top match is distinct (Confidence Gate)
            if results[0]['score'] > 0.4:
                res = results[0]
                ans = f"**{res['q']}**\n\n{res['a']}"
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                st.session_state.last_q = res['q'] # Remember context
            else:
                # Ambiguous match: Show suggestions
                st.markdown("I'm not 100% sure. Did you mean:")
                for r in results:
                    if st.button(f"👉 {r['q']}", key=f"btn_{r['idx']}"):
                        final_ans = f"**{r['q']}**\n\n{r['a']}"
                        st.session_state.messages.append({"role": "assistant", "content": final_ans})
                        st.session_state.last_q = r['q']
                        st.rerun()
        else:
            msg = "I couldn't find a match for that. Could you try rephrasing?"
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.session_state.last_q = "" # Clear context on failure

if st.button("Clear History"):
    st.session_state.messages = []
    st.session_state.last_q = ""
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
