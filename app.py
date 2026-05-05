import streamlit as st
import pandas as pd
import os
import nltk
import string
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Page Config ---
st.set_page_config(layout="wide", page_title="TechM GuruCool")

# --- 2. Sidebar Navigation ---
with st.sidebar:
    st.title("🧭 Navigation")
    page = st.radio("Go to:", ["Chatbot", "Usage Analytics"])
    st.divider()

# --- 3. Shared Functions (Logging & Data) ---
def log_usage(question, user_email):
    log_file = "usage_logs.csv"
    new_entry = pd.DataFrame({
        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'User': [user_email],
        'Question': [question]
    })
    if not os.path.isfile(log_file):
        new_entry.to_csv(log_file, index=False)
    else:
        new_entry.to_csv(log_file, mode='a', header=False, index=False)

@st.cache_data
def load_data():
    if os.path.exists("data.csv"): return pd.read_csv("data.csv")
    return pd.DataFrame({'Question': ['Sample?'], 'Answer': ['Sample Answer.']})

# --- 4. Logic for ANALYTICS PAGE ---
if page == "Usage Analytics":
    st.title("📊 Usage Analytics")
    if os.path.exists("usage_logs.csv"):
        df = pd.read_csv("usage_logs.csv")
        if not df.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Top Questions")
                st.bar_chart(df['Question'].value_counts().head(10))
            with c2:
                st.subheader("User Activity")
                st.write(df['User'].value_counts())
            st.subheader("Raw Logs")
            st.dataframe(df.sort_values(by='Timestamp', ascending=False), use_container_width=True)
        else:
            st.info("No logs yet.")
    else:
        st.error("Log file not found.")

# --- 5. Logic for CHATBOT PAGE ---
else:
    # --- CSS & NLP (Only loaded when on Chat page) ---
    st.markdown("<style>.floating-chat { position: fixed; bottom: 20px; right: 20px; width: 450px; background: white; border-radius: 15px; box-shadow: 0px 10px 25px rgba(0,0,0,0.1); z-index: 1000; border: 1px solid #e0e0e0; padding: 10px; } .bot-header { background: #0078d4; color: white; padding: 10px; border-radius: 10px 10px 0 0; font-weight: bold; margin: -10px -10px 10px -10px; }</style>", unsafe_allow_html=True)
    
    # NLP Setup & Bot Logic (Keep your existing classes here)
    # ... [Insert your FAQBot class and tfidf_preprocess function here] ...
    
    # [Insert the rest of your Chatbot UI logic here]
    st.title("🗺️ Maps Knowledge Portal")
    # (The rest of the code for Login, Chat History, and Input as provided previously)
