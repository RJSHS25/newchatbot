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

# --- 2. NLP & Bot Logic (Cached for Speed) ---
@st.cache_resource
def setup_nltk():
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except: pass

setup_nltk()
lemmatizer = nltk.stem.WordNetLemmatizer()

def tfidf_preprocess(text):
    if pd.isna(text): return ""
    text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

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
                results.append({"idx": i, "score": sims[i], "q": self.df.iloc[i]['Question'], "a": self.df.iloc[i]['Answer']})
        return results

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
    return pd.DataFrame({'Question': ['How to test?'], 'Answer': ['Type a question in chat!']})

# --- 3. Initialize Data & State ---
df_qa = load_data()
bot = FAQBot(df_qa)

if 'messages' not in st.session_state: st.session_state.messages = []
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'temp_results' not in st.session_state: st.session_state.temp_results = []

# --- 4. Sidebar Navigation ---
with st.sidebar:
    st.title("🧭 Navigation")
    # This radio button acts as our "Page Switcher"
    choice = st.radio("Switch View:", ["💬 Chatbot", "📊 Analytics"])
    st.divider()
    if st.button("Clear History"):
        st.session_state.messages = []
        st.session_state.temp_results = []
        st.rerun()

# --- 5. Login Gate ---
if not st.session_state.authenticated:
    st.title("🚀 GuruCool Prototype")
    email = st.text_input("Enter Email to start:")
    if st.button("Login"):
        if "@" in email:
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.rerun()
    st.stop()

# --- 6. PAGE LOGIC ---

if choice == "📊 Analytics":
    st.title("📊 Usage Analytics")
    if os.path.exists("usage_logs.csv"):
        df_logs = pd.read_csv("usage_logs.csv")
        if not df_logs.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top 10 Questions")
                st.bar_chart(df_logs['Question'].value_counts().head(10))
            with col2:
                st.subheader("Active Users")
                st.write(df_logs['User'].value_counts())
            st.subheader("Full Log History")
            st.dataframe(df_logs.sort_values(by='Timestamp', ascending=False), use_container_width=True)
        else:
            st.info("No data logged yet.")
    else:
        st.warning("No log file found yet. Use the chatbot first!")

else:
    # CHATBOT PAGE
    st.title("🗺️ Maps Knowledge Portal")
    st.info(f"Logged in as: {st.session_state.user_email}")

    # CSS for Floating Chat
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

    st.markdown('<div class="floating-chat">', unsafe_allow_html=True)
    st.markdown('<div class="bot-header">🪐 GuruCool AI Support</div>', unsafe_allow_html=True)

    chat_box = st.container(height=380)

    with chat_box:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if st.session_state.temp_results:
            with st.chat_message("assistant"):
                st.write("Please select the most relevant question:")
                for r in st.session_state.temp_results:
                    if st.button(f"👉 {r['q']}", key=f"btn_{r['idx']}"):
                        log_usage(r['q'], st.session_state.user_email)
                        st.session_state.messages.append({"role": "assistant", "content": f"**{r['q']}**\n\n{r['a']}"})
                        st.session_state.temp_results = []
                        st.rerun()

    if prompt := st.chat_input("Ask about savings, mapping, etc..."):
        st.session_state.temp_results = []
        st.session_state.messages.append({"role": "user", "content": prompt})
        results = bot.search(prompt)
        
        if results:
            if results[0]['score'] > 0.8:
                log_usage(results[0]['q'], st.session_state.user_email)
                st.session_state.messages.append({"role": "assistant", "content": f"**{results[0]['q']}**\n\n{results[0]['a']}"})
            else:
                st.session_state.messages.append({"role": "assistant", "content": "I found a few similar topics:"})
                st.session_state.temp_results = results
        else:
            st.session_state.messages.append({"role": "assistant", "content": "I couldn't find a match. Try rephrasing."})
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
