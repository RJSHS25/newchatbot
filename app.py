import streamlit as st
import pandas as pd
import os
import csv
import string
from datetime import datetime
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# ⚙️ CONFIG & SESSION STATE
# ===============================
st.set_page_config(layout="wide", page_title="TechM Maps Portal")

# CSS: Improved to anchor the bot column and clean up the input
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .right-pane {
        border-left: 1px solid #dee2e6;
        padding: 20px;
        background-color: #ffffff;
        min-height: 90vh;
    }
    /* This makes sure chat elements don't overflow the column */
    [data-testid="column"] {
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm GuruCool. How can I help you today?"}]
if 'temp_results' not in st.session_state:
    st.session_state.temp_results = []

# ===============================
# 🔐 AUTHENTICATION
# ===============================
if not st.session_state.authenticated:
    st.title("🔐 TechM Portal Login")
    email = st.text_input("Enter Email:")
    if st.button("Login"):
        if "@" in email:
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.rerun()
    st.stop()

# ===============================
# 📄 DATA & NLP ENGINE
# ===============================
def log_usage(question, user_email):
    log_file = "usage_logs.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(['Timestamp', 'User', 'Question'])
        writer.writerow([timestamp, user_email, question])

@st.cache_data
def load_and_prep_data():
    file_path = "knowledge_base.csv" if os.path.exists("knowledge_base.csv") else "data.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=['Topic', 'Description'])
        df.loc[0] = ["Sample", "Please upload knowledge_base.csv to see real data."]
    
    if 'Question' in df.columns:
        df.rename(columns={'Question': 'Topic', 'Answer': 'Description'}, inplace=True)
    df['profile'] = df['Topic'].fillna('') + " " + df['Description'].fillna('')
    return df

df_kb = load_and_prep_data()

def get_matches_nlp(query, dataframe, top_n=3):
    if dataframe.empty: return []
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(dataframe['profile'])
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    indices = cosine_sim.argsort()[-top_n:][::-1]
    results = []
    for idx in indices:
        if cosine_sim[idx] > 0.1:
            results.append({"idx": idx, "score": cosine_sim[idx], "q": dataframe.iloc[idx]['Topic'], "a": dataframe.iloc[idx]['Description']})
    return results

# ===============================
# ⬅️ SIDEBAR (Analytics Only)
# ===============================
with st.sidebar:
    st.title("🧭 Navigation")
    nav_choice = st.radio("View:", ["🏠 Dashboard", "📊 Analytics"])
    st.divider()
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

# ===============================
# 🏗️ MAIN CONTENT AREA
# ===============================
if nav_choice == "📊 Analytics":
    st.title("📊 Usage Analytics")
    if os.path.exists("usage_logs.csv"):
        df_logs = pd.read_csv("usage_logs.csv")
        st.bar_chart(df_logs['Question'].value_counts().head(10))
        st.dataframe(df_logs, use_container_width=True)
    else:
        st.info("No logs yet.")

else:
    # THE CORE LAYOUT
    main_col, bot_col = st.columns([0.7, 0.3])

    with main_col:
        st.markdown("<h2 style='color:#0078d4;'>🗺️ Maps Knowledge Portal</h2>", unsafe_allow_html=True)
        st.divider()
        st.video("https://www.youtube.com/watch?v=hA_-MkU0Nfw")
        st.markdown("### 🚀 Domains")
        d1, d2, d3 = st.columns(3)
        if d1.button("Linear", use_container_width=True): st.info("Opening Linear...")
        if d2.button("Polygon", use_container_width=True): st.info("Opening Polygon...")
        if d3.button("Signals", use_container_width=True): st.info("Opening Signals...")

    # --- RIGHT SIDE BOT (The ONLY Input) ---
    with bot_col:
        st.markdown('<div class="right-pane">', unsafe_allow_html=True)
        st.subheader("🪐 GuruCool AI")
        
        # We wrap the history and input in a specific container logic
        history_placeholder = st.container(height=500)
        
        with history_placeholder:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
            
            # Clickable Suggestions
            if st.session_state.temp_results:
                for r in st.session_state.temp_results:
                    if st.button(f"👉 {r['q']}", key=f"sug_{r['idx']}"):
                        log_usage(r['q'], st.session_state.user_email)
                        st.session_state.messages.append({"role": "assistant", "content": f"**{r['q']}**\n\n{r['a']}"})
                        st.session_state.temp_results = []
                        st.rerun()

        # BY NESTING THIS INSIDE 'with bot_col', it stays in the right pane.
        # DO NOT call st.chat_input anywhere else in the code.
        if prompt := st.chat_input("Ask GuruCool..."):
            st.session_state.temp_results = []
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            clean_p = prompt.lower().strip().translate(str.maketrans('', '', string.punctuation))
            small_talk = {"hi": "Hello!", "hello": "Hi there!", "thanks": "Welcome!"}
            
            if clean_p in small_talk:
                st.session_state.messages.append({"role": "assistant", "content": small_talk[clean_p]})
            else:
                results = get_matches_nlp(prompt, df_kb)
                if results:
                    if results[0]['score'] > 0.6:
                        log_usage(results[0]['q'], st.session_state.user_email)
                        st.session_state.messages.append({"role": "assistant", "content": f"**{results[0]['q']}**\n\n{results[0]['a']}"})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "I found these similar topics:"})
                        st.session_state.temp_results = results
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "I'm not sure. Try another keyword."})
            st.rerun()

        if st.button("Clear Chat", key="clear_bot"):
            st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm GuruCool."}]
            st.session_state.temp_results = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
