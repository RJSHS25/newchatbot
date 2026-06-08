import streamlit as st
import pandas as pd
import os
import csv
import string    
from datetime import datetime
from fuzzywuzzy import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# ⚙️ CONFIG & SESSION STATE
# ===============================
st.set_page_config(layout="wide", page_title="TechM Finance Portal")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    [data-testid="column"]:nth-child(2) {
        border-left: 1px solid #dee2e6;
        padding-left: 20px;
    }
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stChatInput { padding-bottom: 20px; }
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
# 📄 DATA & SEARCH ENGINE
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
        df.loc[0] = ["Sample", "Knowledge base empty."]
    
    if 'Question' in df.columns:
        df.rename(columns={'Question': 'Topic', 'Answer': 'Description'}, inplace=True)
    df['profile'] = df['Topic'].fillna('') + " " + df['Description'].fillna('')
    return df

df_kb = load_and_prep_data()

def get_combined_matches(query, dataframe, top_n=5):

    if dataframe.empty:
        return []

    query_clean = query.lower().strip()

    results = []

    # ==========================
    # 1. Exact Match
    # ==========================
    exact_matches = dataframe[
        dataframe['Topic']
        .astype(str)
        .str.lower()
        == query_clean
    ]

    for idx, row in exact_matches.iterrows():
        results.append({
            "score": 1.0,
            "q": row['Topic'],
            "a": row['Description'],
            "idx": idx
        })

    # ==========================
    # 2. Partial Match
    # ==========================
    if not results:

        partial_matches = dataframe[
            dataframe['Topic']
            .astype(str)
            .str.lower()
            .str.contains(query_clean, na=False)
        ]

        for idx, row in partial_matches.iterrows():
            results.append({
                "score": 0.95,
                "q": row['Topic'],
                "a": row['Description'],
                "idx": idx
            })

    # ==========================
    # 3. Fuzzy Match Fallback
    # ==========================
    if not results:

        choices = dataframe['Topic'].astype(str).tolist()

        fuzzy_results = process.extract(
            query,
            choices,
            scorer=fuzz.token_set_ratio,
            limit=top_n
        )

        for match in fuzzy_results:

            match_text = match[0]
            score = match[1]

            if score > 70:

                idx_list = dataframe.index[
                    dataframe['Topic'] == match_text
                ].tolist()

                if idx_list:

                    idx = idx_list[0]

                    results.append({
                        "score": score / 100,
                        "q": dataframe.iloc[idx]['Topic'],
                        "a": dataframe.iloc[idx]['Description'],
                        "idx": idx
                    })

    return results


# ===============================
# ⬅️ SIDEBAR
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
        st.dataframe(df_logs, use_container_width=True)
    else:
        st.info("No logs found yet.")
else:
    chat_col, nav_col = st.columns([0.75, 0.25])

    with nav_col:
        st.markdown("<h2 style='color:#0078d4;'>🗺️ Tech Mahindra Finance </h2>", unsafe_allow_html=True)
        st.divider()
        st.video("https://www.youtube.com/watch?v=malw6c993qs")
        
        st.markdown("### 🚀 Domains")
        d1, d2, d3 = st.columns(3)
        if st.button("💰 Finance", use_container_width=True):
            st.info("Opening Finance")
        
        if st.button("📒 Accounts", use_container_width=True):
            st.info("Opening Accounts")
        
        if st.button("👤 Onboarding", use_container_width=True):
            st.info("Opening Onboarding")

    # --- THE RIGHT SIDE BOT (GURUCOOL) ---
    with chat_col:
        st.subheader("🪐 GuruCool AI")
        chat_history_container = st.container(height=750)
        
        with chat_history_container:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
            
            if st.session_state.temp_results:
                st.write("---")
                st.caption("Common matches:")
                for r in st.session_state.temp_results:
                    if st.button(f"👉 {r['q']}", key=f"sug_btn_{r['idx']}", use_container_width=True):
                        log_usage(r['q'], st.session_state.user_email)
                        st.session_state.messages.append({"role": "assistant", "content": f"**{r['q']}**\n\n{r['a']}"})
                        st.session_state.temp_results = []
                        st.rerun()

        if prompt := st.chat_input("Ask GuruCool...", key="bot_input"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.temp_results = []
            
            clean_p = prompt.lower().strip().translate(str.maketrans('', '', string.punctuation))
            small_talk = {"hi": "Hello!", "hello": "Hi there!", "thanks": "You're welcome!"}
            
            if clean_p in small_talk:
                st.session_state.messages.append({"role": "assistant", "content": small_talk[clean_p]})
            else:
                # This is the line that was crashing
                results = get_combined_matches(prompt, df_kb)
                if results:
                    if results[0]['score'] > 0.7:
                        log_usage(results[0]['q'], st.session_state.user_email)
                        st.session_state.messages.append({"role": "assistant", "content": f"**{results[0]['q']}**\n\n{results[0]['a']}"})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "I found a few related topics:"})
                        st.session_state.temp_results = results
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "I couldn't find a match. Could you rephrase?"})
            st.rerun()

        if st.button("Clear Chat", key="reset_chat", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm GuruCool."}]
            st.session_state.temp_results = []
            st.rerun()
