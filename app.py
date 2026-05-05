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

# CSS: Cleaned up to prevent layout jumping and "ghost" elements
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    
    /* Styling for the Right Pane container */
    [data-testid="column"]:nth-child(2) {
        border-left: 1px solid #dee2e6;
        padding-left: 20px;
    }

    /* Hide standard Streamlit branding for a cleaner 'Portal' look */
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Ensure the chat input doesn't overlap content */
    .stChatInput {
        padding-bottom: 20px;
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
        df.loc[0] = ["Sample", "Knowledge base not found. Please upload knowledge_base.csv"]
    
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
        st.bar_chart(df_logs['Question'].value_counts().head(10))
        st.dataframe(df_logs, use_container_width=True)
    else:
        st.info("No logs found yet.")
else:
    # THE CORE LAYOUT: Content (Left) | Bot (Right)
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

    # --- THE RIGHT SIDE BOT (GURUCOOL) ---
    with bot_col:
        st.subheader("🪐 GuruCool AI")
        
        # Fixed height container for the conversation
        chat_history_container = st.container(height=500)
        
        with chat_history_container:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
            
            # Show suggestions if any
            if st.session_state.temp_results:
                st.write("---")
                st.caption("Common matches:")
                for r in st.session_state.temp_results:
                    if st.button(f"👉 {r['q']}", key=f"sug_btn_{r['idx']}", use_container_width=True):
                        log_usage(r['q'], st.session_state.user_email)
                        st.session_state.messages.append({"role": "assistant", "content": f"**{r['q']}**\n\n{r['a']}"})
                        st.session_state.temp_results = []
                        st.rerun()

        # Chat Input (Pinned to the bottom of the column)
        if prompt := st.chat_input("Ask GuruCool...", key="bot_input"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.temp_results = []
            
            # Process response
            clean_p = prompt.lower().strip().translate(str.maketrans('', '', string.punctuation))
            small_talk = {"hi": "Hello!", "hello": "Hi there!", "thanks": "You're welcome!", "bye": "Goodbye!"}
            
            if clean_p in small_talk:
                st.session_state.messages.append({"role": "assistant", "content": small_talk[clean_p]})
            else:
                results = get_matches_nlp(prompt, df_kb)
                if results:
                    # If high confidence, give answer directly
                    if results[0]['score'] > 0.6:
                        log_usage(results[0]['q'], st.session_state.user_email)
                        st.session_state.messages.append({"role": "assistant", "content": f"**{results[0]['q']}**\n\n{results[0]['a']}"})
                    else:
                        # If low confidence, show suggestions
                        st.session_state.messages.append({"role": "assistant", "content": "I found a few related topics. Please click one:"})
                        st.session_state.temp_results = results
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "I'm sorry, I couldn't find a match in the knowledge base."})
            
            st.rerun()

        # Utility button
        if st.button("Clear Chat", key="reset_chat", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm GuruCool. How can I help?"}]
            st.session_state.temp_results = []
            st.rerun()
