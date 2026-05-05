import streamlit as st
import pandas as pd
import os
import csv
from datetime import datetime
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# ⚙️ CONFIG & SESSION STATE
# ===============================
st.set_page_config(layout="wide", page_title="TechM Maps Portal")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm GuruCool. How can I help you today?"}]

# Custom CSS for Right Pane Aesthetics
st.markdown("""
    <style>
    .right-pane {
        border-left: 1px solid #dee2e6;
        padding-left: 20px;
        height: 100%;
    }
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ===============================
# 🔐 AUTHENTICATION
# ===============================
if not st.session_state.authenticated:
    st.title("🔐 TechM Portal Login")
    email = st.text_input("Email")
    if st.button("Login"):
        if "@" in email:
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.rerun()
    st.stop()

# ===============================
# 📄 LOGGING & NLP ENGINE
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
    # Load knowledge_base.csv or data.csv
    if os.path.exists("knowledge_base.csv"):
        df = pd.read_csv("knowledge_base.csv")
    else:
        df = pd.DataFrame(columns=['Topic', 'Description', 'Category'])
    df['profile'] = df['Topic'].fillna('') + " " + df['Description'].fillna('')
    return df

df_kb = load_and_prep_data()

def get_best_match_nlp(query, dataframe):
    if dataframe.empty: return None, 0
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(dataframe['profile'])
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = cosine_sim.argsort()[-1]
    highest_score = cosine_sim[best_idx]
    
    if highest_score < 0.2:
        fuzzy_scores = dataframe['profile'].apply(lambda x: fuzz.partial_ratio(query.lower(), str(x).lower()))
        best_idx = fuzzy_scores.idxmax()
        highest_score = fuzzy_scores.max() / 100
    return dataframe.iloc[best_idx], highest_score

# ===============================
# ⬅️ LEFT SIDEBAR (Navigation & Analytics)
# ===============================
with st.sidebar:
    st.title("🧭 Navigation")
    st.write(f"👤 {st.session_state.user_email}")
    st.divider()
    
    # Internal Navigation Links
    nav_choice = st.radio("Go to:", ["🏠 Home Dashboard", "📊 Usage Analytics"])
    
    st.divider()
    st.subheader("Quick Links")
    st.markdown("- [Company Portal](https://example.com)")
    st.markdown("- [Maps Documentation](https://example.com)")
    
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

# ===============================
# 🏗️ MAIN LAYOUT LOGIC
# ===============================

# 1. SHOW ANALYTICS PAGE
if nav_choice == "📊 Usage Analytics":
    st.title("📊 Usage Analytics")
    if os.path.exists("usage_logs.csv"):
        df_logs = pd.read_csv("usage_logs.csv")
        st.bar_chart(df_logs['Question'].value_counts().head(10))
        st.dataframe(df_logs, use_container_width=True)
    else:
        st.info("No data logged yet.")

# 2. SHOW DASHBOARD + RIGHT CHATBOT
else:
    # Split the main area into Content (70%) and Chatbot (30%)
    main_col, bot_col = st.columns([0.7, 0.3])

    with main_col:
        st.markdown("## 🗺️ Maps Knowledge Portal")
        st.markdown("---")
        st.video("https://www.youtube.com/watch?v=hA_-MkU0Nfw")
        
        st.markdown("### 🚀 Choose Your Domain")
        d1, d2, d3 = st.columns(3)
        domain_data = [
            {"name": "Linear", "col": d1, "img": "images/linear.png", "path": "pages/1_Linear.py"},
            {"name": "Polygon", "col": d2, "img": "images/polygon.png", "path": "pages/2_Polygon.py"},
            {"name": "Signals", "col": d3, "img": "images/signals.png", "path": "pages/3_Signals.py"}
        ]

        for dom in domain_data:
            with dom["col"]:
                if os.path.exists(dom["img"]):
                    st.image(dom["img"], use_container_width=True)
                st.subheader(dom["name"])
                if st.button(f"Open {dom['name']}", key=dom['name']):
                    st.switch_page(dom["path"])

    with bot_col:
        st.markdown('<div class="right-pane">', unsafe_allow_html=True)
        st.subheader("🪐 GuruCool Support")
        
        chat_box = st.container(height=550)
        with chat_box:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        if prompt := st.chat_input("Ask GuruCool..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            result, score = get_best_match_nlp(prompt, df_kb)
            if score > 0.25:
                log_usage(result['Topic'], st.session_state.user_email)
                bot_response = f"**{result['Topic']}**\n\n{result['Description']}"
            else:
                bot_response = "I couldn't find an exact match. Try 'Linear' or 'Signals'."
                
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            st.rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)
