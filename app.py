import streamlit as st
import pandas as pd
import os
import csv
import string
import nltk
from datetime import datetime
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# ⚙️ CONFIG & SESSION STATE
# ===============================
st.set_page_config(layout="wide", page_title="TechM Maps Portal")

# Custom CSS for UI Aesthetics
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .right-pane {
        border-left: 1px solid #dee2e6;
        padding-left: 25px;
        height: 100vh;
        background-color: #ffffff;
    }
    /* Style headers and dividers */
    .main-title { color: #0078d4; font-weight: bold; }
    hr { margin-top: 1rem; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# Initialize Session States
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
    email = st.text_input("Enter Email to start:")
    if st.button("Login"):
        if "@" in email:
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.rerun()
        else:
            st.error("Please enter a valid email.")
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
    # Attempt to load knowledge_base.csv (or your data.csv)
    file_path = "knowledge_base.csv" if os.path.exists("knowledge_base.csv") else "data.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=['Topic', 'Description', 'Category'])
        # Fallback if no file exists
        df.loc[0] = ["Linear", "Linear mapping relates to road lines and lanes.", "Linear"]
        
    # Ensure standard column names for searching
    if 'Question' in df.columns and 'Topic' not in df.columns:
        df.rename(columns={'Question': 'Topic', 'Answer': 'Description'}, inplace=True)
        
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
    
    # Fallback to Fuzzy Logic for conversational or low-score queries
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
    st.write(f"👤 **User:** {st.session_state.user_email}")
    st.divider()
    
    nav_choice = st.radio("Switch View:", ["🏠 Dashboard", "📊 Usage Analytics"])
    
    st.divider()
    st.subheader("Support Links")
    st.markdown("[📋 Company Guidelines](https://example.com)")
    st.markdown("[🛠️ Tooling Documentation](https://example.com)")
    
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

# ===============================
# 🏗️ MAIN CONTENT AREA
# ===============================

# PAGE: ANALYTICS
if nav_choice == "📊 Usage Analytics":
    st.title("📊 Usage Analytics")
    if os.path.exists("usage_logs.csv"):
        try:
            df_logs = pd.read_csv("usage_logs.csv", on_bad_lines='skip')
            if not df_logs.empty:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Top Topics Queried")
                    st.bar_chart(df_logs['Question'].value_counts().head(10))
                with c2:
                    st.subheader("Most Active Users")
                    st.write(df_logs['User'].value_counts())
                st.divider()
                st.subheader("Recent Search History")
                st.dataframe(df_logs.sort_values(by='Timestamp', ascending=False), use_container_width=True)
            else:
                st.info("Log file is empty.")
        except:
            st.error("Error reading logs. File may be corrupted.")
    else:
        st.warning("No usage recorded yet.")

# PAGE: DASHBOARD + RIGHT CHATBOT
else:
    # 70% Content | 30% Chatbot
    main_col, bot_col = st.columns([0.7, 0.3])

    with main_col:
        st.markdown("<h2 class='main-title'>🗺️ Maps Knowledge Portal</h2>", unsafe_allow_html=True)
        st.divider()
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
                    try:
                        st.switch_page(dom["path"])
                    except:
                        st.error(f"Page '{dom['path']}' not found.")

    with bot_col:
        st.markdown('<div class="right-pane">', unsafe_allow_html=True)
        st.subheader("🪐 GuruCool Support")
        
        # Small Talk Definitions
        small_talk = {
            "hi": "Hello! I'm GuruCool. How can I help you today?",
            "hello": "Hi there! What can I help you find in the portal?",
            "thanks": "You're very welcome!",
            "thank you": "Happy to help!",
            "how are you": "I'm doing great! Ready to analyze some map data."
        }

        chat_box = st.container(height=500)
        with chat_box:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        # Process Input
        if prompt := st.chat_input("Ask GuruCool..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Clean prompt for Small Talk check
            clean_p = prompt.lower().strip().translate(str.maketrans('', '', string.punctuation))
            
            if clean_p in small_talk:
                bot_response = small_talk[clean_p]
            else:
                result, score = get_best_match_nlp(prompt, df_kb)
                
                if score > 0.15: # Flexible threshold
                    log_usage(result['Topic'], st.session_state.user_email)
                    bot_response = f"**{result['Topic']}**\n\n{result['Description']}"
                else:
                    bot_response = "I'm not exactly sure about that. Try asking about 'Linear', 'Polygon', or specific mapping tasks."
            
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            st.rerun()

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm GuruCool. How can I help?"}]
            st.rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)
