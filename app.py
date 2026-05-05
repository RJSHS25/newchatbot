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

# Custom CSS for UI Aesthetics (Removed Floating Bot CSS)
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .right-pane {
        border-left: 1px solid #dee2e6;
        padding-left: 25px;
        height: 100vh;
        background-color: #ffffff;
    }
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
    file_path = "knowledge_base.csv" if os.path.exists("knowledge_base.csv") else "data.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=['Topic', 'Description', 'Category'])
        df.loc[0] = ["Linear", "Linear mapping relates to road lines and lanes.", "Linear"]
        
    if 'Question' in df.columns and 'Topic' not in df.columns:
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
        score = cosine_sim[idx]
        if score > 0.1:
            results.append({"idx": idx, "score": score, "q": dataframe.iloc[idx]['Topic'], "a": dataframe.iloc[idx]['Description']})
    return results

# ===============================
# ⬅️ LEFT SIDEBAR (Navigation)
# ===============================
with st.sidebar:
    st.title("🧭 Navigation")
    st.write(f"👤 **User:** {st.session_state.user_email}")
    st.divider()
    nav_choice = st.radio("Switch View:", ["🏠 Dashboard", "📊 Usage Analytics"])
    st.divider()
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

# ===============================
# 🏗️ MAIN CONTENT AREA
# ===============================
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
                st.dataframe(df_logs.sort_values(by='Timestamp', ascending=False), use_container_width=True)
        except:
            st.error("Error reading logs.")
    else:
        st.warning("No usage recorded yet.")

else:
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
                    st.switch_page(dom["path"])

    # RIGHT SIDEBAR CHATBOT
    with bot_col:
        st.markdown('<div class="right-pane">', unsafe_allow_html=True)
        st.subheader("🪐 GuruCool Support")
        
        small_talk = {
            "hi": "Hello! How can I help you today?",
            "hello": "Hi there! What are you looking for in the portal?",
            "thanks": "You're welcome!",
            "thank you": "Happy to help!"
        }

        chat_box = st.container(height=500)
        with chat_box:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

            # Render Suggestions Buttons
            if st.session_state.temp_results:
                st.info("Please select the most relevant topic:")
                for r in st.session_state.temp_results:
                    if st.button(f"👉 {r['q']}", key=f"btn_{r['idx']}"):
                        log_usage(r['q'], st.session_state.user_email)
                        st.session_state.messages.append({"role": "assistant", "content": f"**{r['q']}**\n\n{r['a']}"})
                        st.session_state.temp_results = []
                        st.rerun()

        if prompt := st.chat_input("Ask GuruCool..."):
            st.session_state.temp_results = []
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            clean_p = prompt.lower().strip().translate(str.maketrans('', '', string.punctuation))
            
            if clean_p in small_talk:
                st.session_state.messages.append({"role": "assistant", "content": small_talk[clean_p]})
            else:
                results = get_matches_nlp(prompt, df_kb)
                if results:
                    if results[0]['score'] > 0.6:
                        log_usage(results[0]['q'], st.session_state.user_email)
                        st.session_state.messages.append({"role": "assistant", "content": f"**{results[0]['q']}**\n\n{results[0]['a']}"})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "I found these related topics:"})
                        st.session_state.temp_results = results
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "I'm not sure. Try asking about Linear or Polygon mapping."})
            st.rerun()

        if st.button("Clear History", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help?"}]
            st.session_state.temp_results = []
            st.rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)
