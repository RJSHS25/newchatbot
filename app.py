import streamlit as st
import pandas as pd
import os
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# ⚙️ CONFIG & SESSION STATE
# ===============================
st.set_page_config(layout="wide", page_title="TechM Maps Portal")

# Custom CSS to make the right pane look like a sidebar and fix the chat input
st.markdown("""
    <style>
    /* Add a vertical divider line between content and bot */
    .right-pane {
        border-left: 2px solid #e6e9ef;
        padding-left: 20px;
        height: 100vh;
    }
    /* Style the chat bubble area */
    .chat-scroll {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm GuruCool. How can I help you today?"}]

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
# 📄 DATA & NLP ENGINE
# ===============================
@st.cache_data
def load_and_prep_data():
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
# 🏗️ MAIN LAYOUT (2 COLUMNS)
# ===============================
# Main content gets 70% width, Bot gets 30% width
main_content, right_bot_pane = st.columns([0.7, 0.3])

# --- LEFT COLUMN: MAIN CONTENT ---
with main_content:
    st.markdown("## 🗺️ Maps Knowledge Portal")
    st.info(f"Welcome, {st.session_state.user_email}")
    st.markdown("---")

    # Video Section
    st.video("https://www.youtube.com/watch?v=hA_-MkU0Nfw")

    st.markdown("---")
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
                    st.error("Page not found.")

# --- RIGHT COLUMN: CHATBOT PANE ---
with right_bot_pane:
    st.markdown('<div class="right-pane">', unsafe_allow_html=True)
    st.subheader("🪐 GuruCool Support")
    
    # Large chat area
    chat_history_box = st.container(height=600) 
    
    with chat_history_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input field specifically for the right pane
    if prompt := st.chat_input("Ask GuruCool..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        result, score = get_best_match_nlp(prompt, df_kb)
        if score > 0.25:
            bot_response = f"**{result['Topic']}**\n\n{result['Description']}"
        else:
            bot_response = "I couldn't find an exact match. Try asking about 'Linear' or 'Signals'."
            
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.rerun()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm GuruCool. How can I help?"}]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
