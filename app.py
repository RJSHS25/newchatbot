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
        if "@" in email: # Basic validation
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.rerun()
    st.stop()

# ===============================
# 📄 DATA & NLP ENGINE
# ===============================
@st.cache_data
def load_and_prep_data():
    # Load knowledge_base.csv (Ensure this file exists)
    if os.path.exists("knowledge_base.csv"):
        df = pd.read_csv("knowledge_base.csv")
    else:
        # Fallback empty dataframe if file missing
        df = pd.DataFrame(columns=['Topic', 'Description', 'Category'])
    
    df['profile'] = df['Topic'].fillna('') + " " + df['Description'].fillna('')
    return df

df_kb = load_and_prep_data()

def get_best_match_nlp(query, dataframe):
    if dataframe.empty:
        return None, 0
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(dataframe['profile'])
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    best_idx = cosine_sim.argsort()[-1]
    highest_score = cosine_sim[best_idx]
    
    # Fallback to Fuzzy Logic for low scores
    if highest_score < 0.2:
        fuzzy_scores = dataframe['profile'].apply(lambda x: fuzz.partial_ratio(query.lower(), str(x).lower()))
        best_idx = fuzzy_scores.idxmax()
        highest_score = fuzzy_scores.max() / 100
    
    return dataframe.iloc[best_idx], highest_score

# ===============================
# 🤖 SIDEBAR CHATBOT (Right or Left pane)
# ===============================
with st.sidebar:
    st.title("🪐 GuruCool Support")
    st.write(f"Logged in as: {st.session_state.user_email}")
    st.divider()

    # Chat history container
    # height=400 keeps it compact in the sidebar
    chat_container = st.container(height=450)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat Input field inside Sidebar
    if prompt := st.chat_input("Ask GuruCool..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # NLP Matching
        result, score = get_best_match_nlp(prompt, df_kb)
        
        if score > 0.25:
            bot_response = f"**{result['Topic']}**\n\n{result['Description']}"
        else:
            bot_response = "I couldn't find an exact match. Try asking about 'Linear' or 'Signals'."
            
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.rerun()
    
    if st.button("Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm GuruCool. How can I help?"}]
        st.rerun()

# ===============================
# 🧭 MAIN DASHBOARD CONTENT
# ===============================
st.markdown("## 🗺️ Maps Knowledge Portal")
st.markdown("---")

# Video Section
col_v1, col_v2, col_v3 = st.columns([1, 2, 1])
with col_v2:
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
                st.error(f"Page {dom['path']} not found. Check your 'pages' folder.")
