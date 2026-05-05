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

# Auth & Chat History
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===============================
# 🔐 AUTHENTICATION
# ===============================
# (Keeping your login logic simplified for the demo)
if not st.session_state.authenticated:
    st.title("🔐 Login")
    email = st.text_input("Email")
    if st.button("Login"):
        st.session_state.authenticated = True
        st.rerun()
    st.stop()

# ===============================
# 📄 DATA & NLP ENGINE
# ===============================
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv("knowledge_base.csv")
    # NLP Prep: Combine columns into a single 'search_profile'
    df['profile'] = df['Topic'].fillna('') + " " + df['Description'].fillna('') + " " + df['Project'].fillna('')
    return df

df_kb = load_and_prep_data()

def get_best_match_nlp(query, dataframe):
    """Combines TF-IDF (NLP) and Fuzzy matching for high accuracy."""
    # 1. TF-IDF Cosine Similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(dataframe['profile'])
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # 2. Get top indices from Cosine Similarity
    best_idx = cosine_sim.argsort()[-1]
    highest_score = cosine_sim[best_idx]
    
    # 3. Fallback to Fuzzy if NLP score is low
    if highest_score < 0.2:
        # Use your existing fuzzy logic as a backup
        fuzzy_scores = dataframe['profile'].apply(lambda x: fuzz.partial_ratio(query.lower(), str(x).lower()))
        best_idx = fuzzy_scores.idxmax()
        highest_score = fuzzy_scores.max() / 100 # Normalize to 0-1 range
    
    return dataframe.iloc[best_idx], highest_score

# ===============================
# 🧭 TOP NAVIGATION BAR
# ===============================
nav1, nav2, nav3 = st.columns([3, 1, 1])
with nav1:
    st.markdown("## 🗺️ Maps Knowledge Portal")

with nav3:
    # 🤖 FLOATING CHAT ICON (Using Popover)
    with st.popover("💬 Chat with Me"):
        st.write("### 🪐 GuruCool Chat")
        
        # Display Conversation
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat Input
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get NLP Match
            result_row, score = get_best_match_nlp(prompt, df_kb)
            
            with st.chat_message("assistant"):
                if score > 0.25: # Confidence threshold
                    response = f"I think you're asking about **{result_row['Topic']}**."
                    st.markdown(response)
                    st.write(f"**Description:** {result_row['Description']}")
                    
                    # Quick Actions
                    cat = result_row['Category']
                    if st.button(f"Open {cat} Page", key="nav_btn"):
                        st.switch_page(f"pages/{'1_Linear.py' if cat == 'Linear' else '2_Polygon.py' if cat == 'Polygon' else '3_Signals.py'}")
                else:
                    response = "I'm not quite sure. Could you provide more keywords (e.g., 'Linear boundaries')?"
                    st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})

# ===============================
# 🎥 MAIN DASHBOARD CONTENT
# ===============================
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.video("https://www.youtube.com/watch?v=hA_-MkU0Nfw")

st.markdown("---")
st.markdown("## 🚀 Choose Your Domain")
d1, d2, d3 = st.columns(3)

# Mapping Domains
domain_data = [
    {"name": "Linear", "col": d1, "img": "images/linear.png"},
    {"name": "Polygon", "col": d2, "img": "images/polygon.png"},
    {"name": "Signals", "col": d3, "img": "images/signals.png"}
]

for dom in domain_data:
    with dom["col"]:
        if os.path.exists(dom["img"]):
            st.image(dom["img"], use_container_width=True)
        st.subheader(dom["name"])
        if st.button(f"Open {dom['name']}", key=dom['name']):
            st.switch_page(f"pages/{dom['name']}.py")
