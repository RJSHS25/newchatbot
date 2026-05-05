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

# CSS to force the Chat Popover to the bottom right corner
st.markdown("""
    <style>
    /* Position the popover container */
    div[data-testid="stPopover"] {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 999999;
    }
    /* Style the button into a circular chat icon */
    div[data-testid="stPopover"] > button {
        border-radius: 50% !important;
        width: 70px !important;
        height: 70px !important;
        background-color: #007bff !important;
        color: white !important;
        border: none !important;
        font-size: 30px !important;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===============================
# 🔐 AUTHENTICATION
# ===============================
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
    if not os.path.exists("knowledge_base.csv"):
        # Fallback for demo purposes if file is missing
        return pd.DataFrame(columns=['Topic', 'Description', 'Project', 'Category', 'profile'])
    
    df = pd.read_csv("knowledge_base.csv")
    df['profile'] = df['Topic'].fillna('') + " " + df['Description'].fillna('') + " " + df['Project'].fillna('')
    return df

df_kb = load_and_prep_data()

def get_best_match_nlp(query, dataframe):
    if dataframe.empty: return None, 0
    
    # 1. TF-IDF Cosine Similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(dataframe['profile'])
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    best_idx = cosine_sim.argsort()[-1]
    highest_score = cosine_sim[best_idx]
    
    # 2. Fallback to Fuzzy
    if highest_score < 0.2:
        fuzzy_scores = dataframe['profile'].apply(lambda x: fuzz.partial_ratio(query.lower(), str(x).lower()))
        best_idx = fuzzy_scores.idxmax()
        highest_score = fuzzy_scores.max() / 100 
    
    return dataframe.iloc[best_idx], highest_score

# ===============================
# 🧭 DASHBOARD CONTENT
# ===============================
st.markdown("## 🗺️ Maps Knowledge Portal")
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.video("https://www.youtube.com/watch?v=hA_-MkU0Nfw")

st.markdown("---")
st.markdown("## 🚀 Choose Your Domain")
d1, d2, d3 = st.columns(3)

domain_data = [
    {"name": "Linear", "col": d1, "img": "images/linear.png", "target": "pages/1_Linear.py"},
    {"name": "Polygon", "col": d2, "img": "images/polygon.png", "target": "pages/2_Polygon.py"},
    {"name": "Signals", "col": d3, "img": "images/signals.png", "target": "pages/3_Signals.py"}
]

for dom in domain_data:
    with dom["col"]:
        if os.path.exists(dom["img"]):
            st.image(dom["img"], use_container_width=True)
        st.subheader(dom["name"])
        if st.button(f"Open {dom['name']}", key=f"btn_{dom['name']}"):
            st.switch_page(dom["target"])

# ===============================
# 🤖 FLOATING CHATBOT (Bottom Right)
# ===============================
with st.popover("💬"):
    st.markdown("### 🪐 GuruCool Chat")
    
    # Chat container for scrollable history
    chat_container = st.container(height=300)
    
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask a question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process NLP
        result_row, score = get_best_match_nlp(prompt, df_kb)
        
        if score > 0.25:
            response = f"I think you're asking about **{result_row['Topic']}**.\n\n**Info:** {result_row['Description']}"
            cat = result_row['Category']
            # We suggest the page but let user click to navigate
            response += f"\n\n*Click the button below if you want to open the {cat} page.*"
        else:
            response = "I'm not quite sure. Could you provide more details?"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    # Place a navigation button inside the popover if a match was found
    if st.session_state.messages and "Category" in locals():
        if st.button(f"🚀 Go to {cat} Page"):
            st.switch_page(f"pages/{'1_Linear.py' if cat == 'Linear' else '2_Polygon.py' if cat == 'Polygon' else '3_Signals.py'}")
