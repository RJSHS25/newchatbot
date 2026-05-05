import streamlit as st
import pandas as pd
import os
from fuzzywuzzy import fuzz
from datetime import datetime

# ===============================
# ⚙️ CONFIG & SESSION STATE
# ===============================
st.set_page_config(layout="wide", page_title="Maps Knowledge Portal")

# Auth State
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
# Chat State
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_row' not in st.session_state:
    st.session_state.selected_row = None

# ===============================
# 🔐 AUTHENTICATION
# ===============================
def check_login():
    if not st.session_state.authenticated:
        st.title("🔐 Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            try:
                users = pd.read_csv("allowed_users.csv")
                creds = dict(zip(users["email"], users["password"]))
                if email in creds and creds[email] == password:
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            except Exception:
                st.error("User database missing.")
        st.stop()

check_login()

# ===============================
# 📄 DATA LOADING
# ===============================
@st.cache_data
def load_kb():
    # Loading the sheet you specified
    return pd.read_csv("knowledge_base.csv")

df_kb = load_kb()

# ===============================
# 🤖 SIDEBAR HELP BOT (GuruCool)
# ===============================
with st.sidebar:
    st.title("🪐 GuruCool Help")
    st.markdown("---")
    
    # Input for Agent Question
    user_query = st.text_input("How can I help you today?", placeholder="Type your question...")

    if user_query:
        # Fuzzy matching logic against 'Topic' and 'Description'
        matches = []
        for _, row in df_kb.iterrows():
            text_pool = f"{row['Topic']} {row['Description']}"
            score = fuzz.partial_ratio(user_query.lower(), str(text_pool).lower())
            matches.append((row, score))
        
        top_matches = sorted(matches, key=lambda x: x[1], reverse=True)[:3]

        if top_matches and top_matches[0][1] > 50:
            st.write("🔍 **I found these topics:**")
            
            # Create selection buttons for matches
            for i, (row, score) in enumerate(top_matches):
                if st.button(f"📌 {row['Topic']}", key=f"match_{i}"):
                    st.session_state.selected_row = row
        else:
            st.warning("😕 I couldn't find an exact match. Try 'Linear' or 'Signals'.")

    # Display the "Answer" if a topic is selected
    if st.session_state.selected_row is not None:
        res = st.session_state.selected_row
        st.markdown("---")
        st.success(f"**Topic:** {res['Topic']}")
        st.write(f"**Info:** {res['Description']}")
        
        with st.expander("🛠️ View Properties"):
            st.caption(f"**PCIR:** {res.get('PCIR', 'N/A')}")
            st.caption(f"**Freshdesk:** {res.get('Freshdesk', 'N/A')}")
            
        # Quick Navigation
        cat = res['Category']
        if st.button(f"Go to {cat} Page"):
            if cat == "Linear": st.switch_page("pages/1_Linear.py")
            elif cat == "Polygon": st.switch_page("pages/2_Polygon.py")
            elif cat == "Signals": st.switch_page("pages/3_Signals.py")
            
        if st.button("Reset Chat"):
            st.session_state.selected_row = None
            st.rerun()

# ===============================
# 🗺️ MAIN INTERFACE (TechM_BOT)
# ===============================
st.markdown("## 🗺️ Maps Knowledge Portal")

# Hero Video Section
col_v1, col_v2, col_v3 = st.columns([1, 2, 1])
with col_v2:
    st.video("https://www.youtube.com/watch?v=hA_-MkU0Nfw")

st.markdown("---")
st.markdown("### 🚀 Choose Your Domain")

# Domain Cards
d1, d2, d3 = st.columns(3)

domains = [
    {"name": "Linear", "img": "linear.png", "col": d1, "page": "pages/1_Linear.py"},
    {"name": "Polygon", "img": "polygon.png", "col": d2, "page": "pages/2_Polygon.py"},
    {"name": "Signals", "img": "signals.png", "col": d3, "page": "pages/3_Signals.py"}
]

for dom in domains:
    with dom["col"]:
        img_path = os.path.join("images", dom["img"])
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        st.subheader(dom["name"])
        if st.button(f"Open {dom['name']}", key=f"main_{dom['name']}"):
            st.switch_page(dom["page"])
