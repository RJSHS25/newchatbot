import streamlit as st
import pandas as pd
import os
from fuzzywuzzy import fuzz
from datetime import datetime

# ===============================
# ⚙️ CONFIG & SESSION STATE
# ===============================
st.set_page_config(layout="wide", page_title="Maps Knowledge Portal")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'selected_topic' not in st.session_state:
    st.session_state.selected_topic = None

# ===============================
# 🔐 AUTHENTICATION SYSTEM
# ===============================
def check_login():
    if not st.session_state.authenticated:
        st.title("🔐 Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            try:
                allowed_users_df = pd.read_csv("allowed_users.csv")
                user_credentials = dict(zip(allowed_users_df["email"], allowed_users_df["password"]))
                if email in user_credentials and user_credentials[email] == password:
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            except FileNotFoundError:
                st.error("User database not found.")
        st.stop()

check_login()

# ===============================
# 📄 DATA LOADING
# ===============================
@st.cache_data
def load_data():
    # Loading your specific Knowledge Base
    df = pd.read_csv("knowledge_base.csv")
    return df

df_kb = load_data()

# ===============================
# 🧭 TOP NAV & HEADER
# ===============================
nav1, nav2 = st.columns([3, 2])
with nav1:
    st.markdown("## 🗺️ Maps Knowledge Portal")
with nav2:
    # This acts as the Bot Input
    search_query = st.text_input("🧠 Ask GuruCool (Search Topics):", placeholder="e.g., How to map boundaries?")

# ===============================
# 🤖 SEARCH BOT LOGIC
# ===============================
if search_query:
    st.markdown("---")
    matches = []
    for _, row in df_kb.iterrows():
        # Score based on Topic and Description
        text_to_search = f"{row['Topic']} {row['Description']}"
        score = fuzz.partial_ratio(search_query.lower(), str(text_to_search).lower())
        matches.append((row, score))

    # Get top 3 matches
    top_matches = sorted(matches, key=lambda x: x[1], reverse=True)[:3]

    if top_matches and top_matches[0][1] > 50:
        st.info("🎯 I found these matches in the Knowledge Base:")
        
        # Display results in an expandable format or selection
        match_titles = [f"{m[0]['Category']} | {m[0]['Topic']}" for m in top_matches]
        selected_match = st.radio("Select the specific topic to view details:", match_titles)
        
        if st.button("View Details & Navigate"):
            # Extract the actual topic name
            topic_name = selected_match.split("|")[1].strip()
            st.session_state.selected_topic = df_kb[df_kb["Topic"] == topic_name].iloc[0]
    else:
        st.warning("❓ No close matches found. Try different keywords.")

# ===============================
# 📊 DISPLAY SEARCH RESULT (The "Bot" Answer)
# ===============================
if st.session_state.selected_topic is not None:
    res = st.session_state.selected_topic
    
    with st.expander("📖 Quick View: " + res['Topic'], expanded=True):
        st.write(f"**Description:** {res['Description']}")
        
        # PCIR & Freshdesk (Styled like your second script)
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"📌 **PCIR:** {res.get('PCIR', 'N/A')}")
        with c2:
            st.success(f"🛠️ **Freshdesk:** {res.get('Freshdesk', 'N/A')}")
        
        # NAVIGATION BUTTON
        cat = res['Category']
        if st.button(f"Go to Full {cat} Page ➡️"):
            st.session_state.selected_topic = None # Clear search
            if cat == "Linear": st.switch_page("pages/1_Linear.py")
            elif cat == "Polygon": st.switch_page("pages/2_Polygon.py")
            elif cat == "Signals": st.switch_page("pages/3_Signals.py")

    if st.button("Clear Results"):
        st.session_state.selected_topic = None
        st.rerun()

# ===============================
# 🎥 HOME UI (VIDEO & CARDS)
# ===============================
st.markdown("---")
col_v1, col_v2, col_v3 = st.columns([1, 2, 1])
with col_v2:
    st.video("https://www.youtube.com/watch?v=hA_-MkU0Nfw")

st.markdown("## 🚀 Choose Your Domain")
d1, d2, d3 = st.columns(3)

domains = [
    {"name": "Linear", "img": "linear.png", "col": d1, "page": "pages/1_Linear.py", "cap": "Line mapping & boundaries"},
    {"name": "Polygon", "img": "polygon.png", "col": d2, "page": "pages/2_Polygon.py", "cap": "Area mapping & geometry"},
    {"name": "Signals", "img": "signals.png", "col": d3, "page": "pages/3_Signals.py", "cap": "Traffic signal configurations"}
]

for dom in domains:
    with dom["col"]:
        img_path = os.path.join("images", dom["img"])
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        st.subheader(dom["name"])
        st.caption(dom["cap"])
        if st.button(f"Open {dom['name']}", key=f"btn_{dom['name']}"):
            st.switch_page(dom["page"])
