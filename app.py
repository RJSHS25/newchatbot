import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz
from datetime import datetime

# 📄 Load allowed users
@st.cache_data
def load_allowed_users():
    return pd.read_csv("allowed_users.csv")

allowed_users_df = load_allowed_users()
user_credentials = dict(zip(allowed_users_df["email"], allowed_users_df["password"]))

# 🔐 Session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = ""

# 🔐 Login
if not st.session_state.authenticated:
    st.title("🔐 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if email in user_credentials and user_credentials[email] == password:
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.success("Login successful ✅")
            st.rerun()
        else:
            st.error("Invalid email or password ❌")
    st.stop()

# ✅ Load Q&A data
@st.cache_data
def load_qa_data():
    return pd.read_csv("data.csv")

df = load_qa_data()

# 🧠 Q&A Interface
st.title("🧠 Internal Q&A Chatbot")
user_input = st.text_input("Ask a question:")

if user_input and not st.session_state.selected_question:
    matches = []
    for _, row in df.iterrows():
        score = fuzz.partial_ratio(user_input.lower(), str(row["Question"]).lower())
        matches.append((row["Question"], score))

    # Top 5 matches
    top_matches = sorted(matches, key=lambda x: x[1], reverse=True)[:5]

    if top_matches and top_matches[0][1] > 50:
        st.info("Did you mean one of these?")
        selected = st.radio("Select the closest match:", [q for q, _ in top_matches], key="question_selector")
        if st.button("Show Answer"):
            st.session_state.selected_question = selected
            st.session_state.user_question = user_input
            st.rerun()
    else:
        st.warning("❌ Sorry, I couldn’t find a good match. Try rephrasing your question.")

# ✅ Show selected answer and details
elif st.session_state.selected_question:
    if st.button("🔄 New Question"):
        st.session_state.selected_question = ""
        st.session_state.user_question = ""
        st.rerun()

    matched_q = st.session_state.selected_question
    matched_row = df[df["Question"] == matched_q].iloc[0]

    st.success(f"**Matched Question:** {matched_q}")
    st.markdown(f"**Answer:** {matched_row.get('Answer', '')}")
    st.markdown(f"**Chat Script:** {matched_row.get('Chat Scripts', '')}")
    st.markdown(f"**Email Script:** {matched_row.get('Email Scripts', '')}")
    st.markdown(f"**Voice Script:** {matched_row.get('Voice Scripts', '')}")
    if pd.notna(matched_row.get("Gurucool Link", None)):
        st.markdown(f"[🔗 Gurucool Link]({matched_row['Gurucool Link']})")
    st.markdown(f"**PCIR:** {matched_row.get('PCIR', '')}")

    log_entry = {
        "Email": st.session_state.user_email,
        "Typed Question": st.session_state.user_question,
        "Matched Question": matched_q,
        "Answer": matched_row.get('Answer', ''),
        "Chat Script": matched_row.get('Chat Scripts', ''),
        "Email Script": matched_row.get('Email Scripts', ''),
        "Voice Script": matched_row.get('Voice Scripts', ''),
        "Gurucool Link": matched_row.get('Gurucool Link', ''),
        "PCIR": matched_row.get('PCIR', ''),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    pd.DataFrame([log_entry]).to_csv("chat_logs.csv", mode='a', header=not pd.io.common.file_exists("chat_logs.csv"), index=False)
