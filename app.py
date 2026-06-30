import streamlit as st
import pandas as pd
import os
import csv
import string    
from datetime import datetime
from fuzzywuzzy import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# ⚙️ CONFIG & SESSION STATE
# ===============================


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
    email = st.text_input("Enter Email:")
    if st.button("Login"):
        if "@" in email:
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.rerun()
    st.stop()

# ===============================
# 📄 DATA & SEARCH ENGINE
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
        df = pd.DataFrame(columns=['Topic', 'Description'])
        df.loc[0] = ["Sample", "Knowledge base empty."]
    
    if 'Question' in df.columns:
        df.rename(columns={'Question': 'Topic', 'Answer': 'Description'}, inplace=True)
    df['profile'] = df['Topic'].fillna('') + " " + df['Description'].fillna('')
    return df

df_kb = load_and_prep_data()

def get_combined_matches(query, dataframe, top_n=5):

    if dataframe.empty:
        return []

    query_clean = query.lower().strip()

    results = []

    # ==========================
    # 1. Exact Match
    # ==========================
    exact_matches = dataframe[
        dataframe['Topic']
        .astype(str)
        .str.lower()
        == query_clean
    ]

    for idx, row in exact_matches.iterrows():
        results.append({
            "score": 1.0,
            "q": row['Topic'],
            "a": row['Description'],
            "idx": idx
        })

    # ==========================
    # 2. Partial Match
    # ==========================
    if not results:

        partial_matches = dataframe[
            dataframe['Topic']
            .astype(str)
            .str.lower()
            .str.contains(query_clean, na=False)
        ]

        for idx, row in partial_matches.iterrows():
            results.append({
                "score": 0.95,
                "q": row['Topic'],
                "a": row['Description'],
                "idx": idx
            })

    # ==========================
    # 3. Fuzzy Match Fallback
    # ==========================
    if not results:

        choices = dataframe['Topic'].astype(str).tolist()

        fuzzy_results = process.extract(
            query,
            choices,
            scorer=fuzz.WRatio,
            limit=top_n
        )

        for match in fuzzy_results:

            match_text = match[0]
            score = match[1]

            if score > 55:

                idx_list = dataframe.index[
                    dataframe['Topic'] == match_text
                ].tolist()

                if idx_list:

                    idx = idx_list[0]

                    results.append({
                        "score": score / 100,
                        "q": dataframe.iloc[idx]['Topic'],
                        "a": dataframe.iloc[idx]['Description'],
                        "idx": idx
                    })

    unique_results = []
    seen = set()

    for r in results:
        if r['idx'] not in seen:
            unique_results.append(r)
            seen.add(r['idx'])
    
    return unique_results

# ===============
# Finance Search data
# =========

@st.cache_data
def load_finance_data():
    file_path = "Finance_data.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=["Topic", "Description"])
        df.loc[0] = ["Sample", "Finance database empty."]

    if "Question" in df.columns:
        df.rename(columns={"Question": "Topic", "Answer": "Description"}, inplace=True)

    df["profile"] = df["Topic"].fillna("") + " " + df["Description"].fillna("")
    return df


df_finance = load_finance_data()

# =================
# Adding more
# ==================
elif nav_choice == "📒 Accounts":
    st.title("📒 Accounts Search Engine")
    st.caption("Search Accounts topics from Accounts_data.csv")

    accounts_query = st.text_input("Search Accounts Database:")

    if accounts_query:
        results = get_combined_matches(accounts_query, df_accounts)

        if results:
            best = results[0]
            if best["score"] > 0.7:
                st.success("Best match found")
                st.markdown(f"### {best['q']}")
                st.write(best["a"])
            else:
                st.info("I found a few related Accounts topics:")
                for r in results:
                    with st.expander(f"👉 {r['q']}"):
                        st.write(r["a"])
        else:
            st.warning("No Accounts match found. Try rephrasing.")


elif nav_choice == "👤 Onboarding":
    st.title("👤 Onboarding Search Engine")
    st.caption("Search Onboarding topics from Onboarding_data.csv")

    onboarding_query = st.text_input("Search Onboarding Database:")

    if onboarding_query:
        results = get_combined_matches(onboarding_query, df_onboarding)

        if results:
            best = results[0]
            if best["score"] > 0.7:
                st.success("Best match found")
                st.markdown(f"### {best['q']}")
                st.write(best["a"])
            else:
                st.info("I found a few related Onboarding topics:")
                for r in results:
                    with st.expander(f"👉 {r['q']}"):
                        st.write(r["a"])
        else:
            st.warning("No Onboarding match found. Try rephrasing.")

# ===============================
# ⬅️ SIDEBAR
# ===============================
with st.sidebar:
    st.title("🧭 Navigation")

    nav_choice = st.radio(
        "View:",
        ["🏠 Dashboard", "💰 Finance", "📒 Accounts", "👤 Onboarding", "📊 Analytics"]
    )

    st.divider()

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
# ===============================
# 🏗️ MAIN CONTENT AREA
# ===============================

@st.cache_data
def load_accounts_data():
    file_path = "Accounts_data.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=["Topic", "Description"])
        df.loc[0] = ["Sample", "Accounts database empty."]

    if "Question" in df.columns:
        df.rename(columns={"Question": "Topic", "Answer": "Description"}, inplace=True)

    df["profile"] = df["Topic"].fillna("") + " " + df["Description"].fillna("")
    return df


@st.cache_data
def load_onboarding_data():
    file_path = "Onboarding_data.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=["Topic", "Description"])
        df.loc[0] = ["Sample", "Onboarding database empty."]

    if "Question" in df.columns:
        df.rename(columns={"Question": "Topic", "Answer": "Description"}, inplace=True)

    df["profile"] = df["Topic"].fillna("") + " " + df["Description"].fillna("")
    return df


df_accounts = load_accounts_data()
df_onboarding = load_onboarding_data()
# ===============================
# 🏗️ MAIN CONTENT AREA
# ===============================

if nav_choice == "📊 Analytics":
    st.title("📊 Usage Analytics")

    if os.path.exists("usage_logs.csv"):
        df_logs = pd.read_csv("usage_logs.csv")
        st.dataframe(df_logs, use_container_width=True)
    else:
        st.info("No logs found yet.")

elif nav_choice == "💰 Finance":
    st.title("💰 Finance Search Engine")
    st.caption("Search Finance topics from Finance_data.csv")

    finance_query = st.text_input("Search Finance Database:")

    if finance_query:
        results = get_combined_matches(finance_query, df_finance)

        if results:
            best = results[0]

            if best["score"] > 0.7:
                st.success("Best match found")
                st.markdown(f"### {best['q']}")
                st.write(best["a"])

            else:
                st.info("I found a few related Finance topics:")

                for i, r in enumerate(results):
                    with st.expander(f"👉 {r['q']}"):
                        st.write(r["a"])
        else:
            st.warning("No Finance match found. Try rephrasing.")

else:
    # ===============================
    # 🏠 DASHBOARD PAGE
    # ===============================

    st.markdown("""
    <h1 style='text-align:center;color:#0078d4;'>
    Tech Mahindra Finance Portal
    </h1>
    <p style='text-align:center;color:#666;'>
    Ask GuruCool anything about Materials, GL Accounts and Finance Processes
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    chat_col, nav_col = st.columns([0.85, 0.15])

    with nav_col:
        st.markdown("""
        
        """, unsafe_allow_html=True)

    with chat_col:
        st.subheader("🪐 TechMahindra Finance Guru")

        chat_history_container = st.container()

        with chat_history_container:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

            if st.session_state.temp_results:
                st.write("---")
                st.caption("Common matches:")
                for i, r in enumerate(st.session_state.temp_results):
                    if st.button(f"👉 {r['q']}", key=f"sug_btn_{r['idx']}_{i}", use_container_width=True):
                        log_usage(r['q'], st.session_state.user_email)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"**{r['q']}**\n\n{r['a']}"
                        })
                        st.session_state.temp_results = []
                        st.rerun()

        if prompt := st.chat_input("Ask GuruCool...", key="bot_input"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.temp_results = []

            clean_p = prompt.lower().strip().translate(
                str.maketrans('', '', string.punctuation)
            )

            small_talk = {
                "hi": "Hello!",
                "hello": "Hi there!",
                "thanks": "You're welcome!"
            }

            if clean_p in small_talk:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": small_talk[clean_p]
                })
            else:
                results = get_combined_matches(prompt, df_kb)

                if results:
                    if results[0]['score'] > 0.7:
                        log_usage(results[0]['q'], st.session_state.user_email)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"**{results[0]['q']}**\n\n{results[0]['a']}"
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "I found a few related topics:"
                        })
                        st.session_state.temp_results = results
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I couldn't find a match. Could you rephrase?"
                    })

            st.rerun()

        if st.button("Clear Chat", key="reset_chat", use_container_width=True):
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Hi! I'm GuruCool."
            }]
            st.session_state.temp_results = []
            st.rerun()
