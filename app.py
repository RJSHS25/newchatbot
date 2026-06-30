import streamlit as st
import pandas as pd
import os
import csv
import string
from datetime import datetime
from fuzzywuzzy import fuzz, process

# ===============================
# ⚙️ SESSION STATE
# ===============================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm GuruCool. How can I help you today?"}
    ]

if "temp_results" not in st.session_state:
    st.session_state.temp_results = []

# ===============================
# 🔐 LOGIN
# ===============================
if not st.session_state.authenticated:
    st.title("🔐 TechM Portal Login")
    email = st.text_input("Enter Email:")

    if st.button("Login"):
        if "@" in email:
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.rerun()
        else:
            st.error("Please enter a valid email.")

    st.stop()

# ===============================
# 📝 LOGGING
# ===============================
def log_usage(question, user_email, page_name):
    log_file = "usage_logs.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)

        if not file_exists:
            writer.writerow(["Timestamp", "User", "Page", "Question"])

        writer.writerow([timestamp, user_email, page_name, question])

# ===============================
# 📄 DATA LOADER
# ===============================
@st.cache_data
def load_data(file_name, empty_message):
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        df = pd.DataFrame(columns=["Topic", "Description"])
        df.loc[0] = ["Sample", empty_message]

    if "Question" in df.columns and "Answer" in df.columns:
        df.rename(columns={"Question": "Topic", "Answer": "Description"}, inplace=True)

    if "Topic" not in df.columns:
        df["Topic"] = ""

    if "Description" not in df.columns:
        df["Description"] = ""

    df["Topic"] = df["Topic"].fillna("").astype(str)
    df["Description"] = df["Description"].fillna("").astype(str)
    df["profile"] = df["Topic"] + " " + df["Description"]

    return df

# Main Finance_Dashboard database
if os.path.exists("knowledge_base.csv"):
    df_kb = load_data("knowledge_base.csv", "Knowledge base empty.")
else:
    df_kb = load_data("data.csv", "Knowledge base empty.")

df_finance = load_data("Finance_data.csv", "Finance database empty.")
df_accounts = load_data("Accounts_data.csv", "Accounts database empty.")
df_onboarding = load_data("Onboarding_data.csv", "Onboarding database empty.")

# ===============================
# 🔎 SEARCH ENGINE
# ===============================
def get_combined_matches(query, dataframe, top_n=5):
    if dataframe.empty:
        return []

    query_clean = query.lower().strip()
    results = []

    exact_matches = dataframe[
        dataframe["Topic"].astype(str).str.lower() == query_clean
    ]

    for idx, row in exact_matches.iterrows():
        results.append({
            "score": 1.0,
            "q": row["Topic"],
            "a": row["Description"],
            "idx": idx
        })

    if not results:
        partial_matches = dataframe[
            dataframe["Topic"]
            .astype(str)
            .str.lower()
            .str.contains(query_clean, na=False, regex=False)
        ]

        for idx, row in partial_matches.iterrows():
            results.append({
                "score": 0.95,
                "q": row["Topic"],
                "a": row["Description"],
                "idx": idx
            })

    if not results:
        choices = dataframe["Topic"].astype(str).tolist()

        fuzzy_results = process.extract(
            query,
            choices,
            scorer=fuzz.WRatio,
            limit=top_n
        )

        for match_text, score in fuzzy_results:
            if score > 55:
                idx_list = dataframe.index[dataframe["Topic"] == match_text].tolist()

                if idx_list:
                    idx = idx_list[0]
                    results.append({
                        "score": score / 100,
                        "q": dataframe.iloc[idx]["Topic"],
                        "a": dataframe.iloc[idx]["Description"],
                        "idx": idx
                    })

    unique_results = []
    seen = set()

    for r in results:
        if r["idx"] not in seen:
            unique_results.append(r)
            seen.add(r["idx"])

    return unique_results

# ===============================
# 🔎 REUSABLE SEARCH PAGE
# ===============================
def render_search_page(title, caption, input_label, dataframe, page_name):
    st.title(title)
    st.caption(caption)

    query = st.text_input(input_label)

    if query:
        results = get_combined_matches(query, dataframe)

        if results:
            best = results[0]

            if best["score"] > 0.7:
                log_usage(best["q"], st.session_state.user_email, page_name)
                st.success("Best match found")
                st.markdown(f"### {best['q']}")
                st.write(best["a"])
            else:
                st.info("I found a few related topics:")

                for i, r in enumerate(results):
                    with st.expander(f"👉 {r['q']}"):
                        st.write(r["a"])
        else:
            st.warning("No match found. Try rephrasing.")

# ===============================
# ⬅️ SIDEBAR
# ===============================
with st.sidebar:
    st.title("🧭 Navigation")

    if "nav_choice" not in st.session_state:
        st.session_state.nav_choice = "🏠 Finance Dashbaord"

    menu_options = [
        "🏠 Home",
        "👥 Supplier Onboarding",
        "🔎 PO/Invoice Search Engine",
        "📦 Material Master",
        "📚 Knowledge Repository",
        "📊 Analytics"
    ]

    nav_choice = st.radio(
        "View:",
        menu_options,
        index=menu_options.index(st.session_state.nav_choice)
    )

    st.session_state.nav_choice = nav_choice

# ===============================
# 📊 ANALYTICS PAGE
# ===============================
if nav_choice == "📊 Analytics":
    st.title("📊 Usage Analytics")

    if os.path.exists("usage_logs.csv"):
        df_logs = pd.read_csv("usage_logs.csv")
        st.dataframe(df_logs, use_container_width=True)
    else:
        st.info("No logs found yet.")

# ===============================
# 💰 FINANCE PAGE
# ===============================
elif nav_choice == "📦 Material Master":
    st.title("📦 Material Master")

    tab1, tab2 = st.tabs(["Material Master Search Engine", "PR Creation SAP"])

    with tab1:
        render_search_page(
            title="Material Master Search Engine",
            caption="Search Material Master topics from Finance_data.csv",
            input_label="Search Material Master:",
            dataframe=df_finance,
            page_name="Material Master"
        )

    with tab2:
        st.subheader("PR Creation SAP")
        st.info("Add PR Creation SAP content here.")

# ===============================
# 📒 ACCOUNTS PAGE
# ===============================
elif nav_choice == "📒 Accounts":
    render_search_page(
        title="📒 Accounts Search Engine",
        caption="Search Accounts topics from Accounts_data.csv",
        input_label="Search Accounts Database:",
        dataframe=df_accounts,
        page_name="Accounts"
    )

# ===============================
# 👤 ONBOARDING PAGE
# ===============================
elif nav_choice == "👤 Onboarding":
    render_search_page(
        title="👤 Onboarding Search Engine",
        caption="Search Onboarding topics from Onboarding_data.csv",
        input_label="Search Onboarding Database:",
        dataframe=df_onboarding,
        page_name="Onboarding"
    )


# ===============================
# 🏠 Supplier Onboarding
# ===============================

elif nav_choice == "👥 Supplier Onboarding":
    st.title("👥 Supplier Onboarding")

    tab1, tab2 = st.tabs(["Supplier Order Status", "Sample Guide for Questionnaire"])

    with tab1:
        st.subheader("Supplier Order Status")
        st.info("Add Supplier Order Status search/content here.")

    with tab2:
        st.subheader("Sample Guide for Questionnaire")
        st.info("Add questionnaire guide here.")


elif nav_choice == "🔎 PO/Invoice Search Engine":
    st.title("🔎 PO/Invoice Search Engine")

    tab1, tab2 = st.tabs(["PO Status Search Engine", "Invoice Status"])

    with tab1:
        st.subheader("PO Status Search Engine")
        st.info("Add PO Status search engine here.")

    with tab2:
        st.subheader("Invoice Status")
        st.info("Add Invoice Status search here.")


elif nav_choice == "📚 Knowledge Repository":
    st.title("📚 Knowledge Repository")

    tab1, tab2 = st.tabs(["SOP's", "Process Maps"])

    with tab1:
        st.subheader("SOP's")
        st.info("Add SOP documents or links here.")

    with tab2:
        st.subheader("Process Maps")
        st.info("Add process maps here.")

# ===============================
# 🏠 DASHBOARD CHATBOT PAGE
# ===============================
else:
    st.title("🏢 Tech Mahindra Finance Portal")

    st.markdown(f"""
    ### Welcome **{st.session_state.user_email}** 👋

    This portal provides quick access to Finance, Accounts,
    Onboarding and Enterprise Knowledge.

    Please choose a module from the left navigation panel.
    """)

    st.divider()

    st.subheader("📢 Announcements")

    st.info("""
    • Finance Month End Closing starts on 28th

    • New GST Guidelines published

    • Vendor Master Process Updated

    • Employee Travel Policy Version 5 Released
    """)

    st.divider()

    st.subheader("🚀 Quick Access")

    col1, col2 = st.columns(2)


    with col1:
        st.page_link("app.py", label="💰 Open Finance", icon="💰")
        st.write("Search Finance policies and SOPs.")
    
    with col2:
        st.page_link("app.py", label="📒 Open Accounts", icon="📒")
        st.write("Search GL, Vendor and Accounting processes.")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.page_link("app.py", label="👤 Open Onboarding", icon="👤")
        st.write("New employee onboarding documents.")
    
    with col4:
        st.page_link("app.py", label="📊 Open Analytics", icon="📊")
        st.write("View Portal Usage.")
        
        with col1:
            st.success("💰 Finance")
            st.write("Search Finance policies and SOPs.")
    
        with col2:
            st.success("📒 Accounts")
            st.write("Search GL, Vendor and Accounting processes.")
    
        col3, col4 = st.columns(2)
    
        with col3:
            st.success("👤 Onboarding")
            st.write("New employee onboarding documents.")

    st.divider()

    st.subheader("📌 About this Portal")

    st.write("""
    Welcome to the Tech Mahindra Finance Knowledge Portal.

    This portal provides centralized access to Finance,
    Accounts and Onboarding knowledge.

    Use the navigation menu on the left to explore the available
    knowledge bases.

    Built with ❤️ by Tech Mahindra Finance Team.
    """)
