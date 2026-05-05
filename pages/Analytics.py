import streamlit as st
import pandas as pd
import os

st.set_page_config(layout="wide", page_title="Usage Analytics")

st.title("📊 Usage Analytics")

if os.path.exists("usage_logs.csv"):
    df = pd.read_csv("usage_logs.csv")
    
    if not df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Questions")
            top_q = df['Question'].value_counts().head(10)
            st.bar_chart(top_q)
            
        with col2:
            st.subheader("Most Active Users")
            top_users = df['User'].value_counts().head(10)
            st.write(top_users)
            
        st.divider()
        st.subheader("Recent Activity Log")
        st.dataframe(df.sort_values(by='Timestamp', ascending=False), use_container_width=True)
    else:
        st.info("No logs found yet. Start chatting to see data!")
else:
    st.error("Log file not found. Ensure the chatbot has been used at least once.")

if st.button("Back to Chat"):
    st.switch_page("chatbot.py") # Change this to the exact name of your main file
