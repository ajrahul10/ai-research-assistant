# streamlit_app.py

import streamlit as st
from tools.search_docs import process_pdf, create_vector_store, query_vector_store
from langchain_openai import OpenAIEmbeddings
import os

st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.markdown("<h1 style='text-align: center;'>📄 AI Research Assistant</h1>", unsafe_allow_html=True)
st.markdown("Upload a research PDF, ask any question, and get smart answers from the content.")

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")
    st.subheader("🕘 Chat History")
    if "history" not in st.session_state:
        st.session_state["history"] = []

    for idx, (q, a) in enumerate(st.session_state["history"]):
        with st.expander(f"Q{idx+1}: {q}"):
            st.markdown(f"**Answer:** {a}")

# Main app
uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])
if uploaded_file:
    st.success("PDF uploaded successfully!")
    with st.spinner("🔍 Processing document..."):
        docs = process_pdf(uploaded_file)
        vectorstore = create_vector_store(docs)

    user_query = st.text_input("💬 Ask a question about the document")

    if st.button("🔎 Submit Query") and user_query:
        with st.spinner("Thinking... 🤖"):
            result = query_vector_store(user_query, vectorstore)
            st.markdown(f"### ✅ Answer:\n{result}")
            st.session_state["history"].append((user_query, result))
