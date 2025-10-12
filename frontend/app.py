import streamlit as st
import os
import sys
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, project_root)
from app.rag_chain import ask_with_context
import tempfile


# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(page_title="🧠 MVP (Week 1) - Simple RAG with Memory", layout="centered")

st.title("🧠 MVP (Week 1) - Simple RAG, Simple Memory")
st.markdown("**Author:** Nacho Domínguez")

# -------------------------------------------------------------
# STATE MANAGEMENT
# -------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "docs" not in st.session_state:
    st.session_state["docs"] = []

# -------------------------------------------------------------
# FILE UPLOAD SECTION
# -------------------------------------------------------------
st.subheader("📄 Upload up to 5 documents")

uploaded_files = st.file_uploader(
    "Drag and drop or select files",
    accept_multiple_files=True,
    type=["txt", "pdf", "md"],
)

if uploaded_files:
    st.session_state["docs"] = uploaded_files[:5]
    st.success(f"{len(st.session_state['docs'])} document(s) loaded in memory")

# -------------------------------------------------------------
# CHATBOT SECTION
# -------------------------------------------------------------
st.subheader("💬 Chat with your documents")

user_input = st.text_input("Type your question here:")

if st.button("Ask") and user_input:
    try:
        with st.spinner("Thinking..."):
            answer, sources = ask_with_context(user_input)
        st.session_state["chat_history"].append(("user", user_input))
        st.session_state["chat_history"].append(("assistant", answer))

        # Display the conversation
        for role, message in st.session_state["chat_history"]:
            if role == "user":
                st.markdown(f"**🧍 You:** {message}")
            else:
                st.markdown(f"**🤖 Assistant:** {message}")

        # Display sources
        if sources:
            st.markdown("**Sources used:**")
            for s in sources:
                st.code(s)

    except Exception as e:
        st.error(f"Error: {e}")

# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.divider()
st.caption("This is a minimal MVP to test RAG + LangChain integration.")
