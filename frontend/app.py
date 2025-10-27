"""
AI Language Tutor - Clean Streamlit UI
"""

import streamlit as st
import sys
import os
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.llm_client import llm
from app.conversation.conversation_analyzer import ConversationAnalyzer
from app.memory.mistake_memory import MistakeMemory
from app.memory.short_term_memory import ShortTermMemory

load_dotenv()

st.set_page_config(
    page_title="AI Language Tutor",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def init_components():
    return ConversationAnalyzer(), MistakeMemory()

analyzer, mistake_memory = init_components()


def login_screen():
    st.title("🎓 AI Language Tutor")
    
    with st.form("login"):
        st.text_input("Username", key="user", placeholder="example")
        st.text_input("Password", key="pass", type="password", placeholder="example")
        
        if st.form_submit_button("Login", type="primary"):
            if st.session_state.user == "example" and st.session_state["pass"] == "example":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")


def chat_interface():
    with st.sidebar:
        st.header("Language Tutor")
        st.write(f"User: {st.session_state.username}")
        
        col1, col2 = st.columns(2)
        col1.metric("Messages", len(st.session_state.messages))
        col2.metric("Mistakes", mistake_memory.count_mistakes())
        
        st.button("Clear Chat", on_click=clear_chat, use_container_width=True)
        st.button("Analyze Session", on_click=toggle_analysis, use_container_width=True)
        st.button("View Mistakes", on_click=toggle_mistakes, use_container_width=True)
        st.button("Logout", on_click=logout, use_container_width=True)
    
    st.header("Practice English")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if st.session_state.get('show_analysis'):
        with st.expander("Session Analysis", expanded=True):
            analyze_session()
    
    if st.session_state.get('show_mistakes'):
        with st.expander("Your Mistakes", expanded=True):
            view_mistakes()
    
    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.stm.add_message("user", prompt)
        
        with st.spinner("Thinking..."):
            response = get_response(prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.stm.add_message("assistant", response)
        st.rerun()


def get_response(user_input):
    history = st.session_state.stm.get_messages(last_n=6)
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    
    prompt = f"""You're a friendly English tutor. Correct errors gently.

Recent chat:
{history_text}

Student: {user_input}

If error: "Almost! Try: [correct]"
If correct: "Good!" and continue
Keep short (2-3 sentences)

Response:"""
    
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Error: {str(e)}"


def analyze_session():
    messages = st.session_state.stm.get_messages()
    if len(messages) < 2:
        st.warning("Need more messages to analyze")
        return
    
    with st.spinner("Analyzing..."):
        mistakes = analyzer.analyze_conversation(messages)
    
    if mistakes:
        mistake_memory.store_mistakes_batch(mistakes)
        st.success(f"Found {len(mistakes)} mistakes")
        
        for i, m in enumerate(mistakes, 1):
            st.subheader(f"{i}. {m.get('error_type', '')}")
            col1, col2 = st.columns(2)
            col1.write("**Your answer:**")
            col1.code(m.get('student_input', ''))
            col2.write("**Correct:**")
            col2.code(m.get('corrected_answer', ''))
            st.info(m.get('explanation', ''))
    else:
        st.success("Perfect! No mistakes found")


def view_mistakes():
    mistakes = mistake_memory.get_all_mistakes(limit=10)
    
    if not mistakes:
        st.info("No mistakes recorded yet")
        return
    
    for i, m in enumerate(mistakes, 1):
        st.subheader(f"{i}. {m.get('error_type', '')}")
        col1, col2 = st.columns(2)
        col1.write("**Your answer:**")
        col1.code(m.get('student_input', ''))
        col2.write("**Correct:**")
        col2.code(m.get('corrected_answer', ''))
        st.info(m.get('explanation', ''))


def clear_chat():
    st.session_state.messages = []
    st.session_state.stm.clear()


def toggle_analysis():
    st.session_state.show_analysis = not st.session_state.get('show_analysis', False)


def toggle_mistakes():
    st.session_state.show_mistakes = not st.session_state.get('show_mistakes', False)


def logout():
    st.session_state.clear()


def init_session_state():
    defaults = {
        'logged_in': False,
        'username': 'example',
        'messages': [],
        'stm': ShortTermMemory(),
        'show_analysis': False,
        'show_mistakes': False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def main():
    init_session_state()
    
    if not st.session_state.logged_in:
        login_screen()
    else:
        chat_interface()


if __name__ == "__main__":
    main()