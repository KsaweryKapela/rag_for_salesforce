import streamlit as st
from llm.llm_logic import DocChatEngine

class DocRetrieverChatApp:
    def __init__(self):
        self._setup_page()
        self._ensure_session_messages()
        self.chat_engine = DocChatEngine()

    @staticmethod
    def _ensure_session_messages():
        if "messages" not in st.session_state:
            st.session_state.messages = []

    @staticmethod
    def _setup_page():
        st.set_page_config(page_title="Doc Retriever Chat", page_icon="💬")
        st.title("💬 Document Assistant")
        st.markdown("Ask anything about your documents!")
        if st.button("🔄 Reset Chat"):
            st.session_state.messages = []
            st.rerun()

    @staticmethod
    def _escape_dollars(text: str) -> str:
        return text.replace("$", r"\$")

    def _render_chat_history(self):
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(self._escape_dollars((msg["content"])))

    def run(self):
        self._render_chat_history()

        user_query = st.chat_input("Type your question here…")

        if user_query:
            with st.chat_message("user"):
                st.markdown(self._escape_dollars(user_query))
            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.chat_message("assistant"):
                placeholder = st.empty()
                with st.spinner("Thinking…"):
                    raw_reply = self.chat_engine.process_user_query(user_query)
                placeholder.markdown(self._escape_dollars(raw_reply))

            st.session_state.messages.append({"role": "assistant", "content": raw_reply})
