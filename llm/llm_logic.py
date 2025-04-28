from typing import Any, Deque
from collections import deque

from config import MODEL_ID
from db.initalize_db import chroma_db
from llm.llm_client import google_client
from llm.utils import IntentionDetection, prompt_templates


class DocChatEngine:
    def __init__(self, max_history: int = 10):
        self.db = chroma_db
        self.chat_history: Deque[dict[str, str]] = deque(maxlen=max_history)

    def _format_chat(self) -> str:
        return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in self.chat_history)

    def _make_tool_selector_prompt(self, query: str, titles_and_pages: list[dict[str, Any]]) -> str:
        formatted_titles = "\n".join(
            f"- {item['title']} (pages number: {item['pages']})"
            for item in titles_and_pages
        )
        formatted_history = self._format_chat() or "-"
        prompt_template = prompt_templates["tool_selector_prompt"]
        return prompt_template.format(
            formatted_history=formatted_history,
            query=query,
            formatted_titles=formatted_titles,
        )

    def _make_retrival_prompt(self, query: str, documents: list[str], metadata: list[dict]) -> str:
        formatted_passages = []
        for doc_text, meta in zip(documents, metadata):
            title = meta.get("title", "Unknown Title")
            pages = meta.get("pages", "Unknown Page")
            cleaned_text = doc_text.replace("'", "").replace('"', "").replace("\n", " ").strip()
            formatted_passages.append(f"In '{title}', on page(s) {pages}, it says: \"{cleaned_text}\"")
        combined_passages = "\n\n".join(formatted_passages)
        formatted_history = self._format_chat() or "-"
        prompt_template = prompt_templates["retrival_prompt"]
        return prompt_template.format(
            formatted_history=formatted_history,
            query=query,
            combined_passages=combined_passages,
        )

    def _select_tool_and_prepare(self, query: str) -> IntentionDetection:
        titles_and_pages = self.db.get_titles_and_pages()
        tool_selector_prompt = self._make_tool_selector_prompt(query, titles_and_pages)
        response = google_client.models.generate_content(
            model=MODEL_ID,
            contents=tool_selector_prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": IntentionDetection,
            },
        )
        return response.parsed

    def _answer_question_with_retrieval(self, query: str, documents: list[str], metadata: list[dict]) -> str:
        prompt = self._make_retrival_prompt(query, documents, metadata)
        answer = google_client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        return answer.text

    def process_user_query(self, query: str) -> str:
        self.chat_history.append({"role": "user", "content": query})

        tool_selector_response = self._select_tool_and_prepare(query)
        if tool_selector_response.should_query_vector_db:
            results = self.db.retrieve_documents(
                selected_titles=tool_selector_response.titles,
                search_query=tool_selector_response.query_to_vector_db
            )
            answer = self._answer_question_with_retrieval(
                query, results["documents"][0], results["metadatas"][0]
            )
        else:
            answer = tool_selector_response.answer_to_user

        self.chat_history.append({"role": "assistant", "content": answer})
        return answer
