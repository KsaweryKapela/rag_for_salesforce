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

    def _make_extraction_prompt(self, query: str, titles_and_pages: list[dict[str, Any]]) -> str:
        formatted_titles = "\n".join(
            f"- {item['title']} (pages number: {item['pages']})"
            for item in titles_and_pages
        )
        formatted_history = self._format_chat() or "-"
        prompt_template = prompt_templates["extraction_prompt"]
        return prompt_template.format(
            formatted_history=formatted_history,
            query=query,
            formatted_titles=formatted_titles,
        )

    def _make_answer_prompt(self, query: str, documents: list[str], metadata: list[dict]) -> str:
        formatted_passages = []
        for doc_text, meta in zip(documents, metadata):
            title = meta.get("title", "Unknown Title")
            pages = meta.get("pages", "Unknown Page")
            cleaned_text = doc_text.replace("'", "").replace('"', "").replace("\n", " ").strip()
            formatted_passages.append(f"In '{title}', on page(s) {pages}, it says: \"{cleaned_text}\"")
        combined_passages = "\n\n".join(formatted_passages)
        formatted_history = self._format_chat() or "-"
        prompt_template = prompt_templates["answer_prompt"]
        return prompt_template.format(
            formatted_history=formatted_history,
            query=query,
            combined_passages=combined_passages,
        )

    def _detect_intention_and_prepare(self, query: str) -> IntentionDetection:
        titles_and_pages = self.db.get_titles_and_pages()
        extraction_prompt = self._make_extraction_prompt(query, titles_and_pages)
        extracted_info = google_client.models.generate_content(
            model=MODEL_ID,
            contents=extraction_prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": IntentionDetection,
            },
        )
        return extracted_info.parsed

    def _answer_question_with_retrieval(self, query: str, documents: list[str], metadata: list[dict]) -> str:
        prompt = self._make_answer_prompt(query, documents, metadata)
        answer = google_client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        return answer.text

    def process_user_query(self, query: str) -> str:
        self.chat_history.append({"role": "user", "content": query})

        extracted_info = self._detect_intention_and_prepare(query)
        if extracted_info.should_query_vector_db:
            results = self.db.retrieve_documents(
                selected_titles=extracted_info.titles,
                search_query=extracted_info.query_to_vector_db
            )
            answer = self._answer_question_with_retrieval(
                query, results["documents"][0], results["metadatas"][0]
            )
        else:
            answer = extracted_info.answer_to_user

        self.chat_history.append({"role": "assistant", "content": answer})
        return answer
