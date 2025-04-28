import os
from typing import Any

import chromadb

from db.chunk_files import chunk_all_pdfs_in_dir
from config import DOCS_PATH, DB_PATH
from llm.llm_client import google_ef

class ChromaDB:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.db = self._ensure_db()

    def _create_collection_from_dir(self):
        data = chunk_all_pdfs_in_dir(DOCS_PATH)
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=google_ef
        )

        docs, ids, metas = [], [], []

        for filename, chunk_list in data.items():
            for idx, chunk in enumerate(chunk_list):
                ids.append(f"{filename[:-4]}_{idx}")
                docs.append(chunk["text"])
                page_range = ",".join(map(str, chunk["pages"]))
                metas.append({
                    "title": filename,
                    "source": filename,
                    "pages": page_range
                })

        collection.add(documents=docs, ids=ids, metadatas=metas)
        return collection

    def _open_existing_collection(self):
        return self.client.get_collection(
            name=self.collection_name,
            embedding_function=google_ef
        )

    def _ensure_db(self):
        if not os.path.exists(DB_PATH):
            print(f"Chroma DB '{self.collection_name}' not found. Creating new database...")
            return self._create_collection_from_dir()
        else:
            print(f"Chroma DB '{self.collection_name}' found. Loading existing database...")
            return self._open_existing_collection()

    def get_relevant_passages(self, query: str, top_n: int = 20) -> str:
        query_embedding = google_ef([query])
        results = self.db.query(
            query_embeddings=query_embedding,
            n_results=top_n
        )
        passages = results['documents'][0]
        return "\n\n".join(passages)

    def retrieve_documents(self, selected_titles: list[str], search_query: str, n_results: int = 20):
        return self.db.query(
            where={"title": {"$in": selected_titles}},
            query_texts=[search_query],
            n_results=n_results
        )

    def get_titles_and_pages(self) -> list[dict[str, Any]]:
        titles_with_pages = {}
        all_metadata = self.db.get(include=["metadatas"])["metadatas"]
        for meta in all_metadata:
            title = meta["title"]
            page_raw = meta.get("pages")
            page_numbers = [int(p.strip()) for p in page_raw.split(",") if p.strip().isdigit()]
            page = max(page_numbers) if page_numbers else 0
            titles_with_pages[title] = max(titles_with_pages.get(title, 0), page)
        return [{"title": title, "pages": page} for title, page in titles_with_pages.items()]
