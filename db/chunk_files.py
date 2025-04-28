import nltk
import tiktoken
import os
import fitz

def split_pdf_into_chunks(pdf_path: str,
                          chunk_size_tokens: int = 300,
                          overlap_ratio: float = 0.1):
    """
    PDF ➜ sentences ➜ ~`chunk_size_tokens`-token chunks with ~`overlap_ratio` overlap.
    """

    tok = tiktoken.get_encoding("cl100k_base")
    ct = lambda s: len(tok.encode(s))

    sentences = []
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc):
            txt = page.get_text() or ""
            for s in nltk.sent_tokenize(txt):
                s = s.strip()
                if s:
                    sentences.append((s, idx + 1))

    overlap_tokens = int(chunk_size_tokens * overlap_ratio)
    chunks, chunks_pages = [], []
    i = 0
    N = len(sentences)

    while i < N:
        start_i = i                       # remember where this chunk begins
        chunk_sents, pages, tokens = [], set(), 0


        while i < N:
            s, p = sentences[i]
            s_tokens = ct(s)

            if not chunk_sents and s_tokens >= chunk_size_tokens:

                chunk_sents.append(s)
                pages.add(p)
                tokens += s_tokens
                i += 1
                break

            if tokens + s_tokens > chunk_size_tokens:
                break

            chunk_sents.append(s)
            pages.add(p)
            tokens += s_tokens
            i += 1

        if not chunk_sents:
            break

        chunks.append(" ".join(chunk_sents))
        chunks_pages.append(sorted(pages))

        if i >= N:
            break

        if len(chunk_sents) == 1:
            continue

        back_tokens, back_sentences = 0, 0
        for s in reversed(chunk_sents):
            back_tokens += ct(s)
            back_sentences += 1
            if back_tokens >= overlap_tokens:
                break

        back_sentences = min(back_sentences, len(chunk_sents) - 1)

        i = max(start_i + 1, i - back_sentences)

    return chunks, chunks_pages


def chunk_all_pdfs_in_dir(directory_path: str):
    nltk.download("punkt")

    result = {}

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            print(f"Processing: {filename}")
            chunks, pages = split_pdf_into_chunks(pdf_path)

            file_chunks = [
                {"pages": chunk_pages, "text": chunk_text}
                for chunk_text, chunk_pages in zip(chunks, pages)
            ]
            result[filename] = file_chunks

    return result

