# rag/index.py
from typing import List
from ..memory.long_term import LongTermMemory
from ..utils.logging import logger

class RAGIndex:
    def __init__(self, store: LongTermMemory):
        self.store = store

    def ingest_text(self, doc_id: str, text: str, meta: dict):
        self.store.add_document(doc_id, text, meta)

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        results = self.store.search(query, top_k=top_k)

        if not results:
            logger.debug("RAG: no hits for query %r, falling back to generic query", query)
            # fallback to a generic term that likely appears in most docs
            results = self.store.search("account transaction balance error", top_k=top_k)

        if not results:
            logger.debug("RAG: still no hits after fallback")
            return ""
        
        logger.debug("RAG: %d hit(s) for query %r", len(results), query)

        context_chunks: List[str] = []
        for doc_id, text, meta in results:
            preview = (text[:200] + "...") if len(text) > 200 else text
            logger.debug(
                "RAG chunk used | doc_id=%s | meta=%s | preview=%r",
                doc_id,
                meta,
                preview,
            )
            context_chunks.append(f"[DOC {doc_id} | {meta.get('type')}] {text}")

        return "\n\n".join(context_chunks)

    def delete(self, doc_id: str) -> bool:
        """Delete a document from the store by id."""
        return self.store.delete_document(doc_id)
