# memory/long_term.py
from typing import List, Tuple
import pathlib
import json

class LongTermMemory:
    """
    Minimal long-term memory abstraction.
    You can replace the internals with a real vector DB (Chroma, Qdrant, pgvector).
    """

    def __init__(self, root: str = "./data/vector_store"):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_file = self.root / "index.json"
        if self.index_file.exists():
            self.index = json.loads(self.index_file.read_text(encoding="utf-8"))
        else:
            self.index = []

    def add_document(self, doc_id: str, text: str, meta: dict) -> None:
        self.index.append({"doc_id": doc_id, "text": text, "meta": meta})
        self.index_file.write_text(json.dumps(self.index, indent=2), encoding="utf-8")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, dict]]:
        """
        Very simple token-based search.

        - Split the query into lowercase tokens
        - Score each document by how many times those tokens appear
        - Return top_k documents with score > 0
        """
        scored = []
        # split query into words, drop tiny ones like "a", "to"
        tokens = [t.strip().lower() for t in query.split() if len(t.strip()) > 2]

        if not tokens:
            return []

        for item in self.index:
            text = item["text"].lower()
            # naive bag-of-words overlap count
            score = sum(text.count(tok) for tok in tokens)
            scored.append((score, item["doc_id"], item["text"], item["meta"]))

        # sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[Tuple[str, str, dict]] = [
            (doc_id, text, meta)
            for score, doc_id, text, meta in scored[:top_k]
            if score > 0
        ]
        return results

    def delete_document(self, doc_id: str) -> bool:
        """Remove a doc by id; returns True if something was deleted."""
        before = len(self.index)
        self.index = [item for item in self.index if item["doc_id"] != doc_id]
        if len(self.index) < before:
            self.index_file.write_text(json.dumps(self.index, indent=2), encoding="utf-8")
            return True
        return False
