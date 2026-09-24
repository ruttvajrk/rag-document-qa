"""
sentence_transformer.py
------------------------
Wraps a sentence-transformers model to turn text chunks into
dense vector embeddings for storage in / retrieval from Pinecone.
"""

from typing import List
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of chunk texts (used during ingestion)."""
        embeddings = self.model.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string (used during retrieval)."""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
