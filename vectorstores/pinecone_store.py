"""
pinecone_store.py
-------------------
Wraps the Pinecone client: creates/connects to an index, upserts
chunk embeddings, and runs similarity search queries.

Uses the modern Pinecone SDK (pinecone>=3.0) with serverless indexes.
"""

import hashlib
from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec


class PineconeStore:
    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int,
        cloud: str = "aws",
        region: str = "us-east-1",
        metric: str = "cosine",
    ):
        self.index_name = index_name
        self.pc = Pinecone(api_key=api_key)

        existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

        self.index = self.pc.Index(index_name)

    def upsert(self, chunks: List[Dict], embeddings: List[List[float]], batch_size: int = 100):
        """
        Args:
            chunks: [{"text": ..., "metadata": {...}}, ...]
            embeddings: list of vectors, same length/order as chunks
        """
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            meta = chunk["metadata"]
            # Deterministic ID: the same file/page/chunk always gets the same ID,
            # so re-uploading a PDF overwrites its vectors instead of duplicating them.
            key = f"{meta.get('source')}|{meta.get('page')}|{meta.get('chunk_id')}"
            vector_id = hashlib.sha1(key.encode("utf-8")).hexdigest()
            vectors.append(
                {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        **chunk["metadata"],
                        "text": chunk["text"],
                    },
                }
            )

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch)

        return len(vectors)

    def query(self, query_embedding: List[float], top_k: int = 4) -> List[Dict]:
        """
        Returns:
            [{"text": ..., "source": ..., "page": ..., "score": ...}, ...]
        """
        results = self.index.query(
            vector=query_embedding, top_k=top_k, include_metadata=True
        )

        matches = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            matches.append(
                {
                    "text": metadata.get("text", ""),
                    "source": metadata.get("source", "unknown"),
                    "page": metadata.get("page", "?"),
                    "score": match.get("score", 0.0),
                }
            )
        return matches
