"""
pipeline.py
------------
Orchestrates the full RAG flow:

  INGEST:  PDF(s) -> load -> split into chunks -> embed -> upsert to Pinecone
  QUERY:   question -> embed -> similarity search in Pinecone -> return top chunks

  ASK:     question -> retrieve top chunks -> generate a grounded answer
           using Groq's hosted LLM API

This file wires together the loader, splitter, embedder, vector store,
and generator modules into one end-to-end pipeline.
"""

import time
from typing import Dict
from config import config
from loaders.pdf_loader import PDFLoader
from splitters.text_splitter import TextSplitter
from embeddings.sentence_transformer import SentenceTransformerEmbedder
from vectorstores.pinecone_store import PineconeStore
from generators.groq_generator import GroqGenerator


class RAGPipeline:
    def __init__(self):
        config.validate()

        self.embedder = SentenceTransformerEmbedder(
            model_name=config.EMBEDDING_MODEL_NAME
        )
        self.splitter = TextSplitter(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )
        self.store = PineconeStore(
            api_key=config.PINECONE_API_KEY,
            index_name=config.PINECONE_INDEX_NAME,
            dimension=self.embedder.dimension,
            cloud=config.PINECONE_CLOUD,
            region=config.PINECONE_REGION,
        )
        self.generator = GroqGenerator(
            api_key=config.GROQ_API_KEY, model=config.GROQ_MODEL
        )

    def ingest(self, source: str) -> int:
        """
        Ingest a PDF file or a directory of PDFs into Pinecone.

        Args:
            source: path to a .pdf file or a directory of PDFs

        Returns:
            Number of chunks upserted.
        """
        print(f"[1/4] Loading PDF(s) from: {source}")
        loader = PDFLoader(source)
        documents = loader.load()
        print(f"      -> loaded {len(documents)} page(s) of text")

        print("[2/4] Splitting into chunks...")
        chunks = self.splitter.split_documents(documents)
        print(f"      -> created {len(chunks)} chunk(s)")

        print("[3/4] Generating embeddings...")
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)

        print("[4/4] Upserting to Pinecone...")
        count = self.store.upsert(chunks, embeddings)
        print(f"      -> upserted {count} vector(s) into index '{config.PINECONE_INDEX_NAME}'")

        return count

    def query(self, question: str, top_k: int = None):
        """
        Retrieve the most relevant chunks for a question.

        Args:
            question: natural language query
            top_k: number of chunks to retrieve (defaults to config.TOP_K)

        Returns:
            List of matches: [{"text", "source", "page", "score"}, ...]
        """
        top_k = top_k or config.TOP_K
        query_embedding = self.embedder.embed_query(question)
        matches = self.store.query(query_embedding, top_k=top_k)
        return matches

    def ask(self, question: str, top_k: int = None) -> Dict:
        """
        Full RAG: retrieve relevant chunks, then generate a grounded
        natural-language answer using Groq's hosted LLM API.

        Args:
            question: natural language question
            top_k: number of chunks to retrieve (defaults to config.TOP_K)

        Returns:
            {
                "answer": str,
                "sources": [{"text", "source", "page", "score"}, ...],
                "top_k": int,
                "timing_ms": {"retrieval", "generation", "total"}
            }
        """
        top_k = top_k or config.TOP_K

        start = time.perf_counter()
        matches = self.query(question, top_k=top_k)
        retrieval_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        answer = self.generator.generate(question, matches)
        generation_ms = (time.perf_counter() - start) * 1000

        return {
            "answer": answer,
            "sources": matches,
            "top_k": top_k,
            "timing_ms": {
                "retrieval": round(retrieval_ms, 1),
                "generation": round(generation_ms, 1),
                "total": round(retrieval_ms + generation_ms, 1),
            },
        }
