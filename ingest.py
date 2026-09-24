"""
ingest.py
---------
Command-line ingestion: load every PDF in a folder (or a single PDF)
into the Pinecone index.

Because Pinecone is a cloud database, running this once on your laptop
also fills the index that your deployed demo reads from.

Usage:
    python ingest.py ./data
    python ingest.py ./data/my_file.pdf
"""

import sys
from pipeline import RAGPipeline

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "./data"
    count = RAGPipeline().ingest(source)
    print(f"Done. {count} chunk(s) indexed from {source}")
