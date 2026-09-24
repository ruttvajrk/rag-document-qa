"""
text_splitter.py
-----------------
Splits long document text into overlapping chunks so each chunk stays
within a reasonable size for embedding + retrieval.

Uses a simple, dependency-light recursive character splitter (no
LangChain required), but the interface mirrors LangChain's splitter
so it's easy to swap in later if needed.
"""

from typing import List, Dict


class TextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Order matters: try to split on paragraph, then sentence, then word.
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str) -> List[str]:
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Base case: hard split by character length
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunks.append(text[start:end])
                start = end - self.chunk_overlap
            return chunks

        splits = text.split(separator)
        chunks = []
        current_chunk = ""

        for part in splits:
            candidate = (current_chunk + separator + part) if current_chunk else part
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(part) > self.chunk_size:
                    # Part itself is too big -> recurse with finer separators
                    chunks.extend(self._recursive_split(part, remaining_separators))
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        # Apply overlap between consecutive chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_tail = chunks[i - 1][-self.chunk_overlap:]
                overlapped.append(prev_tail + chunks[i])
            return overlapped

        return chunks

    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Args:
            documents: [{"text": ..., "metadata": {...}}, ...]

        Returns:
            [{"text": chunk, "metadata": {..., "chunk_id": int}}, ...]
        """
        chunked_documents = []
        for doc in documents:
            chunks = self._split_text(doc["text"])
            for i, chunk in enumerate(chunks):
                chunked_documents.append(
                    {
                        "text": chunk,
                        "metadata": {
                            **doc["metadata"],
                            "chunk_id": i,
                        },
                    }
                )
        return chunked_documents
