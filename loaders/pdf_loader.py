"""
pdf_loader.py
-------------
Loads PDF files from a folder (or a single file) and extracts their
raw text along with basic metadata (source filename, page number).
"""

import os
from typing import List, Dict
from pypdf import PdfReader


class PDFLoader:
    def __init__(self, source: str):
        """
        Args:
            source: path to a single .pdf file OR a directory containing PDFs.
        """
        self.source = source

    def _load_single_pdf(self, filepath: str) -> List[Dict]:
        documents = []
        reader = PdfReader(filepath)
        filename = os.path.basename(filepath)

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue  # skip blank/scanned pages with no extractable text
            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "source": filename,
                        "page": page_num,
                    },
                }
            )
        return documents

    def load(self) -> List[Dict]:
        """
        Returns:
            List of dicts: [{"text": ..., "metadata": {"source": ..., "page": ...}}, ...]
        """
        all_documents = []

        if os.path.isdir(self.source):
            pdf_files = [
                f for f in sorted(os.listdir(self.source)) if f.lower().endswith(".pdf")
            ]
            if not pdf_files:
                raise FileNotFoundError(f"No PDF files found in directory: {self.source}")
            for pdf_file in pdf_files:
                filepath = os.path.join(self.source, pdf_file)
                all_documents.extend(self._load_single_pdf(filepath))

        elif os.path.isfile(self.source) and self.source.lower().endswith(".pdf"):
            all_documents.extend(self._load_single_pdf(self.source))

        else:
            raise ValueError(f"Invalid source: {self.source} (must be a .pdf file or a directory)")

        return all_documents
