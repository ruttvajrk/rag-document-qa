"""
main.py
--------
FastAPI application exposing the RAG pipeline as a web service:

  GET  /              -> serves the chat UI (templates/index.html)
  POST /api/ingest     -> upload PDF(s), ingest into Pinecone
  POST /api/ask         -> ask a question, get a generated answer + sources
  GET  /api/health      -> simple health check (useful for Azure)
  GET  /api/stats       -> usage and response-time summary from the SQLite query log

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Run in Docker (see Dockerfile):
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import config
from pipeline import RAGPipeline
from storage.query_log import QueryLogger

app = FastAPI(title="RAG Pipeline with Pinecone")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data"
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# The pipeline connects to Pinecone + loads the embedding model on startup.
# Keeping this as a module-level singleton avoids reloading the model
# on every request.
rag_pipeline: RAGPipeline | None = None
query_logger: QueryLogger | None = None


@app.on_event("startup")
def load_pipeline():
    global rag_pipeline, query_logger
    rag_pipeline = RAGPipeline()
    query_logger = QueryLogger(config.QUERY_LOG_DB)


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    if not config.ALLOW_UPLOADS:
        raise HTTPException(status_code=403, detail="Uploads are disabled on this demo")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024

    saved_paths = []
    for upload in files:
        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, detail=f"Only PDF files are supported: {upload.filename}"
            )
        upload.file.seek(0, os.SEEK_END)
        size = upload.file.tell()
        upload.file.seek(0)
        if size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{upload.filename} is larger than {config.MAX_UPLOAD_MB} MB",
            )
        # Keep only the file name to avoid writing outside the upload folder
        dest = UPLOAD_DIR / Path(upload.filename).name
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved_paths.append(str(dest))

    total_chunks = 0
    for path in saved_paths:
        total_chunks += rag_pipeline.ingest(path)

    return {
        "message": f"Ingested {len(saved_paths)} file(s) successfully",
        "files": [os.path.basename(p) for p in saved_paths],
        "chunks_indexed": total_chunks,
    }


@app.post("/api/ask")
def ask(payload: AskRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = rag_pipeline.ask(payload.question, top_k=payload.top_k)
    query_logger.log(
        question=payload.question,
        answer=result["answer"],
        sources=result["sources"],
        top_k=result["top_k"],
        retrieval_ms=result["timing_ms"]["retrieval"],
        generation_ms=result["timing_ms"]["generation"],
    )
    return result


@app.get("/api/stats")
def stats():
    return query_logger.stats()


@app.get("/api/config")
def public_config():
    """Lets the web page hide the upload box when uploads are disabled."""
    return {"allow_uploads": config.ALLOW_UPLOADS}
