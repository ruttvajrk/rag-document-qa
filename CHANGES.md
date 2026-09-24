# What changed in this version (and why)

Read this before interviews: you should be able to explain every change.

| Change | File | Why |
|---|---|---|
| Removed `.env` from the project, added `.env.example` and `.gitignore` | root | API keys must never be pushed to GitHub. |
| Deterministic vector IDs (SHA-1 of file name + page + chunk number) instead of random UUIDs | `vectorstores/pinecone_store.py` | Uploading the same PDF twice used to create duplicate vectors, so the same text could fill several of the top-4 results. Now a re-upload overwrites the old vectors. |
| Response-time measurement (retrieval and generation separately) | `pipeline.py` | Needed to report honest response times. |
| SQLite query log: `queries` and `sources` tables | `storage/query_log.py` | Every question, answer, timing and retrieved source is stored, so usage can be analysed with SQL. |
| `/api/stats` endpoint | `main.py` | Quick summary: number of questions, average times, declined answers, most-retrieved documents. |
| Example SQL queries (JOIN, GROUP BY, subquery, window function) | `sql/analysis.sql` | Analysis of the query log. |
| Evaluation script | `eval/evaluate.py` | Measures retrieval hit rate, correct refusals and response time on your own test questions. |
| `ingest.py` command-line ingestion | root | Load a folder of PDFs into Pinecone without the web page (used to prepare the public demo). |
| Upload limits: PDF only, max size (`MAX_UPLOAD_MB`), `ALLOW_UPLOADS=false` demo mode | `main.py`, `config.py`, `index.html` | A public demo should not let strangers upload files and use up your Pinecone/Groq quota. |
| File name cleaned with `Path(...).name` before saving | `main.py` | Stops a crafted file name from writing outside the `data` folder. |
| Web page shows response time; hides upload box in demo mode | `templates/index.html` | Visible performance, cleaner demo. |
| Dockerfile: CPU-only PyTorch, embedding model downloaded at build time, runs as non-root user | `Dockerfile` | Smaller image, faster start, and required by Hugging Face Spaces. |
| README rewritten | `README.md` | Architecture, setup, API, evaluation and deployment for recruiters. |
