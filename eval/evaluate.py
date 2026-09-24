"""
evaluate.py
-----------
Measures the running RAG API on a set of test questions and prints the
numbers you can honestly put on a resume.

For each question in eval/questions.csv it records:
  - response time (retrieval, generation, total)
  - for "in_doc" questions: was the expected page among the retrieved sources?
  - for "out_of_doc" questions: did the model correctly decline to answer?

Start the API first (uvicorn main:app, or the Docker container), then run:
    python eval/evaluate.py --url http://localhost:8000 --questions eval/questions.csv

Results are saved to eval/results.csv and eval/summary.json.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import requests

REFUSAL_TEXT = "i don't have enough information"


def page_matches(sources, expected_source, expected_page):
    for s in sources:
        same_page = str(s.get("page")) == str(expected_page)
        same_file = (not expected_source) or s.get("source") == expected_source
        if same_page and same_file:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--questions", default="eval/questions.csv")
    parser.add_argument("--out", default="eval")
    args = parser.parse_args()

    with open(args.questions, newline="", encoding="utf-8") as f:
        questions = [row for row in csv.DictReader(f) if row.get("question", "").strip()]
    if not questions:
        raise SystemExit("No questions found. Fill in eval/questions.csv first.")

    rows = []
    for i, q in enumerate(questions, start=1):
        resp = requests.post(f"{args.url}/api/ask", json={"question": q["question"]}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        declined = REFUSAL_TEXT in data["answer"].lower()

        if q["type"] == "in_doc":
            correct = page_matches(data["sources"], q.get("expected_source", ""), q["expected_page"])
        else:  # out_of_doc
            correct = declined

        rows.append({
            "question": q["question"],
            "type": q["type"],
            "correct": int(correct),
            "declined": int(declined),
            "retrieval_ms": data["timing_ms"]["retrieval"],
            "generation_ms": data["timing_ms"]["generation"],
            "total_ms": data["timing_ms"]["total"],
            "answer": data["answer"],
        })
        print(f"[{i}/{len(questions)}] {'OK ' if correct else 'MISS'} {data['timing_ms']['total']:>8.1f} ms  {q['question'][:60]}")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    in_doc = [r for r in rows if r["type"] == "in_doc"]
    out_doc = [r for r in rows if r["type"] == "out_of_doc"]
    totals = [r["total_ms"] for r in rows]
    summary = {
        "questions_tested": len(rows),
        "retrieval_hit_rate": f"{sum(r['correct'] for r in in_doc)}/{len(in_doc)}" if in_doc else None,
        "correct_refusals": f"{sum(r['correct'] for r in out_doc)}/{len(out_doc)}" if out_doc else None,
        "avg_total_ms": round(statistics.mean(totals), 1),
        "median_total_ms": round(statistics.median(totals), 1),
        "avg_retrieval_ms": round(statistics.mean(r["retrieval_ms"] for r in rows), 1),
        "avg_generation_ms": round(statistics.mean(r["generation_ms"] for r in rows), 1),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        print(f"{k:22}: {v}")
    print(f"\nSaved {out/'results.csv'} and {out/'summary.json'}")


if __name__ == "__main__":
    main()
