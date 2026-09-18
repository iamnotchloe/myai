#!/usr/bin/env python3
"""Create or score human reviews against a saved API evaluation (no API calls).

Template: --report run.json --output pending_reviews.jsonl
Score: --report run.json --reviews completed_reviews.jsonl --output reviewed_run.json

Review every factual claim, including incorrect extra facts. Copy its literal
answer span into claims[].claim. Set supported=true only when source evidence
entails it; attach source_file, page_number and a verbatim quote. For calculations,
attach operands' evidence and explain reasoning in notes. Code checks quote
provenance, not entailment. A refusal without factual assertions may have claims=[].
reviewer and review_complete certify human completeness.

semantic_answer_correct: true only if the answer satisfies the gold answer's
meaning, units, year and entities without contradictions (or correctly refuses).
answer_relevance: 0 off-topic, 1 partly on-topic, 2 directly addresses the query.
completeness: 0 no required facts/behavior, 1 partial, 2 all required facts/behavior.
Scores normalize to [0,1]; faithfulness is per-answer supported claim fraction,
claim_support is the pooled fraction. Empty claims are unscored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .evaluate_api import load_jsonl
    from .metrics import answer_digest, human_review_metrics
except ImportError:
    from evaluate_api import load_jsonl
    from metrics import answer_digest, human_review_metrics


def review_template(row):
    return {
        "query_id": row["query_id"], "answer_sha256": answer_digest(row["answer"]),
        "reviewer": "", "review_complete": False, "semantic_answer_correct": None,
        "answer_relevance": None, "completeness": None, "claims": [], "notes": "",
        "question": row["question"], "answer": row["answer"],
        "gold_answer": row.get("gold_answer"), "answerable": row["answerable"],
        "source_documents": row.get("source_documents", []),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("summary", {}).get("mode") == "retrieval_only":
        parser.error("Human answer reviews require an end-to-end report")
    if args.reviews:
        summary = human_review_metrics(report["results"], load_jsonl(args.reviews))
        report["summary"]["human_review"] = summary
        output = json.dumps(report, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        output = "".join(json.dumps(review_template(row), ensure_ascii=False) + "\n"
                         for row in report["results"] if not row.get("error"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
