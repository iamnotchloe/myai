"""Command-line review workflow for the RAG Bad Case queue."""

from __future__ import annotations

import argparse
import json
import os

from .bad_cases import BadCaseStore, DISPOSITIONS, FAILURE_STAGES
from .config import BAD_CASE_PATH, CURATED_BAD_CASE_PATH, QUERY_TRACE_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review RAG Bad Cases one by one")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List queued cases")
    list_parser.add_argument("--status", default="pending_review")

    review_parser = subparsers.add_parser("review", help="Attribute and label one case")
    review_parser.add_argument("bad_case_id")
    review_parser.add_argument("--stage", required=True, choices=FAILURE_STAGES)
    review_parser.add_argument(
        "--disposition",
        default="evaluation_candidate",
        choices=DISPOSITIONS,
    )
    review_parser.add_argument("--notes", default="")
    review_parser.add_argument("--expected-answer", default="")
    review_parser.add_argument(
        "--expected-pages-json",
        default="[]",
        help='JSON array, e.g. [{"source_file":"report.pdf","page_number":2}]',
    )

    subparsers.add_parser("export", help="Export reviewed non-ignored cases")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    store = BadCaseStore(
        QUERY_TRACE_PATH,
        BAD_CASE_PATH,
        max_traces=int(os.getenv("BAD_CASE_TRACE_LIMIT", "500")),
    )
    if args.command == "list":
        items = store.list_cases(status=args.status or None)
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return
    if args.command == "review":
        expected_pages = json.loads(args.expected_pages_json)
        if not isinstance(expected_pages, list):
            raise ValueError("--expected-pages-json 必须是 JSON 数组")
        item = store.review_case(
            args.bad_case_id,
            failure_stage=args.stage,
            disposition=args.disposition,
            review_notes=args.notes,
            expected_answer=args.expected_answer,
            expected_pages=expected_pages,
        )
        print(json.dumps(item, ensure_ascii=False, indent=2))
        return
    count = store.export_reviewed(CURATED_BAD_CASE_PATH)
    print(f"已导出 {count} 条审核样本到 {CURATED_BAD_CASE_PATH}")


if __name__ == "__main__":
    main()
