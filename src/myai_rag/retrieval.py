"""Shared chunk-level retrieval with injected indexes and no model startup."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Sequence

from .retrieval_config import RetrievalConfig

if TYPE_CHECKING:
    from langchain_core.documents import Document


MANUAL_COMPANY_ALIASES = {
    "滨江消费品有限公司": ["滨江消费品"],
    "澜赋科技有限公司": ["澜赋科技"],
    "阳光传媒集团有限公司": ["阳光传媒"],
    "美好家政服务有限公司": ["美好家政", "美好家政服务"],
    "蓝天旅游有限公司": ["蓝天旅游"],
    "ACME研发有限公司": ["ACME研发", "ACME"],
    "绿源环保有限公司": ["绿源环保"],
    "拓远科技有限公司": ["拓远科技"],
    "医疗先锋股份有限公司": ["医疗先锋"],
    "能源巨星有限公司": ["能源巨星"],
}


@dataclass
class RankedDocument:
    document: Document
    score: float | None
    rank: int
    method: str


def normalize_content(content: str) -> str:
    return re.sub(r"[^\w]", "", re.sub(r"\s+", "", content)).lower()


def document_key(document: Document) -> tuple[str, int, int, str]:
    return (
        str(document.metadata.get("source_file", "")),
        int(document.metadata.get("page", 0)),
        int(document.metadata.get("start_index", -1)),
        normalize_content(document.page_content),
    )


def dedupe_documents(documents: Iterable[Document]) -> list[Document]:
    result = []
    seen = set()
    for document in documents:
        key = document_key(document)
        if document.page_content.strip() and key not in seen:
            result.append(document)
            seen.add(key)
    return result


def tokenize_chinese_bm25(text: str) -> list[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:[._%-][a-z0-9]+)*", lowered)
    for run in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def tokenize_whitespace(text: str) -> list[str]:
    """Historical ablation only; Chinese production retrieval uses char-bigram."""
    return text.lower().split()


def build_company_aliases(companies: Iterable[str]) -> dict[str, str]:
    aliases = {}
    suffixes = ("集团有限公司", "股份有限公司", "有限责任公司", "有限公司")
    for company in sorted(set(companies)):
        candidates = {company, *MANUAL_COMPANY_ALIASES.get(company, [])}
        for suffix in suffixes:
            if company.endswith(suffix):
                candidates.add(company[: -len(suffix)])
        for candidate in sorted(candidates):
            alias = candidate.strip().casefold()
            if len(alias) >= 4:
                aliases[alias] = company
    return aliases


def companies_mentioned_in(question: str, aliases: Mapping[str, str]) -> list[str]:
    lowered = question.casefold()
    result = []
    for alias, company in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if alias.casefold() in lowered and company not in result:
            result.append(company)
    return result


def dense_search(question: str, *, vectorstore, documents: Sequence[Document], k: int = 10,
                 target_companies: set[str] | None = None) -> list[RankedDocument]:
    if k < 1 or not documents or vectorstore is None:
        return []
    # Search before filtering: unrelated global TopK must not hide scoped evidence.
    results = vectorstore.similarity_search_with_score(question, k=len(documents) if target_companies else k)
    if target_companies:
        results = [(doc, score) for doc, score in results if str(doc.metadata.get("company", "")) in target_companies]
    return [RankedDocument(doc, float(score), rank, "dense") for rank, (doc, score) in enumerate(results[:k], 1)]


def bm25_search(question: str, *, bm25_model, documents: Sequence[Document], k: int = 20,
                target_companies: set[str] | None = None,
                tokenizer: Callable[[str], list[str]] = tokenize_chinese_bm25) -> list[RankedDocument]:
    if k < 1 or not documents:
        return []
    scores = bm25_model.get_scores(tokenizer(question))
    indices = sorted(range(len(documents)), key=lambda index: float(scores[index]), reverse=True)
    result = []
    for index in indices:
        document = documents[index]
        if target_companies and str(document.metadata.get("company", "")) not in target_companies:
            continue
        score = float(scores[index])
        if not math.isfinite(score) or score <= 0:
            continue
        result.append(RankedDocument(document, score, len(result) + 1, "bm25"))
        if len(result) >= k:
            break
    return result


def rrf_fuse(rankings: list[list[RankedDocument]], rrf_k: int = 5, limit: int = 30,
             weights: tuple[float, ...] = (2.0, 1.0)) -> list[RankedDocument]:
    if len(weights) != len(rankings):
        raise ValueError("RRF weights must match the number of rankings")
    if rrf_k < 1 or limit < 1:
        raise ValueError("RRF k and limit must be positive")
    scores = {}
    documents = {}
    for ranking, weight in zip(rankings, weights):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("RRF weights must be finite and nonnegative")
        if weight == 0:
            continue
        seen = set()
        for item in ranking:
            key = document_key(item.document)
            if key in seen or not item.document.page_content.strip():
                continue
            seen.add(key)
            documents.setdefault(key, item.document)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + item.rank)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [RankedDocument(documents[key], float(score), rank, "rrf") for rank, (key, score) in enumerate(ordered, 1)]


class RetrievalEngine:
    """The same recall/fusion stages for serving, ablations, and offline replay."""

    def __init__(self, documents, vectorstore, bm25_model, config: RetrievalConfig | None = None):
        self.documents = documents
        self.vectorstore = vectorstore
        self.bm25_model = bm25_model
        self.config = config or RetrievalConfig.from_env()
        self.company_aliases = build_company_aliases(str(doc.metadata.get("company", "")) for doc in documents)
        self.tokenizer = tokenize_chinese_bm25 if self.config.bm25_tokenizer == "char-bigram" else tokenize_whitespace

    def search(self, question: str) -> dict[str, list[RankedDocument]]:
        scope = set(companies_mentioned_in(question, self.company_aliases)) or None
        dense = dense_search(question, vectorstore=self.vectorstore, documents=self.documents,
                             k=self.config.dense_top_k, target_companies=scope)
        sparse = bm25_search(question, bm25_model=self.bm25_model, documents=self.documents,
                             k=self.config.bm25_top_k, target_companies=scope, tokenizer=self.tokenizer)
        fused = rrf_fuse([dense, sparse], self.config.rrf_k, self.config.fused_top_k,
                         (self.config.dense_rrf_weight, self.config.bm25_rrf_weight))
        return {"dense": dense, "bm25": sparse, "rrf": fused}
