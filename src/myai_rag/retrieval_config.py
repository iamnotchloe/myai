"""Validated retrieval settings shared by the API and offline evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from typing import Mapping


@dataclass(frozen=True)
class RetrievalConfig:
    dense_top_k: int = 10
    bm25_top_k: int = 20
    fused_top_k: int = 30
    rrf_k: int = 5
    dense_rrf_weight: float = 2.0
    bm25_rrf_weight: float = 1.0
    bm25_tokenizer: str = "char-bigram"

    def __post_init__(self) -> None:
        for name in ("dense_top_k", "bm25_top_k", "fused_top_k", "rrf_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        weights = (self.dense_rrf_weight, self.bm25_rrf_weight)
        if any(not math.isfinite(value) or value < 0 for value in weights) or not any(weights):
            raise ValueError("RRF weights must be finite, nonnegative, and not both zero")
        if self.bm25_tokenizer not in ("char-bigram", "current"):
            raise ValueError("Unsupported BM25 tokenizer")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "RetrievalConfig":
        source = os.environ if environ is None else environ
        return cls(
            dense_top_k=int(source.get("DENSE_TOP_K", "10")),
            bm25_top_k=int(source.get("BM25_TOP_K", "20")),
            fused_top_k=int(source.get("FUSED_TOP_K", "30")),
            rrf_k=int(source.get("RRF_K", "5")),
            dense_rrf_weight=float(source.get("DENSE_RRF_WEIGHT", "2.0")),
            bm25_rrf_weight=float(source.get("BM25_RRF_WEIGHT", "1.0")),
            bm25_tokenizer=source.get("BM25_TOKENIZER", "char-bigram"),
        )

    def to_dict(self) -> dict:
        return asdict(self)
