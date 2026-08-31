"""RAG retrieval-query completion for multi-turn follow-up questions.

The module turns context-dependent user follow-ups into standalone retrieval
queries.  It deliberately reads only user turns: generated assistant answers
must not become retrieval facts.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Sequence


YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")
FOLLOW_UP_MARKERS = (
    "那",
    "它",
    "该公司",
    "这家公司",
    "这家",
    "上面",
    "刚才",
    "呢",
    "再",
    "相比",
    "比较",
    "对比",
    "换算",
    "分别",
)
COMPARISON_MARKERS = ("相比", "比较", "对比", "区别", "差异")

# Current knowledge base is finance-report oriented.  These terms are used to
# preserve the user's intent when a follow-up only says "相比呢" or "换算呢".
FINANCE_TOPICS = (
    "营业收入",
    "营业总收入",
    "净利润",
    "利润总额",
    "营业利润",
    "总资产",
    "资产总额",
    "总负债",
    "负债总额",
    "负债率",
    "资产负债率",
    "所有者权益",
    "股东权益",
    "现金流",
    "经营活动现金流",
    "研发投入",
    "研发费用",
    "毛利率",
    "同比",
    "增长率",
)


@dataclass(frozen=True)
class QueryRewriteResult:
    """Decision made before retrieval."""

    original_query: str
    rewritten_query: str
    used_history: bool
    inherited_companies: tuple[str, ...] = ()
    inherited_years: tuple[str, ...] = ()
    inherited_topics: tuple[str, ...] = ()
    confidence: float = 1.0
    needs_clarification: bool = False
    clarification_question: str | None = None
    reason: str = "当前问题可独立检索，无需改写。"


def _user_history(
    history: Sequence[Mapping[str, str]], max_user_turns: int
) -> list[str]:
    """Keep recent user queries only and cap pathological payload sizes."""
    turns = []
    for item in history:
        if str(item.get("role", "")).lower() != "user":
            continue
        content = str(item.get("content", "")).strip()
        if content:
            turns.append(content[:500])
    return turns[-max_user_turns:]


def _companies_in(text: str, company_aliases: Mapping[str, str]) -> list[str]:
    lowered = text.casefold()
    found: list[str] = []
    for alias, canonical in sorted(
        company_aliases.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if alias.casefold() in lowered and canonical not in found:
            found.append(canonical)
    return found


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _topics_in(text: str) -> list[str]:
    return [topic for topic in FINANCE_TOPICS if topic in text]


def _is_follow_up(
    question: str, has_history: bool, current_companies: Sequence[str]
) -> bool:
    if not has_history:
        return any(marker in question for marker in FOLLOW_UP_MARKERS)
    return any(marker in question for marker in FOLLOW_UP_MARKERS) or (
        not current_companies and len(question) <= 24
    )


def _latest_context(
    turns: Sequence[str], company_aliases: Mapping[str, str]
) -> tuple[list[str], list[str], list[str]]:
    latest_companies: list[str] = []
    latest_years: list[str] = []
    latest_topics: list[str] = []
    for turn in reversed(turns):
        if not latest_companies:
            latest_companies = _companies_in(turn, company_aliases)
        if not latest_years:
            latest_years = YEAR_PATTERN.findall(turn)
        if not latest_topics:
            latest_topics = _topics_in(turn)
        if latest_companies and latest_years and latest_topics:
            break
    return latest_companies, latest_years, latest_topics


def _clean_follow_up(question: str) -> str:
    cleaned = question.strip().rstrip("？?。！!")
    cleaned = re.sub(r"^(?:那(?:么)?|再|请问)", "", cleaned)
    cleaned = re.sub(r"(?:呢|怎么样|如何)$", "", cleaned)
    cleaned = cleaned.replace("该公司", "").replace("这家公司", "")
    cleaned = re.sub(r"(?<![\w\u4e00-\u9fff])它(?![\w\u4e00-\u9fff])", "", cleaned)
    return cleaned.strip("，, ：:")


def rewrite_retrieval_query(
    question: str,
    history: Sequence[Mapping[str, str]],
    company_aliases: Mapping[str, str],
    *,
    max_user_turns: int = 4,
) -> QueryRewriteResult:
    """Resolve a conversational question into a standalone RAG query.

    The implementation follows the RAG query-rewriting principle: identify
    entities and intent, reduce ambiguity, then retrieve.  It uses deterministic
    slots for the current finance domain so results are testable and do not add
    another model call to every request.
    """
    original = question.strip()
    turns = _user_history(history, max_user_turns)
    current_companies = _companies_in(original, company_aliases)
    current_years = YEAR_PATTERN.findall(original)
    current_topics = _topics_in(original)
    follow_up = _is_follow_up(original, bool(turns), current_companies)

    if not follow_up:
        return QueryRewriteResult(original, original, False)

    history_companies, history_years, history_topics = _latest_context(
        turns, company_aliases
    )
    comparison = any(marker in original for marker in COMPARISON_MARKERS)

    inherited_companies: list[str] = []
    if comparison:
        inherited_companies = [
            company for company in history_companies if company not in current_companies
        ]
    elif not current_companies:
        inherited_companies = history_companies

    inherited_years = history_years if not current_years else []
    inherited_topics = history_topics if not current_topics else []
    companies = _unique([*inherited_companies, *current_companies])
    years = _unique([*current_years, *inherited_years])
    topics = _unique([*current_topics, *inherited_topics])

    deictic = any(
        marker in original
        for marker in ("它", "该公司", "这家公司", "那", "呢", "相比", "换算")
    )
    if deictic and not companies:
        return QueryRewriteResult(
            original_query=original,
            rewritten_query=original,
            used_history=False,
            confidence=0.25,
            needs_clarification=True,
            clarification_question="请补充你要查询的公司名称；我会再从对应报告中检索。",
            reason="追问包含指代词，但当前对话中没有可确认的公司实体。",
        )

    if not (inherited_companies or inherited_years or inherited_topics):
        return QueryRewriteResult(original, original, False)

    company_text = "与".join(companies)
    year_text = "、".join(years)
    topic_text = "、".join(topics)
    cleaned = _clean_follow_up(original)

    # Remove entities, years and already-resolved topic words before assembling
    # a concise standalone query.
    for alias in sorted(company_aliases, key=len, reverse=True):
        cleaned = re.sub(re.escape(alias), "", cleaned, flags=re.IGNORECASE)
    cleaned = YEAR_PATTERN.sub("", cleaned)
    for topic in topics:
        cleaned = cleaned.replace(topic, "")
    cleaned = cleaned.strip("的，, ：:")

    prefix = company_text
    if year_text:
        prefix += f"{year_text}年"
    subject = f"{prefix}{topic_text}"

    if comparison and len(companies) >= 2:
        suffix = cleaned.replace("相比", "").replace("比较", "").replace("对比", "")
        rewritten = f"{subject}{suffix}相比如何？"
    elif "换算" in original:
        transformation = cleaned or "换算"
        rewritten = f"{subject}{transformation}是多少？"
    else:
        suffix = cleaned
        if suffix and suffix not in topic_text:
            rewritten = f"{subject}{suffix}"
            if not rewritten.endswith(("？", "?")):
                rewritten += "？"
        else:
            rewritten = f"{subject}是多少？"

    rewritten = re.sub(r"\s+", "", rewritten)
    return QueryRewriteResult(
        original_query=original,
        rewritten_query=rewritten,
        used_history=True,
        inherited_companies=tuple(inherited_companies),
        inherited_years=tuple(inherited_years),
        inherited_topics=tuple(inherited_topics),
        confidence=0.92 if companies and topics else 0.78,
        reason="根据最近用户问题补齐缺失的公司、年份或财务意图，再执行检索。",
    )
