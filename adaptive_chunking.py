"""元素级、token 感知的自适应分块。

实现严格对应语雀 RAG 1.5.4 的核心流程：
1. 把页面解析为标题、正文和表格样式元素；
2. 分别计算元素 token 数；
3. 当前块加入下一元素将超过 ``max_tokens`` 时结束当前块；
4. 保留完整元素，并用少量完整尾部元素维持跨块上下文。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

from langchain_core.documents import Document


TokenCounter = Callable[[str], int]
SENTENCE_END_RE = re.compile(r"(?<=[。！？；!?])")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
TABLE_SEPARATOR_RE = re.compile(r"\t| {2,}")
TERMINAL_PUNCTUATION = "。！？；!?"


@dataclass(frozen=True)
class AdaptiveChunkConfig:
    """token 上限与完整元素重叠预算。"""

    max_tokens: int = 512
    overlap_tokens: int = 64

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("overlap_tokens 必须位于 [0, max_tokens) 区间")


@dataclass(frozen=True)
class ParsedElement:
    text: str
    kind: str
    start: int
    overlap: bool = False


def approximate_token_count(text: str) -> int:
    """无模型 tokenizer 时的保守估算；构建索引时会注入 BGE tokenizer。"""
    chinese = len(re.findall(r"[\u3400-\u9fff]", text))
    latin_or_number = len(re.findall(r"[A-Za-z0-9]+(?:[._%-][A-Za-z0-9]+)*", text))
    punctuation = len(re.findall(r"[^\w\s\u3400-\u9fff]", text))
    return max(1, chinese + latin_or_number + punctuation)


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    return (
        0 < len(stripped) <= 40
        and not stripped.endswith(tuple(TERMINAL_PUNCTUATION))
        and len(NUMBER_RE.findall(stripped)) <= 1
    )


def _looks_like_table_line(line: str) -> bool:
    cells = [cell.strip() for cell in TABLE_SEPARATOR_RE.split(line) if cell.strip()]
    numeric_cells = sum(bool(NUMBER_RE.search(cell)) for cell in cells)
    return len(cells) >= 3 and numeric_cells >= 2


def _join_soft_wrapped_lines(lines: list[str]) -> str:
    """合并 PDF 布局造成的软换行。"""
    result = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not result:
            result = stripped
            continue
        needs_space = bool(
            re.search(r"[A-Za-z0-9]$", result) and re.match(r"[A-Za-z0-9]", stripped)
        )
        result += (" " if needs_space else "") + stripped
    return result


def parse_page_elements(text: str) -> list[ParsedElement]:
    """把页面解析为标题、正文句子和连续表格行。"""
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    groups: list[tuple[str, list[str]]] = []
    text_lines: list[str] = []
    table_lines: list[str] = []

    def flush_text() -> None:
        nonlocal text_lines
        if text_lines:
            groups.append(("text", text_lines))
            text_lines = []

    def flush_table() -> None:
        nonlocal table_lines
        if table_lines:
            groups.append(("table", table_lines))
            table_lines = []

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            flush_text()
            flush_table()
            continue
        if _looks_like_table_line(raw_line):
            flush_text()
            table_lines.append(line)
            continue
        flush_table()
        if _looks_like_heading(line):
            flush_text()
            groups.append(("heading", [line]))
        else:
            text_lines.append(line)
    flush_text()
    flush_table()

    elements: list[ParsedElement] = []
    cursor = 0
    for kind, lines in groups:
        merged = "\n".join(lines) if kind == "table" else _join_soft_wrapped_lines(lines)
        parts = (
            [part.strip() for part in SENTENCE_END_RE.split(merged) if part.strip()]
            if kind == "text"
            else [merged.strip()]
        )
        for part in parts:
            elements.append(ParsedElement(part, kind, cursor))
            cursor += len(part)
    return elements


def _merge_elements(elements: list[ParsedElement]) -> str:
    merged = ""
    previous_kind = ""
    for element in elements:
        if not merged:
            merged = element.text
        elif element.kind in {"heading", "table", "image"} or previous_kind in {
            "heading",
            "table",
            "image",
        }:
            merged += "\n" + element.text
        else:
            merged += element.text
        previous_kind = element.kind
    return merged.strip()


def _safe_character_cut(text: str, upper_bound: int) -> int:
    """在给定字符上限附近优先选择语义标点。"""
    window = text[: upper_bound + 1]
    cut = max((window.rfind(mark) for mark in "。！？；，、：!?;,"), default=-1) + 1
    return cut if cut >= upper_bound // 2 else upper_bound


def _split_element_to_token_limit(
    element: ParsedElement,
    max_tokens: int,
    count_tokens: TokenCounter,
) -> list[ParsedElement]:
    """单个元素超长时在标点处二分，保证每段不超过 token 上限。"""
    if count_tokens(element.text) <= max_tokens:
        return [element]

    parts: list[ParsedElement] = []
    remaining = element.text
    consumed = 0
    while remaining:
        if count_tokens(remaining) <= max_tokens:
            parts.append(ParsedElement(remaining, element.kind, element.start + consumed))
            break
        low, high = 1, len(remaining)
        while low < high:
            middle = (low + high + 1) // 2
            if count_tokens(remaining[:middle]) <= max_tokens:
                low = middle
            else:
                high = middle - 1
        cut = _safe_character_cut(remaining, low)
        part = remaining[:cut].strip()
        if not part:
            part = remaining[:low]
            cut = low
        parts.append(ParsedElement(part, element.kind, element.start + consumed))
        remaining = remaining[cut:].strip()
        consumed += cut
    return parts


def _select_overlap(
    elements: list[ParsedElement],
    config: AdaptiveChunkConfig,
    count_tokens: TokenCounter,
) -> list[ParsedElement]:
    """仅复制完整尾部元素，避免把表格或事实句再次截断。"""
    if config.overlap_tokens == 0:
        return []
    selected: list[ParsedElement] = []
    for element in reversed(elements):
        if element.kind == "heading" or element.overlap:
            continue
        candidate = [ParsedElement(element.text, element.kind, element.start, True), *selected]
        if count_tokens(_merge_elements(candidate)) > config.overlap_tokens:
            break
        selected = candidate
    return selected


def adaptive_chunk_elements(
    elements: list[ParsedElement],
    config: AdaptiveChunkConfig,
    count_tokens: TokenCounter,
) -> list[list[ParsedElement]]:
    """按语雀 1.5.4 的累计 token 逻辑把元素组成动态大小的块。"""
    normalized: list[ParsedElement] = []
    for element in elements:
        normalized.extend(
            _split_element_to_token_limit(element, config.max_tokens, count_tokens)
        )

    chunks: list[list[ParsedElement]] = []
    current: list[ParsedElement] = []
    for element in normalized:
        candidate = [*current, element]
        if current and count_tokens(_merge_elements(candidate)) > config.max_tokens:
            chunks.append(current)
            current = _select_overlap(current, config, count_tokens)
            candidate = [*current, element]
            if current and count_tokens(_merge_elements(candidate)) > config.max_tokens:
                current = []
        current.append(element)
    if current:
        chunks.append(current)
    return chunks


def adaptive_split_document(
    document: Document,
    config: AdaptiveChunkConfig | None = None,
    count_tokens: TokenCounter = approximate_token_count,
) -> list[Document]:
    """把一个 PDF 页面转换为 token 感知的动态 Document 块。"""
    config = config or AdaptiveChunkConfig()
    element_chunks = adaptive_chunk_elements(
        parse_page_elements(document.page_content), config, count_tokens
    )
    result: list[Document] = []
    for index, elements in enumerate(element_chunks):
        content = _merge_elements(elements)
        non_overlap = [element for element in elements if not element.overlap]
        starts = [element.start for element in non_overlap or elements]
        metadata = dict(document.metadata)
        metadata.update(
            {
                "chunk_strategy": "adaptive_element_token_v1",
                "chunk_max_tokens": config.max_tokens,
                "chunk_token_count": count_tokens(content),
                "chunk_char_count": len(content),
                "chunk_element_count": len(non_overlap),
                "chunk_element_types": sorted(
                    {element.kind for element in non_overlap}
                ),
                "chunk_overlap_tokens": count_tokens(
                    _merge_elements([element for element in elements if element.overlap])
                )
                if any(element.overlap for element in elements)
                else 0,
                "chunk_index_on_page": index,
                "start_index": min(starts),
            }
        )
        result.append(Document(page_content=content, metadata=metadata))
    return result


def adaptive_split_documents(
    documents: list[Document],
    config: AdaptiveChunkConfig | None = None,
    count_tokens: TokenCounter = approximate_token_count,
) -> list[Document]:
    """逐页分块，保持原 PDF 页码引用不变。"""
    config = config or AdaptiveChunkConfig()
    chunks: list[Document] = []
    for document in documents:
        chunks.extend(adaptive_split_document(document, config, count_tokens))
    return chunks
