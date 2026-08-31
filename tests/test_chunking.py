import re
import unittest

from langchain_core.documents import Document

from myai_rag.chunking import (
    AdaptiveChunkConfig,
    adaptive_chunk_elements,
    adaptive_split_document,
    parse_page_elements,
)


def count_test_tokens(text: str) -> int:
    """测试用 tokenizer：中文逐字，连续英文/数字各算一个 token。"""
    return len(re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+|[^\w\s]", text))


class AdaptiveChunkingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AdaptiveChunkConfig(max_tokens=40, overlap_tokens=8)

    def test_elements_accumulate_until_token_limit(self) -> None:
        elements = parse_page_elements(
            "财务概览\n"
            "2021年营业收入8000万元，净利润500万元。"
            "公司持续优化产品结构并提升市场竞争力。"
            "2022年营业收入9200万元，净利润680万元。"
        )
        chunks = adaptive_chunk_elements(elements, self.config, count_test_tokens)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(
                count_test_tokens("".join(element.text for element in chunk))
                <= self.config.max_tokens
                for chunk in chunks
            )
        )

    def test_document_metadata_records_dynamic_token_size(self) -> None:
        text = "财务概览\n" + (
            "2021年营业收入8000万元，净利润500万元，总资产20000万元，总负债8000万元。"
            * 8
        )
        chunks = adaptive_split_document(
            Document(page_content=text, metadata={"page": 0, "source": "demo.pdf"}),
            self.config,
            count_test_tokens,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(chunk.metadata["chunk_token_count"] <= self.config.max_tokens for chunk in chunks)
        )
        self.assertTrue(
            all(
                chunk.metadata["chunk_strategy"] == "adaptive_element_token_v1"
                for chunk in chunks
            )
        )
        self.assertGreater(
            len({chunk.metadata["chunk_char_count"] for chunk in chunks}),
            1,
        )
        self.assertTrue(chunks[0].page_content.startswith("财务概览\n"))

    def test_table_rows_remain_one_element_when_under_limit(self) -> None:
        elements = parse_page_elements(
            "指标    2021年    2022年\n营业收入    8000    9200\n说明文字。"
        )

        self.assertEqual(elements[0].kind, "table")
        self.assertIn("营业收入", elements[0].text)

        chunks = adaptive_chunk_elements(elements, self.config, count_test_tokens)
        self.assertTrue(any(element.kind == "table" for element in chunks[0]))


if __name__ == "__main__":
    unittest.main()
