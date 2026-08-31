#!/usr/bin/env python3
from __future__ import annotations

import unittest
import json
from decimal import Decimal
from pathlib import Path

from structured_finance import StructuredFinanceEngine


ROOT = Path(__file__).resolve().parents[1]


class StructuredFinanceEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = StructuredFinanceEngine(ROOT / "structured_financial_data.json")

    def test_single_company_multiple_metrics(self) -> None:
        result = self.engine.answer(
            "滨江消费品2021年营收和净利润各多少？",
            ["滨江消费品有限公司"],
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("8000万元", result.answer)
        self.assertIn("500万元", result.answer)
        self.assertEqual(["11_滨江消费品有限公司_report.pdf"], result.source_files)

    def test_cross_company_difference(self) -> None:
        result = self.engine.answer(
            "阳光传媒与医疗先锋的营业收入相差多少亿元？",
            ["阳光传媒集团有限公司", "医疗先锋股份有限公司"],
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("238亿元", result.answer)
        self.assertIn("150亿元", result.answer)
        self.assertIn("相差88亿元", result.answer)

    def test_requested_amount_unit_is_respected(self) -> None:
        result = self.engine.answer(
            "蓝天旅游2021年的营业收入和净利润换算成万元是多少？",
            ["蓝天旅游有限公司"],
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("营业收入为10000万元", result.answer)
        self.assertIn("净利润为500万元", result.answer)

    def test_equal_revenue(self) -> None:
        result = self.engine.answer(
            "滨江消费品和澜赋科技营业收入是否相同？",
            ["滨江消费品有限公司", "澜赋科技有限公司"],
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("两者相同", result.answer)

    def test_derived_net_margin(self) -> None:
        result = self.engine.answer(
            "澜赋科技和滨江消费品的净利率各是多少，谁更高？",
            ["澜赋科技有限公司", "滨江消费品有限公司"],
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("25%", result.answer)
        self.assertIn("6.25%", result.answer)
        self.assertIn("澜赋科技有限公司更高", result.answer)

    def test_different_currency_difference_falls_back_to_rag(self) -> None:
        result = self.engine.answer(
            "能源巨星和绿源环保营业收入相差多少？",
            ["能源巨星有限公司", "绿源环保有限公司"],
        )
        self.assertIsNone(result)

    def test_non_financial_event_falls_back_to_rag(self) -> None:
        result = self.engine.answer(
            "滨江消费品进行了哪些治理调整？",
            ["滨江消费品有限公司"],
        )
        self.assertIsNone(result)

    def test_explanation_question_falls_back_to_rag(self) -> None:
        result = self.engine.answer(
            "滨江消费品2021年营业收入为什么变化？",
            ["滨江消费品有限公司"],
        )
        self.assertIsNone(result)

    def test_wrong_financial_year_falls_back_to_rag(self) -> None:
        result = self.engine.answer(
            "滨江消费品2020年营业收入是多少？",
            ["滨江消费品有限公司"],
        )
        self.assertIsNone(result)

    def test_structured_data_values_and_sources_are_valid(self) -> None:
        payload = json.loads(
            (ROOT / "structured_financial_data.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (ROOT / "faiss_index" / "documents_metadata.json").read_text(encoding="utf-8")
        )
        indexed_pages = {
            (
                item["metadata"]["source_file"],
                int(item["metadata"].get("page_label", item["metadata"]["page"] + 1)),
            )
            for item in metadata
        }
        required_metrics = {
            "revenue", "net_profit", "total_assets", "total_liabilities",
            "equity", "cash_flow", "debt_ratio", "asset_liability_ratio", "roe",
        }
        for company, company_data in payload["companies"].items():
            with self.subTest(company=company):
                self.assertEqual(required_metrics, set(company_data["metrics"]))
                self.assertIn(
                    (company_data["source_file"], int(company_data["page_number"])),
                    indexed_pages,
                )
                for metric in company_data["metrics"].values():
                    self.assertGreaterEqual(Decimal(metric["value"]), Decimal("0"))


if __name__ == "__main__":
    unittest.main()
