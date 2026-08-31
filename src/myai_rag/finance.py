"""核心财务指标的确定性查询与计算，不调用或替换任何模型。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


METRIC_LABELS = {
    "revenue": "营业收入",
    "net_profit": "净利润",
    "total_assets": "总资产",
    "total_liabilities": "总负债",
    "equity": "股东权益",
    "cash_flow": "现金流量",
    "debt_ratio": "负债比率",
    "asset_liability_ratio": "资产负债率",
    "roe": "净资产收益率",
    "net_margin": "净利率",
}

METRIC_ALIASES = {
    "资产负债率": "asset_liability_ratio",
    "净资产收益率": "roe",
    "股东权益": "equity",
    "所有者权益": "equity",
    "现金流量": "cash_flow",
    "负债比率": "debt_ratio",
    "营业收入": "revenue",
    "总资产": "total_assets",
    "总负债": "total_liabilities",
    "净利润": "net_profit",
    "净利率": "net_margin",
    "现金流": "cash_flow",
    "净资产": "equity",
    "营收": "revenue",
    "净利": "net_profit",
    "ROE": "roe",
    "roe": "roe",
}

AMOUNT_METRICS = {
    "revenue",
    "net_profit",
    "total_assets",
    "total_liabilities",
    "equity",
    "cash_flow",
}


@dataclass(frozen=True)
class StructuredFinanceResult:
    answer: str
    source_companies: list[str]
    source_files: list[str]
    source_pages: list[int]


class StructuredFinanceEngine:
    def __init__(self, data_path: Path):
        with data_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        self.companies: dict[str, dict] = payload["companies"]
        aliases = sorted(METRIC_ALIASES, key=len, reverse=True)
        self.metric_pattern = re.compile("|".join(re.escape(alias) for alias in aliases), re.I)
        self.metric_alias_lookup = {
            alias.casefold(): metric for alias, metric in METRIC_ALIASES.items()
        }

    def requested_metrics(self, question: str) -> list[str]:
        metrics = []
        for match in self.metric_pattern.finditer(question):
            key = self.metric_alias_lookup.get(match.group(0).casefold())
            if key and key not in metrics:
                metrics.append(key)
        return metrics

    def metric(self, company: str, metric_key: str) -> dict | None:
        company_data = self.companies.get(company)
        if not company_data:
            return None
        if metric_key == "net_margin":
            revenue = Decimal(company_data["metrics"]["revenue"]["value"])
            profit = Decimal(company_data["metrics"]["net_profit"]["value"])
            value = (profit / revenue * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            return {"value": str(value), "display": f"{self.clean_decimal(value)}%"}
        return company_data["metrics"].get(metric_key)

    @staticmethod
    def clean_decimal(value: Decimal) -> str:
        text = format(value, "f")
        return text.rstrip("0").rstrip(".") if "." in text else text

    @staticmethod
    def requested_amount_unit(question: str, currency: str) -> str | None:
        units = (
            ("CNY", "亿元"),
            ("CNY", "万元"),
            ("USD", "亿美元"),
            ("USD", "万美元"),
        )
        for unit_currency, unit in units:
            if unit in question:
                return unit if unit_currency == currency else "currency_mismatch"
        return None

    def format_amount(self, value: Decimal, currency: str, unit: str | None = None) -> str:
        unit_divisors = {
            "亿元": Decimal("100000000"),
            "万元": Decimal("10000"),
            "亿美元": Decimal("100000000"),
            "万美元": Decimal("10000"),
        }
        if unit:
            number = value / unit_divisors[unit]
            return f"{self.clean_decimal(number)}{unit}"
        absolute = abs(value)
        if currency == "CNY":
            if absolute >= Decimal("100000000"):
                number = value / Decimal("100000000")
                return f"{self.clean_decimal(number)}亿元"
            if absolute >= Decimal("10000"):
                number = value / Decimal("10000")
                return f"{self.clean_decimal(number)}万元"
            return f"{self.clean_decimal(value)}元"
        if absolute >= Decimal("100000000"):
            number = value / Decimal("100000000")
            return f"{self.clean_decimal(number)}亿美元"
        if absolute >= Decimal("10000"):
            number = value / Decimal("10000")
            return f"{self.clean_decimal(number)}万美元"
        return f"{self.clean_decimal(value)}美元"

    def display_metric(self, question: str, company: str, metric_key: str) -> str:
        metric = self.metric(company, metric_key)
        assert metric is not None
        if metric_key not in AMOUNT_METRICS:
            return metric["display"]
        currency = self.companies[company]["currency"]
        requested_unit = self.requested_amount_unit(question, currency)
        if requested_unit and requested_unit != "currency_mismatch":
            return self.format_amount(Decimal(metric["value"]), currency, requested_unit)
        return metric["display"]

    def answer(self, question: str, mentioned_companies: list[str]) -> StructuredFinanceResult | None:
        if not mentioned_companies:
            return None
        explanatory_signals = (
            "为什么", "原因", "影响因素", "如何改善", "如何提高", "怎么改善",
            "怎么提高", "分析", "解释", "趋势", "变化原因",
        )
        if any(signal in question for signal in explanatory_signals):
            return None
        metrics = self.requested_metrics(question)
        if not metrics:
            return None
        if any(company not in self.companies for company in mentioned_companies):
            return None
        query_years = {int(year) for year in re.findall(r"(?:19|20)\d{2}", question)}
        if query_years and any(
            int(self.companies[company]["report_year"]) not in query_years
            for company in mentioned_companies
        ):
            return None
        if any(self.metric(company, metric) is None for company in mentioned_companies for metric in metrics):
            return None
        if any(
            metric in AMOUNT_METRICS
            and self.requested_amount_unit(question, self.companies[company]["currency"])
            == "currency_mismatch"
            for company in mentioned_companies
            for metric in metrics
        ):
            return None

        lines = []
        for company in mentioned_companies:
            company_data = self.companies[company]
            facts = "，".join(
                f"{METRIC_LABELS[metric]}为{self.display_metric(question, company, metric)}"
                for metric in metrics
            )
            lines.append(f"{company}{company_data['report_year']}年{facts}")

        conclusion = ""
        compare_words = ("相差", "差多少", "差额", "是否相同", "一样", "谁更高", "谁更大", "谁更多")
        if len(mentioned_companies) == 2 and len(metrics) == 1 and any(word in question for word in compare_words):
            left_company, right_company = mentioned_companies
            metric_key = metrics[0]
            left = Decimal(self.metric(left_company, metric_key)["value"])
            right = Decimal(self.metric(right_company, metric_key)["value"])
            if "相同" in question or "一样" in question:
                conclusion = "两者相同。" if left == right else "两者不相同。"
            elif "相差" in question or "差多少" in question or "差额" in question:
                difference = abs(left - right)
                if metric_key in {"roe", "debt_ratio", "asset_liability_ratio", "net_margin"}:
                    conclusion = f"两者相差{self.clean_decimal(difference)}个百分点。"
                else:
                    left_currency = self.companies[left_company]["currency"]
                    right_currency = self.companies[right_company]["currency"]
                    if left_currency != right_currency:
                        return None
                    requested_unit = self.requested_amount_unit(question, left_currency)
                    conclusion = (
                        f"两者相差{self.format_amount(difference, left_currency, requested_unit)}。"
                    )
            else:
                if left == right:
                    conclusion = "两者相同。"
                else:
                    higher = left_company if left > right else right_company
                    conclusion = f"{higher}更高。"

        answer_text = "；".join(lines) + "。"
        if conclusion:
            answer_text += conclusion
        return StructuredFinanceResult(
            answer=answer_text,
            source_companies=mentioned_companies,
            source_files=[self.companies[company]["source_file"] for company in mentioned_companies],
            source_pages=[self.companies[company]["page_number"] for company in mentioned_companies],
        )
