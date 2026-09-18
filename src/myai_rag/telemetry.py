"""Request-local measurements; absent provider usage is unknown, never zero."""

from contextlib import contextmanager
from time import perf_counter


def new_telemetry() -> dict:
    return {
        "stage_latency_ms": {},
        "llm_usage": None,
        "llm_attempts": 0,
        "reranker_attempts": 0,
        "llm_status": "not_called",
        "reranker_method": "not_called",
        "outcome": "pending",
    }


@contextmanager
def measure_stage(telemetry: dict, stage: str):
    started = perf_counter()
    try:
        yield
    finally:
        timings = telemetry["stage_latency_ms"]
        timings[stage] = round(timings.get(stage, 0) + (perf_counter() - started) * 1000, 3)


def record_usage(telemetry: dict, usage) -> None:
    """Record only counts actually reported by the provider."""
    if not isinstance(usage, dict):
        return
    counts = {
        key: usage[key]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if isinstance(usage.get(key), int) and not isinstance(usage[key], bool) and usage[key] >= 0
    }
    if counts:
        telemetry["llm_usage"] = counts
