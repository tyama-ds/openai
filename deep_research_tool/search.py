import importlib
from dataclasses import dataclass
from typing import Iterable, List, Protocol

from .config import SearchConfig, SearchProvider


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class BaseSearchClient(Protocol):
    def search(self, query: str, max_results: int) -> Iterable[SearchResult]:
        ...


class DuckDuckGoSearchClient:
    def __init__(self, config: SearchConfig):
        self.config = config
        if importlib.util.find_spec("duckduckgo_search") is None:
            raise RuntimeError("duckduckgo_search package is required for DuckDuckGoSearchClient")
        from duckduckgo_search import DDGS

        self._ddgs = DDGS()

    def search(self, query: str, max_results: int) -> List[SearchResult]:
        results = self._ddgs.text(query, max_results=max_results)
        normalized = [
            SearchResult(title=item.get("title", ""), url=item.get("href", ""), snippet=item.get("body", ""))
            for item in results
        ]
        return normalized


class SeleniumSearchClient:
    def __init__(self, config: SearchConfig):
        self.config = config

    def search(self, query: str, max_results: int) -> List[SearchResult]:
        raise NotImplementedError(
            "Selenium-based search is a placeholder. Configure a webdriver and implement dynamic extraction."
        )


class MockSearchClient:
    """Offline-friendly search that fabricates results for testing."""

    def __init__(self, reason: str | None = None):
        self.reason = reason or "Mock search client active"

    def search(self, query: str, max_results: int) -> List[SearchResult]:
        hints = [
            "概要", "主要企業", "市場動向", "リスク", "技術動向", "競合", "投資"  # simple variations
        ]
        fabricated = []
        for idx in range(max_results):
            suffix = hints[idx % len(hints)]
            fabricated.append(
                SearchResult(
                    title=f"Mock result {idx + 1}: {query}",
                    url=f"https://example.com/mock/{idx + 1}",
                    snippet=f"{query} に関する{suffix}の概観（テスト用生成）。{self.reason}",
                )
            )
        return fabricated


def create_search_client(config: SearchConfig) -> BaseSearchClient:
    if config.provider == SearchProvider.DUCKDUCKGO:
        try:
            return DuckDuckGoSearchClient(config)
        except Exception as exc:  # pragma: no cover - offline environment fallback
            return MockSearchClient(reason=str(exc))
    if config.provider == SearchProvider.SELENIUM:
        return SeleniumSearchClient(config)
    raise ValueError(f"Unsupported search provider: {config.provider}")
