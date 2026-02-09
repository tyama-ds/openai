from dataclasses import dataclass
from typing import List

from .config import ResearchQuery
from .llm_clients import BaseLLMClient, LLMResponse, MockLLMClient


@dataclass
class Plan:
    toc: List[str]
    seed_queries: List[str]
    llm_response: LLMResponse


PLAN_PROMPT = """
You are a research planner. Given a research brief and requirements, propose a concise table of contents and seed search queries.
Return a bullet list TOC and a separate bullet list of queries.
"""


def _default_plan(query: ResearchQuery) -> Plan:
    toc = [
        "背景と定義",
        "市場動向と主要プレーヤー",
        "技術トレンド",
        "リスクと課題",
        "まとめと示唆",
    ]
    seed_queries = [
        f"{query.brief} とは",
        f"{query.brief} 最新 動向",
        f"{query.brief} 課題 事例",
    ]
    mock_client = MockLLMClient()
    llm_response = mock_client.generate(f"TOC: {toc}\nQueries: {seed_queries}")
    return Plan(toc=toc, seed_queries=seed_queries, llm_response=llm_response)


def build_plan(query: ResearchQuery, llm_client: BaseLLMClient, additional_context: str | None = None) -> Plan:
    prompt = (
        f"{PLAN_PROMPT}\n"
        f"Research brief: {query.brief}\n"
        f"Requirements: {query.requirements or 'N/A'}\n"
        f"Additional documents: {', '.join(query.additional_documents) or 'None'}\n"
        "Respond with two bullet lists labeled TOC and QUERIES."
    )
    if additional_context:
        prompt += f"\nRelevant excerpts from provided documents:\n{additional_context[:1200]}"
    try:
        response = llm_client.generate(prompt)
    except Exception:
        return _default_plan(query)

    content = response.content
    toc, queries = [], []
    for line in content.splitlines():
        normalized = line.strip("- ")
        if normalized.lower().startswith("toc") or normalized.lower().startswith("queries"):
            continue
        if normalized:
            (toc if len(toc) < 5 else queries).append(normalized)
    if not toc or not queries:
        return _default_plan(query)
    return Plan(toc=toc, seed_queries=queries, llm_response=response)
