import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import LLMProvider, ResearchQuery, ResearchSettings, SearchConfig
from .evidence import EvidenceItem, EvidenceLocker
from .llm_clients import AnthropicClient, BaseLLMClient, MockLLMClient, OpenAIClient
from .planner import Plan, build_plan
from .report import ReportGenerator
from .search import SearchResult, create_search_client


@dataclass
class ResearchIterationResult:
    toc_item: str
    iteration: int
    queries: List[str]
    findings: List[str]
    search_results: List[SearchResult]


class ResearchManager:
    def __init__(
        self,
        llm_provider: LLMProvider = LLMProvider.MOCK,
        search_config: Optional[SearchConfig] = None,
        research_settings: Optional[ResearchSettings] = None,
    ):
        self.search_config = search_config or SearchConfig()
        self.research_settings = research_settings or ResearchSettings()
        self.llm_client = self._build_llm_client(llm_provider)
        self.search_client = create_search_client(self.search_config)
        self.evidence_locker = EvidenceLocker()
        self.additional_context: str | None = None

    @staticmethod
    def _build_llm_client(provider: LLMProvider) -> BaseLLMClient:
        if provider == LLMProvider.OPENAI:
            return OpenAIClient()
        if provider == LLMProvider.ANTHROPIC:
            return AnthropicClient()
        return MockLLMClient()

    def _summarize_findings(self, toc_item: str, snippets: List[str]) -> str:
        if not snippets:
            return f"{toc_item}: 情報不足のため追加調査が必要です。"
        prompt = (
            f"Summarize the following findings for section '{toc_item}'.\n"
            f"Highlight evidence-backed insights and note uncertainties.\n"
            f"Snippets: {' | '.join(snippets[:5])}"
        )
        try:
            response = self.llm_client.generate(prompt, model=self.research_settings.summary_model)
            return response.content
        except Exception:
            return snippets[0]

    def _generate_queries(self, toc_item: str, base_queries: List[str], iteration: int) -> List[str]:
        prompt = (
            f"You are refining search queries for '{toc_item}'.\n"
            f"Existing queries: {base_queries}\n"
            f"Iteration: {iteration}. Provide up to {self.research_settings.max_queries_per_iteration} improved queries."
        )
        try:
            response = self.llm_client.generate(prompt)
            lines = [line.strip("- ") for line in response.content.splitlines() if line.strip()]
            return lines[: self.research_settings.max_queries_per_iteration]
        except Exception:
            return base_queries

    def _process_search_results(
        self, toc_item: str, iteration: int, search_results: List[SearchResult], findings: List[str]
    ) -> None:
        for result in search_results:
            self.evidence_locker.add_entry(
                EvidenceItem(
                    source_type="web",
                    title=result.title,
                    url=result.url,
                    snippet=result.snippet,
                    iteration=iteration,
                    toc_item=toc_item,
                )
            )
            findings.append(f"{result.title}: {result.snippet}")

    def _ingest_additional_documents(self, documents: List[str]) -> str:
        excerpts: List[str] = []
        for doc in documents:
            path = Path(doc)
            if not path.exists():
                excerpts.append(f"[missing] {doc}")
                continue
            if path.suffix.lower() == ".pdf" and importlib.util.find_spec("pdfminer.high_level"):
                from pdfminer.high_level import extract_text

                text = extract_text(str(path))
                excerpts.append(text[:800])
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
                excerpts.append(text[:800])
        return "\n".join(excerpts)

    def run(self, research_query: ResearchQuery) -> Dict[str, List[ResearchIterationResult]]:
        self.additional_context = self._ingest_additional_documents(research_query.additional_documents)
        research_query.additional_context = self.additional_context
        plan: Plan = build_plan(research_query, self.llm_client, additional_context=self.additional_context)
        iterations: Dict[str, List[ResearchIterationResult]] = {toc: [] for toc in plan.toc[: self.research_settings.toc_items]}

        for iteration in range(1, self.research_settings.iterations + 1):
            for toc_item in iterations.keys():
                queries = self._generate_queries(toc_item, plan.seed_queries, iteration)
                search_results = list(self.search_client.search(" ".join(queries), self.search_config.max_results))
                findings: List[str] = []
                if self.additional_context:
                    findings.append(f"[Additional docs context] {self.additional_context[:200]}")
                self._process_search_results(toc_item, iteration, search_results, findings)
                summary = self._summarize_findings(toc_item, findings)
                findings.append(summary)
                iterations[toc_item].append(
                    ResearchIterationResult(
                        toc_item=toc_item,
                        iteration=iteration,
                        queries=queries,
                        findings=findings,
                        search_results=search_results,
                    )
                )

        return iterations

    def generate_report(
        self, research_query: ResearchQuery, iterations: Dict[str, List[ResearchIterationResult]], output_path: str
    ) -> str:
        report_generator = ReportGenerator(self.evidence_locker, self.research_settings)
        document = report_generator.generate(research_query, iterations)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(document, encoding="utf-8")
        # save evidence
        self.evidence_locker.save_json(str(destination.with_suffix(".evidence.json")))
        self.evidence_locker.save_csv(str(destination.with_suffix(".evidence.csv")))
        return str(destination)
