import argparse
from pathlib import Path
from typing import List

from .config import LLMProvider, ResearchQuery, ResearchSettings, SearchConfig, SearchProvider
from .researcher import ResearchManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Research Tool")
    parser.add_argument("brief", help="Research brief or question")
    parser.add_argument("--requirements", help="Additional requirements", default=None)
    parser.add_argument("--llm-provider", choices=[p.value for p in LLMProvider], default=LLMProvider.MOCK.value)
    parser.add_argument("--search-provider", choices=[p.value for p in SearchProvider], default=SearchProvider.DUCKDUCKGO.value)
    parser.add_argument("--iterations", type=int, default=3, help="Number of research iterations")
    parser.add_argument("--max-results", type=int, default=5, help="Max search results per query")
    parser.add_argument("--toc-items", type=int, default=5, help="Number of TOC items to cover")
    parser.add_argument("--additional-pdf", nargs="*", default=[], help="Paths to additional PDF documents")
    parser.add_argument("--output", default="reports/report.md", help="Output report path")
    parser.add_argument("--summary-model", default=None, help="Model override for summaries")
    return parser.parse_args()


def main(args: List[str] | None = None) -> str:
    parsed = parse_args() if args is None else argparse.Namespace(**args)

    llm_provider = LLMProvider(parsed.llm_provider)
    search_provider = SearchProvider(parsed.search_provider)

    search_config = SearchConfig(provider=search_provider, max_results=parsed.max_results)
    research_settings = ResearchSettings(
        iterations=parsed.iterations, toc_items=parsed.toc_items, summary_model=parsed.summary_model
    )

    manager = ResearchManager(
        llm_provider=llm_provider, search_config=search_config, research_settings=research_settings
    )

    research_query = ResearchQuery(
        brief=parsed.brief,
        requirements=parsed.requirements,
        additional_documents=[str(Path(p)) for p in parsed.additional_pdf],
    )

    iterations = manager.run(research_query)
    output_path = manager.generate_report(research_query, iterations, parsed.output)
    return output_path


if __name__ == "__main__":
    main()
