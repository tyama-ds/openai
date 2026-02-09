# Deep Research Tool Plan

## Objectives
- Provide a CLI-first research orchestrator that can run in notebooks as well.
- Support OpenAI and Anthropic LLM providers with configurable models and API keys.
- Offer pluggable web search (DuckDuckGo API or Selenium-driven browsing) while keeping evidence and citations.
- Produce research reports (Markdown baseline) plus exportable evidence lockers (JSON and CSV).
- Enable iterative research loops, optional ingestion of provided documents (e.g., PDFs), and a verification step that highlights uncertainty.

## Architecture Overview
- `config.py`: Dataclasses for API credentials, search settings, and research parameters.
- `llm_clients.py`: Thin client wrappers for OpenAI, Anthropic, and a mock client for offline/testing.
- `search.py`: Search provider interfaces with a DuckDuckGo implementation and a Selenium placeholder for dynamic pages.
- `evidence.py`: EvidenceLocker for tracking citations and exporting them to JSON/CSV.
- `planner.py`: Builds an initial table of contents and seed queries.
- `researcher.py`: ResearchManager that orchestrates planning, iterative querying, summarization, and verification hooks.
- `report.py`: Generates human-readable reports and verification dashboards.
- `cli.py`: Argparse-based entrypoint for CLI use; also friendly for notebook use by importing ResearchManager.

## Data Flows
1. Operator provides a research brief (and optional PDF paths).
2. Planner produces a table of contents (TOC) and seed search queries via LLM or mock heuristics.
3. Research loop runs N iterations: for each TOC item, generate/refine queries, run searches, collect evidence, and summarize findings.
4. EvidenceLocker records sources and exports CSV/JSON for cross-report integration.
5. ReportGenerator compiles Markdown (extensible to docx/pdf) with sections for findings, citations, and verification risk bands.

## Extensibility Notes
- Search provider selection is pluggable via `SearchConfig.provider`.
- LLM provider selection is pluggable via `LLMProvider` enum.
- Iteration counts, max results, and verification sensitivity are configurable in `ResearchSettings`.
- Additional documents (PDFs) can be ingested when `additional_docs` are provided.

## Testing Strategy
- Offline smoke test via `python -m compileall` to ensure syntax correctness.
- CLI help check to confirm argument wiring.
- Optional integration tests can be added once API keys and networked search are available.
