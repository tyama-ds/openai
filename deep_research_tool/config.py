from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"


class SearchProvider(str, Enum):
    DUCKDUCKGO = "duckduckgo"
    SELENIUM = "selenium"


@dataclass
class OpenAIConfig:
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini-2024-07-18"
    base_url: Optional[str] = None


@dataclass
class AnthropicConfig:
    api_key: Optional[str] = None
    model: str = "claude-3-5-sonnet-20240620"


@dataclass
class SearchConfig:
    provider: SearchProvider = SearchProvider.DUCKDUCKGO
    max_results: int = 5
    use_selenium: bool = False


@dataclass
class ResearchSettings:
    iterations: int = 3
    toc_items: int = 5
    max_queries_per_iteration: int = 3
    summary_model: Optional[str] = None
    verification_sensitivity: float = 0.5


@dataclass
class ResearchQuery:
    brief: str
    requirements: Optional[str] = None
    additional_documents: List[str] = field(default_factory=list)
    additional_context: Optional[str] = None
