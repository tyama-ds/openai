import importlib
import os
from dataclasses import dataclass
from typing import Optional

from .config import AnthropicConfig, LLMProvider, OpenAIConfig


@dataclass
class LLMResponse:
    content: str
    provider: LLMProvider
    model: Optional[str] = None


class BaseLLMClient:
    provider: LLMProvider

    def generate(self, prompt: str, model: Optional[str] = None) -> LLMResponse:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    provider = LLMProvider.MOCK

    def generate(self, prompt: str, model: Optional[str] = None) -> LLMResponse:
        synthetic = "\n".join(
            [
                "[MOCK OUTPUT]",
                "This response is generated without calling an external API.",
                f"Prompt digest: {prompt[:120]}...",
            ]
        )
        return LLMResponse(content=synthetic, provider=self.provider, model=model)


class OpenAIClient(BaseLLMClient):
    provider = LLMProvider.OPENAI

    def __init__(self, config: Optional[OpenAIConfig] = None):
        self.config = config or OpenAIConfig()

    def _get_client(self):
        if importlib.util.find_spec("openai") is None:
            raise RuntimeError("openai package is required for OpenAIClient")
        import openai

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = self.config.base_url or os.getenv("OPENAI_BASE_URL")
        openai.api_key = api_key
        if base_url:
            openai.base_url = base_url
        return openai

    def generate(self, prompt: str, model: Optional[str] = None) -> LLMResponse:
        client = self._get_client()
        chosen_model = model or self.config.model
        completion = client.chat.completions.create(
            model=chosen_model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = completion.choices[0].message["content"]
        return LLMResponse(content=content, provider=self.provider, model=chosen_model)


class AnthropicClient(BaseLLMClient):
    provider = LLMProvider.ANTHROPIC

    def __init__(self, config: Optional[AnthropicConfig] = None):
        self.config = config or AnthropicConfig()

    def _get_client(self):
        if importlib.util.find_spec("anthropic") is None:
            raise RuntimeError("anthropic package is required for AnthropicClient")
        import anthropic

        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        return anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, model: Optional[str] = None) -> LLMResponse:
        client = self._get_client()
        chosen_model = model or self.config.model
        message = client.messages.create(
            model=chosen_model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        # anthropic SDK returns content as a list of blocks
        content_blocks = [block.text for block in message.content if hasattr(block, "text")]
        content = "\n".join(content_blocks)
        return LLMResponse(content=content, provider=self.provider, model=chosen_model)
