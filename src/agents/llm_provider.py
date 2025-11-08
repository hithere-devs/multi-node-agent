"""LLM provider abstraction and implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class ToolCall(ABC):
    """Base class for tool call representation."""

    pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Call the LLM with a prompt.

        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system/instruction prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            tools: Optional tool definitions for function calling

        Returns:
            Dictionary with keys:
                - text: The LLM response text
                - tool_calls: List of tool calls (if any)
                - finish_reason: Why the LLM stopped (stop, length, tool_use, etc)
                - usage: Token usage information
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo",
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: Model name (default: gpt-4-turbo)
            base_url: Optional custom base URL
        """
        self.model = model
        self.base_url = base_url

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Call OpenAI API."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            if tools:
                kwargs["tools"] = [
                    {
                        "type": "function",
                        "function": tool,
                    }
                    for tool in tools
                ]

            response = await self.client.chat.completions.create(**kwargs)

            tool_calls = []
            if response.choices[0].message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in response.choices[0].message.tool_calls
                ]

            return {
                "text": response.choices[0].message.content or "",
                "tool_calls": tool_calls,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }

        except Exception as e:
            logger.error("openai_call_failed", error=str(e))
            return {
                "text": "",
                "tool_calls": [],
                "finish_reason": "error",
                "error": str(e),
            }


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, response_text: str = "Mock response"):
        """Initialize mock provider."""
        self.response_text = response_text
        self.call_count = 0

    async def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Return mock response."""
        self.call_count += 1
        return {
            "text": self.response_text,
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
