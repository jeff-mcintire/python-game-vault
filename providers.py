"""
providers.py — LLM provider abstraction layer.

All provider code lives in this single file so it imports cleanly from
the project root on any platform without subdirectory path issues.

Supported providers:
  - Claude  (Anthropic SDK)       env: ANTHROPIC_API_KEY
  - Grok    (OpenAI-compat SDK)   env: XAI_API_KEY

Usage:
    from providers import create_provider, ProviderName
    provider = create_provider(ProviderName.GROK)
    turn = provider.start(system_prompt, user_message, tools)
    while not turn.is_done:
        results = [execute(tc) for tc in turn.tool_calls]
        turn = provider.continue_with_results(results)
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import anthropic
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider names
# ---------------------------------------------------------------------------

class ProviderName(str, Enum):
    CLAUDE = "claude"
    GROK   = "grok"


# ---------------------------------------------------------------------------
# Shared normalized types
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single tool call returned by the LLM."""
    id: str       # provider's tool call id (used to match results)
    name: str     # tool name, e.g. "create_file"
    input: dict   # parsed arguments


@dataclass
class ToolResult:
    """The result of executing a tool call."""
    tool_call_id: str
    content: str


@dataclass
class TurnResult:
    """Normalized result from one LLM turn."""
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def is_done(self) -> bool:
        return len(self.tool_calls) == 0


@dataclass
class ToolDefinition:
    """Provider-agnostic tool schema."""
    name: str
    description: str
    input_schema: dict   # JSON Schema with "properties" and "required"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> ProviderName: ...

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def start(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[ToolDefinition],
    ) -> TurnResult: ...

    @abstractmethod
    def continue_with_results(self, tool_results: list[ToolResult]) -> TurnResult: ...


# ---------------------------------------------------------------------------
# Claude provider
# ---------------------------------------------------------------------------

CLAUDE_DEFAULT_MODEL = "claude-opus-4-5"


class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = CLAUDE_DEFAULT_MODEL):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._system_prompt = ""
        self._messages: list[dict] = []
        self._tools: list[dict] = []

    @property
    def name(self) -> ProviderName:
        return ProviderName.CLAUDE

    @property
    def model(self) -> str:
        return self._model

    def start(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[ToolDefinition],
    ) -> TurnResult:
        self._system_prompt = system_prompt
        self._tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        self._messages = [{"role": "user", "content": user_message}]
        return self._call()

    def continue_with_results(self, tool_results: list[ToolResult]) -> TurnResult:
        self._messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": r.tool_call_id, "content": r.content}
                for r in tool_results
            ],
        })
        return self._call()

    def _call(self) -> TurnResult:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=self._system_prompt,
            tools=self._tools,
            messages=self._messages,
        )
        self._messages.append({"role": "assistant", "content": response.content})

        text = ""
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

        if response.stop_reason == "end_turn" or not tool_calls:
            return TurnResult(text=text)
        return TurnResult(text=text, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Grok provider  (OpenAI-compatible API at api.x.ai)
# ---------------------------------------------------------------------------

GROK_DEFAULT_MODEL = "grok-4-1-fast-reasoning"
GROK_CHAT_MODELS = [
    "grok-4-1-fast-reasoning",      # default — fast + reasoning
    "grok-4-1-fast-non-reasoning",  # fast, no reasoning overhead
    "grok-3",                       # previous flagship
    "grok-3-mini",                  # lightweight / low-cost
]
GROK_BASE_URL = "https://api.x.ai/v1"


class GrokProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = GROK_DEFAULT_MODEL):
        self._client = OpenAI(api_key=api_key, base_url=GROK_BASE_URL)
        self._model = model
        self._messages: list[dict] = []
        self._tools: list[dict] = []

    @property
    def name(self) -> ProviderName:
        return ProviderName.GROK

    @property
    def model(self) -> str:
        return self._model

    def start(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[ToolDefinition],
    ) -> TurnResult:
        self._tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,   # OpenAI uses "parameters" not "input_schema"
                },
            }
            for t in tools
        ]
        self._messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
        return self._call()

    def continue_with_results(self, tool_results: list[ToolResult]) -> TurnResult:
        # OpenAI expects one "tool" message per result, not batched
        for r in tool_results:
            self._messages.append({
                "role": "tool",
                "tool_call_id": r.tool_call_id,
                "content": r.content,
            })
        return self._call()

    def _call(self) -> TurnResult:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": self._messages,
        }
        if self._tools:
            kwargs["tools"] = self._tools

        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        self._messages.append(msg.model_dump(exclude_unset=True))

        text = msg.content or ""
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Grok: non-JSON arguments for {tc.function.name}: "
                        f"{tc.function.arguments!r} — using empty dict"
                    )
                    arguments = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=arguments))

        if response.choices[0].finish_reason == "stop" or not tool_calls:
            return TurnResult(text=text)
        return TurnResult(text=text, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_provider(
    provider_name: ProviderName,
    model: Optional[str] = None,
) -> LLMProvider:
    """
    Instantiate a provider by name, reading API keys from the environment.

    Raises RuntimeError if the required key is missing.
    Raises ValueError for unknown provider names.
    """
    if provider_name == ProviderName.CLAUDE:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        return ClaudeProvider(api_key=api_key, model=model or CLAUDE_DEFAULT_MODEL)

    elif provider_name == ProviderName.GROK:
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("XAI_API_KEY is not set.")
        return GrokProvider(api_key=api_key, model=model or GROK_DEFAULT_MODEL)

    else:
        raise ValueError(f"Unknown provider: {provider_name!r}")