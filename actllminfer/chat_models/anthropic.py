"""Anthropic Claude chat model.

Anthropic's Messages API (``/v1/messages``) is shaped differently from
OpenAI's: ``system`` is a top-level field (not a message), tool calls are
``tool_use`` content blocks, and SSE events are typed
(``content_block_delta``, ``message_delta``, etc.).

This class targets the current public Messages API:
    https://docs.anthropic.com/en/api/messages
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import requests

from ..exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    RateLimitError,
)
from ..language_models.base import BaseChatModel
from ..messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from ..outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from ..utils import SecretStr, filter_none, get_from_env_or_arg, get_optional_env


_DEFAULT_API_VERSION = "2023-06-01"


class ChatAnthropic(BaseChatModel):
    PROVIDER_NAME = "anthropic"
    DEFAULT_BASE_URL = "https://api.anthropic.com"
    API_KEY_ENV = "ANTHROPIC_API_KEY"
    BASE_URL_ENV = "ANTHROPIC_BASE_URL"
    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | SecretStr | None = None,
        base_url: str | None = None,
        anthropic_version: str = _DEFAULT_API_VERSION,
        max_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_sequences: list[str] | None = None,
        request_timeout: float | tuple[float, float] = 60.0,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            max_retries=max_retries,
            request_timeout=request_timeout,
        )
        raw_key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        self.api_key = SecretStr(get_from_env_or_arg(raw_key, self.API_KEY_ENV, "anthropic api_key"))
        self.base_url = (base_url or get_optional_env(None, self.BASE_URL_ENV) or self.DEFAULT_BASE_URL).rstrip("/")
        self.anthropic_version = anthropic_version
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop_sequences = stop_sequences
        self.default_headers = dict(default_headers or {})
        self.extra_body = dict(extra_body or {})
        self._extra_init_kwargs = kwargs

    # ------------------------------------------------------------------ #
    # Payload construction
    # ------------------------------------------------------------------ #

    def _endpoint(self) -> str:
        return f"{self.base_url}/v1/messages"

    def _headers(self) -> dict[str, str]:
        headers = {
            "x-api-key": self.api_key.get_secret_value(),
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(self.default_headers)
        return headers

    def _build_payload(self, messages: list[BaseMessage], stream: bool, **kwargs: Any) -> dict[str, Any]:
        system_text, body = self._split_system(messages)
        payload: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": body,
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": kwargs.pop("temperature", self.temperature),
            "top_p": kwargs.pop("top_p", self.top_p),
            "stop_sequences": kwargs.pop("stop_sequences", kwargs.pop("stop", self.stop_sequences)),
            "stream": stream,
        }
        if system_text:
            payload["system"] = system_text
        if "tools" in kwargs:
            payload["tools"] = self._render_tools(kwargs.pop("tools"))
        if "tool_choice" in kwargs:
            payload["tool_choice"] = kwargs.pop("tool_choice")
        payload = filter_none(payload)
        for k, v in self.extra_body.items():
            payload.setdefault(k, v)
        if kwargs:
            payload.update(kwargs)
        return payload

    @staticmethod
    def _split_system(messages: list[BaseMessage]) -> tuple[str | None, list[dict]]:
        system_parts: list[str] = []
        body: list[dict] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                if isinstance(m.content, str):
                    system_parts.append(m.content)
                else:
                    system_parts.append("".join(p if isinstance(p, str) else p.get("text", "") for p in m.content))
                continue
            body.append(ChatAnthropic._render_message(m))
        return ("\n\n".join(system_parts) if system_parts else None), body

    @staticmethod
    def _render_message(m: BaseMessage) -> dict:
        if isinstance(m, ToolMessage):
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content if isinstance(m.content, str) else m.content,
                    }
                ],
            }
        if isinstance(m, AIMessage):
            content_blocks: list[dict] = []
            if m.content:
                if isinstance(m.content, str):
                    content_blocks.append({"type": "text", "text": m.content})
                else:
                    for p in m.content:
                        if isinstance(p, str):
                            content_blocks.append({"type": "text", "text": p})
                        else:
                            content_blocks.append(p)
            for tc in m.tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.args,
                })
            return {"role": "assistant", "content": content_blocks or [{"type": "text", "text": ""}]}
        # Default: human / user.
        if isinstance(m.content, str):
            return {"role": "user", "content": m.content}
        return {"role": "user", "content": m.content}

    @staticmethod
    def _render_tools(tools: list[Any]) -> list[dict]:
        out: list[dict] = []
        for t in tools:
            if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
                fn = t["function"]
                out.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
            else:
                out.append(t)
        return out

    # ------------------------------------------------------------------ #
    # HTTP
    # ------------------------------------------------------------------ #

    def _post(self, payload: dict[str, Any], *, stream: bool) -> requests.Response:
        try:
            response = requests.post(
                self._endpoint(),
                headers=self._headers(),
                json=payload,
                timeout=self.request_timeout,
                stream=stream,
            )
        except requests.Timeout as exc:
            raise APIConnectionError("anthropic request timed out") from exc
        except requests.RequestException as exc:
            raise APIConnectionError(f"anthropic connection failed: {exc}") from exc
        if 200 <= response.status_code < 300:
            return response
        body = response.text
        if response.status_code in (401, 403):
            raise AuthenticationError(f"anthropic auth failed ({response.status_code}): {body}")
        if response.status_code == 429:
            raise RateLimitError(f"anthropic rate limited (429): {body}")
        raise APIError(f"anthropic HTTP {response.status_code}: {body}", status_code=response.status_code, body=body)

    # ------------------------------------------------------------------ #
    # Generate
    # ------------------------------------------------------------------ #

    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        payload = self._build_payload(messages, stream=False, **kwargs)
        response = self._post(payload, stream=False)
        try:
            data = response.json()
        except ValueError as exc:
            raise APIError("anthropic returned non-JSON response", status_code=response.status_code, body=response.text) from exc
        return self._parse_response(data)

    def _stream(self, messages: list[BaseMessage], **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stream=True, **kwargs)
        response = self._post(payload, stream=True)
        try:
            current_tool_calls: dict[int, ToolCall] = {}
            current_tool_args_buf: dict[int, str] = {}
            for event in self._iter_sse(response):
                etype = event.get("type")
                if etype == "content_block_start":
                    block = event.get("content_block") or {}
                    if block.get("type") == "tool_use":
                        idx = event.get("index", 0)
                        current_tool_calls[idx] = ToolCall(id=block.get("id", ""), name=block.get("name", ""), args={})
                        current_tool_args_buf[idx] = ""
                elif etype == "content_block_delta":
                    delta = event.get("delta") or {}
                    dtype = delta.get("type")
                    if dtype == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield ChatGenerationChunk(message=AIMessageChunk(content=text))
                    elif dtype == "input_json_delta":
                        idx = event.get("index", 0)
                        current_tool_args_buf[idx] = current_tool_args_buf.get(idx, "") + delta.get("partial_json", "")
                elif etype == "content_block_stop":
                    idx = event.get("index", 0)
                    if idx in current_tool_calls:
                        tc = current_tool_calls.pop(idx)
                        raw = current_tool_args_buf.pop(idx, "")
                        try:
                            tc.args = json.loads(raw) if raw else {}
                        except json.JSONDecodeError:
                            tc.args = {"_raw": raw}
                        yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_calls=[tc]))
                elif etype == "message_delta":
                    delta = event.get("delta") or {}
                    info = filter_none({
                        "stop_reason": delta.get("stop_reason"),
                        "stop_sequence": delta.get("stop_sequence"),
                    })
                    if info:
                        yield ChatGenerationChunk(message=AIMessageChunk(content=""), generation_info=info)
                elif etype == "message_stop":
                    return
        finally:
            response.close()

    @staticmethod
    def _iter_sse(response: requests.Response) -> Iterator[dict[str, Any]]:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #

    def _parse_response(self, data: dict[str, Any]) -> ChatResult:
        content_blocks = data.get("content") or []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in content_blocks:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        args=block.get("input", {}) or {},
                    )
                )
        ai = AIMessage(content="".join(text_parts), tool_calls=tool_calls)
        usage = data.get("usage") or {}
        llm_output = filter_none({
            "model": data.get("model"),
            "id": data.get("id"),
            "stop_reason": data.get("stop_reason"),
            "usage": usage,
            "provider": self.PROVIDER_NAME,
        })
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=ai,
                    generation_info=filter_none({"stop_reason": data.get("stop_reason")}),
                )
            ],
            llm_output=llm_output,
        )


__all__ = ["ChatAnthropic"]
