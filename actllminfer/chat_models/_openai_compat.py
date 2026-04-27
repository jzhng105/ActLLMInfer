"""Shared OpenAI-compatible chat backend.

Many Chinese LLM providers (Moonshot/Kimi, DeepSeek, ZhipuAI/GLM v4) and
OpenAI itself all speak the same ``/v1/chat/completions`` JSON. Subclassing
``OpenAICompatibleChat`` is the easiest way to add a new provider — usually
you only override ``DEFAULT_BASE_URL`` and the env-var name for the API key.
"""

from __future__ import annotations

import json
from typing import Any, Iterator, Sequence

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
    ToolCall,
)
from ..outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from ..utils import SecretStr, filter_none, get_from_env_or_arg, get_optional_env


class OpenAICompatibleChat(BaseChatModel):
    """Base class for ``/v1/chat/completions``-style providers."""

    #: Subclasses set these.
    PROVIDER_NAME: str = "openai-compatible"
    DEFAULT_BASE_URL: str = "https://api.openai.com/v1"
    API_KEY_ENV: str = "OPENAI_API_KEY"
    BASE_URL_ENV: str | None = "OPENAI_BASE_URL"
    DEFAULT_MODEL: str = "gpt-4o-mini"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | SecretStr | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        default_headers: dict[str, str] | None = None,
        request_timeout: float | tuple[float, float] = 60.0,
        max_retries: int = 2,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            max_retries=max_retries,
            request_timeout=request_timeout,
        )
        raw_key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        self.api_key = SecretStr(get_from_env_or_arg(raw_key, self.API_KEY_ENV, f"{self.PROVIDER_NAME} api_key"))
        env_base = self.BASE_URL_ENV
        self.base_url = (
            (base_url or (get_optional_env(None, env_base) if env_base else None) or self.DEFAULT_BASE_URL)
            .rstrip("/")
        )
        self.organization = organization
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop
        self.default_headers = dict(default_headers or {})
        self.extra_body = dict(extra_body or {})
        self._extra_init_kwargs = kwargs

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #

    def _chat_endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        headers.update(self.default_headers)
        return headers

    def _build_payload(self, messages: list[BaseMessage], stream: bool, **kwargs: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": [self._render_message(m) for m in messages],
            "stream": stream,
            "temperature": kwargs.pop("temperature", self.temperature),
            "top_p": kwargs.pop("top_p", self.top_p),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "stop": kwargs.pop("stop", self.stop),
        }
        for key in ("tools", "tool_choice", "response_format", "seed", "user", "presence_penalty", "frequency_penalty"):
            if key in kwargs:
                payload[key] = kwargs.pop(key)
        payload = filter_none(payload)
        if self.extra_body:
            for k, v in self.extra_body.items():
                payload.setdefault(k, v)
        if kwargs:
            payload.update(kwargs)
        return payload

    @staticmethod
    def _render_message(msg: BaseMessage) -> dict[str, Any]:
        return msg.to_dict()

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        payload = self._build_payload(messages, stream=False, **kwargs)
        response = self._post(payload, stream=False)
        try:
            data = response.json()
        except ValueError as exc:
            raise APIError(f"{self.PROVIDER_NAME} returned non-JSON response", status_code=response.status_code, body=response.text) from exc
        return self._parse_response(data)

    def _stream(self, messages: list[BaseMessage], **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stream=True, **kwargs)
        response = self._post(payload, stream=True)
        try:
            for chunk in self._iter_sse(response):
                parsed = self._parse_stream_chunk(chunk)
                if parsed is not None:
                    yield parsed
        finally:
            response.close()

    # ------------------------------------------------------------------ #
    # HTTP plumbing
    # ------------------------------------------------------------------ #

    def _post(self, payload: dict[str, Any], *, stream: bool) -> requests.Response:
        try:
            response = requests.post(
                self._chat_endpoint(),
                headers=self._build_headers(),
                json=payload,
                timeout=self.request_timeout,
                stream=stream,
            )
        except requests.Timeout as exc:
            raise APIConnectionError(f"{self.PROVIDER_NAME} request timed out") from exc
        except requests.RequestException as exc:
            raise APIConnectionError(f"{self.PROVIDER_NAME} connection failed: {exc}") from exc
        self._raise_for_status(response)
        return response

    @staticmethod
    def _iter_sse(response: requests.Response) -> Iterator[dict[str, Any]]:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue

    def _raise_for_status(self, response: requests.Response) -> None:
        if 200 <= response.status_code < 300:
            return
        body = response.text
        if response.status_code == 401 or response.status_code == 403:
            raise AuthenticationError(f"{self.PROVIDER_NAME} auth failed ({response.status_code}): {body}")
        if response.status_code == 429:
            raise RateLimitError(f"{self.PROVIDER_NAME} rate limited (429): {body}")
        raise APIError(
            f"{self.PROVIDER_NAME} HTTP {response.status_code}: {body}",
            status_code=response.status_code,
            body=body,
        )

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #

    def _parse_response(self, data: dict[str, Any]) -> ChatResult:
        choices = data.get("choices") or []
        if not choices:
            raise APIError(f"{self.PROVIDER_NAME} returned no choices", body=json.dumps(data)[:500])
        generations: list[ChatGeneration] = []
        for choice in choices:
            message = choice.get("message") or {}
            tool_calls = self._parse_tool_calls(message.get("tool_calls") or [])
            additional: dict[str, Any] = {}
            if message.get("reasoning_content"):
                additional["reasoning_content"] = message["reasoning_content"]
            ai = AIMessage(
                content=message.get("content") or "",
                additional_kwargs=additional,
                tool_calls=tool_calls,
            )
            generations.append(
                ChatGeneration(
                    message=ai,
                    generation_info={
                        "finish_reason": choice.get("finish_reason"),
                        "index": choice.get("index"),
                    },
                )
            )
        llm_output = {
            "model": data.get("model"),
            "id": data.get("id"),
            "usage": data.get("usage") or {},
            "system_fingerprint": data.get("system_fingerprint"),
            "provider": self.PROVIDER_NAME,
        }
        return ChatResult(generations=generations, llm_output=filter_none(llm_output))

    def _parse_tool_calls(self, raw_calls: Sequence[dict[str, Any]]) -> list[ToolCall]:
        out: list[ToolCall] = []
        for tc in raw_calls:
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments", "")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) and args_raw else (args_raw if isinstance(args_raw, dict) else {})
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
            out.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), args=args))
        return out

    def _parse_stream_chunk(self, data: dict[str, Any]) -> ChatGenerationChunk | None:
        choices = data.get("choices") or []
        if not choices:
            return None
        choice = choices[0]
        delta = choice.get("delta") or {}
        content = delta.get("content") or ""
        additional: dict[str, Any] = {}
        if delta.get("reasoning_content"):
            additional["reasoning_content"] = delta["reasoning_content"]
        tool_calls: list[ToolCall] = []
        for tc in delta.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args_str = fn.get("arguments", "")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) and args_str else {}
            except json.JSONDecodeError:
                args = {"_raw": args_str}
            tool_calls.append(ToolCall(id=tc.get("id") or "", name=fn.get("name") or "", args=args))
        chunk_msg = AIMessageChunk(content=content, additional_kwargs=additional, tool_calls=tool_calls)
        return ChatGenerationChunk(
            message=chunk_msg,
            generation_info={"finish_reason": choice.get("finish_reason")},
        )


__all__ = ["OpenAICompatibleChat"]
