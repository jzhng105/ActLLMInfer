"""Tests for the LiteLLM-style unified completion / embedding endpoint."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

import actllminfer.chat_models._openai_compat as openai_compat
import actllminfer.embeddings.openai_compat as embeddings_compat
from actllminfer import (
    ConfigurationError,
    Router,
    acompletion,
    aembedding,
    completion,
    embedding,
    reset_client_cache,
    supported_chat_providers,
    supported_embedding_providers,
)
from actllminfer.chat_models import anthropic as anthropic_module
from actllminfer.exceptions import APIError, RateLimitError


# ---------------------------------------------------------------------------
# Shared HTTP mocks (mirror the chat-model test harness)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, body: Any, lines: list[str] | None = None):
        self.status_code = status_code
        self._body = body
        self._lines = lines or []
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self) -> Any:
        if isinstance(self._body, str):
            return json.loads(self._body)
        return self._body

    def iter_lines(self, decode_unicode: bool = False):
        for line in self._lines:
            yield line

    def close(self) -> None:
        pass


class _Recorder:
    def __init__(self, response: _FakeResponse | list[_FakeResponse]):
        self.responses = response if isinstance(response, list) else [response]
        self.calls: list[dict] = []
        self._idx = 0

    def __call__(self, url, headers=None, json=None, timeout=None, stream=False):  # noqa: A002
        self.calls.append({
            "url": url,
            "headers": dict(headers or {}),
            "json": json,
            "timeout": timeout,
            "stream": stream,
        })
        resp = self.responses[min(self._idx, len(self.responses) - 1)]
        self._idx += 1
        return resp


def _openai_response_body(text: str = "Hello", tool_calls: list[dict] | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": text}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-xyz",
        "model": "test-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": msg}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_client_cache()
    yield
    reset_client_cache()


# ---------------------------------------------------------------------------
# completion() — basic dispatch & response shape
# ---------------------------------------------------------------------------


def test_completion_returns_openai_shaped_response(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    recorder = _Recorder(_FakeResponse(200, _openai_response_body("hi from kimi")))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    resp = completion(
        model="kimi/moonshot-v1-8k",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.3,
    )

    assert resp.object == "chat.completion"
    assert resp["choices"][0]["message"]["content"] == "hi from kimi"
    assert resp.choices[0].message.content == "hi from kimi"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage["total_tokens"] == 7
    assert resp.provider == "moonshot"

    sent = recorder.calls[0]
    assert sent["url"] == "https://api.moonshot.cn/v1/chat/completions"
    assert sent["json"]["model"] == "moonshot-v1-8k"
    assert sent["json"]["temperature"] == 0.3


def test_completion_accepts_colon_separator(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    recorder = _Recorder(_FakeResponse(200, _openai_response_body("ok")))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    resp = completion(model="deepseek:deepseek-chat", messages="hi")
    assert resp.text == "ok"
    assert recorder.calls[0]["json"]["model"] == "deepseek-chat"


def test_completion_accepts_plain_string_input(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    recorder = _Recorder(_FakeResponse(200, _openai_response_body("ok")))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    completion(model="kimi/moonshot-v1-8k", messages="hello world")
    assert recorder.calls[0]["json"]["messages"] == [{"role": "user", "content": "hello world"}]


def test_completion_infers_provider_from_model_prefix(monkeypatch):
    monkeypatch.setenv("ZHIPUAI_API_KEY", "k")
    recorder = _Recorder(_FakeResponse(200, _openai_response_body("ok")))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    resp = completion(model="glm-4-plus", messages="hi")
    assert resp.provider == "zhipuai"
    assert recorder.calls[0]["url"].startswith("https://open.bigmodel.cn")


def test_completion_explicit_provider_kwarg(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    recorder = _Recorder(_FakeResponse(200, _openai_response_body("ok")))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    resp = completion(model="some-custom-id", provider="kimi", messages="hi")
    assert resp.provider == "moonshot"
    assert recorder.calls[0]["json"]["model"] == "some-custom-id"


def test_completion_unknown_provider_raises():
    with pytest.raises(ConfigurationError):
        completion(model="totally-unknown-model-id", messages="hi")


def test_completion_unknown_provider_prefix_raises():
    with pytest.raises(ConfigurationError):
        completion(model="someprovider/whatever", provider="someprovider", messages="hi")


# ---------------------------------------------------------------------------
# completion() — tool calls
# ---------------------------------------------------------------------------


def test_completion_serializes_tool_calls_to_openai_shape(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    body = _openai_response_body(
        text="",
        tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
        }],
    )
    recorder = _Recorder(_FakeResponse(200, body))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    resp = completion(model="deepseek/deepseek-chat", messages="weather?")
    tc = resp.choices[0].message["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Beijing"}


def test_completion_forwards_tools_kwarg(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    recorder = _Recorder(_FakeResponse(200, _openai_response_body("ok")))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    completion(model="kimi/moonshot-v1-8k", messages="hi", tools=tools)
    assert recorder.calls[0]["json"]["tools"] == tools


# ---------------------------------------------------------------------------
# completion() — streaming
# ---------------------------------------------------------------------------


def test_completion_stream_yields_openai_shaped_chunks(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    sse_lines = [
        'data: {"choices": [{"delta": {"content": "Hel"}}]}',
        '',
        'data: {"choices": [{"delta": {"content": "lo"}}]}',
        '',
        'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
        '',
        'data: [DONE]',
    ]
    monkeypatch.setattr(openai_compat.requests, "post", _Recorder(_FakeResponse(200, "", lines=sse_lines)))

    chunks = list(completion(model="kimi/moonshot-v1-8k", messages="hi", stream=True))
    contents = [c.choices[0].delta.get("content", "") for c in chunks]
    assert "".join(contents) == "Hello"
    assert {c.object for c in chunks} == {"chat.completion.chunk"}
    # All chunks of one response share an id.
    assert len({c.id for c in chunks}) == 1


# ---------------------------------------------------------------------------
# completion() — Anthropic dispatches to /v1/messages
# ---------------------------------------------------------------------------


def test_completion_routes_anthropic_to_messages_api(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    body = {
        "id": "msg_1",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "hi from claude"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 4, "output_tokens": 6},
    }
    recorder = _Recorder(_FakeResponse(200, body))
    monkeypatch.setattr(anthropic_module.requests, "post", recorder)

    resp = completion(
        model="anthropic/claude-sonnet-4-6",
        messages=[
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ],
        max_tokens=64,
    )
    assert resp.choices[0].message.content == "hi from claude"
    sent = recorder.calls[0]
    assert sent["url"] == "https://api.anthropic.com/v1/messages"
    assert sent["json"]["system"] == "be helpful"


# ---------------------------------------------------------------------------
# Init- vs per-call kwarg routing & client caching
# ---------------------------------------------------------------------------


def test_completion_separates_init_kwargs_from_call_kwargs(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "ignored")
    recorder = _Recorder(_FakeResponse(200, _openai_response_body("ok")))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    completion(
        model="kimi/moonshot-v1-8k",
        messages="hi",
        api_key="explicit-key",
        base_url="https://example.test/v1",
        temperature=0.7,
    )
    sent = recorder.calls[0]
    assert sent["url"] == "https://example.test/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer explicit-key"
    assert sent["json"]["temperature"] == 0.7


def test_completion_reuses_cached_client(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    recorder = _Recorder([
        _FakeResponse(200, _openai_response_body("one")),
        _FakeResponse(200, _openai_response_body("two")),
    ])
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    from actllminfer import unified

    completion(model="kimi/moonshot-v1-8k", messages="hi")
    completion(model="kimi/moonshot-v1-8k", messages="hi again")
    assert len(unified._CHAT_CLIENT_CACHE) == 1
    assert len(recorder.calls) == 2


# ---------------------------------------------------------------------------
# embedding()
# ---------------------------------------------------------------------------


def test_embedding_returns_openai_shape(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    body = {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
            {"object": "embedding", "embedding": [0.3, 0.4], "index": 1},
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }
    recorder = _Recorder(_FakeResponse(200, body))
    monkeypatch.setattr(embeddings_compat.requests, "post", recorder)

    resp = embedding(
        model="openai/text-embedding-3-small",
        input=["hello", "world"],
    )
    assert resp.object == "list"
    assert len(resp.data) == 2
    assert resp.data[0].embedding == [0.1, 0.2]
    assert resp.data[1].index == 1
    assert resp.model == "text-embedding-3-small"
    assert resp.provider == "openai"


def test_embedding_accepts_single_string_input(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    body = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1], "index": 0}],
        "model": "text-embedding-3-small",
    }
    recorder = _Recorder(_FakeResponse(200, body))
    monkeypatch.setattr(embeddings_compat.requests, "post", recorder)

    resp = embedding(model="openai/text-embedding-3-small", input="hi")
    assert recorder.calls[0]["json"]["input"] == ["hi"]
    assert len(resp.data) == 1


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def test_router_falls_back_on_rate_limit(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")

    recorder = _Recorder([
        _FakeResponse(429, "rate-limited"),
        _FakeResponse(200, _openai_response_body("from-fallback")),
    ])
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    router = Router([
        "kimi/moonshot-v1-8k",
        "deepseek/deepseek-chat",
    ])
    resp = router.completion(messages="hi", max_retries=0)
    assert resp.text == "from-fallback"
    assert "moonshot" in recorder.calls[0]["url"]
    assert "deepseek" in recorder.calls[1]["url"]


def test_router_raises_when_all_fail(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setattr(
        openai_compat.requests,
        "post",
        _Recorder([_FakeResponse(500, "boom"), _FakeResponse(500, "boom")]),
    )

    router = Router(["kimi/moonshot-v1-8k", "deepseek/deepseek-chat"])
    with pytest.raises(APIError):
        router.completion(messages="hi", max_retries=0)


def test_router_requires_models():
    with pytest.raises(ConfigurationError):
        Router([])


def test_router_entry_dict_supplies_default_params(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    recorder = _Recorder(_FakeResponse(200, _openai_response_body("ok")))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    router = Router([{"model": "kimi/moonshot-v1-8k", "temperature": 0.1}])
    router.completion(messages="hi")
    assert recorder.calls[0]["json"]["temperature"] == 0.1


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------


def test_acompletion_runs_sync_call_in_thread(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    recorder = _Recorder(_FakeResponse(200, _openai_response_body("async-hi")))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    resp = asyncio.run(
        acompletion(model="kimi/moonshot-v1-8k", messages="hi")
    )
    assert resp.text == "async-hi"


def test_acompletion_streaming_async_iterator(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    sse_lines = [
        'data: {"choices": [{"delta": {"content": "He"}}]}',
        '',
        'data: {"choices": [{"delta": {"content": "y"}}]}',
        '',
        'data: [DONE]',
    ]
    monkeypatch.setattr(openai_compat.requests, "post", _Recorder(_FakeResponse(200, "", lines=sse_lines)))

    async def _collect() -> str:
        out: list[str] = []
        stream = await acompletion(model="kimi/moonshot-v1-8k", messages="hi", stream=True)
        async for chunk in stream:
            out.append(chunk.choices[0].delta.get("content", ""))
        return "".join(out)

    assert asyncio.run(_collect()) == "Hey"


def test_aembedding_runs_sync_call_in_thread(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    body = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
        "model": "text-embedding-3-small",
    }
    monkeypatch.setattr(embeddings_compat.requests, "post", _Recorder(_FakeResponse(200, body)))

    resp = asyncio.run(aembedding(model="openai/text-embedding-3-small", input="hi"))
    assert resp.data[0].embedding == [0.1, 0.2]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_supported_provider_lists_include_expected_names():
    chat = supported_chat_providers()
    assert "openai" in chat and "anthropic" in chat and "kimi" in chat
    emb = supported_embedding_providers()
    assert "openai" in emb and "zhipuai" in emb
