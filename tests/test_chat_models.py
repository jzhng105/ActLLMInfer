"""End-to-end tests of the chat-model surface using a mocked HTTP layer."""

from __future__ import annotations

import json
from typing import Any

import pytest

import actllminfer.chat_models._openai_compat as openai_compat
from actllminfer import (
    BaseCallbackHandler,
    ChatDeepSeek,
    ChatGLM,
    ChatHuggingFace,
    ChatKimi,
    ChatMiniMax,
    ChatMoonshot,
    ChatOpenAI,
    ChatPromptTemplate,
    ChatZhipuAI,
    ConfigurationError,
    StrOutputParser,
    init_chat_model,
)
from actllminfer.chat_models import anthropic as anthropic_module


# ---------------------------------------------------------------------------
# HTTP mocking helpers
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
    """Drop-in replacement for ``requests.post`` that records the call."""

    def __init__(self, response: _FakeResponse):
        self.response = response
        self.calls: list[dict] = []

    def __call__(self, url, headers=None, json=None, timeout=None, stream=False):  # noqa: A002
        self.calls.append({
            "url": url,
            "headers": dict(headers or {}),
            "json": json,
            "timeout": timeout,
            "stream": stream,
        })
        return self.response


def _openai_response_body(text: str = "Hello there", tool_calls: list[dict] | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": text}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "model": "test-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": msg}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


# ---------------------------------------------------------------------------
# Construction / configuration
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_configuration_error(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        ChatMoonshot()


def test_kimi_is_alias_of_moonshot():
    assert ChatKimi is ChatMoonshot


def test_glm_is_alias_of_zhipuai():
    assert ChatGLM is ChatZhipuAI


def test_provider_default_endpoints(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("ZHIPUAI_API_KEY", "k")
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("HF_TOKEN", "k")

    assert ChatMoonshot()._chat_endpoint() == "https://api.moonshot.cn/v1/chat/completions"
    assert ChatDeepSeek()._chat_endpoint() == "https://api.deepseek.com/v1/chat/completions"
    assert ChatZhipuAI()._chat_endpoint() == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    assert ChatMiniMax()._chat_endpoint() == "https://api.minimax.chat/v1/text/chatcompletion_v2"
    assert ChatOpenAI()._chat_endpoint() == "https://api.openai.com/v1/chat/completions"
    assert anthropic_module.ChatAnthropic()._endpoint() == "https://api.anthropic.com/v1/messages"
    assert ChatHuggingFace()._chat_endpoint() == "https://router.huggingface.co/v1/chat/completions"

# ---------------------------------------------------------------------------
# OpenAI-compatible providers (one shared backend)
# ---------------------------------------------------------------------------


def test_openai_compatible_invoke_returns_ai_message(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    recorder = _Recorder(_FakeResponse(200, _openai_response_body("hi from kimi")))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    llm = ChatMoonshot(model="moonshot-v1-8k", temperature=0.5)
    msg = llm.invoke("hello")

    assert msg.content == "hi from kimi"
    sent = recorder.calls[0]
    assert sent["url"] == "https://api.moonshot.cn/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer test-key"
    assert sent["json"]["model"] == "moonshot-v1-8k"
    assert sent["json"]["messages"] == [{"role": "user", "content": "hello"}]
    assert sent["json"]["temperature"] == 0.5
    assert sent["json"]["stream"] is False


def test_openai_compatible_parses_tool_calls(monkeypatch):
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

    llm = ChatDeepSeek(model="deepseek-chat")
    msg = llm.invoke("weather?")
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "get_weather"
    assert msg.tool_calls[0].args == {"city": "Beijing"}


def test_with_tools_binds_tool_spec_to_payload(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    recorder = _Recorder(_FakeResponse(200, _openai_response_body("ok")))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    llm = ChatMoonshot().with_tools(tools)
    llm.invoke("hi")
    assert recorder.calls[0]["json"]["tools"] == tools


def test_streaming_yields_chunks_and_aggregates(monkeypatch):
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
    recorder = _Recorder(_FakeResponse(200, "", lines=sse_lines))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    llm = ChatMoonshot()
    pieces = [chunk.text for chunk in llm.stream("hi")]
    assert "".join(pieces) == "Hello"


def test_callbacks_observe_token_stream(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    sse_lines = [
        'data: {"choices": [{"delta": {"content": "Hi"}}]}',
        '',
        'data: {"choices": [{"delta": {"content": "!"}}]}',
        '',
        'data: [DONE]',
    ]
    monkeypatch.setattr(openai_compat.requests, "post", _Recorder(_FakeResponse(200, "", lines=sse_lines)))

    seen: list[str] = []

    class H(BaseCallbackHandler):
        def on_llm_new_token(self, token, *, chunk=None, **kwargs):
            seen.append(token)

    llm = ChatMoonshot()
    list(llm.stream("hi", callbacks=[H()]))
    assert "".join(seen) == "Hi!"


def test_authentication_error_raised_on_401(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    monkeypatch.setattr(openai_compat.requests, "post", _Recorder(_FakeResponse(401, "nope")))
    from actllminfer import AuthenticationError

    with pytest.raises(AuthenticationError):
        ChatMoonshot().invoke("hi")


# ---------------------------------------------------------------------------
# Chain composition
# ---------------------------------------------------------------------------


def test_prompt_pipe_model_pipe_parser_end_to_end(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    monkeypatch.setattr(
        openai_compat.requests, "post", _Recorder(_FakeResponse(200, _openai_response_body("translated")))
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You translate to {language}."),
        ("user", "{text}"),
    ])
    chain = prompt | ChatMoonshot() | StrOutputParser()
    result = chain.invoke({"language": "French", "text": "Hello"})
    assert result == "translated"


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def test_anthropic_invoke_parses_messages_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    body = {
        "id": "msg_1",
        "model": "claude-sonnet-4-6",
        "content": [
            {"type": "text", "text": "hi from claude"},
            {"type": "tool_use", "id": "tu_1", "name": "f", "input": {"x": 1}},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 4},
    }
    recorder = _Recorder(_FakeResponse(200, body))
    monkeypatch.setattr(anthropic_module.requests, "post", recorder)

    llm = anthropic_module.ChatAnthropic(model="claude-sonnet-4-6", max_tokens=64)
    msg = llm.invoke([("system", "be helpful"), ("user", "hi")])

    sent = recorder.calls[0]
    assert sent["url"] == "https://api.anthropic.com/v1/messages"
    assert sent["headers"]["x-api-key"] == "k"
    assert sent["headers"]["anthropic-version"]
    assert sent["json"]["system"] == "be helpful"
    assert sent["json"]["messages"][0]["role"] == "user"
    assert msg.content == "hi from claude"
    assert msg.tool_calls[0].name == "f"
    assert msg.tool_calls[0].args == {"x": 1}


def test_anthropic_translates_openai_tool_spec(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    body = {
        "id": "msg_1",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {},
    }
    recorder = _Recorder(_FakeResponse(200, body))
    monkeypatch.setattr(anthropic_module.requests, "post", recorder)

    tools = [{
        "type": "function",
        "function": {"name": "lookup", "description": "do it", "parameters": {"type": "object", "properties": {}}},
    }]
    anthropic_module.ChatAnthropic().with_tools(tools).invoke("hi")
    sent_tools = recorder.calls[0]["json"]["tools"]
    assert sent_tools[0]["name"] == "lookup"
    assert "input_schema" in sent_tools[0]
    assert "function" not in sent_tools[0]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_explicit_provider_prefix(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    llm = init_chat_model("kimi:moonshot-v1-8k")
    assert isinstance(llm, ChatMoonshot)
    assert llm.model == "moonshot-v1-8k"


def test_factory_infers_glm_provider(monkeypatch):
    monkeypatch.setenv("ZHIPUAI_API_KEY", "k")
    llm = init_chat_model("glm-4-plus")
    assert isinstance(llm, ChatZhipuAI)


def test_factory_infers_deepseek_provider(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    llm = init_chat_model("deepseek-reasoner", temperature=0)
    assert isinstance(llm, ChatDeepSeek)
    assert llm.temperature == 0


def test_factory_infers_minimax_provider(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    assert isinstance(init_chat_model("abab6.5s-chat"), ChatMiniMax)


def test_factory_unknown_model_requires_explicit_provider():
    with pytest.raises(ConfigurationError):
        init_chat_model("some-totally-unknown-model")

# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------


def test_huggingface_invoke_routes_to_hf_router(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    recorder = _Recorder(_FakeResponse(200, _openai_response_body("hi from hf")))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    llm = ChatHuggingFace(model="meta-llama/Llama-3.3-70B-Instruct")
    msg = llm.invoke("hello")

    sent = recorder.calls[0]
    assert msg.content == "hi from hf"
    assert sent["url"] == "https://router.huggingface.co/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer hf-test"
    assert sent["json"]["model"] == "meta-llama/Llama-3.3-70B-Instruct"


def test_huggingface_falls_back_to_legacy_token_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "legacy-token")
    recorder = _Recorder(_FakeResponse(200, _openai_response_body("ok")))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    ChatHuggingFace().invoke("hi")
    assert recorder.calls[0]["headers"]["Authorization"] == "Bearer legacy-token"


def test_huggingface_missing_token_raises_configuration_error(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    with pytest.raises(ConfigurationError):
        ChatHuggingFace()


def test_huggingface_custom_base_url_for_dedicated_endpoint(monkeypatch):
    """Pointing base_url at a TGI / dedicated Inference Endpoint just works."""
    monkeypatch.setenv("HF_TOKEN", "k")
    recorder = _Recorder(_FakeResponse(200, _openai_response_body("ok")))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    llm = ChatHuggingFace(
        model="tgi",
        base_url="https://my-endpoint.example.com/v1",
    )
    llm.invoke("hi")
    assert recorder.calls[0]["url"] == "https://my-endpoint.example.com/v1/chat/completions"


def test_factory_huggingface_provider_prefix(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "k")
    llm = init_chat_model("huggingface:meta-llama/Llama-3.3-70B-Instruct")
    assert isinstance(llm, ChatHuggingFace)
    assert llm.model == "meta-llama/Llama-3.3-70B-Instruct"


def test_factory_hf_short_prefix(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "k")
    llm = init_chat_model("hf:Qwen/Qwen2.5-72B-Instruct")
    assert isinstance(llm, ChatHuggingFace)
    assert llm.model == "Qwen/Qwen2.5-72B-Instruct"