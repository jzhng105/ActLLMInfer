"""Regression tests for NVIDIA NIM model-id namespace handling.

NIM model ids always include a publisher namespace (``meta/llama-...``,
``nvidia/llama-...``, ``mistralai/...``). The unified endpoint must not strip
the ``nvidia/`` prefix when the remaining model id has no further ``/`` —
otherwise NIM 404s on the resulting bare ``llama-3.1-nemotron-70b-instruct``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import actllminfer.chat_models._openai_compat as openai_compat
from actllminfer import completion, reset_client_cache
from actllminfer.unified import _split_provider_and_model


class _FakeResponse:
    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self) -> Any:
        return self._body

    def iter_lines(self, decode_unicode: bool = False):
        return iter(())

    def close(self) -> None:
        pass


class _Recorder:
    def __init__(self, response: _FakeResponse):
        self.response = response
        self.calls: list[dict] = []

    def __call__(self, url, headers=None, json=None, timeout=None, stream=False):  # noqa: A002
        self.calls.append({"url": url, "json": json})
        return self.response


def _nim_response() -> dict:
    return {
        "id": "chatcmpl-1",
        "model": "nvidia/llama-3.1-nemotron-70b-instruct",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_client_cache()
    yield
    reset_client_cache()


def test_split_keeps_nvidia_publisher_namespace_in_model_id():
    # ``nvidia/<single segment>`` — the head is the publisher namespace,
    # not the provider prefix. Defer to ``_infer_provider``.
    provider, model = _split_provider_and_model("nvidia/llama-3.1-nemotron-70b-instruct")
    assert provider is None
    assert model == "nvidia/llama-3.1-nemotron-70b-instruct"


def test_split_still_strips_nvidia_provider_when_publisher_differs():
    # ``nvidia/meta/...`` — the head IS the provider prefix; the model id
    # retains its ``meta/`` publisher.
    provider, model = _split_provider_and_model("nvidia/meta/llama-3.3-70b-instruct")
    assert provider == "nvidia"
    assert model == "meta/llama-3.3-70b-instruct"


def test_split_strips_nvidia_nim_alias():
    provider, model = _split_provider_and_model("nvidia_nim/nvidia/llama-3.1-nemotron-70b-instruct")
    assert provider == "nvidia_nim"
    assert model == "nvidia/llama-3.1-nemotron-70b-instruct"


def test_split_unaffected_for_other_providers():
    provider, model = _split_provider_and_model("openai/gpt-4o-mini")
    assert provider == "openai"
    assert model == "gpt-4o-mini"


def test_completion_preserves_nvidia_publisher_namespace(monkeypatch):
    """End-to-end: ``nvidia/llama-3.1-nemotron-70b-instruct`` reaches NIM intact."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    recorder = _Recorder(_FakeResponse(200, _nim_response()))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    resp = completion(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        messages="hello",
    )

    assert resp.provider == "nvidia"
    sent = recorder.calls[0]
    assert sent["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    # The publisher namespace MUST survive — NIM 404s without it.
    assert sent["json"]["model"] == "nvidia/llama-3.1-nemotron-70b-instruct"


def test_completion_with_nvidia_nim_alias_and_publisher(monkeypatch):
    """``nvidia_nim/nvidia/...`` is the unambiguous LiteLLM-style form."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    recorder = _Recorder(_FakeResponse(200, _nim_response()))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    completion(
        model="nvidia_nim/nvidia/llama-3.1-nemotron-70b-instruct",
        messages="hello",
    )

    sent = recorder.calls[0]
    assert sent["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert sent["json"]["model"] == "nvidia/llama-3.1-nemotron-70b-instruct"


def test_completion_with_meta_publisher_prefix(monkeypatch):
    """``nvidia/meta/llama-...`` keeps the ``meta/`` publisher and routes to NIM."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    recorder = _Recorder(_FakeResponse(200, {
        "id": "x",
        "model": "meta/llama-3.3-70b-instruct",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
        "usage": {},
    }))
    monkeypatch.setattr(openai_compat.requests, "post", recorder)

    completion(
        model="nvidia/meta/llama-3.3-70b-instruct",
        messages="hello",
    )

    sent = recorder.calls[0]
    assert sent["json"]["model"] == "meta/llama-3.3-70b-instruct"
