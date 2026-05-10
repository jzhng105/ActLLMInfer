"""OpenAI-compatible embeddings clients."""

from __future__ import annotations

from typing import Any

import requests

from ..exceptions import APIConnectionError, APIError, AuthenticationError, RateLimitError
from ..utils import SecretStr, get_from_env_or_arg, get_optional_env
from .base import BaseEmbeddings


class OpenAICompatibleEmbeddings(BaseEmbeddings):
    PROVIDER_NAME: str = "openai-compatible"
    DEFAULT_BASE_URL: str = "https://api.openai.com/v1"
    API_KEY_ENV: str = "OPENAI_API_KEY"
    BASE_URL_ENV: str | None = "OPENAI_BASE_URL"
    DEFAULT_MODEL: str = "text-embedding-3-small"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | SecretStr | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        dimensions: int | None = None,
        request_timeout: float | tuple[float, float] = 60.0,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        raw_key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        self.api_key = SecretStr(get_from_env_or_arg(raw_key, self.API_KEY_ENV, f"{self.PROVIDER_NAME} api_key"))
        env_base = self.BASE_URL_ENV
        self.base_url = ((base_url or (get_optional_env(None, env_base) if env_base else None) or self.DEFAULT_BASE_URL).rstrip("/"))
        self.model = model or self.DEFAULT_MODEL
        self.organization = organization
        self.dimensions = dimensions
        self.request_timeout = request_timeout
        self.default_headers = dict(default_headers or {})

    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts, **kwargs)

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        vectors = self._embed([text], **kwargs)
        return vectors[0]

    def _embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        payload: dict[str, Any] = {"model": kwargs.pop("model", self.model), "input": texts}
        dims = kwargs.pop("dimensions", self.dimensions)
        if dims is not None:
            payload["dimensions"] = dims
        payload.update(kwargs)
        response = self._post(payload)
        data = response.json()
        rows = data.get("data") or []
        if not rows:
            raise APIError(f"{self.PROVIDER_NAME} returned no embeddings", status_code=response.status_code, body=response.text)
        rows = sorted(rows, key=lambda r: r.get("index", 0))
        return [r.get("embedding") or [] for r in rows]

    def _post(self, payload: dict[str, Any]) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        headers.update(self.default_headers)
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=self.request_timeout,
            )
        except requests.Timeout as exc:
            raise APIConnectionError(f"{self.PROVIDER_NAME} request timed out") from exc
        except requests.RequestException as exc:
            raise APIConnectionError(f"{self.PROVIDER_NAME} connection failed: {exc}") from exc
        if 200 <= response.status_code < 300:
            return response
        body = response.text
        if response.status_code in (401, 403):
            raise AuthenticationError(f"{self.PROVIDER_NAME} auth failed ({response.status_code}): {body}")
        if response.status_code == 429:
            raise RateLimitError(f"{self.PROVIDER_NAME} rate limited (429): {body}")
        raise APIError(f"{self.PROVIDER_NAME} HTTP {response.status_code}: {body}", status_code=response.status_code, body=body)
