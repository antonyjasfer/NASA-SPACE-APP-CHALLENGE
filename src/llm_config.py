"""LLM provider chain: Meta Llama API first, NVIDIA NIM second (only if its key is set).

Every setting (key, base URL, model) comes from env vars or Streamlit secrets; nothing is
hard-coded here (see .streamlit/secrets.toml.example). `chat()` never raises: if every
provider fails it returns `LLMResult(ok=False, ...)`, which the app turns into an
"AI temporarily unavailable" message plus the retrieved source passages.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

UNAVAILABLE_MESSAGE = "AI temporarily unavailable. Showing the retrieved source passages instead."

# (provider name, env-var prefix). Order = priority.
PROVIDER_ORDER: tuple[tuple[str, str], ...] = (("meta", "META"), ("nvidia", "NVIDIA"))


@dataclass(frozen=True)
class Provider:
    name: str
    api_key: str
    base_url: str
    model: str

    def __repr__(self) -> str:  # never leak the key into logs
        return f"Provider(name={self.name!r}, base_url={self.base_url!r}, model={self.model!r})"


@dataclass(frozen=True)
class LLMResult:
    ok: bool
    text: str
    provider: str | None = None
    model: str | None = None
    error: str | None = None


def _streamlit_secrets() -> Mapping[str, str]:
    try:
        import streamlit as st

        return dict(st.secrets)
    except Exception:  # noqa: BLE001 - no secrets file, or not running under Streamlit
        return {}


def get_setting(name: str, secrets: Mapping[str, str] | None = None) -> str | None:
    """Env var first, then Streamlit secrets. Empty strings count as unset."""
    value = os.environ.get(name)
    if not value:
        value = (secrets if secrets is not None else _streamlit_secrets()).get(name)
    return str(value).strip() if value else None


def load_providers(secrets: Mapping[str, str] | None = None) -> list[Provider]:
    """Providers whose key, base URL and model are all configured, in priority order."""
    secrets = secrets if secrets is not None else _streamlit_secrets()
    providers = []
    for name, prefix in PROVIDER_ORDER:
        key = get_setting(f"{prefix}_API_KEY", secrets)
        url = get_setting(f"{prefix}_BASE_URL", secrets)
        model = get_setting(f"{prefix}_MODEL", secrets)
        if key and url and model:
            providers.append(Provider(name, key, url, model))
    return providers


def _default_client_factory(provider: Provider):
    from openai import OpenAI

    return OpenAI(api_key=provider.api_key, base_url=provider.base_url, timeout=30, max_retries=0)


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    # Connection / timeout errors carry no status code.
    return type(exc).__name__ in {"APIConnectionError", "APITimeoutError", "RateLimitError"}


CACHE_MAX = 256
_CACHE: dict[str, LLMResult] = {}


def _cache_key(messages: Sequence[Mapping[str, str]], temperature: float, max_tokens: int,
               providers: Sequence[Provider]) -> str:
    raw = json.dumps({"m": list(messages), "t": temperature, "n": max_tokens,
                      "p": [(p.name, p.model) for p in providers]}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def chat(
    messages: Sequence[Mapping[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 800,
    providers: Sequence[Provider] | None = None,
    client_factory: Callable[[Provider], object] = _default_client_factory,
    max_attempts: int = 2,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    use_cache: bool = True,
) -> LLMResult:
    """Try each provider with exponential backoff on retryable errors; never raises."""
    try:
        return _chat(messages, temperature, max_tokens, providers, client_factory,
                     max_attempts, base_delay, sleep, use_cache)
    except Exception as exc:  # noqa: BLE001 - contract: the app must never crash on LLM errors
        return LLMResult(False, UNAVAILABLE_MESSAGE, error=f"internal: {type(exc).__name__}")


def _chat(messages, temperature, max_tokens, providers, client_factory,
          max_attempts, base_delay, sleep, use_cache) -> LLMResult:
    providers = list(providers) if providers is not None else load_providers()
    if not providers:
        return LLMResult(False, UNAVAILABLE_MESSAGE, error="no LLM provider configured")
    key = _cache_key(messages, temperature, max_tokens, providers)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    errors = []
    for provider in providers:
        for attempt in range(max_attempts):
            try:
                client = client_factory(provider)
                resp = client.chat.completions.create(
                    model=provider.model,
                    messages=list(messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = (resp.choices[0].message.content or "").strip()
                if not text:
                    raise ValueError("empty completion")
                result = LLMResult(True, text, provider.name, provider.model)
                if use_cache:
                    if len(_CACHE) >= CACHE_MAX:
                        _CACHE.pop(next(iter(_CACHE)))
                    _CACHE[key] = result
                return result
            except Exception as exc:  # noqa: BLE001 - retry, then fall through to the next provider
                errors.append(f"{provider.name}: {type(exc).__name__}")
                if not _is_retryable(exc) or attempt == max_attempts - 1:
                    break
                sleep(base_delay * 2**attempt)
    return LLMResult(False, UNAVAILABLE_MESSAGE, error=" | ".join(errors))
