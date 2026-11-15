"""LLM provider chain: Meta first, NVIDIA NIM second (only if its key is set).

Meta (api.meta.ai) speaks the OpenAI-compatible RESPONSES API by default (META_API_STYLE=responses):
    client.responses.create(model, instructions=<system prompt>, input=<question + passages>)
META_API_STYLE=chat keeps the old Llama API chat-completions path. Settings come from env vars or
Streamlit secrets (see .streamlit/secrets.toml.example): MODEL_API_KEY (or META_API_KEY),
META_BASE_URL, META_MODEL, META_API_STYLE; NVIDIA_API_KEY / NVIDIA_BASE_URL / NVIDIA_MODEL.
`chat()` never raises: if every provider fails it returns `LLMResult(ok=False, ...)`, which the
app turns into an "AI temporarily unavailable" message plus the retrieved source passages.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

UNAVAILABLE_MESSAGE = "AI temporarily unavailable. Showing the retrieved source passages instead."
log = logging.getLogger("fire_margin.llm")
_TOKEN_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[^\s'\"]+"),
    re.compile(r"(?i)(api[_-]?key['\"]?\s*[:=]\s*['\"]?)[^\s'\",}]+"),
    re.compile(r"\b(?:sk|nvapi|LLM|meta)[-_][A-Za-z0-9._\-]{6,}"),
    re.compile(r"[A-Za-z0-9_\-]{32,}"),
)

META_DEFAULT_BASE_URL = "https://api.meta.ai/v1"
META_DEFAULT_MODEL = "muse-wonder-1.0"
API_STYLES = ("responses", "chat")


@dataclass(frozen=True)
class Provider:
    name: str
    api_key: str
    base_url: str
    model: str
    api_style: str = "chat"          # "responses" or "chat"

    def __repr__(self) -> str:  # never leak the key into logs
        return (f"Provider(name={self.name!r}, base_url={self.base_url!r}, model={self.model!r}, "
                f"api_style={self.api_style!r})")


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

        to_dict = getattr(st.secrets, "to_dict", None)   # sections ([meta]) become plain dicts
        return to_dict() if callable(to_dict) else dict(st.secrets)
    except Exception:  # noqa: BLE001 - no secrets file, or not running under Streamlit
        return {}


def clean_value(value: object) -> str | None:
    """Strip whitespace and one pair of surrounding quotes; empty -> None."""
    if value is None:
        return None
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text or None


def get_setting(name: str, secrets: Mapping[str, object] | None = None) -> str | None:
    """Env var first, then Streamlit secrets. Whitespace/quotes stripped; empty counts as unset."""
    value = clean_value(os.environ.get(name))
    if not value:
        value = clean_value((secrets if secrets is not None else _streamlit_secrets()).get(name))
    return value


def _section(secrets: Mapping[str, object], name: str) -> Mapping[str, object]:
    sec = secrets.get(name)
    return sec if isinstance(sec, Mapping) else {}


def meta_settings(secrets: Mapping[str, object] | None = None) -> dict[str, str | None]:
    """Meta settings from MODEL_API_KEY / META_API_KEY / META_* or a [meta] secrets section."""
    secrets = secrets if secrets is not None else _streamlit_secrets()
    sec = _section(secrets, "meta")
    key = (get_setting("MODEL_API_KEY", secrets) or get_setting("META_API_KEY", secrets)
           or clean_value(sec.get("api_key")))
    base_url = get_setting("META_BASE_URL", secrets) or clean_value(sec.get("base_url")) or META_DEFAULT_BASE_URL
    model = get_setting("META_MODEL", secrets) or clean_value(sec.get("model")) or META_DEFAULT_MODEL
    style = (get_setting("META_API_STYLE", secrets) or clean_value(sec.get("api_style")) or "").lower()
    if style not in API_STYLES:   # unset or typo: infer (old Llama API endpoints speak chat)
        style = "chat" if "llama.com" in base_url else "responses"
    return {"api_key": key, "base_url": base_url, "model": model, "api_style": style}


def load_providers(secrets: Mapping[str, object] | None = None) -> list[Provider]:
    """Configured providers in priority order: Meta (if a key is set), then NVIDIA (if fully set)."""
    secrets = secrets if secrets is not None else _streamlit_secrets()
    providers = []
    m = meta_settings(secrets)
    if m["api_key"]:
        providers.append(Provider("meta_responses" if m["api_style"] == "responses" else "meta",
                                  str(m["api_key"]), str(m["base_url"]), str(m["model"]), str(m["api_style"])))
    nv_key, nv_url, nv_model = (get_setting(f"NVIDIA_{k}", secrets) for k in ("API_KEY", "BASE_URL", "MODEL"))
    if nv_key and nv_url and nv_model:
        providers.append(Provider("nvidia", nv_key, nv_url, nv_model, "chat"))
    return providers


def redact(text: object, secrets_to_hide: Sequence[str | None] = ()) -> str:
    """Remove known keys and anything token-shaped from a message."""
    out = str(text)
    for k in secrets_to_hide:
        if k and len(k) >= 4:
            out = out.replace(k, "[REDACTED]")
    for pat in _TOKEN_PATTERNS:
        out = pat.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", out)
    return out


def _status(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def response_text(resp: object) -> str:
    """Text of a Responses-API result: output_text, else walk output[*].content[*].text."""
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    parts = []
    for item in getattr(resp, "output", None) or []:
        if getattr(item, "type", "message") != "message":       # skip reasoning / tool items
            continue
        for content in getattr(item, "content", None) or []:
            if getattr(content, "type", "output_text") != "output_text":   # skip refusals etc.
                continue
            t = getattr(content, "text", None)
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
    return "\n".join(parts)


def _call(client, provider: Provider, messages: Sequence[Mapping[str, str]], temperature: float,
          max_tokens: int) -> str:
    if provider.api_style == "responses":
        instructions = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        user_input = "\n\n".join(m["content"] for m in messages if m.get("role") != "system")
        # Only parameters the brief specifies: unknown extras could be rejected by api.meta.ai.
        if not instructions:
            return response_text(client.responses.create(model=provider.model, input=user_input))
        try:
            resp = client.responses.create(model=provider.model, instructions=instructions, input=user_input)
        except Exception as exc:
            if _status(exc) not in (400, 422):
                raise
            # Some servers reject `instructions`: retry once with the system text inside `input`.
            resp = client.responses.create(model=provider.model, input=f"{instructions}\n\n{user_input}")
        return response_text(resp)
    resp = client.chat.completions.create(model=provider.model, messages=list(messages),
                                          temperature=temperature, max_tokens=max_tokens)
    return (resp.choices[0].message.content or "").strip()


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
_CACHE_LOCK = threading.Lock()


def _cache_put(key: str, result: LLMResult) -> None:
    """Thread-safe bounded insert; a cache problem must never turn a good answer into a failure."""
    try:
        with _CACHE_LOCK:
            while len(_CACHE) >= CACHE_MAX:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = result
    except Exception:  # noqa: BLE001
        pass


def _cache_key(messages: Sequence[Mapping[str, str]], temperature: float, max_tokens: int,
               providers: Sequence[Provider]) -> str:
    raw = json.dumps({"m": list(messages), "t": temperature, "n": max_tokens,
                      "p": [(p.name, p.base_url, p.model, p.api_style) for p in providers]}, sort_keys=True, default=str)
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
                text = _call(client, provider, messages, temperature, max_tokens)
                if not text:
                    raise ValueError("empty completion")
                result = LLMResult(True, text, provider.name, provider.model)
                if use_cache:
                    _cache_put(key, result)
                return result
            except Exception as exc:  # noqa: BLE001 - retry, then fall through to the next provider
                errors.append(f"{provider.name}: {type(exc).__name__}")
                log.warning("LLM provider %s failed (attempt %d): %s status=%s %s", provider.name, attempt + 1,
                            type(exc).__name__, _status(exc), redact(exc, [provider.api_key])[:300])
                if not _is_retryable(exc) or attempt == max_attempts - 1:
                    break
                sleep(base_delay * 2**attempt)
    return LLMResult(False, UNAVAILABLE_MESSAGE, error=" | ".join(errors))


# ----------------------------------------------------------------- connection check (app sidebar)
CHECK_MESSAGES = [{"role": "system", "content": "You are a connection test. Reply with the single word OK."},
                  {"role": "user", "content": "Reply with OK."}]


def connection_report(secrets: Mapping[str, object] | None = None,
                      client_factory: Callable[[Provider], object] = _default_client_factory) -> list[dict]:
    """One live test call per provider. Never returns or logs a key (only key_found True/False)."""
    secrets = secrets if secrets is not None else _streamlit_secrets()
    m = meta_settings(secrets)
    meta = Provider("meta_responses" if m["api_style"] == "responses" else "meta", m["api_key"] or "",
                    str(m["base_url"]), str(m["model"]), str(m["api_style"]))
    nv = [get_setting(f"NVIDIA_{k}", secrets) for k in ("API_KEY", "BASE_URL", "MODEL")]
    rows = []
    for p in (meta, Provider("nvidia", nv[0] or "", nv[1] or "(not set)", nv[2] or "(not set)", "chat")):
        row: dict = {"provider": p.name, "base_url": p.base_url, "model": p.model, "api_style": p.api_style,
                     "key_found": bool(p.api_key)}
        if not p.api_key:
            rows.append({**row, "ok": False, "result": "skipped: no key found"})
            continue
        try:
            text = _call(client_factory(p), p, CHECK_MESSAGES, 0.0, 16)
            row.update(ok=bool(text), result=("OK: " + redact(text, [p.api_key])[:80]) if text else "empty reply")
        except Exception as exc:  # noqa: BLE001 - this is a diagnostic: report, never raise
            status = _status(exc)
            row.update(ok=False, error_class=type(exc).__name__, status_code=status,
                       result=redact(exc, [p.api_key])[:300])
            if status == 404:
                row["hint"] = f"Model '{p.model}' or base URL not found: check META_MODEL / META_BASE_URL."
                try:
                    row["available_models"] = [md.id for md in getattr(client_factory(p), "models").list()][:30]
                except Exception as list_exc:  # noqa: BLE001
                    row["available_models_error"] = f"{type(list_exc).__name__}: " + redact(list_exc, [p.api_key])[:200]
            log.warning("Connection check %s failed: %s status=%s", p.name, row.get("error_class"), status)
        rows.append(row)
    return rows
