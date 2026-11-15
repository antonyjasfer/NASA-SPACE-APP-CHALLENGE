from types import SimpleNamespace

from src.llm_config import UNAVAILABLE_MESSAGE, Provider, chat, load_providers

ALL_KEYS = ("MODEL_API_KEY", "META_API_KEY", "META_BASE_URL", "META_MODEL", "META_API_STYLE",
            "NVIDIA_API_KEY", "NVIDIA_BASE_URL", "NVIDIA_MODEL")
META = Provider("meta", "k1", "https://meta.example/v1", "m1")
META_R = Provider("meta_responses", "k3", "https://api.meta.ai/v1", "muse-wonder-1.0", "responses")
NVIDIA = Provider("nvidia", "k2", "https://nv.example/v1", "m2")


class FakeError(Exception):
    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status_code = status


def fake_factory(behaviour):
    """behaviour: provider name -> list of outcomes (Exception or reply text)."""
    calls = []

    def factory(provider):
        def create(**kw):
            calls.append(provider.name)
            outcome = behaviour[provider.name].pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            msg = SimpleNamespace(content=outcome)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    return factory, calls


MSG = [{"role": "user", "content": "hi"}]


def test_no_provider_gives_clean_unavailable(monkeypatch):
    for v in ALL_KEYS:
        monkeypatch.delenv(v, raising=False)
    assert load_providers(secrets={}) == []
    res = chat(MSG, providers=[], use_cache=False)
    assert not res.ok and res.text == UNAVAILABLE_MESSAGE


def test_meta_first_nvidia_only_with_key(monkeypatch):
    for v in ALL_KEYS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("META_API_KEY", "a")          # legacy name still accepted
    secrets = {"NVIDIA_BASE_URL": "u2", "NVIDIA_MODEL": "m2"}
    assert [p.name for p in load_providers(secrets)] == ["meta_responses"]
    secrets["NVIDIA_API_KEY"] = "b"
    assert [p.name for p in load_providers(secrets)] == ["meta_responses", "nvidia"]


def test_meta_defaults_and_styles(monkeypatch):
    for v in ALL_KEYS:
        monkeypatch.delenv(v, raising=False)
    meta = load_providers(secrets={"MODEL_API_KEY": "k"})[0]
    assert (meta.api_style, meta.base_url, meta.model) == ("responses", "https://api.meta.ai/v1", "muse-wonder-1.0")
    chat_style = load_providers(secrets={"MODEL_API_KEY": "k", "META_API_STYLE": "chat",
                                         "META_BASE_URL": "https://x/v1", "META_MODEL": "llama"})[0]
    assert (chat_style.name, chat_style.api_style, chat_style.model) == ("meta", "chat", "llama")


def test_retry_then_fallback(monkeypatch):
    factory, calls = fake_factory({"meta": [FakeError(429), FakeError(503), FakeError(500)],
                                   "nvidia": ["answer"]})
    sleeps = []
    res = chat(MSG, providers=[META, NVIDIA], client_factory=factory, sleep=sleeps.append,
               use_cache=False, max_attempts=3)
    assert res.ok and res.text == "answer" and res.provider == "nvidia"
    assert calls == ["meta", "meta", "meta", "nvidia"]
    assert sleeps == [2.0, 4.0]


def test_non_retryable_skips_to_next_and_all_fail_is_clean():
    factory, calls = fake_factory({"meta": [FakeError(401)], "nvidia": [FakeError(400)]})
    res = chat(MSG, providers=[META, NVIDIA], client_factory=factory, sleep=lambda s: None, use_cache=False)
    assert not res.ok and res.text == UNAVAILABLE_MESSAGE
    assert calls == ["meta", "nvidia"]
    assert "k1" not in (res.error or "")


def test_key_not_in_repr():
    assert "k1" not in repr(META)


def test_never_raises_on_odd_status_or_unserializable_message():
    factory, _ = fake_factory({"meta": [FakeError("429")]})
    res = chat(MSG, providers=[META], client_factory=factory, sleep=lambda s: None, use_cache=False)
    assert not res.ok
    res = chat([{"role": "user", "content": object()}], providers=[META],
               client_factory=fake_factory({"meta": ["x"]})[0], use_cache=True)
    assert res.ok  # default=str in the cache key keeps this from raising


def test_empty_completion_falls_back():
    factory, calls = fake_factory({"meta": [""], "nvidia": ["ok"]})
    res = chat(MSG, providers=[META, NVIDIA], client_factory=factory, sleep=lambda s: None, use_cache=False)
    assert res.ok and res.provider == "nvidia" and calls == ["meta", "nvidia"]


# ----------------------------------------------------------------- Responses API (api.meta.ai)
def responses_factory(make_resp):
    seen = {}

    def factory(provider):
        def create(**kw):
            seen.update(kw)
            return make_resp()
        return SimpleNamespace(responses=SimpleNamespace(create=create),
                               chat=SimpleNamespace(completions=SimpleNamespace(create=None)))
    return factory, seen


def test_responses_path_returns_output_text():
    factory, seen = responses_factory(lambda: SimpleNamespace(output_text="Grounded answer [1].", output=[]))
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q + passages"}]
    res = chat(msgs, providers=[META_R], client_factory=factory, use_cache=False)
    assert res.ok and res.text == "Grounded answer [1]." and res.provider == "meta_responses"
    assert seen == {"model": "muse-wonder-1.0", "instructions": "SYS", "input": "Q + passages"}


def test_responses_walks_output_content_when_output_text_empty():
    item = SimpleNamespace(content=[SimpleNamespace(text="Part one."), SimpleNamespace(text="Part two.")])
    factory, _ = responses_factory(lambda: SimpleNamespace(output_text="", output=[item]))
    res = chat(MSG, providers=[META_R], client_factory=factory, use_cache=False)
    assert res.ok and res.text == "Part one.\nPart two."


def test_responses_empty_output_falls_back_to_passages():
    from src.knowledge import Retriever, load_chunks
    from src.rag import answer

    factory, _ = responses_factory(lambda: SimpleNamespace(output_text="", output=[]))
    a = answer("What are cool flames?", Retriever(load_chunks()),
               chat_fn=lambda m: chat(m, providers=[META_R], client_factory=factory, use_cache=False,
                                      sleep=lambda s: None))
    assert a.status == "unavailable" and a.text.startswith(UNAVAILABLE_MESSAGE) and a.hits


def test_missing_key_gives_unavailable_notice(monkeypatch):
    for v in ALL_KEYS:
        monkeypatch.delenv(v, raising=False)
    res = chat(MSG, providers=load_providers(secrets={}), use_cache=False)
    assert not res.ok and res.text == UNAVAILABLE_MESSAGE and res.error == "no LLM provider configured"


def test_responses_walk_skips_reasoning_and_refusal_items():
    reasoning = SimpleNamespace(type="reasoning", content=[SimpleNamespace(type="reasoning_text", text="secret CoT")])
    msg = SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal", refusal="no"),
                                                   SimpleNamespace(type="output_text", text="Answer.")])
    factory, _ = responses_factory(lambda: SimpleNamespace(output_text="", output=[reasoning, msg]))
    res = chat(MSG, providers=[META_R], client_factory=factory, use_cache=False)
    assert res.ok and res.text == "Answer."


def test_old_llama_endpoint_keeps_chat_style(monkeypatch):
    for v in ALL_KEYS:
        monkeypatch.delenv(v, raising=False)
    p = load_providers(secrets={"META_API_KEY": "k", "META_BASE_URL": "https://api.llama.com/compat/v1/"})[0]
    assert p.api_style == "chat"
    typo = load_providers(secrets={"MODEL_API_KEY": "k", "META_API_STYLE": "respones"})[0]
    assert typo.api_style == "responses"
