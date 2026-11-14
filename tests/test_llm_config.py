from types import SimpleNamespace

from src.llm_config import UNAVAILABLE_MESSAGE, Provider, chat, load_providers

META = Provider("meta", "k1", "https://meta.example/v1", "m1")
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
    for v in ("META_API_KEY", "META_BASE_URL", "META_MODEL", "NVIDIA_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    assert load_providers(secrets={}) == []
    res = chat(MSG, providers=[], use_cache=False)
    assert not res.ok and res.text == UNAVAILABLE_MESSAGE


def test_meta_first_nvidia_only_with_key(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "a")
    monkeypatch.setenv("META_BASE_URL", "u")
    monkeypatch.setenv("META_MODEL", "m")
    secrets = {"NVIDIA_BASE_URL": "u2", "NVIDIA_MODEL": "m2"}
    assert [p.name for p in load_providers(secrets)] == ["meta"]
    secrets["NVIDIA_API_KEY"] = "b"
    assert [p.name for p in load_providers(secrets)] == ["meta", "nvidia"]


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
