"""P3/P4: knowledge base, grounded retrieval, RAG fallback, literature provenance, app start-up."""
import json
from pathlib import Path

import pytest

from src.catalog import build_literature
from src.knowledge import (
    CHUNKS_PATH,
    Retriever,
    build_chunks,
    load_chunks,
    split_text,
    tokenize,
)
from src.llm_config import UNAVAILABLE_MESSAGE, LLMResult
from src.rag import REFUSAL, STARTER_QUESTIONS, answer, build_messages

OFF_TOPIC = ["best pizza recipe in Naples", "who won the football world cup", "recommend a good movie",
             "explain blockchain", "what is the capital of France", "stock price of Tesla tomorrow",
             "what is the weather in Moon base", "fire safety at my house"]


@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_chunks())


def test_chunks_file_schema_and_pages():
    rows = [json.loads(line) for line in CHUNKS_PATH.read_text().splitlines()]
    assert len(rows) > 150
    assert all({"id", "source", "page_or_id", "text"} <= set(r) for r in rows)
    guide = [r for r in rows if r["id"].startswith("guide_")]
    assert guide and all(r["page_or_id"].isdigit() and 1 <= int(r["page_or_id"]) <= 56 for r in guide)
    assert max(len(r["text"]) for r in guide) <= 820
    kinds = {r["id"].split("_")[0] for r in rows}
    assert {"guide", "card", "lit", "lesson"} <= kinds


def test_chunks_file_is_current():
    assert [c.id for c in load_chunks()] == [c.id for c in build_chunks()]


def test_split_and_tokenize():
    parts = split_text("word " * 500, size=800, overlap=120)
    assert all(len(p) <= 800 for p in parts) and len(parts) > 2
    assert tokenize("Moon-base O2 extinguishment") == ["moon", "base", "o2", "exting"]


@pytest.mark.parametrize("q", STARTER_QUESTIONS)
def test_starter_questions_are_grounded(retriever, q):
    hits = retriever.search(q)
    assert 1 <= len(hits) <= 5


@pytest.mark.parametrize("q", OFF_TOPIC)
def test_off_topic_questions_are_refused(retriever, q):
    assert retriever.search(q) == []
    calls = []
    a = answer(q, retriever, chat_fn=lambda *a, **k: calls.append(1))
    assert a.status == "refused" and a.text == REFUSAL and not calls   # LLM never called


def test_prompt_requires_citations_and_only_retrieved_passages(retriever):
    hits = retriever.search("Why do droplets burst at high oxygen?")
    msgs = build_messages("Why do droplets burst at high oxygen?", hits)
    assert "ONLY the numbered source passages" in msgs[0]["content"] and REFUSAL in msgs[0]["content"]
    assert all(h.chunk.cite() in msgs[1]["content"] for h in hits)


def test_fallback_when_every_provider_fails(retriever):
    a = answer("What did Apollo 1 teach us?", retriever,
               chat_fn=lambda *a, **k: LLMResult(False, UNAVAILABLE_MESSAGE, error="all down"))
    assert a.status == "unavailable" and a.text.startswith(UNAVAILABLE_MESSAGE)
    assert a.hits and all(h.chunk.cite() in a.text for h in a.hits)


def test_fallback_when_chat_raises(retriever):
    def boom(*a, **k):
        raise RuntimeError("network")
    assert answer("What are cool flames?", retriever, chat_fn=boom).status == "unavailable"


def test_uncited_answer_is_not_shown_as_grounded(retriever):
    a = answer("What are cool flames?", retriever,
               chat_fn=lambda *a, **k: LLMResult(True, "Cool flames burn at low temperature.", "meta", "m"))
    assert a.status == "unverified" and "only the source passages are shown" in a.text


def test_foreign_citation_is_rejected(retriever):
    a = answer("What are cool flames?", retriever,
               chat_fn=lambda *a, **k: LLMResult(True, "They are hot. [Wikipedia, p. 1]", "meta", "m"))
    assert a.status == "unverified"


def test_numbered_citations_are_expanded_and_accepted(retriever):
    hits = retriever.search("What are cool flames?")
    a = answer("What are cool flames?", retriever,
               chat_fn=lambda *a, **k: LLMResult(True, "Cool flames form after hot-flame extinction [1].", "meta", "m"))
    assert a.status == "answered" and hits[0].chunk.cite() in a.text


def test_model_refusal_detected(retriever):
    a = answer("What are cool flames?", retriever,
               chat_fn=lambda *a, **k: LLMResult(True, '"I don\'t have NASA evidence for that."', "meta", "m"))
    assert a.status == "refused" and a.text == REFUSAL


def test_long_question_is_truncated(retriever):
    seen = {}

    def spy(messages, **k):
        seen["q"] = messages[1]["content"]
        return LLMResult(False, UNAVAILABLE_MESSAGE)
    answer("What are cool flames? " + "cool flames " * 600, retriever, chat_fn=spy)
    assert len(seen["q"]) < 6000


def test_no_secrets_means_unavailable_not_crash(retriever, monkeypatch):
    for v in ("MODEL_API_KEY", "META_API_KEY", "META_BASE_URL", "META_MODEL", "NVIDIA_API_KEY",
              "NVIDIA_BASE_URL", "NVIDIA_MODEL"):
        monkeypatch.delenv(v, raising=False)
    a = answer("Is a Moon base at 34% O2 safe from fire?", retriever)
    assert a.status == "unavailable" and a.hits


def test_literature_rows_have_verified_flag_and_url():
    lit = build_literature()
    rows = lit[lit["kind"] == "literature"]
    assert len(rows) >= 6
    assert rows["value_verified"].all()
    assert rows["url"].str.match(r"https://").all()


def test_app_starts_without_secrets(monkeypatch):
    from streamlit.testing.v1 import AppTest

    for v in ("MODEL_API_KEY", "META_API_KEY", "NVIDIA_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["Catalog & Ranking", "Flame Predictor", "Oxygen-Margin Map",
                                          "Ask Fire-Margin", "Lessons & Earth Impact"]
    assert any("No AI key is configured" in i.value for i in at.info)
