"""Tests for the memo regeneration endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from app.backend import main
from app.backend.config import Settings
from app.backend.main import get_settings


@pytest.fixture(autouse=True)
def override_settings(monkeypatch, tmp_path):
    """Provide deterministic settings + env for each test."""

    def _override() -> Settings:
        return Settings(
            POLICY_AGENT_MODEL="gpt-test",
            POLICY_AGENT_CACHE_PATH=str(tmp_path / "policy_cache.sqlite"),
            POLICY_AGENT_USE_CACHED=True,
            POLICY_AGENT_TEMPERATURE=0.1,
            OPENAI_API_KEY="fake-key",
            LANGSMITH_PROJECT=None,
        )

    main.app.dependency_overrides[get_settings] = _override
    yield
    main.app.dependency_overrides.pop(get_settings, None)


@pytest.fixture()
def client():
    return TestClient(main.app)


def test_resubmit_requires_summary_content(client: TestClient) -> None:
    response = client.post(
        "/rollouts/resubmit",
        json={
            "run_id": "run-1",
            "question": "Where should FEMA focus resilience dollars?",
            "summary": {
                "top_arguments": [],
                "top_articles": [],
                "top_people": [],
            },
            "notes": [],
            "directives": "",
        },
    )
    assert response.status_code == 400
    assert "Provide at least one top argument" in response.json()["detail"]


def test_resubmit_returns_memo(monkeypatch, client: TestClient) -> None:
    def fake_renderer(question, summary, notes, directives, model_name, temperature, api_key, tool_events, prior_memo):
        assert question.startswith("Where should FEMA")
        assert summary.top_arguments == ["Argument A"]
        assert summary.top_articles[0].paperId == "paper-1"
        assert api_key == "fake-key"
        assert model_name == "gpt-test"
        assert tool_events[0]["tool"] == "search_semantic_scholar"
        assert prior_memo == "Older memo"
        return "Draft memo body"

    monkeypatch.setattr(main, "_render_memo_from_summary", fake_renderer)

    response = client.post(
        "/rollouts/resubmit",
        json={
            "run_id": "run-1",
            "question": "Where should FEMA focus resilience dollars?",
            "summary": {
                "top_arguments": ["Argument A"],
                "top_articles": [
                    {"paper_id": "paper-1", "title": "Paper One", "authors": ["Doe"], "reason_chosen": "Great stats"}
                ],
                "top_people": ["Jane Doe"],
            },
            "notes": ["Remember to highlight coastal impacts."],
            "directives": "Keep it short.",
            "tool_events": [
                {"type": "tool_result", "tool": "search_semantic_scholar", "step": 2, "result": {"query": "fema"}}],
            "prior_memo": "Older memo",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"memo": "Draft memo body"}
