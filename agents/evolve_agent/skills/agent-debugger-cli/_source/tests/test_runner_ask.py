import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_debugger_core.runtime.runner import run_agent, RunnerResult, RunnerError


def _wrap(payload):
    inner = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return json.dumps({
        "success": True,
        "message": "Result submitted and task completed.",
        "status": "TASK_COMPLETED",
        "task_completed": True,
        "output": {"result": inner},
    }, ensure_ascii=False)


def _scripted_agent(payload: dict):
    fake_agent = MagicMock()
    fake_agent.run.return_value = _wrap(payload)
    return fake_agent


def test_ask_mode_happy_path(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_BASE_URL", "u")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AHE_HOME", str(tmp_path))
    (tmp_path / "evolve_agent" / "tools").mkdir(parents=True)
    (tmp_path / "evolve_agent" / "tools" / "__init__.py").write_text("")

    fake_agent = _scripted_agent({"mode": "ask", "answer": "42"})

    with patch("agent_debugger_core.runtime.runner._build_agent", return_value=fake_agent):
        result = run_agent(
            trace_paths=[Path("/fake/trace.json")],
            mode="ask",
            question="why?",
        )

    assert isinstance(result, RunnerResult)
    assert result.mode == "ask"
    assert result.answer == "42"
    assert result.budget_exceeded is False


def test_ask_mode_accepts_response_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_BASE_URL", "u")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AHE_HOME", str(tmp_path))
    (tmp_path / "evolve_agent" / "tools").mkdir(parents=True)
    (tmp_path / "evolve_agent" / "tools" / "__init__.py").write_text("")

    fake_agent = _scripted_agent({"mode": "ask", "response": "because"})

    with patch("agent_debugger_core.runtime.runner._build_agent", return_value=fake_agent):
        result = run_agent(
            trace_paths=[Path("/fake/trace.json")],
            mode="ask",
            question="why?",
        )

    assert result.answer == "because"


def test_ask_mode_retries_invalid_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_BASE_URL", "u")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AHE_HOME", str(tmp_path))
    (tmp_path / "evolve_agent" / "tools").mkdir(parents=True)
    (tmp_path / "evolve_agent" / "tools" / "__init__.py").write_text("")

    fake_agent = MagicMock()
    fake_agent.run.side_effect = [
        _wrap({"mode": "ask", "foo": "bar"}),
        _wrap({"mode": "ask", "answer": "fixed"}),
    ]

    with patch("agent_debugger_core.runtime.runner._build_agent", return_value=fake_agent):
        result = run_agent(
            trace_paths=[Path("/fake/trace.json")],
            mode="ask",
            question="why?",
        )

    assert result.answer == "fixed"
    assert fake_agent.run.call_count == 2


def test_ask_mode_accepts_check_payload_response(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_BASE_URL", "u")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AHE_HOME", str(tmp_path))
    (tmp_path / "evolve_agent" / "tools").mkdir(parents=True)
    (tmp_path / "evolve_agent" / "tools" / "__init__.py").write_text("")

    fake_agent = _scripted_agent({"mode": "check", "issues": [], "response": "brief diagnosis"})

    with patch("agent_debugger_core.runtime.runner._build_agent", return_value=fake_agent):
        result = run_agent(
            trace_paths=[Path("/fake/trace.json")],
            mode="ask",
            question="why?",
        )

    assert result.answer == "brief diagnosis"


def test_ask_mode_accepts_check_payload_issues(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_BASE_URL", "u")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AHE_HOME", str(tmp_path))
    (tmp_path / "evolve_agent" / "tools").mkdir(parents=True)
    (tmp_path / "evolve_agent" / "tools" / "__init__.py").write_text("")

    fake_agent = _scripted_agent({
        "mode": "check",
        "issues": [
            {
                "summary": "Assistant ignored a failing test",
                "trace_id": "t1",
                "message_index": 15,
            }
        ],
    })

    with patch("agent_debugger_core.runtime.runner._build_agent", return_value=fake_agent):
        result = run_agent(
            trace_paths=[Path("/fake/trace.json")],
            mode="ask",
            question="why?",
        )

    assert "Assistant ignored a failing test" in result.answer
