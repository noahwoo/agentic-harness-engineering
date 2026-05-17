#!/usr/bin/env python3
"""Web app for browsing and analyzing agent debug trajectories."""

import asyncio
import json
import os
import time
import threading
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

os.environ["AHE_HOME"] = os.environ.get(
    "AHE_HOME",
    "/mnt/cfs_bj_mt/workspace/jianmin/git/fork/agentic-harness-engineering",
)

TRACE_DIR = Path("/mnt/cfs_bj_mt/workspace/jianmin/datasets/cleaned-20250512")
BASE_DIR = Path(__file__).parent
QC_RESULTS_PATH = BASE_DIR / "debug_results" / "qc_results.jsonl"

app = FastAPI(title="Agent Trajectory Debugger")

# ── Startup data ─────────────────────────────────────────────────

TRACE_INDEX: dict[str, dict] = {}
QC_RESULTS: dict[str, dict] = {}


def _load_qc_results():
    if not QC_RESULTS_PATH.exists():
        return
    with open(QC_RESULTS_PATH) as f:
        for line in f:
            r = json.loads(line)
            QC_RESULTS[r["trace_id"]] = r


def _build_trace_index():
    for p in sorted(TRACE_DIR.glob("*.jsonl")):
        trace_id = p.stem
        try:
            with open(p) as f:
                data = json.loads(f.readline())
            msgs = data.get("messages", [])
            msg_count = len(msgs)
            total_chars = sum(len(str(m.get("content", ""))) for m in msgs)
        except Exception:
            msg_count = 0
            total_chars = 0
        qc = QC_RESULTS.get(trace_id, {})
        TRACE_INDEX[trace_id] = {
            "trace_id": trace_id,
            "path": str(p),
            "msg_count": msg_count,
            "total_chars": total_chars,
            "issues_count": qc.get("issues_count", -1),
            "status": qc.get("status", "unknown"),
        }


@app.on_event("startup")
def startup():
    _load_qc_results()
    _build_trace_index()
    print(f"[web] Loaded {len(QC_RESULTS)} QC results, {len(TRACE_INDEX)} traces")


# ── Static files ─────────────────────────────────────────────────

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(
        html_path.read_text(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ── API endpoints ────────────────────────────────────────────────

@app.get("/api/traces")
async def list_traces(
    search: str = Query("", description="Filter by trace_id substring"),
    issue_type: str = Query("", description="Filter by issue type"),
    min_issues: int = Query(-1),
    max_issues: int = Query(999),
):
    results = []
    for t in TRACE_INDEX.values():
        if search and search not in t["trace_id"]:
            continue
        ic = t["issues_count"]
        if ic < min_issues or ic > max_issues:
            continue
        if issue_type:
            qc = QC_RESULTS.get(t["trace_id"], {})
            types = {i.get("issue_type", "") for i in qc.get("issues", [])}
            if issue_type not in types:
                continue
        results.append(t)
    return results


@app.get("/api/traces/{trace_id}")
async def get_trace(trace_id: str):
    info = TRACE_INDEX.get(trace_id)
    if not info:
        return JSONResponse({"error": "not found"}, 404)
    with open(info["path"]) as f:
        data = json.loads(f.readline())
    msgs = data.get("messages", [])
    processed = []
    for i, m in enumerate(msgs):
        role = m.get("role", "unknown")
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        tc = m.get("tool_calls", [])
        tcid = m.get("tool_call_id", "")
        processed.append({
            "index": i,
            "role": role,
            "content": content or "",
            "content_len": len(str(content or "")),
            "tool_calls": tc,
            "tool_call_id": tcid,
        })
    return {"trace_id": trace_id, "msg_count": len(msgs), "messages": processed}


@app.get("/api/traces/{trace_id}/raw")
async def get_trace_raw(trace_id: str):
    info = TRACE_INDEX.get(trace_id)
    if not info:
        return JSONResponse({"error": "not found"}, 404)
    with open(info["path"]) as f:
        data = json.loads(f.readline())
    return data


@app.get("/api/traces/{trace_id}/qc")
async def get_qc(trace_id: str):
    qc = QC_RESULTS.get(trace_id)
    if not qc:
        return JSONResponse({"error": "no QC result"}, 404)
    return qc


# ── Live analysis with SSE ───────────────────────────────────────

def _run_analysis(trace_path: str, mode: str, question: str,
                  max_iterations: int, queue: asyncio.Queue, loop):
    """Run adb analysis in a thread, pushing events to the async queue."""
    import openai
    from agent_debugger_core.runtime.bootstrap import ensure_tools_importable
    ensure_tools_importable()
    from agent_debugger_core.cli.llm_resolver import resolve_llm_settings
    from agent_debugger_core.runtime.runner import (
        AGENT_CONFIG_PATH, _build_user_message, _parse_run_output, BudgetExceeded,
    )
    from agent_debugger_core.trace_io import normalize_trace
    from nexau import Agent, AgentConfig

    def push(event):
        loop.call_soon_threadsafe(queue.put_nowait, event)

    call_counter = [0]
    total_tokens = [0]

    original_create = openai.resources.chat.completions.Completions.create

    def interceptor(*args, **kwargs):
        call_counter[0] += 1
        call_num = call_counter[0]
        t0 = time.time()

        messages = kwargs.get("messages", [])

        # Extract tool results that appeared since the last LLM call
        # (these are the messages appended after the previous assistant turn)
        tool_results = []
        for m in reversed(messages):
            if m.get("role") == "tool":
                content = m.get("content", "")
                tool_results.append({
                    "tool_call_id": m.get("tool_call_id", ""),
                    "content": content if len(content) <= 2000
                              else content[:2000] + f"\n... [{len(content)} chars total]",
                })
            elif m.get("role") == "assistant":
                break
        tool_results.reverse()

        push({"type": "llm_start", "call": call_num,
              "messages_count": len(messages),
              "tool_results": tool_results if tool_results else None})

        response = original_create(*args, **kwargs)
        elapsed = round(time.time() - t0, 1)

        choice = response.choices[0] if response.choices else None
        usage = response.usage
        event = {
            "type": "llm_end",
            "call": call_num,
            "elapsed": elapsed,
            "finish_reason": choice.finish_reason if choice else "?",
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
        }

        if choice:
            msg = choice.message
            tc = msg.tool_calls if hasattr(msg, "tool_calls") and msg.tool_calls else []
            if tc:
                tools = []
                for c in tc:
                    fn = c.function
                    tools.append({"name": fn.name, "arguments": fn.arguments or ""})
                event["tool_calls"] = tools
            else:
                content = msg.content or ""
                event["content"] = content
                event["content_len"] = len(content)

        if usage:
            total_tokens[0] += usage.total_tokens
            event["total_tokens_so_far"] = total_tokens[0]

        push(event)
        return response

    try:
        push({"type": "status", "message": "Normalizing trace..."})
        normalized_path, trace_id = normalize_trace(
            Path(trace_path), trace_type="openai_messages"
        )

        push({"type": "status", "message": "Configuring LLM..."})
        llm = resolve_llm_settings()
        os.environ["LLM_MODEL"] = llm["model"]
        os.environ["LLM_BASE_URL"] = llm["base_url"]
        os.environ["LLM_API_KEY"] = llm["api_key"]
        os.environ.setdefault("LLM_API_TYPE", "openai_chat_completion")

        config = AgentConfig.from_yaml(config_path=AGENT_CONFIG_PATH)
        config.llm_config.model = llm["model"]
        config.llm_config.base_url = llm["base_url"]
        config.llm_config.api_key = llm["api_key"]
        if max_iterations > 0:
            config.max_iterations = max_iterations

        # Render system prompt with actual max_iterations so the agent
        # knows its real budget.
        prompt_path = AGENT_CONFIG_PATH.parent / "system_prompt.md"
        from jinja2 import Template
        rendered_prompt = Template(prompt_path.read_text()).render(
            max_iterations=config.max_iterations,
        )
        config.system_prompt = rendered_prompt
        config.system_prompt_type = "string"

        push({"type": "config", "model": llm["model"],
              "max_iterations": config.max_iterations})

        agent = Agent(config=config)
        user_msg = _build_user_message(
            [Path(normalized_path)], mode, question if mode == "ask" else None
        )

        push({"type": "status", "message": f"Running agent ({mode})..."})

        openai.resources.chat.completions.Completions.create = interceptor
        t0 = time.time()
        try:
            run_output = agent.run(message=user_msg)
        finally:
            openai.resources.chat.completions.Completions.create = original_create

        elapsed = round(time.time() - t0, 1)

        try:
            payload = _parse_run_output(run_output)
            push({"type": "complete", "elapsed": elapsed,
                  "llm_calls": call_counter[0],
                  "total_tokens": total_tokens[0], "result": payload})
        except BudgetExceeded as be:
            fallback = be.fallback_text or ""
            # Try to extract partial JSON findings from the fallback text
            result = {"response": fallback}
            try:
                start = fallback.find("{")
                if start >= 0:
                    import re
                    m = re.search(r"\{.*\}", fallback, flags=re.DOTALL)
                    if m:
                        parsed = json.loads(m.group(0))
                        if isinstance(parsed, dict):
                            result = parsed
            except (json.JSONDecodeError, Exception):
                pass
            push({"type": "complete", "elapsed": elapsed,
                  "llm_calls": call_counter[0],
                  "total_tokens": total_tokens[0],
                  "budget_exceeded": True,
                  "result": result})
        except Exception as e:
            push({"type": "error", "message": f"Parse error: {e}",
                  "raw": str(run_output)[:500]})
    except Exception as e:
        push({"type": "error", "message": str(e)})
    finally:
        push({"type": "done"})


@app.get("/api/traces/{trace_id}/analyze")
async def analyze_trace(
    request: Request,
    trace_id: str,
    max_iterations: int = Query(25),
):
    info = TRACE_INDEX.get(trace_id)
    if not info:
        return JSONResponse({"error": "not found"}, 404)

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    thread = threading.Thread(
        target=_run_analysis,
        args=(info["path"], "check", "", max_iterations, queue, loop),
        daemon=True,
    )
    thread.start()

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": "{}"}
                continue
            yield {"event": "message", "data": json.dumps(event, ensure_ascii=False)}
            if event.get("type") in ("done", "error"):
                break

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/traces/{trace_id}/ask")
async def ask_trace(request: Request, trace_id: str):
    info = TRACE_INDEX.get(trace_id)
    if not info:
        return JSONResponse({"error": "not found"}, 404)

    body = await request.json()
    question = body.get("question", "这条 trace 存在什么问题？")
    max_iterations = body.get("max_iterations", 25)

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    thread = threading.Thread(
        target=_run_analysis,
        args=(info["path"], "ask", question, max_iterations, queue, loop),
        daemon=True,
    )
    thread.start()

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": "{}"}
                continue
            yield {"event": "message", "data": json.dumps(event, ensure_ascii=False)}
            if event.get("type") in ("done", "error"):
                break

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8899)
