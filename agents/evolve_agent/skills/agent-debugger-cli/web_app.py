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

TRACE_DIRS = [
    Path("/mnt/cfs_bj_mt/workspace/jianmin/datasets/cleaned-20250512"),
    Path("/mnt/cfs_bj_mt/workspace/jianmin/datasets/cleaned-20260518"),
]
TRACE_DIR = TRACE_DIRS[0]
BASE_DIR = Path(__file__).parent
QC_RESULTS_PATH = BASE_DIR / "debug_results" / "qc_results.jsonl"
ANALYSIS_RESULTS_PATH = BASE_DIR / "debug_results" / "analysis_results.jsonl"
ANNOTATIONS_PATH = BASE_DIR / "debug_results" / "annotations.jsonl"

app = FastAPI(title="Agent Trajectory Debugger")

# ── Startup data ─────────────────────────────────────────────────

TRACE_INDEX: dict[str, dict] = {}
QC_RESULTS: dict[str, dict] = {}
ANALYSIS_RESULTS: dict[str, dict] = {}
ANNOTATIONS: dict[str, dict] = {}
_analysis_lock = threading.Lock()
_annotations_lock = threading.Lock()

# Thread-safe interceptor registry for parallel analyses.
# Each analysis thread registers its interceptor keyed by thread ID.
# A single global wrapper dispatches to the correct interceptor based on
# the calling thread, avoiding cross-contamination between concurrent analyses.
_interceptor_registry: dict[int, callable] = {}
_interceptor_lock = threading.Lock()
_original_create = None  # stashed reference to the real openai create method
_original_anthropic_create = None  # stashed reference to anthropic messages.create


def _load_qc_results():
    if not QC_RESULTS_PATH.exists():
        return
    with open(QC_RESULTS_PATH) as f:
        for line in f:
            r = json.loads(line)
            QC_RESULTS[r["trace_id"]] = r


def _load_analysis_results():
    if not ANALYSIS_RESULTS_PATH.exists():
        return
    with open(ANALYSIS_RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ANALYSIS_RESULTS[r["trace_id"]] = r


def _load_annotations():
    if not ANNOTATIONS_PATH.exists():
        return
    with open(ANNOTATIONS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ANNOTATIONS[r["trace_id"]] = r


def _extract_env_os(msgs) -> str:
    """Extract OS platform from <env> block in system prompt."""
    import re
    for m in msgs:
        if m.get("role") == "system":
            content = str(m.get("content", ""))
            match = re.search(r"(?:操作系统平台|Platform):\s*(\S+)", content)
            if match:
                v = match.group(1).lower()
                if "win" in v:
                    return "win32"
                if "darwin" in v or "mac" in v:
                    return "mac"
                if "linux" in v:
                    wd = re.search(r"(?:工作目录|Working directory):\s*(\S+)", content)
                    if wd and wd.group(1).startswith("/Users/"):
                        return "mac"
                    return "linux"
                return v
            break
    return "unknown"


def _build_trace_index():
    for p in sorted(TRACE_DIR.glob("*.jsonl")):
        trace_id = p.stem
        try:
            with open(p) as f:
                data = json.loads(f.readline())
            msgs = data.get("messages", [])
            msg_count = len(msgs)
            total_chars = sum(len(str(m.get("content", ""))) for m in msgs)
            env_os = _extract_env_os(msgs)
        except Exception:
            msg_count = 0
            total_chars = 0
            env_os = "unknown"
        qc = QC_RESULTS.get(trace_id, {})
        ann = ANNOTATIONS.get(trace_id, {})
        tags = list(ann.get("tags", []))
        if env_os not in tags:
            tags.append(env_os)
        TRACE_INDEX[trace_id] = {
            "trace_id": trace_id,
            "path": str(p),
            "msg_count": msg_count,
            "total_chars": total_chars,
            "issues_count": qc.get("issues_count", -1),
            "status": qc.get("status", "unknown"),
            "has_analysis": trace_id in ANALYSIS_RESULTS,
            "tags": tags,
            "description": ann.get("description", ""),
            "env_os": env_os,
        }


@app.on_event("startup")
def startup():
    _load_qc_results()
    _load_analysis_results()
    _load_annotations()
    _build_trace_index()
    print(f"[web] Loaded {len(QC_RESULTS)} QC results, "
          f"{len(ANALYSIS_RESULTS)} analysis results, "
          f"{len(ANNOTATIONS)} annotations, {len(TRACE_INDEX)} traces")


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

@app.get("/api/datasets")
async def list_datasets():
    global TRACE_DIR
    return [{"path": str(d), "name": d.name, "active": d == TRACE_DIR} for d in TRACE_DIRS]


@app.post("/api/datasets/switch")
async def switch_dataset(request: Request):
    global TRACE_DIR
    body = await request.json()
    path = Path(body.get("path", ""))
    if path not in TRACE_DIRS:
        return JSONResponse({"error": "invalid dataset path"}, 400)
    TRACE_DIR = path
    TRACE_INDEX.clear()
    _build_trace_index()
    return {"active": str(TRACE_DIR), "trace_count": len(TRACE_INDEX)}

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


@app.get("/api/traces/{trace_id}/analysis")
async def get_analysis(trace_id: str):
    record = ANALYSIS_RESULTS.get(trace_id)
    if not record:
        return JSONResponse({"error": "no analysis result"}, 404)
    return record


@app.get("/api/traces/{trace_id}/annotation")
async def get_annotation(trace_id: str):
    ann = ANNOTATIONS.get(trace_id)
    if not ann:
        return JSONResponse({"error": "no annotation"}, 404)
    return ann


@app.put("/api/traces/{trace_id}/annotation")
async def put_annotation(request: Request, trace_id: str):
    body = await request.json()
    tags = body.get("tags", [])
    description = body.get("description", "")
    record = {"trace_id": trace_id, "tags": tags, "description": description}
    with _annotations_lock:
        ANNOTATIONS[trace_id] = record
        ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ANNOTATIONS_PATH, "w") as f:
            for r in ANNOTATIONS.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if trace_id in TRACE_INDEX:
            TRACE_INDEX[trace_id]["tags"] = tags
            TRACE_INDEX[trace_id]["description"] = description
    return record


# ── Live analysis with SSE ───────────────────────────────────────

def _dispatcher_create(*args, **kwargs):
    """Global replacement for openai Completions.create that routes to the
    correct per-thread interceptor, or falls through to the real method."""
    fn = None
    if _interceptor_registry:
        fn = next(iter(_interceptor_registry.values()))
    if fn:
        return fn("openai", *args, **kwargs)
    return _original_create(*args, **kwargs)


def _convert_tools_to_anthropic(kwargs):
    """Convert tools from OpenAI format to Anthropic format if needed."""
    tools = kwargs.get("tools")
    if tools and isinstance(tools, list) and tools and tools[0].get("type") == "function":
        kwargs["tools"] = [
            {"name": t["function"]["name"], "description": t["function"].get("description", ""), "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}})}
            for t in tools
        ]


def _dispatcher_anthropic_create(*args, **kwargs):
    """Global replacement for anthropic Messages.create."""
    # nexau may call from a worker thread different from the one that registered,
    # so pick any active interceptor (only one analysis runs at a time).
    fn = None
    if _interceptor_registry:
        fn = next(iter(_interceptor_registry.values()))
    if fn:
        return fn("anthropic", *args, **kwargs)
    kwargs.pop("tool_choice", None)
    _convert_tools_to_anthropic(kwargs)
    return _original_anthropic_create(*args, **kwargs)


def _install_interceptor(tid: int, fn, use_anthropic: bool = False):
    """Register a thread's interceptor and ensure the global dispatcher is active."""
    global _original_create, _original_anthropic_create
    import openai
    with _interceptor_lock:
        if _original_create is None:
            _original_create = openai.resources.chat.completions.Completions.create
            openai.resources.chat.completions.Completions.create = _dispatcher_create
        if use_anthropic and _original_anthropic_create is None:
            import anthropic
            _original_anthropic_create = anthropic.resources.messages.Messages.create
            anthropic.resources.messages.Messages.create = _dispatcher_anthropic_create
        _interceptor_registry[tid] = fn


def _uninstall_interceptor(tid: int):
    """Unregister a thread's interceptor; restore original when no threads remain."""
    global _original_anthropic_create
    import openai
    with _interceptor_lock:
        _interceptor_registry.pop(tid, None)
        if not _interceptor_registry and _original_create is not None:
            openai.resources.chat.completions.Completions.create = _original_create
        if not _interceptor_registry and _original_anthropic_create is not None:
            import anthropic
            anthropic.resources.messages.Messages.create = _original_anthropic_create
            _original_anthropic_create = None


def _run_analysis(trace_id: str, trace_path: str, mode: str, question: str,
                  max_iterations: int, queue: asyncio.Queue, loop,
                  use_claude: bool = False):
    """Run adb analysis in a thread, pushing events to the async queue."""
    from agent_debugger_core.runtime.bootstrap import ensure_tools_importable
    ensure_tools_importable()
    from agent_debugger_core.cli.llm_resolver import resolve_llm_settings
    from agent_debugger_core.runtime.runner import (
        AGENT_CONFIG_PATH, _build_user_message, _parse_run_output, BudgetExceeded,
    )
    from agent_debugger_core.trace_io import normalize_trace
    from nexau import Agent, AgentConfig

    collected_events = []

    def push(event):
        collected_events.append(event)
        loop.call_soon_threadsafe(queue.put_nowait, event)

    call_counter = [0]
    total_tokens = [0]

    def interceptor(api_type, *args, **kwargs):
        call_counter[0] += 1
        call_num = call_counter[0]
        t0 = time.time()

        messages = kwargs.get("messages", [])

        # Extract tool results that appeared since the last LLM call
        tool_results = []
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "tool":
                content = m.get("content", "")
                tool_results.append({
                    "tool_call_id": m.get("tool_call_id", ""),
                    "content": content if len(content) <= 2000
                              else content[:2000] + f"\n... [{len(content)} chars total]",
                })
            elif isinstance(m, dict) and m.get("role") == "assistant":
                break
        tool_results.reverse()

        push({"type": "llm_start", "call": call_num,
              "messages_count": len(messages),
              "tool_results": tool_results if tool_results else None})

        if api_type == "anthropic":
            kwargs.pop("tool_choice", None)
            _convert_tools_to_anthropic(kwargs)
            response = _original_anthropic_create(*args, **kwargs)
        else:
            response = _original_create(*args, **kwargs)
        elapsed = round(time.time() - t0, 1)

        event = {"type": "llm_end", "call": call_num, "elapsed": elapsed}

        if api_type == "anthropic":
            # Anthropic response format
            usage = response.usage
            event["finish_reason"] = response.stop_reason or "?"
            event["prompt_tokens"] = usage.input_tokens if usage else 0
            event["completion_tokens"] = usage.output_tokens if usage else 0
            # Check for tool use
            tool_use_blocks = [b for b in (response.content or []) if b.type == "tool_use"]
            if tool_use_blocks:
                tools = []
                for b in tool_use_blocks:
                    tools.append({"name": b.name, "arguments": json.dumps(b.input) if isinstance(b.input, dict) else str(b.input)})
                event["tool_calls"] = tools
            else:
                text_blocks = [b.text for b in (response.content or []) if b.type == "text"]
                content = "\n".join(text_blocks)
                event["content"] = content
                event["content_len"] = len(content)
            tok = (usage.input_tokens + usage.output_tokens) if usage else 0
            total_tokens[0] += tok
            event["total_tokens_so_far"] = total_tokens[0]
        else:
            # OpenAI response format
            choice = response.choices[0] if response.choices else None
            usage = response.usage
            event["finish_reason"] = choice.finish_reason if choice else "?"
            event["prompt_tokens"] = usage.prompt_tokens if usage else 0
            event["completion_tokens"] = usage.completion_tokens if usage else 0
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
        normalized_path, _norm_trace_id = normalize_trace(
            Path(trace_path), trace_type="openai_messages"
        )

        push({"type": "status", "message": "Configuring LLM..."})
        if use_claude:
            claude_cfg = _load_claude_settings()
            llm = {"model": claude_cfg["model"], "base_url": claude_cfg["base_url"].rstrip("/"),
                   "api_key": claude_cfg["api_key"]}
            os.environ["LLM_API_TYPE"] = "anthropic_chat_completion"
        else:
            llm = resolve_llm_settings()
            os.environ.setdefault("LLM_API_TYPE", "openai_chat_completion")
        os.environ["LLM_MODEL"] = llm["model"]
        os.environ["LLM_BASE_URL"] = llm["base_url"]
        os.environ["LLM_API_KEY"] = llm["api_key"]

        config = AgentConfig.from_yaml(config_path=AGENT_CONFIG_PATH)
        config.llm_config.model = llm["model"]
        config.llm_config.base_url = llm["base_url"]
        config.llm_config.api_key = llm["api_key"]
        if use_claude:
            config.llm_config.api_type = "anthropic_chat_completion"
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

        # Register this thread's interceptor and install global dispatcher
        tid = threading.current_thread().ident
        _install_interceptor(tid, interceptor, use_anthropic=use_claude)
        t0 = time.time()
        try:
            run_output = agent.run(message=user_msg)
        finally:
            _uninstall_interceptor(tid)

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
        _save_analysis(trace_id, collected_events)


def _save_analysis(trace_id: str, events: list):
    """Persist collected analysis events to disk."""
    from datetime import datetime, timezone
    complete_evt = next((e for e in events if e.get("type") == "complete"), None)
    if not complete_evt:
        return
    record = {
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed": complete_evt.get("elapsed"),
        "llm_calls": complete_evt.get("llm_calls"),
        "total_tokens": complete_evt.get("total_tokens"),
        "budget_exceeded": complete_evt.get("budget_exceeded", False),
        "result": complete_evt.get("result"),
        "events": events,
    }
    with _analysis_lock:
        ANALYSIS_RESULTS[trace_id] = record
        ANALYSIS_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Rewrite the file to handle updates for the same trace_id
        with open(ANALYSIS_RESULTS_PATH, "w") as f:
            for r in ANALYSIS_RESULTS.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if trace_id in TRACE_INDEX:
            TRACE_INDEX[trace_id]["has_analysis"] = True
    print(f"[web] Saved analysis for {trace_id}")


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
        args=(trace_id, info["path"], "check", "", max_iterations, queue, loop),
        kwargs={"use_claude": True},
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


def _load_claude_settings():
    """Load Claude API settings from ~/.claude/settings.json."""
    settings_path = Path.home() / ".claude" / "settings.json"
    if not settings_path.exists():
        raise RuntimeError("~/.claude/settings.json not found")
    data = json.loads(settings_path.read_text())
    env = data.get("env", {})
    return {
        "base_url": env.get("ANTHROPIC_BASE_URL", ""),
        "api_key": env.get("ANTHROPIC_AUTH_TOKEN", ""),
        "model": env.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "Claude Sonnet 4.6"),
    }


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
        args=(trace_id, info["path"], "ask", question, max_iterations, queue, loop),
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
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8899)
