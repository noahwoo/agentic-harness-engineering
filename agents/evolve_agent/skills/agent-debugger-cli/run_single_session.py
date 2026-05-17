#!/usr/bin/env python3
"""Run a single adb check session with full LLM API call interception.

Patches the HTTP client to capture every LLM request/response, revealing
the complete agent loop: LLM calls, tool_calls decisions, tool results.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ["AHE_HOME"] = "/mnt/cfs_bj_mt/workspace/jianmin/git/fork/agentic-harness-engineering"

from agent_debugger_core.runtime.bootstrap import ensure_tools_importable
ensure_tools_importable()

from agent_debugger_core.cli.llm_resolver import resolve_llm_settings
from agent_debugger_core.runtime.runner import (
    AGENT_CONFIG_PATH,
    _build_user_message,
    _parse_run_output,
    BudgetExceeded,
)
from nexau import Agent, AgentConfig

LLM_LOG = []
SEP = "=" * 70


def intercept_openai_create(original_create):
    """Wrap the OpenAI chat.completions.create to log requests/responses."""
    def wrapper(*args, **kwargs):
        call_num = len(LLM_LOG) + 1
        t0 = time.time()

        # Extract request info
        model = kwargs.get("model", "?")
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools", [])
        tool_names = [t.get("function", {}).get("name", "?") for t in tools] if tools else []

        print(f"\n  ┌─ LLM CALL #{call_num} (model={model})")
        print(f"  │  Messages: {len(messages)}")
        # Show the last few messages (most relevant)
        for m in messages[-3:]:
            role = m.get("role", "?")
            content = str(m.get("content", ""))[:200]
            tc = m.get("tool_calls", [])
            tcid = m.get("tool_call_id", "")
            if tc:
                tc_names = [c.get("function", {}).get("name", "?") for c in tc]
                print(f"  │  ... [{role}] → tool_calls: {tc_names}")
            elif tcid:
                print(f"  │  ... [{role}] (tool_call_id={tcid[:20]}) {content[:100]}")
            else:
                print(f"  │  ... [{role}] {content[:150]}")
        if tool_names:
            print(f"  │  Available tools: {tool_names}")

        # Call original
        response = original_create(*args, **kwargs)
        elapsed = time.time() - t0

        # Parse response
        choice = response.choices[0] if response.choices else None
        if choice:
            msg = choice.message
            finish = choice.finish_reason
            tc = msg.tool_calls if hasattr(msg, "tool_calls") and msg.tool_calls else []
            content = msg.content or ""

            if tc:
                print(f"  │")
                print(f"  │  Response: TOOL CALLS ({finish})")
                for c in tc:
                    fn = c.function
                    args_str = fn.arguments[:200] if fn.arguments else ""
                    print(f"  │    → {fn.name}({args_str})")
            else:
                print(f"  │")
                print(f"  │  Response: TEXT ({finish}, {len(content)} chars)")
                print(f"  │    {content[:300]}")
                if len(content) > 300:
                    print(f"  │    ... [{len(content)} total]")

        usage = response.usage
        if usage:
            print(f"  │  Tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
        print(f"  └─ {elapsed:.1f}s")

        LLM_LOG.append({
            "call": call_num,
            "messages_count": len(messages),
            "has_tool_calls": bool(choice and hasattr(choice.message, "tool_calls") and choice.message.tool_calls),
            "finish_reason": choice.finish_reason if choice else "?",
            "content_len": len(content) if choice else 0,
            "elapsed": round(elapsed, 1),
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
        })

        return response
    return wrapper


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run a single adb session with full LLM call tracing"
    )
    parser.add_argument("trace", help="Path to .jsonl trace file")
    parser.add_argument("--mode", choices=["check", "ask"], default="check")
    parser.add_argument("--question", "-q", default="这条 trace 存在什么问题？")
    parser.add_argument("--max-iterations", "-n", type=int, default=None,
                        help="Max agent iterations (default: 25 from agent_config.yaml)")
    return parser.parse_args()


def main():
    args = parse_args()
    trace_path = Path(args.trace).resolve()
    mode = args.mode
    question = args.question
    max_iterations = args.max_iterations

    # Normalize trace
    from agent_debugger_core.trace_io import normalize_trace
    normalized_path, trace_id = normalize_trace(trace_path, trace_type="openai_messages")

    print(SEP)
    print(f"  ADB SINGLE SESSION — FULL LLM TRACE")
    print(SEP)
    print(f"  Input:      {trace_path.name}")
    print(f"  Normalized: {normalized_path}")
    print(f"  Trace ID:   {trace_id}")
    print(f"  Mode:       {mode}")
    if max_iterations is not None:
        print(f"  Max iter:   {max_iterations}")
    print(SEP)

    # Show normalized trace structure
    with open(normalized_path) as f:
        data = json.loads(f.read())
    msgs = data.get("messages", [])
    total_chars = sum(len(str(m.get("content", ""))) for m in msgs)
    print(f"\n[INPUT TRACE] {len(msgs)} messages, {total_chars:,} chars total")
    print("-" * 70)
    for i, m in enumerate(msgs):
        role = m.get("role", "?")
        content = str(m.get("content", ""))
        tc = m.get("tool_calls", [])
        tcid = m.get("tool_call_id", "")
        line = f"  [{i:3d}] {role:10s}"
        if tc:
            names = [c.get("function", {}).get("name", "?") for c in tc]
            line += f"  → tool_calls: {names}"
        elif tcid:
            line += f"  (tool_call_id={tcid[:24]})"
        line += f"  [{len(content):,} chars]"
        print(line)

    # Configure LLM
    llm = resolve_llm_settings()
    os.environ["LLM_MODEL"] = llm["model"]
    os.environ["LLM_BASE_URL"] = llm["base_url"]
    os.environ["LLM_API_KEY"] = llm["api_key"]
    os.environ.setdefault("LLM_API_TYPE", "openai_chat_completion")

    print(f"\n[LLM] model={llm['model']}, base_url={llm['base_url']}")

    # Build agent
    config = AgentConfig.from_yaml(config_path=AGENT_CONFIG_PATH)
    config.llm_config.model = llm["model"]
    config.llm_config.base_url = llm["base_url"]
    config.llm_config.api_key = llm["api_key"]
    if max_iterations is not None:
        config.max_iterations = max_iterations

    print(f"[AGENT] max_iterations={config.max_iterations}, tools={[t.name for t in config.tools]}")

    agent = Agent(config=config)

    # Patch OpenAI client to intercept LLM calls
    import openai
    original_create = openai.resources.chat.completions.Completions.create
    openai.resources.chat.completions.Completions.create = intercept_openai_create(original_create)
    print(f"[PATCH] Intercepted openai.chat.completions.create")

    # Build user message
    user_msg = _build_user_message([Path(normalized_path)], mode, question)
    print(f"\n[USER MESSAGE]")
    print("-" * 70)
    print(user_msg)
    print("-" * 70)

    # Run agent
    print(f"\n{SEP}")
    print(f"  AGENT LOOP START")
    print(SEP)

    t0 = time.time()
    try:
        run_output = agent.run(message=user_msg)
    except Exception as e:
        print(f"\n[AGENT EXCEPTION] {type(e).__name__}: {e}")
        run_output = str(e)
    elapsed = time.time() - t0

    # Restore original
    openai.resources.chat.completions.Completions.create = original_create

    print(f"\n{SEP}")
    print(f"  AGENT LOOP END — {elapsed:.1f}s, {len(LLM_LOG)} LLM calls")
    print(SEP)

    # LLM call summary
    if LLM_LOG:
        print(f"\n[LLM CALL SUMMARY]")
        print("-" * 70)
        total_prompt = 0
        total_completion = 0
        for entry in LLM_LOG:
            tc_flag = "→ TOOL_CALLS" if entry["has_tool_calls"] else f"→ TEXT ({entry['content_len']}c)"
            print(f"  #{entry['call']:2d}  msgs={entry['messages_count']:3d}  "
                  f"{entry['finish_reason']:12s}  {tc_flag:25s}  "
                  f"tokens={entry['prompt_tokens']}+{entry['completion_tokens']}  "
                  f"{entry['elapsed']}s")
            total_prompt += entry["prompt_tokens"]
            total_completion += entry["completion_tokens"]
        print(f"\n  Total: {len(LLM_LOG)} calls, "
              f"{total_prompt}+{total_completion}={total_prompt+total_completion} tokens, "
              f"{sum(e['elapsed'] for e in LLM_LOG):.1f}s LLM time")

    # Final output
    print(f"\n{SEP}")
    print(f"  FINAL OUTPUT")
    print(SEP)
    try:
        payload = _parse_run_output(run_output)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except BudgetExceeded as be:
        print(f"[BUDGET EXCEEDED]")
        print(be.fallback_text[:1000])
    except Exception as e:
        print(f"[PARSE ERROR] {e}")
        print(f"\nRaw output ({len(str(run_output))} chars):")
        print(str(run_output)[:2000])

    print(f"\n[Session complete] {elapsed:.1f}s, {len(LLM_LOG)} LLM calls")


if __name__ == "__main__":
    main()
