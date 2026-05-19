"""Unit test to verify Claude model service can be invoked correctly via Anthropic API."""
import json
from pathlib import Path

import anthropic


def load_settings():
    data = json.loads((Path.home() / ".claude" / "settings.json").read_text())
    env = data.get("env", {})
    return {
        "base_url": env.get("ANTHROPIC_BASE_URL", ""),
        "api_key": env.get("ANTHROPIC_AUTH_TOKEN", ""),
        "model": env.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "Claude Sonnet 4.6"),
    }


def convert_tools_to_anthropic(tools):
    """Convert OpenAI-format tools to Anthropic format."""
    return [
        {"name": t["function"]["name"], "description": t["function"].get("description", ""),
         "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}})}
        for t in tools
    ]


SAMPLE_TOOLS_OPENAI = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from disk",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    }},
    {"type": "function", "function": {
        "name": "search_file_content",
        "description": "Search for content in files",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string"}}, "required": ["query"]}
    }},
]


def test_tools_format_conversion():
    """Test that OpenAI tools are correctly converted to Anthropic format."""
    converted = convert_tools_to_anthropic(SAMPLE_TOOLS_OPENAI)
    assert len(converted) == 2
    assert converted[0]["name"] == "read_file"
    assert "input_schema" in converted[0]
    assert converted[0]["input_schema"]["type"] == "object"
    assert "type" not in converted[0] or converted[0].get("type") == "custom"
    print("PASS: tools format conversion")


def test_claude_basic_call():
    """Test basic Claude API call without tools."""
    settings = load_settings()
    client = anthropic.Anthropic(base_url=settings["base_url"], api_key=settings["api_key"])
    response = client.messages.create(
        model=settings["model"],
        max_tokens=100,
        messages=[{"role": "user", "content": "Reply with exactly: hello"}],
    )
    assert response.content and len(response.content) > 0
    assert response.content[0].type == "text"
    print(f"PASS: basic call, response: {response.content[0].text[:50]}")


def test_claude_call_with_tools():
    """Test Claude API call with tools in Anthropic format."""
    settings = load_settings()
    client = anthropic.Anthropic(base_url=settings["base_url"], api_key=settings["api_key"])
    tools = convert_tools_to_anthropic(SAMPLE_TOOLS_OPENAI)
    response = client.messages.create(
        model=settings["model"],
        max_tokens=200,
        messages=[{"role": "user", "content": "Read the file at /tmp/test.txt"}],
        tools=tools,
    )
    assert response.content and len(response.content) > 0
    has_tool_use = any(b.type == "tool_use" for b in response.content)
    has_text = any(b.type == "text" for b in response.content)
    assert has_tool_use or has_text, f"Unexpected content types: {[b.type for b in response.content]}"
    if has_tool_use:
        tool_block = next(b for b in response.content if b.type == "tool_use")
        assert tool_block.name == "read_file"
        assert isinstance(tool_block.input, dict)
        print(f"PASS: tool call, name={tool_block.name}, input={tool_block.input}")
    else:
        print(f"PASS: text response (model chose not to use tool): {response.content[0].text[:80]}")


def test_claude_tool_result_roundtrip():
    """Test a full tool-use roundtrip: call -> tool_result -> final response."""
    settings = load_settings()
    client = anthropic.Anthropic(base_url=settings["base_url"], api_key=settings["api_key"])
    tools = convert_tools_to_anthropic(SAMPLE_TOOLS_OPENAI)

    # First call - should request tool use
    resp1 = client.messages.create(
        model=settings["model"],
        max_tokens=200,
        messages=[{"role": "user", "content": "Read the file at /tmp/test.txt"}],
        tools=tools,
    )

    tool_blocks = [b for b in resp1.content if b.type == "tool_use"]
    if not tool_blocks:
        print("SKIP: model did not use tool, cannot test roundtrip")
        return

    tool_block = tool_blocks[0]

    # Serialize assistant content blocks properly for Bedrock
    assistant_content = []
    for b in resp1.content:
        if b.type == "text":
            assistant_content.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            assistant_content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})

    # Second call - provide tool result
    resp2 = client.messages.create(
        model=settings["model"],
        max_tokens=200,
        messages=[
            {"role": "user", "content": "Read the file at /tmp/test.txt"},
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_block.id, "content": "hello world"}
            ]},
        ],
        tools=tools,
    )
    assert resp2.content and len(resp2.content) > 0
    print(f"PASS: roundtrip, stop_reason={resp2.stop_reason}, content_types={[b.type for b in resp2.content]}")


def test_claude_nexau_agent():
    """Test that nexau Agent can complete a simple task using Claude model."""
    import os
    import sys
    sys.path.insert(0, "/mnt/cfs_bj_mt/workspace/jianmin/git/fork/agentic-harness-engineering/agents/evolve_agent/skills/agent-debugger-cli")
    sys.path.insert(0, "/mnt/cfs_bj_mt/workspace/jianmin/git/fork/agentic-harness-engineering/agents/evolve_agent/skills/agent-debugger-cli/_source")
    os.environ["AHE_HOME"] = "/mnt/cfs_bj_mt/workspace/jianmin/git/fork/agentic-harness-engineering"

    from agent_debugger_core.runtime.bootstrap import ensure_tools_importable
    ensure_tools_importable()
    from agent_debugger_core.runtime.runner import _parse_run_output
    from nexau import Agent, AgentConfig

    settings = load_settings()
    config_path = Path("/mnt/cfs_bj_mt/workspace/jianmin/git/fork/agentic-harness-engineering/agents/evolve_agent/skills/agent-debugger-cli/_source/agent_debugger_core/runtime/agent_config.yaml")

    os.environ["LLM_MODEL"] = settings["model"]
    os.environ["LLM_BASE_URL"] = settings["base_url"]
    os.environ["LLM_API_KEY"] = settings["api_key"]
    os.environ["LLM_API_TYPE"] = "anthropic_chat_completion"

    config = AgentConfig.from_yaml(config_path=config_path)
    config.llm_config.model = settings["model"]
    config.llm_config.base_url = settings["base_url"]
    config.llm_config.api_key = settings["api_key"]
    config.llm_config.api_type = "anthropic_chat_completion"
    config.max_iterations = 3
    config.system_prompt = "You are a test agent. Complete the task immediately by calling complete_task with the result."
    config.system_prompt_type = "string"

    agent = Agent(config=config)

    # Install interceptor to strip tool_choice and convert tools format (same as web_app)
    import anthropic as anthropic_mod
    _original = anthropic_mod.resources.messages.Messages.create

    def patched_create(*args, **kwargs):
        kwargs.pop("tool_choice", None)
        tools = kwargs.get("tools")
        if tools and isinstance(tools, list) and tools and isinstance(tools[0], dict) and tools[0].get("type") == "function":
            kwargs["tools"] = [
                {"name": t["function"]["name"], "description": t["function"].get("description", ""),
                 "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}})}
                for t in tools
            ]
        return _original(*args, **kwargs)

    anthropic_mod.resources.messages.Messages.create = patched_create
    try:
        run_output = agent.run(message='Return this JSON via complete_task: {"mode": "test", "response": "hello"}')
    finally:
        anthropic_mod.resources.messages.Messages.create = _original

    print(f"Raw agent output (first 500 chars): {str(run_output)[:500]}")

    try:
        payload = _parse_run_output(run_output)
        print(f"PASS: nexau agent with Claude, parsed payload: {payload}")
    except Exception as e:
        print(f"FAIL: nexau agent with Claude, parse error: {e}")
        print(f"  raw output: {str(run_output)[:800]}")
        raise


if __name__ == "__main__":
    test_tools_format_conversion()
    test_claude_basic_call()
    test_claude_call_with_tools()
    test_claude_tool_result_roundtrip()
    test_claude_nexau_agent()
    print("\nAll tests passed!")
