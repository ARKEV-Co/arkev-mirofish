"""The proxy with a fake `claude` CLI: OpenAI shapes in and out, tool calls, the subscription
token alone in the CLI's environment, auth, and refusals."""
import json
import os
import stat
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import proxy  # noqa: E402

FAKE = """#!/usr/bin/env python3
import json, os, sys
prompt = sys.stdin.read()
argv = sys.argv
record = {"argv": argv, "env": sorted(os.environ), "prompt": prompt}
open(os.path.join(os.environ["HOME"], "calls.jsonl"), "a").write(json.dumps(record) + "\\n")
answer = open(os.path.join(os.environ["HOME"], "answer.txt")).read()
print(answer)
"""


@pytest.fixture
def cli(tmp_path, monkeypatch):
    fake = tmp_path / "claude"
    fake.write_text(FAKE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(proxy, "CLAUDE_BIN", str(fake))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dead-key")
    monkeypatch.setenv("PROXY_API_KEY", "internal")

    def answer(result=None, is_error=False):
        (tmp_path / "answer.txt").write_text(json.dumps({"result": result, "is_error": is_error}))

    def calls():
        path = tmp_path / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    return answer, calls


def test_plain_chat_is_an_openai_completion_on_the_subscription_only(cli):
    answer, calls = cli
    answer("Hello from Claude")
    out = proxy.complete({"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "Be brief."},
                                                               {"role": "user", "content": "hi"}]})
    assert out["object"] == "chat.completion" and out["choices"][0]["message"]["content"] == "Hello from Claude"
    assert out["model"] == proxy.DEFAULT_MODEL  # a non-Claude model name gets the default
    call = calls()[0]
    assert "CLAUDE_CODE_OAUTH_TOKEN" in call["env"] and "ANTHROPIC_API_KEY" not in call["env"]
    argv = call["argv"]
    assert argv[argv.index("--tools") + 1] == "" and "--strict-mcp-config" in argv
    assert argv[argv.index("--system-prompt") + 1] == "Be brief." and call["prompt"].endswith("User: hi\n\nAssistant:")


TOOLS = [{"type": "function", "function": {"name": "create_post", "description": "Post on the feed",
                                            "parameters": {"type": "object", "properties": {"content": {"type": "string"}}}}}]


def test_tools_become_tool_calls(cli):
    answer, calls = cli
    answer('```json\n{"tool_calls":[{"name":"create_post","arguments":{"content":"Love it"}},{"name":"rm_rf","arguments":{}}]}\n```')
    out = proxy.complete({"messages": [{"role": "user", "content": "react to the idea"}], "tools": TOOLS, "tool_choice": "required"})
    choice = out["choices"][0]
    assert choice["finish_reason"] == "tool_calls" and choice["message"]["content"] is None
    (tc,) = choice["message"]["tool_calls"]  # an unknown function is dropped
    assert tc["function"]["name"] == "create_post" and json.loads(tc["function"]["arguments"]) == {"content": "Love it"}
    system = calls()[0]["argv"][calls()[0]["argv"].index("--system-prompt") + 1]
    assert "create_post" in system and "You MUST call" in system


def test_a_text_answer_with_tools_and_the_tool_round_trip(cli):
    answer, calls = cli
    answer('{"content":"Nothing to do."}')
    out = proxy.complete({"messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
                                                              "function": {"name": "create_post", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "posted"}], "tools": TOOLS})
    assert out["choices"][0]["message"]["content"] == "Nothing to do." and out["choices"][0]["finish_reason"] == "stop"
    prompt = calls()[0]["prompt"]
    assert "Called tools:" in prompt and "Tool result (call call_1):\nposted" in prompt


def test_json_mode_and_refusals(cli, monkeypatch):
    answer, calls = cli
    answer('{"a": 1}')
    proxy.complete({"messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_object"}})
    assert "ONLY a valid JSON object" in calls()[0]["argv"][calls()[0]["argv"].index("--system-prompt") + 1]
    answer("Invalid API key · Fix external API key", is_error=True)
    with pytest.raises(proxy.ProxyError) as exc:
        proxy.complete({"messages": [{"role": "user", "content": "x"}]})
    assert exc.value.status == 401
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN")
    with pytest.raises(proxy.ProxyError) as exc:
        proxy.complete({"messages": [{"role": "user", "content": "x"}]})
    assert exc.value.status == 503 and "setup-token" in str(exc.value)
    with pytest.raises(proxy.ProxyError):
        proxy.complete({"messages": []})


def test_http_needs_the_key_and_streams_when_asked(cli):
    answer, _ = cli
    answer("streamed")
    server = ThreadingHTTPServer(("127.0.0.1", 0), proxy.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True}).encode()

    def post(key):
        req = urllib.request.Request(base + "/v1/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        return urllib.request.urlopen(req, timeout=10)

    with pytest.raises(urllib.error.HTTPError) as exc:
        post("wrong")
    assert exc.value.code == 401
    text = post("internal").read().decode()
    assert '"content": "streamed"' in text and text.endswith("data: [DONE]\n\n")
    assert json.loads(urllib.request.urlopen(base + "/health", timeout=10).read())["status"] == "ok"
    server.shutdown()


def test_the_key_can_come_from_a_file(tmp_path, monkeypatch):
    monkeypatch.delenv("PROXY_API_KEY", raising=False)
    (tmp_path / "proxy_key").write_text("from-file\n")
    monkeypatch.setenv("PROXY_API_KEY_FILE", str(tmp_path / "proxy_key"))
    assert proxy.proxy_key() == "from-file"
