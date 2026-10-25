"""
Claude subscription proxy: an OpenAI-compatible /v1/chat/completions endpoint answered by the
official Claude Code CLI (`claude -p`) signed in with the owner's subscription token
(CLAUDE_CODE_OAUTH_TOKEN, from `claude setup-token`).

MiroFish's own LLM client and OASIS/CAMEL (the simulated people) speak the OpenAI API; this lets
both run on the Claude subscription. Unlike a text-only proxy, it supports OpenAI tool calls:
OASIS's people act (post, like, comment, follow) through tools, so the tools are described to
Claude and its JSON answer is returned as `tool_calls`.

Safety:
- Only the subscription token reaches the CLI (never ANTHROPIC_API_KEY or anything else of the
  container's environment), Claude gets no tools of its own, no MCP servers, no settings files,
  and each call runs in an empty temporary directory.
- Callers must send `Authorization: Bearer <PROXY_API_KEY>`; the proxy listens only on the
  internal network.
- At most PROXY_MAX_CONCURRENT CLI calls run at once (the subscription has usage limits); the
  rest wait up to PROXY_QUEUE_SECONDS.

Adapted from JayFarei/MiroFish's claude_openai_proxy.py (AGPL-3.0), with tool calls, auth and
the lockdown added. Standard library only.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("claude-proxy")

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
DEFAULT_MODEL = os.environ.get("PROXY_MODEL", "claude-sonnet-5-5")
TIMEOUT = float(os.environ.get("PROXY_TIMEOUT_SECONDS", "300"))
QUEUE_SECONDS = float(os.environ.get("PROXY_QUEUE_SECONDS", "900"))
MAX_BODY = 4 << 20
MAX_PROMPT = 400_000
# Claude model names the CLI accepts; anything else a client sends (e.g. "gpt-4o-mini", a
# MiroFish default) is answered by the default model.
CLAUDE_MODEL = re.compile(r"^(claude-[a-z0-9.-]+|opus|sonnet|haiku)$")

_slots = threading.BoundedSemaphore(max(1, min(int(os.environ.get("PROXY_MAX_CONCURRENT", "3")), 16)))
_stats = {"calls": 0, "failures": 0, "waiting": 0}
_stats_lock = threading.Lock()


class ProxyError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def child_env() -> dict[str, str]:
    """The CLI's environment: the subscription token and the bare minimum, nothing else."""
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    token_file = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN_FILE", "").strip()
    if not token and token_file:
        with open(token_file, encoding="utf-8") as f:
            token = f.read().strip()
    if not token:
        raise ProxyError(503, "CLAUDE_CODE_OAUTH_TOKEN is not set: run `claude setup-token` and set it")
    env = {k: os.environ[k] for k in ("HOME", "PATH", "LANG", "TMPDIR") if os.environ.get(k)}
    env.update(CLAUDE_CODE_OAUTH_TOKEN=token, NO_COLOR="1", DISABLE_AUTOUPDATER="1",
               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
    return env


def text_of(content) -> str:
    """OpenAI message content (a string or a list of parts) as text; images are noted, not sent."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            elif isinstance(p, dict):
                parts.append(f"[{p.get('type', 'part')} omitted]")
        return "\n".join(parts)
    return str(content)


def render(messages: list, tools: list | None, tool_choice, json_mode: bool) -> tuple[str, str]:
    """(system prompt, conversation prompt) for the CLI."""
    system, lines = [], []
    for m in messages:
        role = m.get("role", "user")
        if role in ("system", "developer"):
            system.append(text_of(m.get("content")))
        elif role == "tool":
            lines.append(f"Tool result (call {m.get('tool_call_id', '')}):\n{text_of(m.get('content'))}")
        elif role == "assistant":
            said = text_of(m.get("content"))
            calls = m.get("tool_calls") or []
            if calls:
                said = (said + "\n" if said else "") + "Called tools: " + json.dumps(
                    [{"id": c.get("id"), "name": (c.get("function") or {}).get("name"),
                      "arguments": (c.get("function") or {}).get("arguments")} for c in calls], ensure_ascii=False)
            lines.append(f"Assistant: {said}")
        else:
            lines.append(f"User: {text_of(m.get('content'))}")
    if tools:
        specs = []
        for t in tools:
            f = t.get("function") if isinstance(t, dict) else None
            if f:
                specs.append({"name": f.get("name"), "description": f.get("description", ""),
                              "parameters": f.get("parameters", {})})
        must = tool_choice == "required" or (isinstance(tool_choice, dict) and tool_choice.get("type") == "function")
        forced = (tool_choice.get("function") or {}).get("name") if isinstance(tool_choice, dict) else None
        rule = ("You MUST call " + (f"the function {forced!r}" if forced else "one or more functions") + ".") if must \
            else "Call functions when they are the right way to act; otherwise answer in text."
        system.append(
            "You can call these functions:\n" + json.dumps(specs, ensure_ascii=False) + "\n" + rule +
            '\nReply with ONLY one JSON object, no prose around it: {"tool_calls":[{"name":"<function>",'
            '"arguments":{...}}]} to call functions, or {"content":"<your answer>"} to answer in text.')
    elif json_mode:
        system.append("Reply with ONLY a valid JSON object, no prose and no code fences.")
    prompt = "\n\n".join(lines) + "\n\nAssistant:"
    if len(prompt) > MAX_PROMPT:
        raise ProxyError(413, "the conversation is too long")
    return "\n\n".join(s for s in system if s), prompt


JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def parse(text: str, tools: list | None) -> tuple[str | None, list]:
    """(content, tool_calls) from Claude's answer."""
    if not tools:
        return text, []
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    m = JSON_OBJECT.search(cleaned)
    if not m:
        return text, []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return text, []
    names = {(t.get("function") or {}).get("name") for t in tools if isinstance(t, dict)}
    calls = []
    for c in data.get("tool_calls") or []:
        if not isinstance(c, dict) or c.get("name") not in names:
            continue
        args = c.get("arguments", {})
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": c["name"],
                                   "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}})
    if calls:
        return None, calls
    content = data.get("content")
    return (content if isinstance(content, str) else text), []


def call_claude(system: str, prompt: str, model: str) -> str:
    argv = [CLAUDE_BIN, "-p", "--output-format", "json", "--model", model, "--tools", "",
            "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--setting-sources", "",
            "--no-session-persistence"]
    if system:
        argv += ["--system-prompt", system]
    with _stats_lock:
        _stats["waiting"] += 1
    got = _slots.acquire(timeout=QUEUE_SECONDS)
    with _stats_lock:
        _stats["waiting"] -= 1
    if not got:
        raise ProxyError(429, "too many Claude calls are waiting; try again later")
    try:
        env = child_env()
        with tempfile.TemporaryDirectory(prefix="claude-proxy-") as cwd:
            start = time.monotonic()
            try:
                out = subprocess.run(argv, input=prompt, capture_output=True, text=True, timeout=TIMEOUT,
                                     cwd=cwd, env=env)
            except subprocess.TimeoutExpired:
                raise ProxyError(504, "Claude took too long") from None
            except OSError as exc:
                raise ProxyError(503, f"the Claude CLI could not start: {exc}") from None
            log.info("claude -p model=%s %.1fs exit=%s prompt=%d", model, time.monotonic() - start, out.returncode, len(prompt))
        try:
            envelope = json.loads(out.stdout)
        except json.JSONDecodeError:
            envelope = None
        if not isinstance(envelope, dict):
            raise ProxyError(502, f"Claude CLI failed (exit {out.returncode})")
        text = envelope.get("result") or ""
        if envelope.get("is_error") or not text:
            reason = (text or "no answer")[:200]
            status = 401 if re.search(r"auth|login|oauth|api key|expired|credit", reason, re.I) else 502
            raise ProxyError(status, f"Claude: {reason}")
        return text
    finally:
        _slots.release()


def complete(body: dict) -> dict:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ProxyError(400, "messages is required")
    model = str(body.get("model") or "")
    model = model if CLAUDE_MODEL.match(model) else DEFAULT_MODEL
    tools = body.get("tools") or None
    fmt = body.get("response_format") or {}
    json_mode = isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema")
    system, prompt = render(messages, tools, body.get("tool_choice"), json_mode)
    with _stats_lock:
        _stats["calls"] += 1
    try:
        text = call_claude(system, prompt, model)
    except ProxyError:
        with _stats_lock:
            _stats["failures"] += 1
        raise
    content, calls = parse(text, tools)
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return {"id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion", "created": int(time.time()),
            "model": model, "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def stream_chunks(result: dict) -> bytes:
    """A finished completion as a server-sent event stream, for clients that ask to stream."""
    msg = result["choices"][0]["message"]
    delta = {"role": "assistant", "content": msg.get("content")}
    if msg.get("tool_calls"):
        delta["tool_calls"] = [dict(c, index=i) for i, c in enumerate(msg["tool_calls"])]
    chunk = {"id": result["id"], "object": "chat.completion.chunk", "created": result["created"], "model": result["model"],
             "choices": [{"index": 0, "delta": delta, "finish_reason": result["choices"][0]["finish_reason"]}]}
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    server_version = "claude-proxy"

    def log_message(self, fmt, *args):  # requests are logged by call_claude
        pass

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode())

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": {"message": message, "type": "server_error" if status >= 500 else "invalid_request_error"}})

    def _authorized(self) -> bool:
        want = os.environ.get("PROXY_API_KEY", "")
        got = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        return bool(want) and hmac.compare_digest(got.encode(), want.encode())

    def do_GET(self):
        if self.path == "/health":
            with _stats_lock:
                return self._json(200, {"status": "ok", **_stats})
        if not self._authorized():
            return self._error(401, "a valid Bearer key is required")
        if self.path.rstrip("/") == "/v1/models":
            return self._json(200, {"object": "list", "data": [{"id": DEFAULT_MODEL, "object": "model", "owned_by": "anthropic"}]})
        self._error(404, "not found")

    def do_POST(self):
        if not self._authorized():
            return self._error(401, "a valid Bearer key is required")
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            return self._error(404, "not found")
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            return self._error(413, "the request body is missing or too large")
        try:
            body = json.loads(self.rfile.read(length))
            result = complete(body)
        except json.JSONDecodeError:
            return self._error(400, "the body is not JSON")
        except ProxyError as exc:
            log.warning("request failed: %s", exc)
            return self._error(exc.status, str(exc))
        if body.get("stream"):
            return self._send(200, stream_chunks(result), "text/event-stream")
        self._json(200, result)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    if not os.environ.get("PROXY_API_KEY"):
        raise SystemExit("PROXY_API_KEY must be set (the key MiroFish sends as its LLM API key)")
    if not shutil.which(CLAUDE_BIN):
        raise SystemExit(f"the Claude CLI ({CLAUDE_BIN}) is not installed")
    host, port = os.environ.get("PROXY_HOST", "0.0.0.0"), int(os.environ.get("PROXY_PORT", "8082"))
    log.info("claude-proxy on %s:%s, model=%s", host, port, DEFAULT_MODEL)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
