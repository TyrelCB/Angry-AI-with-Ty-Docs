#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOME_DIR="${HOME:-/home/tyrel}"

UPSTREAM_URL="http://192.168.6.181:9090/v1"
SEARXNG_URL="http://127.0.0.1:18910"
MODEL_NAME="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
ADAPTER_HOST="127.0.0.1"
ADAPTER_PORT="4141"
ENABLE_CODEX=0
ENABLE_WEB_MCP=0
ENABLE_OPENCODE=0

usage() {
  cat <<'EOF'
Usage:
  setup-dramatic-local-llm-demo.sh [--all] [--codex] [--web-mcp] [--opencode]
                                   [--upstream-url URL] [--searxng-url URL]
                                   [--model-name NAME] [--adapter-port PORT]

Defaults:
  --all               Set up everything from the session summary.
  --codex             Install the Codex llama.cpp adapter and Codex config.
  --web-mcp           Install the local_web MCP server and register it in Codex.
  --opencode          Add local_web and Qwen 3.6 to OpenCode config.

Examples:
  ./setup-dramatic-local-llm-demo.sh --all
  ./setup-dramatic-local-llm-demo.sh --codex --web-mcp
  ./setup-dramatic-local-llm-demo.sh --opencode --upstream-url http://127.0.0.1:9090/v1

Docs in this repo:
  codex-llamacpp-installation.md
  codex-llamacpp-setup.md
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      ENABLE_CODEX=1
      ENABLE_WEB_MCP=1
      ENABLE_OPENCODE=1
      shift
      ;;
    --codex)
      ENABLE_CODEX=1
      shift
      ;;
    --web-mcp)
      ENABLE_WEB_MCP=1
      shift
      ;;
    --opencode)
      ENABLE_OPENCODE=1
      shift
      ;;
    --upstream-url)
      UPSTREAM_URL="$2"
      shift 2
      ;;
    --searxng-url)
      SEARXNG_URL="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --adapter-port)
      ADAPTER_PORT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ $ENABLE_CODEX -eq 0 && $ENABLE_WEB_MCP -eq 0 && $ENABLE_OPENCODE -eq 0 ]]; then
  ENABLE_CODEX=1
  ENABLE_WEB_MCP=1
  ENABLE_OPENCODE=1
fi

mkdir -p "$HOME_DIR/.local/bin" "$HOME_DIR/.config/systemd/user" "$HOME_DIR/.codex" "$HOME_DIR/.config/opencode"

write_codex_proxy() {
  cat > "$HOME_DIR/.local/bin/codex-llamacpp-proxy.py" <<'PY'
#!/usr/bin/env python3

import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "http://192.168.6.181:9090/v1").rstrip("/")
LISTEN_HOST = os.environ.get("LISTEN_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "4141"))
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "600"))


def log(message: str) -> None:
    sys.stderr.write(f"[codex-llamacpp-proxy] {message}\n")
    sys.stderr.flush()


def now_ts() -> int:
    return int(time.time())


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def json_dumps(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [stringify_content(item) for item in value]
        return "".join(part for part in parts if part)
    if isinstance(value, dict):
        part_type = value.get("type")
        if part_type in {"input_text", "output_text", "text"}:
            return stringify_content(value.get("text"))
        if part_type == "refusal":
            return stringify_content(value.get("refusal"))
        if part_type in {"input_image", "image_url"}:
            return "[image]"
        if "content" in value:
            return stringify_content(value["content"])
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return str(value)


def response_input_to_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instructions = payload.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": stringify_content(instructions)})

    response_input = payload.get("input")
    if isinstance(response_input, str):
        messages.append({"role": "user", "content": response_input})
        return messages

    if not isinstance(response_input, list):
        return messages

    for item in response_input:
        if not isinstance(item, dict):
            text = stringify_content(item)
            if text:
                messages.append({"role": "user", "content": text})
            continue

        item_type = item.get("type")
        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or make_id("call")
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "{}"),
                            },
                        }
                    ],
                }
            )
            continue

        if item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or item.get("id") or make_id("call"),
                    "content": stringify_content(item.get("output")),
                }
            )
            continue

        role = item.get("role")
        if role:
            message: dict[str, Any] = {"role": role, "content": stringify_content(item.get("content"))}
            if "tool_call_id" in item:
                message["tool_call_id"] = item["tool_call_id"]
            if "tool_calls" in item and isinstance(item["tool_calls"], list):
                message["tool_calls"] = item["tool_calls"]
            messages.append(message)
            continue

        text = stringify_content(item)
        if text:
            messages.append({"role": "user", "content": text})

    return messages


def response_tools_to_chat_tools(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return None

    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        if "function" in tool and isinstance(tool["function"], dict):
            converted.append({"type": "function", "function": tool["function"]})
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            }
        )
    return converted or None


def response_tool_choice_to_chat_tool_choice(payload: dict[str, Any]) -> Any:
    tool_choice = payload.get("tool_choice")
    if tool_choice in {None, "auto", "none", "required"}:
        return tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function" and "function" in tool_choice:
            return tool_choice
        if tool_choice.get("type") == "function":
            return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    return None


def build_chat_request(payload: dict[str, Any], stream: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": payload.get("model") or DEFAULT_MODEL,
        "messages": response_input_to_messages(payload),
        "stream": stream,
    }

    if "temperature" in payload:
        body["temperature"] = payload["temperature"]
    if "top_p" in payload:
        body["top_p"] = payload["top_p"]
    if "max_output_tokens" in payload:
        body["max_tokens"] = payload["max_output_tokens"]
    elif "max_tokens" in payload:
        body["max_tokens"] = payload["max_tokens"]

    tools = response_tools_to_chat_tools(payload)
    if tools is not None:
        body["tools"] = tools

    tool_choice = response_tool_choice_to_chat_tool_choice(payload)
    if tool_choice is not None:
        body["tool_choice"] = tool_choice

    if payload.get("parallel_tool_calls") is not None:
        body["parallel_tool_calls"] = payload["parallel_tool_calls"]

    return body


def map_chat_to_response(chat: dict[str, Any]) -> dict[str, Any]:
    choice = (chat.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = stringify_content(message.get("content"))
    response_id = make_id("resp")
    output: list[dict[str, Any]] = []

    if text:
        output.append(
            {
                "id": make_id("msg"),
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        output.append(
            {
                "id": tool_call.get("id") or make_id("fc"),
                "type": "function_call",
                "call_id": tool_call.get("id") or make_id("call"),
                "name": function.get("name", ""),
                "arguments": function.get("arguments", "{}"),
                "status": "completed",
            }
        )

    usage = chat.get("usage") or {}
    return {
        "id": response_id,
        "object": "response",
        "created_at": now_ts(),
        "status": "completed",
        "model": chat.get("model") or DEFAULT_MODEL,
        "output": output,
        "output_text": text,
        "parallel_tool_calls": False,
        "reasoning": {"effort": None, "summary": None},
        "store": False,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "input_tokens_details": {"cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        log(f"{self.address_string()} - {fmt % args}")

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        return json.loads(body.decode("utf-8") or "{}")

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json_dumps(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_upstream_error(self, exc: Exception) -> None:
        if isinstance(exc, urllib.error.HTTPError):
            raw = exc.read()
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                payload = {"error": {"message": raw.decode("utf-8", errors="replace") or str(exc)}}
            self.send_json(exc.code, payload)
            return
        self.send_json(502, {"error": {"message": str(exc)}})

    def upstream_request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{UPSTREAM_BASE_URL}{path}"
        data = json_dumps(payload) if payload is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            self.send_json(200, {"status": "ok", "upstream": UPSTREAM_BASE_URL, "model": DEFAULT_MODEL})
            return
        if parsed.path == "/v1/models":
            try:
                upstream = self.upstream_request("GET", "/models")
            except Exception as exc:
                self.send_upstream_error(exc)
                return
            self.send_json(200, upstream)
            return
        self.send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/v1/responses":
            self.handle_responses()
            return
        if parsed.path == "/v1/chat/completions":
            self.handle_chat_passthrough()
            return
        self.send_json(404, {"error": {"message": "not found"}})

    def handle_chat_passthrough(self) -> None:
        try:
            payload = self.read_json_body()
            upstream = self.upstream_request("POST", "/chat/completions", payload)
        except Exception as exc:
            self.send_upstream_error(exc)
            return
        self.send_json(200, upstream)

    def handle_responses(self) -> None:
        try:
            payload = self.read_json_body()
            chat_request = build_chat_request(payload, stream=False)
            upstream = self.upstream_request("POST", "/chat/completions", chat_request)
            response_payload = map_chat_to_response(upstream)
        except Exception as exc:
            self.send_upstream_error(exc)
            return
        self.send_json(200, response_payload)


def main() -> None:
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    log(f"listening on http://{LISTEN_HOST}:{LISTEN_PORT} -> {UPSTREAM_BASE_URL}")
    server.serve_forever()


if __name__ == "__main__":
    main()
PY
  chmod 755 "$HOME_DIR/.local/bin/codex-llamacpp-proxy.py"
}

write_local_web_mcp() {
  cat > "$HOME_DIR/.local/bin/local-web-mcp.py" <<'PY'
#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from mcp.server.fastmcp import FastMCP


SEARXNG_BASE_URL = os.environ.get("LOCAL_WEB_SEARXNG_URL", "http://127.0.0.1:18910").rstrip("/")
FETCH_TIMEOUT = float(os.environ.get("LOCAL_WEB_FETCH_TIMEOUT", "20"))
DEFAULT_MAX_FETCH_CHARS = int(os.environ.get("LOCAL_WEB_MAX_FETCH_CHARS", "12000"))
DEFAULT_SEARCH_LIMIT = int(os.environ.get("LOCAL_WEB_SEARCH_LIMIT", "5"))
USER_AGENT = os.environ.get("LOCAL_WEB_USER_AGENT", "local-web-mcp/1.0")

mcp = FastMCP(
    "local-web",
    instructions=(
        "Use search_web to search the web via the local SearxNG instance. "
        "Use fetch_url to read a specific page after you identify a promising result."
    ),
)


class HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.skip_depth = 0
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self.in_title = True
            return
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1
            return
        if tag in {"p", "div", "section", "article", "li", "ul", "ol", "br", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
            return
        if tag in {"script", "style", "noscript", "svg"} and self.skip_depth > 0:
            self.skip_depth -= 1
            return
        if tag in {"p", "div", "section", "article", "li", "ul", "ol"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self.in_title:
            self.title_parts.append(text)
            return
        self.text_parts.append(text)

    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    def text(self) -> str:
        text = "".join(self.text_parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


def http_get_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs are supported.")
    return url


@mcp.tool()
def search_web(query: str, limit: int = DEFAULT_SEARCH_LIMIT, category: str = "general") -> dict:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    limit = clamp(limit, 1, 10)
    params = {
        "q": query,
        "format": "json",
        "language": "en-US",
        "safesearch": "1",
        "categories": category,
    }
    url = f"{SEARXNG_BASE_URL}/search?{urllib.parse.urlencode(params)}"
    try:
        payload = http_get_json(url)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach local SearxNG at {SEARXNG_BASE_URL}") from exc

    results = []
    for item in payload.get("results", [])[:limit]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "source": item.get("engine", ""),
                "engines": item.get("engines", []),
                "published": item.get("publishedDate"),
                "category": item.get("category", ""),
            }
        )
    return {
        "query": query,
        "category": category,
        "result_count": len(results),
        "results": results,
        "suggestions": payload.get("suggestions", []),
    }


@mcp.tool()
def fetch_url(url: str, max_chars: int = DEFAULT_MAX_FETCH_CHARS) -> dict:
    url = normalize_url(url.strip())
    max_chars = clamp(max_chars, 500, 50000)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/json,text/plain;q=0.9,*/*;q=0.5",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:
            body = response.read()
            final_url = response.geturl()
            content_type = response.headers.get("Content-Type", "application/octet-stream")
            status_code = getattr(response, "status", 200)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Fetch failed with HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Fetch failed for {url}: {exc.reason}") from exc

    charset = "utf-8"
    match = re.search(r"charset=([A-Za-z0-9._-]+)", content_type)
    if match:
        charset = match.group(1)
    text = body.decode(charset, errors="replace")

    title = ""
    extracted_text = text
    if "html" in content_type:
        parser = HTMLTextExtractor()
        parser.feed(text)
        title = parser.title()
        extracted_text = parser.text()
    elif "json" in content_type:
        try:
            extracted_text = json.dumps(json.loads(text), ensure_ascii=True, indent=2)
        except json.JSONDecodeError:
            pass

    extracted_text = extracted_text[:max_chars]
    return {
        "url": final_url,
        "requested_url": url,
        "status_code": status_code,
        "content_type": content_type,
        "title": title,
        "text": extracted_text,
        "truncated": len(extracted_text) == max_chars,
    }


if __name__ == "__main__":
    mcp.run()
PY
  chmod 755 "$HOME_DIR/.local/bin/local-web-mcp.py"
}

write_systemd_service() {
  cat > "$HOME_DIR/.config/systemd/user/codex-llamacpp-proxy.service" <<EOF
[Unit]
Description=Codex llama.cpp Responses adapter
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$HOME_DIR
Environment=UPSTREAM_BASE_URL=$UPSTREAM_URL
Environment=LISTEN_HOST=$ADAPTER_HOST
Environment=LISTEN_PORT=$ADAPTER_PORT
Environment=DEFAULT_MODEL=$MODEL_NAME
ExecStart=/usr/bin/python3 $HOME_DIR/.local/bin/codex-llamacpp-proxy.py
Restart=always
RestartSec=2
StandardOutput=append:$HOME_DIR/.codex/llamacpp-proxy.log
StandardError=append:$HOME_DIR/.codex/llamacpp-proxy.log

[Install]
WantedBy=default.target
EOF
}

update_codex_config() {
  python3 - "$HOME_DIR/.codex/config.toml" "$MODEL_NAME" "$ADAPTER_HOST" "$ADAPTER_PORT" "$SEARXNG_URL" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
model_name = sys.argv[2]
adapter_host = sys.argv[3]
adapter_port = sys.argv[4]
searxng_url = sys.argv[5]

text = path.read_text() if path.exists() else ""
if not text:
    text = 'model = "gpt-5.4"\nmodel_reasoning_effort = "medium"\n\n'

def upsert_table(src: str, table: str, body: str) -> str:
    pattern = re.compile(rf'(?ms)^\[{re.escape(table)}\]\n.*?(?=^\[|\Z)')
    block = f'[{table}]\n{body.rstrip()}\n\n'
    if pattern.search(src):
        return pattern.sub(block, src, count=1)
    if not src.endswith("\n"):
        src += "\n"
    return src + "\n" + block

text = upsert_table(
    text,
    "profiles.llama",
    f'model_provider = "llamacpp_proxy"\nmodel = "{model_name}"',
)
text = upsert_table(
    text,
    "model_providers.llamacpp_proxy",
    f'name = "llama.cpp Responses proxy"\nbase_url = "http://{adapter_host}:{adapter_port}/v1"\nwire_api = "responses"',
)
text = upsert_table(
    text,
    "mcp_servers.local_web",
    'command = "/usr/bin/python3"\nargs = ["/home/tyrel/.local/bin/local-web-mcp.py"]\nstartup_timeout_sec = 20\ntool_timeout_sec = 60\nsupports_parallel_tool_calls = true\ndefault_tools_approval_mode = "approve"',
)
text = upsert_table(
    text,
    "mcp_servers.local_web.env",
    f'LOCAL_WEB_SEARXNG_URL = "{searxng_url}"',
)
path.write_text(text)
PY
}

update_opencode_config() {
  python3 - "$HOME_DIR/.config/opencode/opencode.json" "$MODEL_NAME" "$SEARXNG_URL" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
model_name = sys.argv[2]
searxng_url = sys.argv[3]

if path.exists():
    config = json.loads(path.read_text())
else:
    config = {"$schema": "https://opencode.ai/config.json"}

config.setdefault("$schema", "https://opencode.ai/config.json")
config.setdefault("provider", {})
config.setdefault("mcp", {})

config["mcp"]["local_web"] = {
    "type": "local",
    "command": ["/usr/bin/python3", "/home/tyrel/.local/bin/local-web-mcp.py"],
    "enabled": True,
    "timeout": 20000,
    "environment": {"LOCAL_WEB_SEARXNG_URL": searxng_url},
}

llama = config["provider"].setdefault("llama", {})
llama.setdefault("name", "llama.cpp")
llama.setdefault("npm", "@ai-sdk/openai-compatible")
llama.setdefault("options", {})
llama["options"].setdefault("apiKey", "EMPTY")
llama["options"].setdefault("baseURL", "http://127.0.0.1:9090/v1")
models = llama.setdefault("models", {})
models["qwen3.6-35b"] = {
    "limit": {"context": 262144, "output": 32768},
    "name": model_name,
}

path.write_text(json.dumps(config, indent=2) + "\n")
PY
}

verify_codex() {
  curl -fsS "http://${ADAPTER_HOST}:${ADAPTER_PORT}/healthz" >/dev/null
}

verify_web_mcp() {
  curl -fsS "${SEARXNG_URL}/search?q=test&format=json" >/dev/null
}

verify_opencode() {
  opencode mcp list >/dev/null
  opencode models llama >/dev/null
}

if [[ $ENABLE_CODEX -eq 1 ]]; then
  echo "==> Installing Codex adapter"
  write_codex_proxy
  write_systemd_service
  update_codex_config
  if command -v systemctl >/dev/null 2>&1; then
    systemctl --user daemon-reload
    systemctl --user enable --now codex-llamacpp-proxy.service
  fi
  verify_codex
fi

if [[ $ENABLE_WEB_MCP -eq 1 ]]; then
  echo "==> Installing local_web MCP"
  write_local_web_mcp
  update_codex_config
  verify_web_mcp
fi

if [[ $ENABLE_OPENCODE -eq 1 ]]; then
  echo "==> Updating OpenCode"
  write_local_web_mcp
  update_opencode_config
  verify_opencode
fi

cat <<EOF

Done.

References:
  $REPO_DIR/codex-llamacpp-installation.md
  $REPO_DIR/codex-llamacpp-setup.md

Try:
  codex --profile llama
  opencode mcp list
  opencode run --dangerously-skip-permissions -m llama/qwen3.6-35b 'Use the local_web tool to search for "OpenAI Codex GitHub" and reply with only the best result URL.'
EOF
