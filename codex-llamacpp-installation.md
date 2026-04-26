# Codex to llama.cpp Installation Instructions

These steps install a local adapter so Codex CLI can use a remote llama.cpp server that only exposes Chat Completions.

This guide assumes:

- Codex CLI is already installed
- Python 3 is available at `/usr/bin/python3`
- The remote llama.cpp server is reachable at `http://192.168.6.181:9090/v1`
- The target model is `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`

## 1. Verify the remote llama.cpp server

Check that the remote host answers `models`:

```bash
curl http://192.168.6.181:9090/v1/models
```

Check that chat completions work:

```bash
curl -X POST http://192.168.6.181:9090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "messages": [
      {
        "role": "user",
        "content": "reply with exactly ok"
      }
    ],
    "max_tokens": 8,
    "temperature": 0
  }'
```

## 2. Install the adapter script

Create `~/.local/bin/codex-llamacpp-proxy.py` with this content:

```python
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

    def write_sse(self, event: str, data: Any) -> None:
        payload = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
        self.wfile.write(f"event: {event}\n".encode("utf-8"))
        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def write_sse_done(self) -> None:
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

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
            stream = bool(payload.get("stream"))
            chat_request = build_chat_request(payload, stream=False)
            upstream = self.upstream_request("POST", "/chat/completions", chat_request)
            response_payload = map_chat_to_response(upstream)
        except Exception as exc:
            self.send_upstream_error(exc)
            return

        if not stream:
            self.send_json(200, response_payload)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        text_item = None
        function_items: list[dict[str, Any]] = []
        for item in response_payload["output"]:
            if item.get("type") == "message" and text_item is None:
                text_item = item
            elif item.get("type") == "function_call":
                function_items.append(item)

        created_response = dict(response_payload)
        created_response["status"] = "in_progress"
        self.write_sse("response.created", {"type": "response.created", "response": created_response})

        output_index = 0
        if text_item is not None:
            message_stub = {
                "id": text_item["id"],
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            }
            self.write_sse(
                "response.output_item.added",
                {"type": "response.output_item.added", "output_index": output_index, "item": message_stub},
            )
            part = {"type": "output_text", "text": "", "annotations": []}
            self.write_sse(
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    "output_index": output_index,
                    "item_id": text_item["id"],
                    "content_index": 0,
                    "part": part,
                },
            )
            text = text_item["content"][0]["text"]
            chunk_size = 64
            for start in range(0, len(text), chunk_size):
                chunk = text[start : start + chunk_size]
                self.write_sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "output_index": output_index,
                        "item_id": text_item["id"],
                        "content_index": 0,
                        "delta": chunk,
                    },
                )
            self.write_sse(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "output_index": output_index,
                    "item_id": text_item["id"],
                    "content_index": 0,
                    "text": text,
                },
            )
            self.write_sse(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "output_index": output_index,
                    "item_id": text_item["id"],
                    "content_index": 0,
                    "part": text_item["content"][0],
                },
            )
            self.write_sse(
                "response.output_item.done",
                {"type": "response.output_item.done", "output_index": output_index, "item": text_item},
            )
            output_index += 1

        for function_item in function_items:
            self.write_sse(
                "response.output_item.added",
                {"type": "response.output_item.added", "output_index": output_index, "item": function_item},
            )
            self.write_sse(
                "response.output_item.done",
                {"type": "response.output_item.done", "output_index": output_index, "item": function_item},
            )
            output_index += 1

        self.write_sse("response.completed", {"type": "response.completed", "response": response_payload})
        self.write_sse_done()


def main() -> None:
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    log(f"listening on http://{LISTEN_HOST}:{LISTEN_PORT} -> {UPSTREAM_BASE_URL}")
    server.serve_forever()


if __name__ == "__main__":
    main()
```

Make it executable:

```bash
chmod 755 /home/tyrel/.local/bin/codex-llamacpp-proxy.py
```

## 3. Add the user service

Create `~/.config/systemd/user/codex-llamacpp-proxy.service`:

```ini
[Unit]
Description=Codex llama.cpp Responses adapter
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/tyrel
Environment=UPSTREAM_BASE_URL=http://192.168.6.181:9090/v1
Environment=LISTEN_HOST=127.0.0.1
Environment=LISTEN_PORT=4141
Environment=DEFAULT_MODEL=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
ExecStart=/usr/bin/python3 /home/tyrel/.local/bin/codex-llamacpp-proxy.py
Restart=always
RestartSec=2
StandardOutput=append:/home/tyrel/.codex/llamacpp-proxy.log
StandardError=append:/home/tyrel/.codex/llamacpp-proxy.log

[Install]
WantedBy=default.target
```

Enable and start it:

```bash
systemctl --user daemon-reload
systemctl --user enable --now codex-llamacpp-proxy.service
```

## 4. Update Codex config

Edit `~/.codex/config.toml` and add:

```toml
[profiles.llama]
model_provider = "llamacpp_proxy"
model = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"

[model_providers.llamacpp_proxy]
name = "llama.cpp Responses proxy"
base_url = "http://127.0.0.1:4141/v1"
wire_api = "responses"
```

## 5. Verify the adapter

Health check:

```bash
curl http://127.0.0.1:4141/healthz
```

List models:

```bash
curl http://127.0.0.1:4141/v1/models
```

Test the Responses endpoint:

```bash
curl -X POST http://127.0.0.1:4141/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "input": [
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": "reply with exactly ok"
          }
        ]
      }
    ],
    "max_output_tokens": 8
  }'
```

## 6. Verify Codex

Run a one-shot test:

```bash
codex --profile llama exec --skip-git-repo-check 'Reply with exactly ok.'
```

Start an interactive session:

```bash
codex --profile llama
```

## Troubleshooting

Check the service:

```bash
systemctl --user status codex-llamacpp-proxy.service
```

Restart the service:

```bash
systemctl --user restart codex-llamacpp-proxy.service
```

View logs:

```bash
tail -f /home/tyrel/.codex/llamacpp-proxy.log
```

If Codex fails but the remote server is healthy:

- confirm the adapter is listening on `127.0.0.1:4141`
- confirm `~/.codex/config.toml` points `profiles.llama` at `llamacpp_proxy`
- confirm the model name matches the remote llama.cpp model exactly
- confirm the remote server still answers `http://192.168.6.181:9090/v1/models`
