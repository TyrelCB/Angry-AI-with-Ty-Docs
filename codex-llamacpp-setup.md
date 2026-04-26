# Codex to llama.cpp Setup

This machine is configured so OpenAI Codex CLI can use the remote llama.cpp server at `192.168.6.181:9090`.

Because current Codex expects the OpenAI **Responses API** and upstream llama.cpp exposes **Chat Completions**, a local adapter translates between the two.

## Current behavior

- Remote model host: `http://192.168.6.181:9090/v1`
- Remote model: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- Local adapter endpoint for Codex: `http://127.0.0.1:4141/v1`
- Codex profile name: `llama`
- Adapter is installed as a persistent user systemd service and restarts automatically

## How it works

Request flow:

1. `codex --profile llama`
2. Codex sends `POST /v1/responses` to `127.0.0.1:4141`
3. The local adapter converts the request to `POST /v1/chat/completions`
4. The adapter forwards that request to llama.cpp at `192.168.6.181:9090`
5. The adapter converts the llama.cpp response back into a Responses API object for Codex

## Files

Codex config:

- `/home/tyrel/.codex/config.toml`

Adapter script:

- `/home/tyrel/.local/bin/codex-llamacpp-proxy.py`

User service:

- `/home/tyrel/.config/systemd/user/codex-llamacpp-proxy.service`

Logs:

- `/home/tyrel/.codex/llamacpp-proxy.log`

## Codex configuration

The `llama` profile in `~/.codex/config.toml` is configured to use a custom provider:

```toml
[profiles.llama]
model_provider = "llamacpp_proxy"
model = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"

[model_providers.llamacpp_proxy]
name = "llama.cpp Responses proxy"
base_url = "http://127.0.0.1:4141/v1"
wire_api = "responses"
```

## Service configuration

The adapter runs as a user service:

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
```

## Commands

Start Codex against llama.cpp:

```bash
codex --profile llama
```

Run a one-shot prompt:

```bash
codex --profile llama exec --skip-git-repo-check 'Reply with exactly ok.'
```

Check service status:

```bash
systemctl --user status codex-llamacpp-proxy.service
```

Restart the adapter:

```bash
systemctl --user restart codex-llamacpp-proxy.service
```

Tail adapter logs:

```bash
tail -f /home/tyrel/.codex/llamacpp-proxy.log
```

Health check:

```bash
curl http://127.0.0.1:4141/healthz
```

List models through the adapter:

```bash
curl http://127.0.0.1:4141/v1/models
```

Direct Responses API test:

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

## Notes

- The adapter exists because Codex no longer supports `wire_api = "chat"`.
- The remote llama.cpp server itself was confirmed healthy before the adapter was added.
- The adapter is bound to `127.0.0.1`, so only local processes on this machine can use it directly.
