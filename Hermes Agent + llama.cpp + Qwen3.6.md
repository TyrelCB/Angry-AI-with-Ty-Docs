# Hermes Agent + LLaMA.cpp + Qwen3.6


## Step 1: Download Qwen 3.6 Model

<!-- ![developer](Gemma4.png) -->
<img src="Qwen3.6.png" width="80%" style="display: block; margin: 0 auto;">

#### Hugging Face Model Card:

[https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)

```bash
pip install huggingface_hub
```

```bash
mkdir -p ~/models
```

```bash
hf download unsloth/Qwen3.6-35B-A3B-GGUF --include "Qwen3.6-35B-A3B-GGUF" --local-dir ~/models/
```

## Step 2: Setup LLaMA.cpp:
![developer](llama.cpp-logo.png)
### 2-A: Build llama.cpp from source (CUDA)

[https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release --target llama-server -j$(nproc)
```

Binary will be at `./build/bin/llama-server`

> Recomend exporting this to PATH

```bash
echo 'export PATH="$PATH:$HOME/llama.cpp/build/bin"' >> ~/.bashrc && source ~/.bashrc
```



### 2-B: Launch llama-server

```bash
~/llama.cpp/build/bin/llama-server \
  --model ~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --alias Qwen3.6-35B-A3B \
  --host 0.0.0.0 \
  --port 9090 \
  --gpu-layers 99 \
  --ctx-size $((262144*4)) \
  --parallel 4 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --flash-attn on \
  --reasoning off \
  --jinja \
  --batch-size 32768 \
  --ubatch-size 2048 \
  --cont-batching \
  --no-context-shift \
  --defrag-thold 0.1 \
  2>&1 | tee ~/llama-server.log
```

Useful Flags:
- `--n-gpu-layers 99` — offload all layers to GPU (reduce if you hit VRAM limits)
- `--ctx-size` — Total context size (gets divided by parallel)
- `--alias` — sets the model name exposed on the API (Hermes uses this as the `model` value)
- `--host 0.0.0.0` - Listen on all interfaces
- `--port 9090` - 8080 by default
- `--reasoning off` - easy toggle for thinking mode (CoT typically slows down agentic coding)
- `--parallel` - concurency for agents and sub-agents
- `--cache-type-k/v` - Key/Value quantization (save vram)
- `--flash-attn on` - Speed up inference and reduces memory [***warning: can reduce quality***]
- `--batch-size` - Tokes processed per forward pass (scheduling)
- `--ubatch-size` - Physical maximum batch size (execution)
- `--cont-batching` - Continuous batching (processing multiple requests simultaneously)
- `--metrics` - enables endpoint that exposes real-time performance and usage data

Verify it's running: `curl http://192.168.6.181:9090/v1/models`

***Hermes auto-detects context length from `/v1/models` when it can — set `context_length` manually below if your alias doesn't expose it.***

> if you want to run **more than one model on llama.cpp** you need sperate instances of `llama-server` on a different ports `--port`

> llama.cpp also has a web interface `http://192.168.6.181:9090` that is useful for Chat GPT style Q&A and to see things like Prompt Processing t/s and Generation t/s
>
> key and valude quantization can be set asyncronusly which will probably be a go to best practice for near future tech like turboquant

#### Reference table for setting total context

| Context| bits (n) | value (2^n) | --parallel 4 | --parallel 8 |
|:---:|:---:|:---:|:---:|:---:|
| ~1k | 10 | 1024 | 256 | 128 |
| ~2k | 11 | 2048 | 512 | 256 |
| ~4k | 12 | 4096 | 1024 | 512 |
| ~8k | 13 | 8192 | 2048 | 1024 |
| ~16k | 14 | 16384 | 4096 | 2048 |
| ~32k | 15 | 32768 | 8192 | 4096 |
| ~64k | 16 | 65536 | 16384 | 8192 |
| ~128k | 17 | 131072 | 32768 | 16384 |
| ~256k | 18 | 262144 | 65536 | 32768 |
| ~512k | 19 | 524288 | 131072 | 65536 |
| 1M | 20 | 1048576 | 262144 | 131072 |
| 2M | 21 | 2097152 | 524288 | 262144 |


<img src="https://raw.githubusercontent.com/NousResearch/hermes-agent/main/assets/banner.png" width="100%" style="display: block; margin: 0 auto;">

## Step 3: Install Hermes Agent

[https://github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)

One-line installer (Linux / macOS / WSL2):

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

The installer drops everything under `~/.hermes/` and adds the `hermes` command to your PATH. Re-source your shell or open a new terminal:

```bash
source ~/.bashrc
```

Verify the install:

```bash
hermes --version
hermes doctor
```

> Data, config, sessions, and logs all live under `$HERMES_HOME` (default `~/.hermes`). Override with `--hermes-home PATH` at install time if you want it elsewhere.

## Step 4: Hermes Config

Create / edit: `~/.hermes/config.yaml`

**Replace `base_url` with the `IP` and `PORT` of your llama-server**

```yaml
model:
  # Must match the --alias you set on llama-server (or the GGUF filename if --alias is unset)
  default: "Qwen3.6-35B-A3B"

  # "llamacpp" is an alias for "custom" — any OpenAI-compatible endpoint
  provider: "llamacpp"

  base_url: "http://192.168.6.181:9090/v1"

  # llama.cpp doesn't require auth, but the OpenAI client still wants *something*
  api_key: "dummy"

  # Optional: pin context_length if auto-detect from /v1/models is wrong
  # context_length: 1048576
```

> Hermes also reads `~/.hermes/.env` for secrets if you prefer to keep `api_key` out of the YAML:
> ```
> OPENAI_API_KEY=dummy
> OPENAI_BASE_URL=http://192.168.6.181:9090/v1
> ```

Pick model / provider interactively instead (writes the same config for you):

```bash
hermes model
```

Or run the full guided setup:

```bash
hermes setup
```

## Step 5: Launch 🚀

```bash
hermes
```

One-shot prompt (non-interactive):

```bash
hermes --model Qwen3.6-35B-A3B --provider llamacpp "reply with exactly ok"
```


### Observations:

- `provider: "llamacpp"`, `"vllm"`, and `"ollama"` are all just aliases for `"custom"` — they all route through Hermes' OpenAI-compatible client
- The `model` value in `config.yaml` must match the `--alias` you gave llama-server (or the GGUF filename if you didn't set one) — check `curl http://<host>:9090/v1/models` to see what llama.cpp is advertising
- Hermes auto-detects context length from the upstream `/v1/models` response when it can; set `context_length` manually only if that detection comes back wrong
- `hermes doctor` will diagnose most config / connectivity issues
- Token generation rate and prompt eval rate are visible in `llama-server` stdout, same as with Claude Code

### Issues:

#### Context length mismatch

If Hermes thinks the context window is smaller than what llama.cpp actually serves, it will compress history too aggressively. Pin `context_length` under `model:` to match the `--ctx-size / --parallel` math from Step 2-B.

#### Tool call parsing

Smaller / heavily quantized models can produce malformed JSON for tool calls. Bumping to a Q8 quant or shrinking the active toolset (`hermes tools`) usually clears it up.
