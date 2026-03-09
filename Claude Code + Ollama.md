# Claude Code with custom Model on Ollama

## Ollama Config:

1. Download Ollama from: https://ollama.com/download or use curl:

   `curl -fsSL https://ollama.com/install.sh | sh`

3. Verify `ollama -v`
4. Check if deamon/process is running `ollama ps`
5. Download Model `ollama pull qwen3.5:35b`
6. Run model  **Optional: for direct chatting and seeing promt eval rate and token generation rates**
   
   `ollama run qwen3.5:35b --verbose --think=false`
7. If using locally you can use: `ollama launch claude --model qwen3.5:35b`
8. Add and Modify Environment Variables in the systemd service file:

`sudo nano /etc/systemd/system/ollama.service`
```
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_KV_CACHE_TYPE=q8"
Environment="OLLAMA_FLASH_ATTENTION=true"
```
9. Reload and Restart service:

   `sudo systemctl daemon-reload && sudo systemctl restart ollama.service`

10. Verify Env Variables:

    `systemctl show ollama --property=Environment`
12. Check listening port `sudo ss -antlp | grep ollama`
13. Create Modelfile `nano modelfile`
```
from qwen3.5:35b

PARAMETER num_ctx 32768
PARAMETER temperature 0
```
12. Create a new model "default_model" from the modelfile

    `ollama create default_model -f modelfile`

14. Follow Logs:

    `journalctl -f -u ollama.service`
    
***Suggested Context Length of at least 65536 tokens***


## Claude Config:

`~/.claude/ollama.settings.json`

***Replace ANTHROPIC_BASE_URL with the IP and PORT of your Ollama server***

```bash
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://192.168.6.181:11434/",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "API_TIMEOUT_MS": "3000000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": 1,
    "CLAUDE_CODE_ATTRIBUTION_HEADER": 0,
    "ANTHROPIC_MODEL": "default_model",
    "ANTHROPIC_SMALL_FAST_MODEL": "default_model",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "default_model",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "default_model",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "default_model"
  }
}
```

## Launch Paramaters 

`claude --settings ~/.claude/ollama.settings.json`


# Observations:

*Claude code looks like it will retry 12 times before giving up*

# Issues:

## Context Lmitations (16k is not enough for anything! 32k fails very quickly) 

```
Mar 04 23:58:56 spark-ee93 ollama[439352]: time=2026-03-04T23:58:56.964-07:00 level=WARN source=runner.go:187 msg="truncating input prompt" limit=32768 prompt=34748 keep=4 new=32768
Mar 04 23:59:54 spark-ee93 ollama[439352]: [GIN] 2026/03/04 - 23:59:54 | 200 | 58.398272961s |   192.168.6.154 | POST     "/v1/messages?beta=true"
Mar 05 00:00:11 spark-ee93 ollama[439352]: [GIN] 2026/03/05 - 00:00:11 | 200 | 16.867790384s |   192.168.6.154 | POST     "/v1/messages?beta=true"
Mar 05 00:03:26 spark-ee93 ollama[439352]: time=2026-03-05T00:03:26.266-07:00 level=WARN source=qwen3.go:108 msg="qwen3 tool call parsing failed" error="failed to parse JSON: unexpected end of JSON input"
Mar 05 00:03:26 spark-ee93 ollama[439352]: [GIN] 2026/03/05 - 00:03:26 | 200 |         2m57s |   192.168.6.154 | POST     "/v1/messages?beta=true"
Mar 05 00:06:37 spark-ee93 ollama[439352]: time=2026-03-05T00:06:37.924-07:00 level=WARN source=qwen3.go:108 msg="qwen3 tool call parsing failed" error="failed to parse JSON: unexpected end of JSON input"
Mar 05 00:06:37 spark-ee93 ollama[439352]: [GIN] 2026/03/05 - 00:06:37 | 500 |         3m11s |   192.168.6.154 | POST     "/v1/messages?beta=true"
Mar 05 00:09:06 spark-ee93 ollama[439352]: time=2026-03-05T00:09:06.581-07:00 level=WARN source=qwen3.go:108 msg="qwen3 tool call parsing failed" error="failed to parse JSON: unexpected end of JSON input"
```

❯ create another variation                                                                                                 
  ⎿  API Error: 500 {"type":"error","error":{"type":"api_error","message":"failed to parse JSON: unexpected end of JSON 
     input"},"request_id":"req_094bbdc3abddc71a6ab6b031"}                                                                  
                                                                                                                         
✻ Baked for 35m 0s                                 
                                                                                                                           
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
❯                                                                                                                          
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ⏵⏵ accept edits on (shift+tab to cycle)                                                                    33462 tokens  
