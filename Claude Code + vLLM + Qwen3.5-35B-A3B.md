# Docker needs to be installed

```docker -v```

Docker version 29.1.3, build f52814d

# vllm bashrc 

```
# Start dgx-vllm-nvfp4 container (test model)
start-vllm-test () {
  sudo docker rm -f dgx-vllm-nvfp4 2>/dev/null

  sudo docker run -d --name dgx-vllm-nvfp4 \
    --network host --gpus all --ipc=host \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
    -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
    -e VLLM_NVFP4_GEMM_BACKEND=marlin \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e MODEL="Sehyo/Qwen3.5-35B-A3B-NVFP4" \
    -e PORT=8888 \
    -e GPU_MEMORY_UTIL=0.75 \
    -e MAX_MODEL_LEN=131072 \
    -e MAX_NUM_SEQS=16 \
    -e VLLM_EXTRA_ARGS='--served-model-name default_model --enable-auto-tool-choice --tool-call-parser hermes --speculative-config {"method":"mtp","num_speculative_tokens":2} --no-enable-chunked-prefill --attention-backend flashinfer --kv-cache-dtype fp8' \
    dgx-vllm-nvfp4-mtp:v22 serve
}
```

## Check docker status

`docker ps`

## Show docker logs

`docker logs dgx-vllm-nvfp4`

## Once its runing logs will show:

```
(APIServer pid=1) INFO 03-04 07:22:27 [api_server.py:500] Starting vLLM API server 0 on http://0.0.0.0:8888
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:38] Available routes are:
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /openapi.json, Methods: GET, HEAD
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /docs, Methods: GET, HEAD
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /docs/oauth2-redirect, Methods: GET, HEAD
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /redoc, Methods: GET, HEAD
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /tokenize, Methods: POST
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /detokenize, Methods: POST
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /load, Methods: GET
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /version, Methods: GET
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /health, Methods: GET
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /metrics, Methods: GET
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /v1/models, Methods: GET
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /ping, Methods: GET
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /ping, Methods: POST
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /invocations, Methods: POST
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /v1/chat/completions, Methods: POST
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /v1/chat/completions/render, Methods: POST
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /v1/responses, Methods: POST
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /v1/completions, Methods: POST
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /v1/completions/render, Methods: POST
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /v1/messages, Methods: POST
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /inference/v1/generate, Methods: POST
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=1) INFO 03-04 07:22:27 [launcher.py:47] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
```

# Downloading the model

`export HF_TOKEN=hf_YOUR_TOKEN_HERE`
`hf download Sehyo/Qwen3.5-35B-A3B-NVFP4`

```
Fetching 14 files: 100%|███████████████████████████████████| 14/14 [07:33<00:00, 32.39s/it]
Download complete: 100%|███████████████████████████████| 23.4G/23.4G [07:33<00:00, 189MB/s]/home/tyrel/.cache/huggingface/hub/models--Sehyo--Qwen3.5-35B-A3B-NVFP4/snapshots/1393d72b1697950a0afb77db6e24c7ede49b761b
Download complete: 100%|██████████████████████████████| 23.4G/23.4G [07:33<00:00, 51.5MB/s]
```


