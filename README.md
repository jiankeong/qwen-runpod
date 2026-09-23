# Qwen3.8-27B GGUF on RunPod Serverless

This worker launches current CUDA `llama.cpp`, downloads Unsloth's 16.5 GB
`UD-Q4_K_M` GGUF on first boot, and exposes it through RunPod queue jobs. The
handler preserves the OpenAI chat/completions response shape.

The Docker build pins the verified multi-architecture CUDA 12 image from the
current `ggml-org/llama.cpp` GitHub Container Registry namespace. It uses the
`full-cuda` runtime so `llama-server` and its split implementation library ship
together.

## RunPod Hub

The repository includes `.runpod/hub.json` and `.runpod/tests.json`, so a GitHub
release can be indexed and tested by RunPod Hub. The Hub listing uses the
**Language** category, one GPU with at least 24 GB VRAM, a 30 GB container disk,
and a non-streaming chat-completion smoke test on the datacenter-oriented 48 GB
NVIDIA A40 pool. The worker registers with RunPod immediately while llama.cpp
downloads the model; the first job waits for readiness. Hub validation uses the
smaller Q2 quant and 4K context to keep a cold-cache test inside its deadline,
while normal deployments continue to default to Q4_K_M and 16K.

## 1. Build and push

```bash
chmod +x deploy.sh
IMAGE=ghcr.io/YOUR_USER/qwen38-runpod:latest ./deploy.sh
```

## 2. RunPod endpoint settings

1. Create a **25 GB or larger network volume** and mount it at `/runpod-volume`.
   The first worker caches the 16.5 GB model there; later cold starts reuse it.
2. Create a Serverless template from the pushed image. Container disk: **10 GB**.
3. Create an endpoint with that template. Select **24 GB+ VRAM** GPUs (L4, RTX
   4090, A5000, A6000, L40/L40S, A100). Start with workers `0–1`, idle timeout
   `60 s`, execution timeout `1200 s`, and FlashBoot enabled where available.
4. Add the environment variables from `.env.example`. `HF_TOKEN` is optional
   for this public model.

`CONTEXT_SIZE=16384` is chosen so Q4 weights, KV cache, and runtime buffers fit a
24 GB GPU. Use 32–48 GB VRAM before raising context or `PARALLEL`.

## 3. Invoke

```bash
curl -sS "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input":{"messages":[{"role":"user","content":"用一句话介绍你自己"}],"temperature":0.7,"top_p":0.8,"max_tokens":256}}'
```

For asynchronous work, replace `runsync` with `run`; poll the returned job ID.
The worker also accepts `{"input":{"prompt":"..."}}` and routes it to
`/v1/completions`. Streaming queue jobs are rejected explicitly.

## Local tests

```bash
python3 -m unittest discover -s tests -v
```
