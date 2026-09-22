FROM ghcr.io/ggerganov/llama.cpp:server-cuda

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --break-system-packages runpod==1.8.1

WORKDIR /worker
COPY handler.py /worker/handler.py

ENV MODEL="unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M" \
    MODEL_FILE="Qwen3.8-27B-UD-Q4_K_M.gguf" \
    CONTEXT_SIZE="16384" \
    PARALLEL="1" \
    GPU_LAYERS="999" \
    HF_HOME="/runpod-volume/huggingface"

ENTRYPOINT []
CMD ["python3", "-u", "/worker/handler.py"]
