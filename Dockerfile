FROM ghcr.io/ggml-org/llama.cpp:full-cuda@sha256:1276d8b3f09fe1cb05b8d7369f64a5a9c15619fb05a61f3df1d261a4b53c96d9

USER root
RUN test -f /app/libllama-server-impl.so \
    && printf '/app\n' > /etc/ld.so.conf.d/llama-cpp.conf \
    && ldconfig \
    && /app/llama-server --version

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --break-system-packages runpod==1.8.1

ARG MODEL_URL="https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf?download=true"
RUN mkdir -p /models \
    && curl --fail --location --retry 5 --retry-delay 5 \
        --output /models/Qwen3.8-27B-UD-Q4_K_M.gguf "$MODEL_URL" \
    && test "$(stat -c %s /models/Qwen3.8-27B-UD-Q4_K_M.gguf)" -gt 15000000000

WORKDIR /worker
COPY handler.py /worker/handler.py

ENV MODEL="unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M" \
    MODEL_FILE="Qwen3.8-27B-UD-Q4_K_M.gguf" \
    MODEL_PATH="/models/Qwen3.8-27B-UD-Q4_K_M.gguf" \
    CONTEXT_SIZE="16384" \
    PARALLEL="1" \
    GPU_LAYERS="999" \
    HF_HOME="/runpod-volume/huggingface"

ENTRYPOINT []
CMD ["python3", "-u", "/worker/handler.py"]
