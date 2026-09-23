"""RunPod Serverless adapter for an OpenAI-compatible llama.cpp server."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any


LLAMA_HOST = os.getenv("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.getenv("LLAMA_PORT", "8080"))
LLAMA_BASE_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}"
ALLOWED_ROUTES = {"/v1/chat/completions", "/v1/completions"}


def mock_pipeline_enabled() -> bool:
    """Return whether this worker is running the model-independent Hub check."""
    return os.getenv("USE_MOCK_PIPELINE", "0") == "1"


def build_server_command() -> list[str]:
    """Build the llama-server command entirely from environment settings."""
    model = os.getenv("MODEL", "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M")
    command = [
        os.getenv("LLAMA_SERVER_BIN", "/app/llama-server"),
        "--host", LLAMA_HOST,
        "--port", str(LLAMA_PORT),
    ]
    model_path = os.getenv("MODEL_PATH")
    if model_path:
        command.extend(["--model", model_path])
    else:
        command.extend([
            "--hf-repo", model.split(":", 1)[0],
            "--hf-file", os.getenv("MODEL_FILE", "Qwen3.8-27B-UD-Q4_K_M.gguf"),
        ])
    command.extend([
        "--n-gpu-layers", os.getenv("GPU_LAYERS", "999"),
        "--ctx-size", os.getenv("CONTEXT_SIZE", "16384"),
        "--parallel", os.getenv("PARALLEL", "1"),
        "--cache-type-k", os.getenv("CACHE_TYPE_K", "q8_0"),
        "--cache-type-v", os.getenv("CACHE_TYPE_V", "q8_0"),
        "--flash-attn", "on",
        "--jinja",
    ])
    if os.getenv("HF_TOKEN"):
        command.extend(["--hf-token", os.environ["HF_TOKEN"]])
    return command


def wait_until_ready(timeout: int = 1800) -> None:
    """Wait through model download/loading until llama.cpp reports healthy."""
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{LLAMA_BASE_URL}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"llama-server did not become ready: {last_error}")


def forward(route: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{LLAMA_BASE_URL}{route}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=int(os.getenv("REQUEST_TIMEOUT", "900"))) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama.cpp returned HTTP {exc.code}: {detail}") from exc


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Accept RunPod job.input and return the native OpenAI-compatible response."""
    payload = job.get("input")
    if not isinstance(payload, dict):
        raise ValueError("job.input must be a JSON object")

    if mock_pipeline_enabled():
        if payload != {"healthcheck": True}:
            raise ValueError("Smoke-test worker accepts only input.healthcheck=true")
        return {"status": "ok", "worker": "qwen3.8-27b-gguf"}

    route = payload.pop("route", None)
    if route is None:
        route = "/v1/chat/completions" if "messages" in payload else "/v1/completions"
    if route not in ALLOWED_ROUTES:
        raise ValueError(f"route must be one of {sorted(ALLOWED_ROUTES)}")
    if payload.get("stream"):
        raise ValueError("stream=true is unsupported by RunPod queue jobs; use stream=false")

    # Register the RunPod worker before the large GGUF finishes downloading.
    # The first accepted job waits here while llama.cpp becomes ready.
    wait_until_ready(int(os.getenv("STARTUP_TIMEOUT", "1800")))
    payload.setdefault("model", os.getenv("MODEL", "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"))
    return forward(route, payload)


def main() -> None:
    if mock_pipeline_enabled():
        import runpod

        runpod.serverless.start({"handler": handler})
        return

    cache_dir = os.getenv("HF_HOME", "/runpod-volume/huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    server = subprocess.Popen(build_server_command())
    try:
        import runpod

        runpod.serverless.start({"handler": handler})
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
