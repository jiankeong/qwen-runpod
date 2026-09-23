import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RunpodHubConfigTests(unittest.TestCase):
    def test_dockerfile_uses_resolvable_official_llama_image(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        first_line = dockerfile.splitlines()[0]
        self.assertTrue(first_line.startswith("FROM ghcr.io/ggml-org/llama.cpp:full-cuda@sha256:"))
        self.assertRegex(first_line, re.compile(r"@sha256:[0-9a-f]{64}$"))
        self.assertNotIn("ghcr.io/ggerganov/", dockerfile)
        self.assertIn("test -f /app/libllama-server-impl.so", dockerfile)
        self.assertIn("/etc/ld.so.conf.d/llama-cpp.conf", dockerfile)
        self.assertIn("/app/llama-server --version", dockerfile)
        self.assertIn('LLAMA_CACHE="/runpod-volume/huggingface/hub"', dockerfile)

    def test_hub_configuration(self):
        config = json.loads((ROOT / ".runpod" / "hub.json").read_text())
        self.assertEqual(config["type"], "serverless")
        self.assertEqual(config["category"], "language")
        self.assertEqual(config["config"]["runsOn"], "GPU")
        self.assertGreaterEqual(config["config"]["containerDiskInGb"], 30)
        env = {item["key"]: item.get("value") for item in config["config"]["env"]}
        self.assertEqual(env["LLAMA_CACHE"], "/runpod-volume/huggingface/hub")
        self.assertIn("ADA_24", config["config"]["gpuIds"])

    def test_hub_smoke_test_matches_handler_input(self):
        config = json.loads((ROOT / ".runpod" / "tests.json").read_text())
        smoke_test = config["tests"][0]
        self.assertEqual(smoke_test["input"], {"healthcheck": True})
        self.assertEqual(smoke_test["timeout"], 30_000)
        self.assertEqual(config["config"]["gpuTypeId"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(config["config"]["gpuCount"], 1)
        env = {item["key"]: item["value"] for item in config["config"]["env"]}
        self.assertEqual(env["USE_MOCK_PIPELINE"], "1")
        hub = json.loads((ROOT / ".runpod" / "hub.json").read_text())
        self.assertTrue(set(config["config"]["allowedCudaVersions"]) <= set(hub["config"]["allowedCudaVersions"]))


if __name__ == "__main__":
    unittest.main()
