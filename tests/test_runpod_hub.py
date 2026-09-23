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

    def test_hub_configuration(self):
        config = json.loads((ROOT / ".runpod" / "hub.json").read_text())
        self.assertEqual(config["type"], "serverless")
        self.assertEqual(config["category"], "language")
        self.assertEqual(config["config"]["runsOn"], "GPU")
        self.assertGreaterEqual(config["config"]["containerDiskInGb"], 30)
        self.assertIn("ADA_24", config["config"]["gpuIds"])

    def test_hub_smoke_test_matches_handler_input(self):
        config = json.loads((ROOT / ".runpod" / "tests.json").read_text())
        smoke_test = config["tests"][0]
        self.assertIn("messages", smoke_test["input"])
        self.assertFalse(smoke_test["input"]["stream"])
        self.assertGreaterEqual(smoke_test["timeout"], 1_800_000)
        self.assertEqual(config["config"]["gpuTypeId"], "NVIDIA A40")
        self.assertEqual(config["config"]["gpuCount"], 1)
        env = {item["key"]: item["value"] for item in config["config"]["env"]}
        self.assertEqual(env["CONTEXT_SIZE"], "4096")
        self.assertEqual(env["MODEL_FILE"], "Qwen3.8-27B-UD-Q2_K_XL.gguf")


if __name__ == "__main__":
    unittest.main()
