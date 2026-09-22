import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RunpodHubConfigTests(unittest.TestCase):
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
        self.assertEqual(config["config"]["gpuCount"], 1)


if __name__ == "__main__":
    unittest.main()
