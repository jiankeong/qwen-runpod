import json
import os
import unittest
from unittest.mock import MagicMock, patch

import handler


class HandlerTests(unittest.TestCase):
    def setUp(self):
        self.ready = patch("handler.wait_until_ready").start()
        self.addCleanup(patch.stopall)

    @patch("handler.forward")
    def test_chat_route_and_defaults(self, forward):
        forward.return_value = {"choices": [{"message": {"content": "ok"}}]}
        result = handler.handler({"input": {"messages": [{"role": "user", "content": "你好"}]}})
        route, payload = forward.call_args.args
        self.assertEqual(route, "/v1/chat/completions")
        self.assertIn("model", payload)
        self.assertEqual(result["choices"][0]["message"]["content"], "ok")
        self.ready.assert_called_once_with(1800)

    @patch("handler.forward")
    def test_completion_route(self, forward):
        forward.return_value = {"choices": [{"text": "ok"}]}
        handler.handler({"input": {"prompt": "hello"}})
        self.assertEqual(forward.call_args.args[0], "/v1/completions")

    def test_rejects_streaming(self):
        with self.assertRaisesRegex(ValueError, "stream=true"):
            handler.handler({"input": {"messages": [], "stream": True}})

    def test_rejects_unknown_route(self):
        with self.assertRaisesRegex(ValueError, "route must be"):
            handler.handler({"input": {"route": "/health", "prompt": "x"}})

    def test_server_command_uses_q4_model(self):
        with patch.dict(os.environ, {}, clear=True):
            command = handler.build_server_command()
        self.assertIn("Qwen3.8-27B-UD-Q4_K_M.gguf", command)
        self.assertIn("16384", command)


if __name__ == "__main__":
    unittest.main()
