"""Unit tests for handlers/cli_gateway_handler.py (+ the two thin instances).

Stdlib-only like the sibling suites: `litellm` is stubbed into sys.modules
before import, and the HTTP boundary (`_post_run` / `_apost_run`) is
monkeypatched so nothing hits the network.
"""

import asyncio
import os
import sys
import types
import unittest

# --- stub `litellm` before importing the handler ---
_fake_litellm = types.ModuleType("litellm")


class _CustomLLM:
    def __init__(self, *a, **kw):
        pass


class _ModelResponse:
    def __init__(self):
        self.choices = []
        self.model = None
        self.usage = None


class _Message:
    def __init__(self, content=None, role="assistant", tool_calls=None):
        self.content = content
        self.role = role
        self.tool_calls = tool_calls


class _Choices:
    def __init__(self, index=0, message=None, finish_reason="stop"):
        self.index = index
        self.message = message
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _APIConnectionError(Exception):
    def __init__(self, message="", llm_provider=None, model=None):
        super().__init__(message)
        self.message = message
        self.llm_provider = llm_provider
        self.model = model


_fake_litellm.CustomLLM = _CustomLLM
_fake_litellm.ModelResponse = _ModelResponse
_fake_litellm.Message = _Message
_fake_litellm.Choices = _Choices
_fake_litellm.Usage = _Usage
_fake_litellm.exceptions = types.SimpleNamespace(APIConnectionError=_APIConnectionError)

sys.modules["litellm"] = _fake_litellm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handlers import cli_gateway_handler as cgh  # noqa: E402
from handlers.opencode_handler import opencode_llm  # noqa: E402
from handlers.claudecode_handler import claudecode_llm  # noqa: E402


GATEWAY_OK = {
    "ok": True,
    "response": "hello from gateway",
    "usage": {"inputTokens": 12, "outputTokens": 5, "totalTokens": 17},
    "durationMs": 900,
}


class CliGatewayHandlerTest(unittest.TestCase):
    def setUp(self):
        self._post = cgh._post_run
        os.environ["OPENCODE_TOKEN"] = "tok-opencode"
        os.environ["CLAUDECODE_TOKEN"] = "tok-claude"

    def tearDown(self):
        cgh._post_run = self._post
        os.environ.pop("OPENCODE_TOKEN", None)
        os.environ.pop("CLAUDECODE_TOKEN", None)
        os.environ.pop("OPENCODE_BASE", None)

    def test_happy_path_flattens_messages_and_maps_response(self):
        seen = {}

        def fake_post(base, token, payload, timeout):
            seen.update(base=base, token=token, payload=payload, timeout=timeout)
            return dict(GATEWAY_OK)

        cgh._post_run = fake_post
        resp = opencode_llm.completion(
            model="opencode/default",
            messages=[
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": [{"type": "text", "text": "again"}]},
            ],
            timeout=30,
        )
        self.assertEqual(seen["base"], "http://192.168.0.93:8100")
        self.assertEqual(seen["token"], "tok-opencode")
        self.assertEqual(seen["payload"]["timeoutMs"], 30000)
        prompt = seen["payload"]["prompt"]
        self.assertIn("System:\nBe terse.", prompt)
        self.assertIn("User: hi", prompt)
        self.assertIn("Assistant: hello", prompt)
        self.assertTrue(prompt.endswith("User: again"))
        self.assertEqual(resp.choices[0].message.content, "hello from gateway")
        self.assertEqual(resp.usage.prompt_tokens, 12)
        self.assertEqual(resp.usage.total_tokens, 17)

    def test_base_env_override(self):
        os.environ["OPENCODE_BASE"] = "http://127.0.0.1:9999/"
        seen = {}

        def fake_post(base, token, payload, timeout):
            seen["base"] = base
            return dict(GATEWAY_OK)

        cgh._post_run = fake_post
        opencode_llm.completion(model="m", messages=[{"role": "user", "content": "x"}])
        self.assertEqual(seen["base"], "http://127.0.0.1:9999")

    def test_missing_token_raises_valueerror(self):
        os.environ.pop("CLAUDECODE_TOKEN", None)
        with self.assertRaises(ValueError):
            claudecode_llm.completion(model="m", messages=[{"role": "user", "content": "x"}])

    def test_gateway_failure_raises_provider_error_for_fallback(self):
        def fake_post(base, token, payload, timeout):
            raise RuntimeError("connect timeout")

        cgh._post_run = fake_post
        with self.assertRaises(_APIConnectionError):
            opencode_llm.completion(model="m", messages=[{"role": "user", "content": "x"}])

    def test_error_body_raises_provider_error(self):
        cgh._post_run = lambda *a, **kw: {"ok": False, "errorKind": "cli-status", "message": "boom"}
        with self.assertRaises(_APIConnectionError):
            opencode_llm.completion(model="m", messages=[{"role": "user", "content": "x"}])

    def test_inline_image_raises_clear_provider_error(self):
        cgh._post_run = lambda *a, **kw: dict(GATEWAY_OK)
        with self.assertRaises(_APIConnectionError) as ctx:
            opencode_llm.completion(
                model="m",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "look"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                        ],
                    }
                ],
            )
        self.assertIn("images are not supported", str(ctx.exception))

    def test_remote_image_url_passes_as_text(self):
        seen = {}

        def fake_post(base, token, payload, timeout):
            seen["prompt"] = payload["prompt"]
            return dict(GATEWAY_OK)

        cgh._post_run = fake_post
        opencode_llm.completion(
            model="m",
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}],
        )
        self.assertIn("[image: https://x/y.png]", seen["prompt"])

    def test_tools_raise_provider_error(self):
        cgh._post_run = lambda *a, **kw: dict(GATEWAY_OK)
        with self.assertRaises(_APIConnectionError) as ctx:
            claudecode_llm.completion(
                model="m",
                messages=[{"role": "user", "content": "x"}],
                optional_params={"tools": [{"function": {"name": "t"}}]},
            )
        self.assertIn("tool calling is not supported", str(ctx.exception))

    def test_per_request_model_override_via_optional_params(self):
        seen = {}

        def fake_post(base, token, payload, timeout):
            seen["payload"] = payload
            return dict(GATEWAY_OK)

        cgh._post_run = fake_post
        opencode_llm.completion(
            model="m",
            messages=[{"role": "user", "content": "x"}],
            optional_params={"model": "opencode/mimo-v2.5-free"},
        )
        self.assertEqual(seen["payload"]["model"], "opencode/mimo-v2.5-free")

    def test_async_path_uses_apost_run(self):
        orig = cgh._apost_run
        seen = {}

        async def fake_apost(base, token, payload, timeout):
            seen["payload"] = payload
            return dict(GATEWAY_OK)

        cgh._apost_run = fake_apost
        try:
            resp = asyncio.run(
                claudecode_llm.acompletion(model="m", messages=[{"role": "user", "content": "async hi"}])
            )
        finally:
            cgh._apost_run = orig
        self.assertEqual(resp.choices[0].message.content, "hello from gateway")
        self.assertIn("async hi", seen["payload"]["prompt"])


if __name__ == "__main__":
    unittest.main()
