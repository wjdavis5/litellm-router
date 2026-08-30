"""Unit tests for handlers/agy_handler.py.

Runs with plain Python 3 / stdlib unittest and NO pip installs: `litellm` is
stubbed into sys.modules before the handler is imported, and the handler's HTTP
boundary (`_post_run` / `_post_file`) is monkeypatched so nothing hits the
network.
"""

import base64
import json
import os
import sys
import types
import unittest

# --- stub `litellm` before importing the handler -------------------------
_fake_litellm = types.ModuleType("litellm")


class CustomLLM:  # no-op base
    pass


class Message:
    def __init__(self, content=None, role=None, **kw):
        self.content = content
        self.role = role
        self.tool_calls = kw.get("tool_calls")


class Choices:
    def __init__(self, index=0, message=None, finish_reason=None, **kw):
        self.index = index
        self.message = message
        self.finish_reason = finish_reason


class ModelResponse:
    def __init__(self, *a, **kw):
        self.choices = []
        self.model = None
        self.usage = None


class Usage:
    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=0, **kw):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


_fake_exceptions = types.ModuleType("litellm.exceptions")


class APIConnectionError(Exception):
    def __init__(self, message=None, llm_provider=None, model=None, **kw):
        super().__init__(message)
        self.message = message
        self.llm_provider = llm_provider
        self.model = model


_fake_exceptions.APIConnectionError = APIConnectionError
_fake_litellm.CustomLLM = CustomLLM
_fake_litellm.Message = Message
_fake_litellm.Choices = Choices
_fake_litellm.ModelResponse = ModelResponse
_fake_litellm.Usage = Usage
_fake_litellm.exceptions = _fake_exceptions

sys.modules["litellm"] = _fake_litellm
sys.modules["litellm.exceptions"] = _fake_exceptions

# repo root on the path so `handlers` is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handlers import agy_handler  # noqa: E402


class AgyHandlerTest(unittest.TestCase):
    def setUp(self):
        self._saved_token = os.environ.get("AGY_TOKEN")
        os.environ["AGY_TOKEN"] = "test-token"
        os.environ.pop("AGY_BASE", None)
        # save/restore the monkeypatched boundary fns
        self._orig_post_run = agy_handler._post_run
        self._orig_post_file = agy_handler._post_file

    def tearDown(self):
        agy_handler._post_run = self._orig_post_run
        agy_handler._post_file = self._orig_post_file
        if self._saved_token is None:
            os.environ.pop("AGY_TOKEN", None)
        else:
            os.environ["AGY_TOKEN"] = self._saved_token

    def test_text_messages_map_response(self):
        seen = {}

        def fake_run(base, token, payload, timeout):
            seen["payload"] = payload
            seen["base"] = base
            seen["token"] = token
            return {"ok": True, "agy": {"status": "SUCCESS", "response": "hello from agy"}}

        agy_handler._post_run = fake_run

        resp = agy_handler.agy_llm.completion(
            model="agy",
            messages=[
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi there"},
            ],
        )

        self.assertEqual(resp.choices[0].message.content, "hello from agy")
        self.assertEqual(resp.choices[0].message.role, "assistant")
        self.assertEqual(seen["token"], "test-token")
        # prompt carries both the system and user turns
        self.assertIn("be terse", seen["payload"]["prompt"])
        self.assertIn("hi there", seen["payload"]["prompt"])
        self.assertEqual(seen["payload"]["outputFormat"], "json")
        self.assertEqual(seen["payload"]["effort"], "high")

    def test_base64_image_uploads_then_references_it(self):
        calls = {"files": 0}
        stored_path = "/mnt/agy-share/uploads/abc123.png"

        def fake_file(base, token, name, data):
            calls["files"] += 1
            calls["name"] = name
            calls["bytes"] = data
            return stored_path

        seen = {}

        def fake_run(base, token, payload, timeout):
            seen["payload"] = payload
            return {"ok": True, "agy": {"status": "SUCCESS", "response": "an image"}}

        agy_handler._post_file = fake_file
        agy_handler._post_run = fake_run

        raw = b"\x89PNG\r\n\x1a\nfake-png-bytes"
        data_url = "data:image/png;base64," + base64.b64encode(raw).decode()

        resp = agy_handler.agy_llm.completion(
            model="agy",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )

        self.assertEqual(calls["files"], 1)
        self.assertEqual(calls["bytes"], raw)
        self.assertTrue(calls["name"].endswith(".png"))
        # the /run prompt references the uploaded path
        self.assertIn(stored_path, seen["payload"]["prompt"])
        self.assertIn("what is this?", seen["payload"]["prompt"])
        self.assertEqual(resp.choices[0].message.content, "an image")

    def test_agy_500_raises_provider_exception(self):
        def fake_run(base, token, payload, timeout):
            raise RuntimeError("HTTP 500 Server Error")

        agy_handler._post_run = fake_run

        with self.assertRaises(APIConnectionError):
            agy_handler.agy_llm.completion(
                model="agy",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_ok_false_raises_provider_exception(self):
        def fake_run(base, token, payload, timeout):
            return {"ok": False, "errorKind": "agy-status"}

        agy_handler._post_run = fake_run

        with self.assertRaises(APIConnectionError):
            agy_handler.agy_llm.completion(
                model="agy",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_missing_token_raises_value_error(self):
        os.environ.pop("AGY_TOKEN", None)

        def fake_run(base, token, payload, timeout):
            raise AssertionError("_post_run should not be reached without a token")

        agy_handler._post_run = fake_run

        with self.assertRaises(ValueError):
            agy_handler.agy_llm.completion(
                model="agy",
                messages=[{"role": "user", "content": "hi"}],
            )

    # --- tool / function calling ----------------------------------------
    _TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "turn_on",
                "description": "Turn a device on",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        }
    ]

    def test_tools_request_maps_structured_tool_calls(self):
        seen = {}

        def fake_run(base, token, payload, timeout):
            seen["payload"] = payload
            return {
                "ok": True,
                "agy": {
                    "status": "SUCCESS",
                    "structured_output": {
                        "tool_calls": [{"name": "turn_on", "arguments": {"name": "kitchen light"}}],
                        "content": "",
                    },
                },
            }

        agy_handler._post_run = fake_run
        resp = agy_handler.agy_llm.completion(
            model="agy",
            messages=[{"role": "user", "content": "turn on the kitchen light"}],
            optional_params={"tools": self._TOOLS, "tool_choice": "auto"},
        )

        choice = resp.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertIsNone(choice.message.content)
        self.assertEqual(len(choice.message.tool_calls), 1)
        tc = choice.message.tool_calls[0]
        self.assertEqual(tc["type"], "function")
        self.assertEqual(tc["function"]["name"], "turn_on")
        # arguments are serialized to a JSON string (OpenAI shape)
        self.assertEqual(json.loads(tc["function"]["arguments"]), {"name": "kitchen light"})
        # a jsonSchema was sent and the tool catalogue reached the prompt
        self.assertIn("jsonSchema", seen["payload"])
        self.assertIn("turn_on", seen["payload"]["prompt"])

    def test_tools_request_without_a_call_returns_content(self):
        def fake_run(base, token, payload, timeout):
            return {
                "ok": True,
                "agy": {
                    "status": "SUCCESS",
                    "structured_output": {"tool_calls": [], "content": "It is sunny."},
                },
            }

        agy_handler._post_run = fake_run
        resp = agy_handler.agy_llm.completion(
            model="agy",
            messages=[{"role": "user", "content": "weather?"}],
            optional_params={"tools": self._TOOLS},
        )
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertEqual(resp.choices[0].message.content, "It is sunny.")
        self.assertIsNone(resp.choices[0].message.tool_calls)

    def test_tool_choice_none_disables_the_schema(self):
        seen = {}

        def fake_run(base, token, payload, timeout):
            seen["payload"] = payload
            return {"ok": True, "agy": {"status": "SUCCESS", "response": "plain answer"}}

        agy_handler._post_run = fake_run
        resp = agy_handler.agy_llm.completion(
            model="agy",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"tools": self._TOOLS, "tool_choice": "none"},
        )
        self.assertNotIn("jsonSchema", seen["payload"])
        self.assertEqual(resp.choices[0].message.content, "plain answer")

    def test_prompt_renders_prior_tool_calls_and_results(self):
        seen = {}

        def fake_run(base, token, payload, timeout):
            seen["payload"] = payload
            return {
                "ok": True,
                "agy": {"status": "SUCCESS", "structured_output": {"tool_calls": [], "content": "done"}},
            }

        agy_handler._post_run = fake_run
        agy_handler.agy_llm.completion(
            model="agy",
            messages=[
                {"role": "user", "content": "turn on the light"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "turn_on", "arguments": '{"name":"light"}'}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "name": "turn_on", "content": "Success"},
            ],
            optional_params={"tools": self._TOOLS},
        )
        prompt = seen["payload"]["prompt"]
        self.assertIn("Assistant called: turn_on", prompt)
        self.assertIn("Tool result (turn_on): Success", prompt)

    def test_response_format_json_schema_returns_structured_content(self):
        seen = {}

        def fake_run(base, token, payload, timeout):
            seen["payload"] = payload
            return {
                "ok": True,
                "agy": {"status": "SUCCESS", "structured_output": {"summary": "a cat", "count": 1}},
            }

        agy_handler._post_run = fake_run
        resp = agy_handler.agy_llm.completion(
            model="agy",
            messages=[{"role": "user", "content": "describe the scene"}],
            optional_params={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "scene",
                        "schema": {
                            "type": "object",
                            "properties": {"summary": {"type": "string"}, "count": {"type": "integer"}},
                        },
                    },
                }
            },
        )
        # the schema was forwarded to agy as jsonSchema ...
        self.assertIn("jsonSchema", seen["payload"])
        self.assertIn("summary", seen["payload"]["jsonSchema"])
        # ... and agy's structured_output is returned as JSON-string content
        self.assertEqual(json.loads(resp.choices[0].message.content), {"summary": "a cat", "count": 1})


if __name__ == "__main__":
    unittest.main()
