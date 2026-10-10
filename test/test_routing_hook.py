"""Unit tests for handlers/routing_hook.py.

Runs with plain Python 3 / stdlib unittest and NO pip installs: `litellm` (and
its `integrations.custom_logger.CustomLogger`) is stubbed into sys.modules
before the hook is imported, and the controller HTTP boundary (`_apost_ensure`)
is monkeypatched so nothing hits the network.
"""

import asyncio
import copy
import os
import sys
import types
import unittest

# --- stub `litellm.integrations.custom_logger` before importing the hook ---
_fake_litellm = types.ModuleType("litellm")
_fake_integrations = types.ModuleType("litellm.integrations")
_fake_custom_logger = types.ModuleType("litellm.integrations.custom_logger")


class CustomLogger:  # no-op base
    def __init__(self, *a, **kw):
        pass


_fake_custom_logger.CustomLogger = CustomLogger
_fake_integrations.custom_logger = _fake_custom_logger
_fake_litellm.integrations = _fake_integrations

sys.modules["litellm"] = _fake_litellm
sys.modules["litellm.integrations"] = _fake_integrations
sys.modules["litellm.integrations.custom_logger"] = _fake_custom_logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handlers import routing_hook  # noqa: E402


class FakeCache:
    """Minimal DualCache stand-in: sync get_cache / set_cache over a dict."""

    def __init__(self):
        self.store = {}

    def get_cache(self, key, **kw):
        return self.store.get(key)

    def set_cache(self, key, value, ttl=None, **kw):
        self.store[key] = value


class RoutingHookTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig = routing_hook._apost_ensure

    def tearDown(self):
        routing_hook._apost_ensure = self._orig

    async def test_default_model_rewritten_to_deepseek(self):
        hook = routing_hook.RouterHook()
        data = {"model": "gpt-4", "messages": []}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
        self.assertEqual(out["model"], "deepseek")

    async def test_no_model_defaults_to_deepseek(self):
        hook = routing_hook.RouterHook()
        data = {"messages": []}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
        self.assertEqual(out["model"], "deepseek")

    async def test_caller_named_model_is_overridden_by_cost_policy(self):
        """The router picks, not the caller: naming a real backend is ignored."""
        hook = routing_hook.RouterHook()
        for name in ("opencode", "claude-code", "agy", "cloud", "gpt-4o"):
            data = {"model": name, "messages": []}
            out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
            self.assertEqual(out["model"], "deepseek", name)

    async def test_image_request_routed_to_multimodal_chain(self):
        """Images must not land on a chain that falls back into a text-only gateway."""
        hook = routing_hook.RouterHook()
        data = {"model": "opencode", "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]},
        ]}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
        self.assertEqual(out["model"], "deepseek-multimodal")

    async def test_tool_request_routed_to_multimodal_chain(self):
        hook = routing_hook.RouterHook()
        data = {"messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "f"}}]}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
        self.assertEqual(out["model"], "deepseek-multimodal")

    async def test_plain_text_content_array_stays_on_text_chain(self):
        """A content ARRAY with no image part is still plain text."""
        hook = routing_hook.RouterHook()
        data = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
        self.assertEqual(out["model"], "deepseek")

    async def test_pin_prefix_reaches_a_specific_backend(self):
        hook = routing_hook.RouterHook()
        for target in ("opencode", "agy", "claude-code"):
            data = {"model": "pin:" + target, "messages": []}
            out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
            self.assertEqual(out["model"], target, target)

    async def test_unknown_models_still_rewritten_to_deepseek(self):
        hook = routing_hook.RouterHook()
        data = {"model": "gpt-5.6", "messages": []}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
        self.assertEqual(out["model"], "deepseek")

    async def test_ensure_running_called_once_under_concurrency(self):
        counter = {"n": 0}

        async def fake_ensure(url, token, timeout):
            counter["n"] += 1
            await asyncio.sleep(0.02)  # hold the in-flight window open
            return 200

        routing_hook._apost_ensure = fake_ensure

        hook = routing_hook.RouterHook()
        cache = FakeCache()
        datas = [{"model": "pin:ollama-local"} for _ in range(3)]

        await asyncio.gather(
            *[hook.async_pre_call_hook(None, cache, d, "completion") for d in datas]
        )

        self.assertEqual(counter["n"], 1)  # exactly one ensure-running POST
        for d in datas:
            self.assertEqual(d["model"], "ollama-local")  # stayed local, ollama is up

    async def test_ensure_running_timeout_falls_back_to_default(self):
        async def fake_ensure(url, token, timeout):
            raise asyncio.TimeoutError()

        routing_hook._apost_ensure = fake_ensure

        hook = routing_hook.RouterHook()
        data = {"model": "pin:ollama-local"}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
        self.assertEqual(out["model"], "deepseek")

    async def test_non_200_falls_back_to_default(self):
        async def fake_ensure(url, token, timeout):
            return 503

        routing_hook._apost_ensure = fake_ensure

        hook = routing_hook.RouterHook()
        data = {"model": "pin:ollama-local"}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
        self.assertEqual(out["model"], "deepseek")

    async def test_embeddings_call_type_unchanged(self):
        called = {"n": 0}

        async def fake_ensure(url, token, timeout):
            called["n"] += 1
            return 200

        routing_hook._apost_ensure = fake_ensure

        hook = routing_hook.RouterHook()
        data = {"model": "text-embedding-3-small"}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "embeddings")
        self.assertEqual(out["model"], "text-embedding-3-small")  # untouched
        self.assertEqual(called["n"], 0)


SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
}


def _schema_kwargs(model="deepseek/deepseek-flash", messages=None):
    return {
        "model": model,
        "messages": messages if messages is not None else [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Invent a person."},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "p", "strict": True, "schema": SCHEMA},
        },
    }


class DeepseekStructuredOutputTest(unittest.IsolatedAsyncioTestCase):
    """DeepSeek rejects response_format json_schema (HTTP 400 "This response_format
    type is unavailable now"); the deployment hook rewrites it, only for DeepSeek."""

    async def test_json_schema_becomes_json_object_with_schema_in_prompt(self):
        hook = routing_hook.RouterHook()
        kw = _schema_kwargs()
        out = await hook.async_pre_call_deployment_hook(kw, "acompletion")
        self.assertEqual(out["response_format"], {"type": "json_object"})
        sysmsg = out["messages"][0]
        self.assertEqual(sysmsg["role"], "system")
        self.assertTrue(sysmsg["content"].startswith("You are terse."))
        self.assertIn("JSON", sysmsg["content"])
        self.assertIn('"required": ["name"]', sysmsg["content"])
        self.assertEqual(out["messages"][1], kw["messages"][1])

    async def test_original_request_is_not_mutated_for_fallbacks(self):
        hook = routing_hook.RouterHook()
        kw = _schema_kwargs()
        before = copy.deepcopy(kw)
        await hook.async_pre_call_deployment_hook(kw, "acompletion")
        self.assertEqual(kw, before)  # fallbacks (agy) still see json_schema

    async def test_no_system_message_gets_one_prepended(self):
        hook = routing_hook.RouterHook()
        kw = _schema_kwargs(messages=[{"role": "user", "content": "hi"}])
        out = await hook.async_pre_call_deployment_hook(kw, "acompletion")
        self.assertEqual([m["role"] for m in out["messages"]], ["system", "user"])
        self.assertIn("JSON Schema", out["messages"][0]["content"])

    async def test_system_message_with_content_parts(self):
        hook = routing_hook.RouterHook()
        kw = _schema_kwargs(messages=[
            {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
            {"role": "user", "content": "hi"},
        ])
        out = await hook.async_pre_call_deployment_hook(kw, "acompletion")
        parts = out["messages"][0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "Be brief."})
        self.assertIn("JSON Schema", parts[1]["text"])

    async def test_multimodal_message_is_preserved(self):
        hook = routing_hook.RouterHook()
        img = {"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}
        out = await hook.async_pre_call_deployment_hook(
            _schema_kwargs(messages=[img]), "acompletion")
        self.assertEqual(out["messages"][-1], img)

    async def test_schemaless_json_schema_still_downgrades(self):
        hook = routing_hook.RouterHook()
        kw = _schema_kwargs()
        kw["response_format"] = {"type": "json_schema"}
        out = await hook.async_pre_call_deployment_hook(kw, "acompletion")
        self.assertEqual(out["response_format"], {"type": "json_object"})
        self.assertIn("JSON object", out["messages"][0]["content"])

    async def test_json_object_without_json_word_gets_instruction(self):
        hook = routing_hook.RouterHook()
        kw = {"model": "deepseek/deepseek-flash",
              "messages": [{"role": "user", "content": "Give a record"}],
              "response_format": {"type": "json_object"}}
        out = await hook.async_pre_call_deployment_hook(kw, "acompletion")
        self.assertEqual(out["response_format"], {"type": "json_object"})
        self.assertIn("JSON", out["messages"][0]["content"])

    async def test_json_object_that_mentions_json_is_untouched(self):
        hook = routing_hook.RouterHook()
        kw = {"model": "deepseek/deepseek-flash",
              "messages": [{"role": "user", "content": "Reply in Json please"}],
              "response_format": {"type": "json_object"}}
        self.assertIsNone(await hook.async_pre_call_deployment_hook(kw, "acompletion"))

    async def test_non_deepseek_deployments_keep_json_schema(self):
        hook = routing_hook.RouterHook()
        for model in ("agy/default", "opencode/default", "claudecode/default"):
            kw = _schema_kwargs(model=model)
            self.assertIsNone(
                await hook.async_pre_call_deployment_hook(kw, "acompletion"), model)

    async def test_provider_flag_also_identifies_deepseek(self):
        hook = routing_hook.RouterHook()
        kw = _schema_kwargs(model="deepseek-flash")
        kw["custom_llm_provider"] = "deepseek"
        out = await hook.async_pre_call_deployment_hook(kw, "acompletion")
        self.assertEqual(out["response_format"], {"type": "json_object"})

    async def test_plain_and_text_requests_untouched(self):
        hook = routing_hook.RouterHook()
        plain = {"model": "deepseek/deepseek-flash",
                 "messages": [{"role": "user", "content": "hi"}]}
        text = dict(plain, response_format={"type": "text"})
        self.assertIsNone(await hook.async_pre_call_deployment_hook(plain, "acompletion"))
        self.assertIsNone(await hook.async_pre_call_deployment_hook(text, "acompletion"))


if __name__ == "__main__":
    unittest.main()
