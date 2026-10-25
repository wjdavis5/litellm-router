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

    async def test_foreign_model_name_lands_on_generic(self):
        hook = routing_hook.RouterHook()
        data = {"model": "gpt-4", "messages": []}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
        self.assertEqual(out["model"], "generic")
        self.assertEqual(out["metadata"]["caller_model"], "gpt-4")

    async def test_no_model_defaults_to_generic(self):
        hook = routing_hook.RouterHook()
        data = {"messages": []}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
        self.assertEqual(out["model"], "generic")

    async def test_legacy_backend_names_map_onto_their_groups(self):
        """The single-route era's names keep working: each means the group it used to be."""
        hook = routing_hook.RouterHook()
        expect = {"opencode": "generic", "claude-code": "generic", "agy": "generic", "cloud": "generic",
                  "deepseek": "generic", "DeepSeek": "generic", "deepseek-multimodal": "vision",
                  "gemini-pro": "reasoning", "local-bonsai": "generic", "gpt-4o": "generic"}
        for name, group in expect.items():
            data = {"model": name, "messages": []}
            out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
            self.assertEqual(out["model"], group, name)

    async def test_use_case_groups_are_kept_as_named(self):
        hook = routing_hook.RouterHook()
        for group in ("generic", "generic-api", "reasoning", "code", "web", "vision"):
            data = {"model": group, "messages": [{"role": "user", "content": "hi"}]}
            out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
            self.assertEqual(out["model"], group, group)

    async def test_image_request_always_goes_to_vision(self):
        """Images must not land on a chain that falls back into a text-only gateway."""
        hook = routing_hook.RouterHook()
        for name in ("opencode", "code", "generic", "whatever"):
            data = {"model": name, "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ]},
            ]}
            out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
            self.assertEqual(out["model"], "vision", name)

    async def test_frigate_vision_is_kept_with_images_and_as_named(self):
        """Frigate's own load-balanced image group survives the images-force-vision rule."""
        hook = routing_hook.RouterHook()
        image_msg = [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}]
        for name in ("frigate-vision", "Frigate-Vision"):
            out = await hook.async_pre_call_hook(None, FakeCache(), {"model": name, "messages": list(image_msg)}, "completion")
            self.assertEqual(out["model"], "frigate-vision", name)
        out = await hook.async_pre_call_hook(None, FakeCache(), {"model": "frigate-vision", "messages": [{"role": "user", "content": "hi"}]}, "completion")
        self.assertEqual(out["model"], "frigate-vision")
        # Frigate's old name still means the shared vision group.
        out = await hook.async_pre_call_hook(None, FakeCache(), {"model": "deepseek-multimodal", "messages": list(image_msg)}, "completion")
        self.assertEqual(out["model"], "vision")

    async def test_tool_request_moves_to_a_tool_capable_group(self):
        hook = routing_hook.RouterHook()
        tools = [{"type": "function", "function": {"name": "f"}}]
        for name, group in {"generic": "generic-api", "code": "generic-api", "opencode": "generic-api",
                            None: "generic-api", "reasoning": "reasoning", "vision": "vision"}.items():
            data = {"messages": [{"role": "user", "content": "hi"}], "tools": tools}
            if name:
                data["model"] = name
            out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
            self.assertEqual(out["model"], group, name)

    async def test_plain_text_content_array_stays_generic(self):
        """A content ARRAY with no image part is still plain text."""
        hook = routing_hook.RouterHook()
        data = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
        self.assertEqual(out["model"], "generic")

    async def test_pin_prefix_reaches_a_specific_backend(self):
        hook = routing_hook.RouterHook()
        for target in ("opencode", "agy", "claude-code"):
            data = {"model": "pin:" + target, "messages": []}
            out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
            self.assertEqual(out["model"], target, target)

    async def test_unknown_models_still_rewritten_to_generic(self):
        hook = routing_hook.RouterHook()
        data = {"model": "gpt-5.6", "messages": []}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
        self.assertEqual(out["model"], "generic")

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
        self.assertEqual(out["model"], "generic")

    async def test_non_200_falls_back_to_default(self):
        async def fake_ensure(url, token, timeout):
            return 503

        routing_hook._apost_ensure = fake_ensure

        hook = routing_hook.RouterHook()
        data = {"model": "pin:ollama-local"}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
        self.assertEqual(out["model"], "generic")

    async def test_embeddings_land_on_the_embeddings_group(self):
        called = {"n": 0}

        async def fake_ensure(url, token, timeout):
            called["n"] += 1
            return 200

        routing_hook._apost_ensure = fake_ensure

        hook = routing_hook.RouterHook()
        for name, group in {"text-embedding-3-small": "embeddings", "embeddings": "embeddings",
                            "embeddings-gemini": "embeddings-gemini", None: "embeddings"}.items():
            data = {"input": "x"}
            if name:
                data["model"] = name
            out = await hook.async_pre_call_hook(None, FakeCache(), data, "aembedding")
            self.assertEqual(out["model"], group, name)
        self.assertEqual(called["n"], 0)

    async def test_transcription_and_rerank_families(self):
        hook = routing_hook.RouterHook()
        out = await hook.async_pre_call_hook(None, FakeCache(), {"model": "whisper-1"}, "transcription")
        self.assertEqual(out["model"], "transcription")
        out = await hook.async_pre_call_hook(None, FakeCache(), {"model": "anything"}, "atranscription")
        self.assertEqual(out["model"], "transcription")
        out = await hook.async_pre_call_hook(None, FakeCache(), {"model": "rerank-x"}, "rerank")
        self.assertEqual(out["model"], "rerank")

    async def test_pin_works_on_every_family(self):
        hook = routing_hook.RouterHook()
        out = await hook.async_pre_call_hook(None, FakeCache(), {"model": "pin:embeddings-gemini"}, "aembedding")
        self.assertEqual(out["model"], "embeddings-gemini")
        out = await hook.async_pre_call_hook(None, FakeCache(), {"model": "pin:transcription"}, "transcription")
        self.assertEqual(out["model"], "transcription")

    async def test_unowned_call_types_pass_through(self):
        hook = routing_hook.RouterHook()
        for call_type in ("amoderation", "aspeech", "asearch", "aresponses", "anthropic_messages"):
            data = {"model": "keep-me"}
            out = await hook.async_pre_call_hook(None, FakeCache(), data, call_type)
            self.assertEqual(out["model"], "keep-me", call_type)

    def test_router_call_record_reads_the_standard_logging_object(self):
        kwargs = {
            "standard_logging_object": {"model_group": "generic", "model_id": "generic-gemini-flash",
                                        "custom_llm_provider": "gemini", "model": "gemini-flash-latest",
                                        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                                        "response_time": 1.234},
            "litellm_params": {"metadata": {"caller_model": "gpt-4"}},
            "call_type": "acompletion",
        }
        rec = routing_hook.router_call_record(kwargs, "ok")
        self.assertEqual(rec["router_call"], "ok")
        self.assertEqual(rec["group"], "generic")
        self.assertEqual(rec["deployment"], "generic-gemini-flash")
        self.assertEqual(rec["provider"], "gemini")
        self.assertEqual(rec["caller_model"], "gpt-4")
        self.assertEqual(rec["tokens"], 15)
        self.assertEqual(rec["seconds"], 1.23)
        # never raises on garbage
        self.assertEqual(routing_hook.router_call_record(None, "error")["router_call"], "error")


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

    async def test_deepseek_behind_opencode_zen_is_also_translated(self):
        """A DeepSeek model reached through opencode's Zen/Go endpoint has the same response_format rules."""
        hook = routing_hook.RouterHook()
        kw = _schema_kwargs(model="openai/deepseek-v4.1-flash")
        kw["custom_llm_provider"] = "openai"
        kw["api_base"] = "https://opencode.ai/inference/go/openai/v1"
        out = await hook.async_pre_call_deployment_hook(kw, "acompletion")
        self.assertIsNotNone(out)
        self.assertEqual(out["response_format"], {"type": "json_object"})
        kw = _schema_kwargs(model="openai/glm-5.3-flash")
        kw["custom_llm_provider"] = "openai"
        kw["api_base"] = "https://opencode.ai/inference/go/openai/v1"
        self.assertIsNone(await hook.async_pre_call_deployment_hook(kw, "acompletion"))

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


class DeclaredCapabilitiesTests(unittest.TestCase):
    """litellm drops image parts for a DeepSeek model its map does not mark as vision-capable, and that map can
    silently be the bundled fallback (2026-10-11). The hook declares the capability itself."""

    IMAGE_REQUEST = {"model": "anything", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]}]}

    def setUp(self):
        self.calls = []
        self._saved = {k: getattr(_fake_litellm, k, None) for k in ("register_model", "model_cost")}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                if hasattr(_fake_litellm, k):
                    delattr(_fake_litellm, k)
            else:
                setattr(_fake_litellm, k, v)

    def _install(self):
        """An empty live map (the bundled-fallback state) and a recording register_model."""
        _fake_litellm.model_cost = {}

        def register_model(mapping):
            self.calls.append(copy.deepcopy(mapping))
            for key, info in mapping.items():
                _fake_litellm.model_cost.setdefault(key, {}).update(info)

        _fake_litellm.register_model = register_model

    def test_declares_vision_for_the_deepseek_model(self):
        self._install()
        self.assertFalse(routing_hook._capabilities_declared())
        self.assertTrue(routing_hook.declare_model_capabilities())
        entry = _fake_litellm.model_cost["deepseek/deepseek-flash"]
        self.assertTrue(entry["supports_vision"])
        self.assertEqual(entry["litellm_provider"], "deepseek")
        self.assertTrue(routing_hook._capabilities_declared())

    def test_never_raises_when_litellm_cannot_register(self):
        if hasattr(_fake_litellm, "register_model"):
            delattr(_fake_litellm, "register_model")
        self.assertFalse(routing_hook.declare_model_capabilities())

        def boom(mapping):
            raise RuntimeError("nope")

        _fake_litellm.register_model = boom
        self.assertFalse(routing_hook.declare_model_capabilities())

    def test_an_image_request_redeclares_a_capability_the_live_map_lost(self):
        self._install()
        hook = routing_hook.RouterHook()
        out = asyncio.run(hook.async_pre_call_hook(None, None, copy.deepcopy(self.IMAGE_REQUEST), "acompletion"))
        self.assertEqual(out["model"], routing_hook.VISION_GROUP)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(routing_hook._capabilities_declared())
        asyncio.run(hook.async_pre_call_hook(None, None, copy.deepcopy(self.IMAGE_REQUEST), "acompletion"))
        self.assertEqual(len(self.calls), 1, "already declared: must not register again on every request")

    def test_a_text_request_does_not_touch_the_map(self):
        self._install()
        hook = routing_hook.RouterHook()
        out = asyncio.run(hook.async_pre_call_hook(
            None, None, {"model": "x", "messages": [{"role": "user", "content": "hi"}]}, "acompletion"))
        self.assertEqual(out["model"], "generic")
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
