"""Unit tests for handlers/routing_hook.py.

Runs with plain Python 3 / stdlib unittest and NO pip installs: `litellm` (and
its `integrations.custom_logger.CustomLogger`) is stubbed into sys.modules
before the hook is imported, and the controller HTTP boundary (`_apost_ensure`)
is monkeypatched so nothing hits the network.
"""

import asyncio
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

    async def test_default_model_rewritten_to_agy(self):
        hook = routing_hook.RouterHook()
        data = {"model": "gpt-4", "messages": []}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
        self.assertEqual(out["model"], "agy")

    async def test_no_model_defaults_to_agy(self):
        hook = routing_hook.RouterHook()
        data = {"messages": []}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
        self.assertEqual(out["model"], "agy")

    async def test_cli_gateway_models_pass_through_untouched(self):
        hook = routing_hook.RouterHook()
        for name in ("opencode", "claude-code", "agy"):
            data = {"model": name, "messages": []}
            out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
            self.assertEqual(out["model"], name, name)

    async def test_unknown_models_still_rewritten_to_agy(self):
        hook = routing_hook.RouterHook()
        data = {"model": "gpt-5.6", "messages": []}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
        self.assertEqual(out["model"], "agy")

    async def test_ensure_running_called_once_under_concurrency(self):
        counter = {"n": 0}

        async def fake_ensure(url, token, timeout):
            counter["n"] += 1
            await asyncio.sleep(0.02)  # hold the in-flight window open
            return 200

        routing_hook._apost_ensure = fake_ensure

        hook = routing_hook.RouterHook()
        cache = FakeCache()
        datas = [{"model": "ollama-local"} for _ in range(3)]

        await asyncio.gather(
            *[hook.async_pre_call_hook(None, cache, d, "completion") for d in datas]
        )

        self.assertEqual(counter["n"], 1)  # exactly one ensure-running POST
        for d in datas:
            self.assertEqual(d["model"], "ollama-local")  # stayed local, ollama is up

    async def test_ensure_running_timeout_falls_back_to_cloud(self):
        async def fake_ensure(url, token, timeout):
            raise asyncio.TimeoutError()

        routing_hook._apost_ensure = fake_ensure

        hook = routing_hook.RouterHook()
        data = {"model": "ollama-local"}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
        self.assertEqual(out["model"], "cloud")

    async def test_non_200_falls_back_to_cloud(self):
        async def fake_ensure(url, token, timeout):
            return 503

        routing_hook._apost_ensure = fake_ensure

        hook = routing_hook.RouterHook()
        data = {"model": "ollama-local"}
        out = await hook.async_pre_call_hook(None, FakeCache(), data, "completion")
        self.assertEqual(out["model"], "cloud")

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


if __name__ == "__main__":
    unittest.main()
