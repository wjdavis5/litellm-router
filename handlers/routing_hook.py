"""litellm CustomLogger that steers routing before each call.

Default behaviour: every chat/completion request is redirected to the ``agy``
model unless the caller explicitly asked for a local ollama target. When an
ollama target is requested, the hook asks the DEPLOYED controller
(``POST {CONTROLLER_URL}/api/services/ollama/ensure-running``) to spin ollama up
on demand, waits for readiness, and on timeout/failure rewrites the request to
the ``cloud`` fallback model.

Concurrency is coalesced: N simultaneous ollama requests trigger at most ONE
ensure-running POST. The in-flight ensure is tracked as a shared awaitable
stored in the passed DualCache under a short-lived key, guarded by an asyncio
lock, so every concurrent caller awaits the same result.

The controller HTTP call goes through the module-level ``_apost_ensure``
function so tests can monkeypatch it and never touch the network. ``httpx`` is
imported lazily inside it so this module imports cleanly without it installed.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

from litellm.integrations.custom_logger import CustomLogger


DEFAULT_CONTROLLER_URL = "http://192.168.0.34:8765"
ENSURING_KEY = "ollama-ensuring"

# call types this hook acts on; anything else (embeddings, moderation, ...) is
# passed through untouched.
_COMPLETION_CALL_TYPES = {
    "completion",
    "acompletion",
    "text_completion",
    "atext_completion",
}

# explicit ollama routes we leave alone (they trigger ensure-running instead of
# the agy redirect).
_OLLAMA_MODELS = {"ollama-local"}


def _controller_url() -> str:
    return os.environ.get("CONTROLLER_URL", DEFAULT_CONTROLLER_URL).rstrip("/")


def _controller_token() -> Optional[str]:
    return os.environ.get("CONTROLLER_TOKEN")


def _is_ollama_target(model: Any) -> bool:
    if not isinstance(model, str):
        return False
    return model in _OLLAMA_MODELS or model.startswith("ollama")


async def _apost_ensure(url: str, token: Optional[str], timeout: float) -> int:
    """POST the controller's ensure-running endpoint. Returns HTTP status code.

    Raises on timeout / transport error (caller treats that as "not ready").
    """
    import httpx

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{url}/api/services/ollama/ensure-running",
            headers=headers,
        )
        return resp.status_code


def _cache_get(cache: Any, key: str) -> Any:
    if cache is None:
        return None
    getter = getattr(cache, "get_cache", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:
            return None
    return None


def _cache_set(cache: Any, key: str, value: Any, ttl: int) -> None:
    if cache is None:
        return
    setter = getattr(cache, "set_cache", None)
    if callable(setter):
        try:
            setter(key, value, ttl=ttl)
        except TypeError:
            try:
                setter(key, value)
            except Exception:
                pass
        except Exception:
            pass


class RouterHook(CustomLogger):
    """Pre-call routing steering with on-demand ollama warm-up."""

    def __init__(self, ensure_timeout: float = 60.0, ensure_ttl: int = 30):
        super().__init__()
        self.ensure_timeout = ensure_timeout
        self.ensure_ttl = ensure_ttl
        self._lock = asyncio.Lock()

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        # only steer chat/text completions; leave embeddings/etc untouched
        if call_type not in _COMPLETION_CALL_TYPES:
            return data

        model = data.get("model")

        if not _is_ollama_target(model):
            # default: send everything to agy unless already agy
            if model != "agy":
                data["model"] = "agy"
            return data

        # explicit ollama target -> ensure ollama is up on demand
        ready = await self._ensure_ollama(cache)
        if not ready:
            # fall back to cloud when warm-up failed or timed out
            data["model"] = "cloud"
        return data

    async def _ensure_ollama(self, cache) -> bool:
        """Coalesced ensure-running: at most one POST across concurrent callers."""
        async with self._lock:
            inflight = _cache_get(cache, ENSURING_KEY)
            if inflight is None:
                inflight = asyncio.ensure_future(self._do_ensure())
                _cache_set(cache, ENSURING_KEY, inflight, ttl=self.ensure_ttl)
        try:
            return await inflight
        except Exception:
            return False

    async def _do_ensure(self) -> bool:
        url = _controller_url()
        token = _controller_token()
        try:
            status = await asyncio.wait_for(
                _apost_ensure(url, token, self.ensure_timeout),
                timeout=self.ensure_timeout,
            )
        except (asyncio.TimeoutError, Exception):
            return False
        return status == 200


# instance registered by config.yaml litellm_settings.callbacks
hook_instance = RouterHook()
