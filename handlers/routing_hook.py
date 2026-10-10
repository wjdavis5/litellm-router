"""litellm CustomLogger that steers routing before each call.

**The router picks the model, not the caller** (2026-09-19). Every completion
request is rewritten to the cheapest backend that has the capability the request
actually needs. Callers describe *what they need*; cost policy lives here, in
one place, instead of being re-litigated in every consumer's config.

- plain text            -> ``deepseek``             (DeepSeek-V4.1-Flash)
- inline images / tools -> ``deepseek-multimodal``  (same upstream, different
                                                     fallback chain)

Both aliases hit the same DeepSeek API. They differ only in what they fall back
TO: the ``opencode``/``claude-code`` CLI gateways are text-only and raise on
images or tools, so a multimodal request must never fall back into them. Its
chain contains only ``agy``, which uploads images via ``POST /files`` and
supports tools through ``--json-schema``.

A caller-supplied model name is IGNORED. The one exception is the operator
escape hatch ``pin:<model>``, which reaches a specific backend directly — that
exists so a fallback backend can be smoke-tested or A/B-compared before it is
relied on. Nothing in the lab uses it in normal operation.

``pin:ollama-local`` additionally asks the DEPLOYED controller
(``POST {CONTROLLER_URL}/api/services/ollama/ensure-running``) to spin ollama up
on demand, falling back to ``deepseek`` if warm-up fails. NOTE: ``ollama-local``
is currently commented out of ``config.yaml``'s model_list, so that pin errors
until it is re-enabled. The machinery is kept because re-enabling is documented.

**Structured output on DeepSeek** (2026-10-06): the DeepSeek API accepts only
``response_format`` ``text`` and ``json_object`` � ``json_schema`` is rejected
with HTTP 400 "This response_format type is unavailable now", and a
``json_object`` request whose prompt never says "json" is rejected too. Callers
(HA's Ollama integration through ollama-shim sends ``format`` as ``json_schema``)
must not have to know that, so ``async_pre_call_deployment_hook`` rewrites such a
request, per deployment and only when the deployment is DeepSeek, into
``json_object`` plus the schema stated in the prompt. It is a DEPLOYMENT hook
rather than part of the pre-call hook on purpose: the fallback backends keep
the original ``json_schema`` (``agy`` turns it into ``--json-schema``).

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
import json
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

# ollama routes, reachable only via the pin escape hatch now; naming one still
# triggers on-demand warm-up before the request is served.
_OLLAMA_MODELS = {"ollama-local"}

# Cost policy targets. Both are the DeepSeek API; they differ only in fallback
# chain (see the module docstring and config.yaml).
DEFAULT_MODEL = "deepseek"
MULTIMODAL_MODEL = "deepseek-multimodal"

# Operator escape hatch. "pin:opencode" -> opencode, bypassing the cost policy.
PIN_PREFIX = "pin:"


def _controller_url() -> str:
    return os.environ.get("CONTROLLER_URL", DEFAULT_CONTROLLER_URL).rstrip("/")


def _controller_token() -> Optional[str]:
    return os.environ.get("CONTROLLER_TOKEN")


def _is_ollama_target(model: Any) -> bool:
    if not isinstance(model, str):
        return False
    return model in _OLLAMA_MODELS or model.startswith("ollama")


def _has_image_part(content: Any) -> bool:
    """True if an OpenAI content array carries an ``image_url`` part."""
    if not isinstance(content, (list, tuple)):
        return False
    return any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for part in content
    )


def _needs_rich_capabilities(data: Any) -> bool:
    """True if this request needs more than plain text completion.

    Images and tool/function calling are exactly the things the text-only CLI
    gateways raise on, so these requests need a fallback chain that excludes
    them. Kept deliberately conservative: when in doubt we do NOT claim a
    request is text-only, because guessing wrong sends an image into a backend
    that cannot serve it.
    """
    if not isinstance(data, dict):
        return False
    if data.get("tools") or data.get("functions"):
        return True
    messages = data.get("messages")
    if not isinstance(messages, (list, tuple)):
        return False
    return any(
        isinstance(msg, dict) and _has_image_part(msg.get("content"))
        for msg in messages
    )


# ---- DeepSeek structured-output translation -------------------------------

def _is_deepseek_deployment(kwargs: Any) -> bool:
    """True if this per-deployment call goes to the DeepSeek API."""
    if not isinstance(kwargs, dict):
        return False
    model = kwargs.get("model")
    return kwargs.get("custom_llm_provider") == "deepseek" or (
        isinstance(model, str) and model.startswith("deepseek/")
    )


def _content_text(content: Any) -> str:
    """All text of one message's content (string or OpenAI content parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def _mentions_json(messages: Any) -> bool:
    if not isinstance(messages, (list, tuple)):
        return False
    return any(
        isinstance(m, dict) and "json" in _content_text(m.get("content")).lower()
        for m in messages
    )


def _json_instruction(schema: Any) -> str:
    text = "Respond with a single valid JSON object and nothing else."
    if isinstance(schema, dict) and schema:
        text += (
            " The object must conform to this JSON Schema:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
    return text


def _with_instruction(messages: Any, instruction: str) -> list:
    """Copy of ``messages`` with ``instruction`` appended to the first system
    message, or prepended as a new system message if there is none. The caller's
    list (shared with fallback attempts) is never mutated."""
    out = [dict(m) if isinstance(m, dict) else m for m in (messages or [])]
    for i, m in enumerate(out):
        if isinstance(m, dict) and m.get("role") in ("system", "developer"):
            content = m.get("content")
            if isinstance(content, str):
                m["content"] = (content + "\n\n" + instruction) if content else instruction
            elif isinstance(content, (list, tuple)):
                m["content"] = list(content) + [{"type": "text", "text": instruction}]
            else:
                m["content"] = instruction
            return out
    return [{"role": "system", "content": instruction}] + out


def translate_response_format_for_deepseek(kwargs: dict) -> Optional[dict]:
    """Return a copy of ``kwargs`` DeepSeek will accept, or None if unchanged.

    - ``json_schema``  -> ``json_object`` + the schema stated in the prompt.
    - ``json_object`` with no mention of "json" in the prompt -> same format plus
      a one-line instruction (DeepSeek 400s without it).
    - anything else (``text``, none, unknown) is left alone.
    """
    rf = kwargs.get("response_format")
    if not isinstance(rf, dict):
        return None
    rtype = rf.get("type")
    if rtype == "json_schema":
        spec = rf.get("json_schema")
        schema = spec.get("schema") if isinstance(spec, dict) else None
        new = dict(kwargs)
        new["response_format"] = {"type": "json_object"}
        new["messages"] = _with_instruction(kwargs.get("messages"), _json_instruction(schema))
        return new
    if rtype == "json_object" and not _mentions_json(kwargs.get("messages")):
        new = dict(kwargs)
        new["messages"] = _with_instruction(kwargs.get("messages"), _json_instruction(None))
        return new
    return None


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

        # Operator escape hatch: "pin:<model>" bypasses the cost policy and
        # reaches one specific backend. For smoke-testing a fallback backend or
        # A/B-comparing models — not used by any consumer in normal operation.
        if isinstance(model, str) and model.startswith(PIN_PREFIX):
            target = model[len(PIN_PREFIX):].strip()
            data["model"] = target
            if _is_ollama_target(target) and not await self._ensure_ollama(cache):
                # warm-up failed/timed out -> serve it rather than erroring.
                # (never "cloud", which is keyless/dead — see config.yaml)
                data["model"] = DEFAULT_MODEL
            return data

        # Cost policy: the ROUTER picks the model, never the caller. Whatever
        # the caller named is discarded; the request goes to the cheapest
        # backend that has the capability it needs. Capability, not preference,
        # is the only thing that changes the answer.
        data["model"] = (
            MULTIMODAL_MODEL if _needs_rich_capabilities(data) else DEFAULT_MODEL
        )
        return data

    async def async_pre_call_deployment_hook(self, kwargs, call_type):
        """Per-deployment rewrite: DeepSeek takes no ``json_schema`` (see the
        module docstring). Runs after the router picked a deployment, so
        fallbacks to the CLI gateways / agy still see the caller's original
        ``response_format``. Returning None leaves the request untouched."""
        if not _is_deepseek_deployment(kwargs):
            return None
        return translate_response_format_for_deepseek(kwargs)

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
