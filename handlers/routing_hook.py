"""litellm CustomLogger that steers routing before each call.

**The router picks the backend, the caller names the USE CASE** (2026-10-20;
before that, 2026-09-19 to 2026-10-20, every request went to one DeepSeek
route). Callers send a use-case group as the model name and litellm spreads
that group's traffic across every subscription the lab pays for, by weight,
with cooldowns and fallbacks (``config.yaml``). The groups:

- ``generic``       plain text chat: DeepSeek API, Gemini Flash, the opencode
                    gateway's Go + Z.ai coding-plan models, ...
- ``generic-api``   the API-only, tool-capable members of ``generic`` (the CLI
                    gateways cannot take ``tools``); tool requests land here
- ``vision``        requests carrying images (DeepSeek, Gemini; agy as fallback)
- ``reasoning``     harder problems: the same Flash pair today (Will: the Pro
                    models are older and worse), Go's models when enabled
- ``code``          code generation: the coding subscriptions first, API spill-over
- ``web``           web-grounded answers (Perplexity)
- ``embeddings``    ONE vector space per group name (never load-shared across
                    models: vectors from different models are incomparable)
- ``transcription`` the lab whisper server (no cloud fallback: audio stays home)
- ``rerank``        reserved

A caller that names something else gets the family default (``generic`` for
chat, ``embeddings`` for embeddings, ``transcription`` for audio), except that
the legacy names from the single-route era (``deepseek``, ``opencode``,
``agy``, ...) map onto the group they used to mean (``CALLER_ALIASES``), so no
consumer had to change on the day the groups appeared. A chat request that
carries images goes to ``vision`` whatever it named; one that carries ``tools``
goes to ``generic-api`` unless it named a group that can take them.

The operator escape hatch ``pin:<deployment>`` still reaches one deployment
directly, for smoke-testing or A/B comparison, on every call type.
``pin:ollama-local`` additionally asks the DEPLOYED controller
(``POST {CONTROLLER_URL}/api/services/ollama/ensure-running``) to spin ollama up
on demand, falling back to ``generic`` if warm-up fails (``ollama-local`` is
commented out of ``config.yaml``, so that pin errors until it is re-enabled).

**Structured output on DeepSeek** (2026-10-06): the DeepSeek API accepts only
``response_format`` ``text`` and ``json_object``; ``json_schema`` is rejected
with HTTP 400, and a ``json_object`` request whose prompt never says "json" is
rejected too. ``async_pre_call_deployment_hook`` rewrites such a request, per
deployment and only when the deployment is DeepSeek-hosted (the API, or a
DeepSeek model reached through opencode's Zen/Go endpoint), into
``json_object`` plus the schema stated in the prompt. Fallback deployments keep
the original ``json_schema`` (``agy`` turns it into ``--json-schema``).

**Which subscription served what:** ``async_log_success_event`` /
``async_log_failure_event`` print one JSON line per request (``router_call``)
with the group, the deployment id, the provider, tokens and latency. The pod's
stdout is shipped to Loki by alloy-pods, so Grafana can count calls per
deployment with LogQL; there is no Prometheus in the lab.

Concurrency of the ollama warm-up is coalesced: N simultaneous requests
trigger at most ONE ensure-running POST (a shared awaitable in the passed
DualCache under a short-lived key, guarded by an asyncio lock). The controller
HTTP call goes through the module-level ``_apost_ensure`` so tests can
monkeypatch it; ``httpx`` is imported lazily inside it.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Optional

from litellm.integrations.custom_logger import CustomLogger


DEFAULT_CONTROLLER_URL = "http://192.168.0.34:8765"
ENSURING_KEY = "ollama-ensuring"

# Call-type families this hook steers. litellm's proxy names them as in
# ``litellm/types/utils.py`` (``CallTypesLiteral``); the transcription route
# arrives as the sync name. Anything else (moderation, speech, search,
# responses, messages) passes through untouched.
_COMPLETION_CALL_TYPES = {
    "completion",
    "acompletion",
    "text_completion",
    "atext_completion",
}
_EMBED_CALL_TYPES = {"embedding", "aembedding"}
_TRANSCRIBE_CALL_TYPES = {"transcription", "atranscription"}
_RERANK_CALL_TYPES = {"rerank", "arerank"}

# ollama routes, reachable only via the pin escape hatch now; naming one still
# triggers on-demand warm-up before the request is served.
_OLLAMA_MODELS = {"ollama-local"}

# Use-case groups, by family. A caller that names one of these keeps it.
TEXT_GROUPS = {"generic", "generic-api", "reasoning", "code", "web"}
VISION_GROUP = "vision"
CHAT_GROUPS = TEXT_GROUPS | {VISION_GROUP}
EMBED_GROUPS = {"embeddings", "embeddings-gemini", "embeddings-local"}
STT_GROUPS = {"transcription"}
RERANK_GROUPS = {"rerank"}

# Groups whose members all take OpenAI ``tools``; a tool request that named
# another text group is moved to ``generic-api`` (the CLI gateways raise on tools).
TOOL_CAPABLE_GROUPS = {"generic-api", "reasoning", VISION_GROUP}

DEFAULT_BY_FAMILY = {
    "chat": "generic",
    "embed": "embeddings",
    "stt": "transcription",
    "rerank": "rerank",
}

# Legacy names from the single-route era (and common foreign names), mapped onto
# the group they meant. Compared case-insensitively.
CALLER_ALIASES = {
    "deepseek": "generic",
    "deepseek-multimodal": VISION_GROUP,
    "deepseek-chat": "generic",
    "deepseek-reasoner": "reasoning",
    "deepseek-pro": "reasoning",
    "opencode": "generic",
    "claude-code": "generic",
    "agy": "generic",
    "cloud": "generic",
    "gemini": "generic",
    "gemini-pro": "reasoning",
    "local-bonsai": "generic",
    "ollama-local": "generic",
    "any": "generic",
    "default": "generic",
    "auto": "generic",
    "chat": "generic",
    "text": "generic",
    "embed": "embeddings",
    "whisper": "transcription",
    "stt": "transcription",
}

# Operator escape hatch. "pin:opencode" -> opencode, bypassing the policy.
PIN_PREFIX = "pin:"

# Capabilities this router DEPENDS ON, declared here instead of trusted to litellm's downloaded model map.
#
# litellm's DeepSeek transform forwards image parts only when ``supports_vision("deepseek-flash", "deepseek")`` is
# true; otherwise it collapses the message to its text and the image is silently dropped. It reads that capability
# from a map fetched from GitHub at process start, and when the fetch fails it falls back to a map bundled in the
# package that has NO ``deepseek-flash`` entry. 2026-10-11: the pod started before DNS was up after a host reboot
# ("Failed to fetch remote model cost map ... Falling back to local backup"), and for 32 hours every image request
# reached DeepSeek as text only. imageAnalysis stored "No image was provided" and invented descriptions for every
# camera frame and stamped real people "Not a person". Nothing errored, so nothing fell back and nothing alerted.
REQUIRED_MODEL_INFO: dict[str, dict[str, Any]] = {
    "deepseek/deepseek-flash": {
        "litellm_provider": "deepseek",
        "mode": "chat",
        "supports_vision": True,
        "supports_function_calling": True,
    },
}


def _capabilities_declared() -> bool:
    """True when litellm's live map already says every required capability (dict lookups only)."""
    try:
        import litellm

        cost = getattr(litellm, "model_cost", None) or {}
        return all(
            all(cost.get(key, {}).get(field) == value for field, value in info.items())
            for key, info in REQUIRED_MODEL_INFO.items()
        )
    except Exception:
        return False


def declare_model_capabilities() -> bool:
    """Register ``REQUIRED_MODEL_INFO`` with litellm. Returns True when litellm accepted it.

    Called at import (proxy start) and again from the pre-call hook if the live map has lost an entry, so a failed or
    later-reloaded model map can never turn the vision route blind again. Never raises: a router that cannot
    declare a capability must still serve text.
    """
    try:
        import litellm

        register = getattr(litellm, "register_model", None)
        if register is None:
            return False
        register({key: dict(info) for key, info in REQUIRED_MODEL_INFO.items()})
        return True
    except Exception:
        return False


declare_model_capabilities()


def _controller_url() -> str:
    return os.environ.get("CONTROLLER_URL", DEFAULT_CONTROLLER_URL).rstrip("/")


def _controller_token() -> Optional[str]:
    return os.environ.get("CONTROLLER_TOKEN")


def _is_ollama_target(model: Any) -> bool:
    if not isinstance(model, str):
        return False
    return model in _OLLAMA_MODELS or model.startswith("ollama")


def _family(call_type: Any) -> Optional[str]:
    if call_type in _COMPLETION_CALL_TYPES:
        return "chat"
    if call_type in _EMBED_CALL_TYPES:
        return "embed"
    if call_type in _TRANSCRIBE_CALL_TYPES:
        return "stt"
    if call_type in _RERANK_CALL_TYPES:
        return "rerank"
    return None


_GROUPS_BY_FAMILY = {
    "chat": CHAT_GROUPS,
    "embed": EMBED_GROUPS,
    "stt": STT_GROUPS,
    "rerank": RERANK_GROUPS,
}


def _has_image_part(content: Any) -> bool:
    """True if an OpenAI content array carries an ``image_url`` part."""
    if not isinstance(content, (list, tuple)):
        return False
    return any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for part in content
    )


def _has_images(data: Any) -> bool:
    """True if any message carries an inline image."""
    if not isinstance(data, dict):
        return False
    messages = data.get("messages")
    if not isinstance(messages, (list, tuple)):
        return False
    return any(
        isinstance(msg, dict) and _has_image_part(msg.get("content"))
        for msg in messages
    )


def _has_tools(data: Any) -> bool:
    """True if the request asks for tool / function calling."""
    return isinstance(data, dict) and bool(data.get("tools") or data.get("functions"))


def _needs_rich_capabilities(data: Any) -> bool:
    """Images or tools: exactly what the text-only CLI gateways raise on.

    Kept deliberately conservative: when in doubt we do NOT claim a request is
    text-only, because guessing wrong sends an image into a backend that cannot
    serve it.
    """
    return _has_images(data) or _has_tools(data)


def resolve_group(name: Any, family: str, data: Any = None) -> str:
    """The use-case group a request belongs to.

    1. a group name of this family is kept as named;
    2. a legacy alias maps to the group it meant;
    3. anything else gets the family default;
    then, for chat only, images force ``vision`` and tools force a
    tool-capable group (``generic-api`` unless the named group takes tools).
    """
    groups = _GROUPS_BY_FAMILY[family]
    key = name.strip().lower() if isinstance(name, str) else ""
    if key in groups:
        group = key
    elif key in CALLER_ALIASES and CALLER_ALIASES[key] in groups:
        group = CALLER_ALIASES[key]
    else:
        group = DEFAULT_BY_FAMILY[family]
    if family == "chat":
        if _has_images(data):
            group = VISION_GROUP
        elif _has_tools(data) and group not in TOOL_CAPABLE_GROUPS:
            group = "generic-api"
    return group


# ---- DeepSeek structured-output translation -------------------------------

def _is_deepseek_deployment(kwargs: Any) -> bool:
    """True if this per-deployment call is served by a DeepSeek model: the
    DeepSeek API itself, or a DeepSeek model reached through opencode's Zen/Go
    OpenAI-compatible endpoint (same upstream rules on ``response_format``)."""
    if not isinstance(kwargs, dict):
        return False
    model = kwargs.get("model")
    if kwargs.get("custom_llm_provider") == "deepseek":
        return True
    if isinstance(model, str) and model.startswith("deepseek/"):
        return True
    api_base = kwargs.get("api_base") or (kwargs.get("litellm_params") or {}).get("api_base") or ""
    return (
        isinstance(model, str)
        and "deepseek" in model.lower()
        and isinstance(api_base, str)
        and "opencode.ai" in api_base
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


# ---- per-request accounting line (Loki) ------------------------------------

def _dig(obj: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            obj = getattr(obj, key, None)
        if obj is None:
            return None
    return obj


def router_call_record(kwargs: Any, status: str, start_time: Any = None, end_time: Any = None) -> dict:
    """One flat dict per served request: which group, which deployment, which
    provider, how many tokens, how long. Every field is best-effort; the line is
    never allowed to raise into litellm."""
    slo = _dig(kwargs, "standard_logging_object") or {}
    lp = _dig(kwargs, "litellm_params") or {}
    meta = _dig(lp, "metadata") or {}
    model_info = _dig(lp, "model_info") or {}
    rec = {
        "router_call": status,
        "group": slo.get("model_group") or meta.get("model_group"),
        "deployment": slo.get("model_id") or model_info.get("id") or meta.get("model_id"),
        "provider": slo.get("custom_llm_provider") or _dig(kwargs, "custom_llm_provider"),
        "model": slo.get("model") or _dig(kwargs, "model"),
        "api_base": slo.get("api_base") or lp.get("api_base"),
        "caller_model": meta.get("caller_model") or slo.get("requested_model"),
        "call_type": _dig(kwargs, "call_type"),
        "tokens_in": slo.get("prompt_tokens"),
        "tokens_out": slo.get("completion_tokens"),
        "tokens": slo.get("total_tokens"),
    }
    try:
        if start_time is not None and end_time is not None:
            rec["seconds"] = round((end_time - start_time).total_seconds(), 2)
        elif slo.get("response_time") is not None:
            rec["seconds"] = round(float(slo["response_time"]), 2)
    except Exception:
        pass
    return {k: v for k, v in rec.items() if v is not None}


class RouterHook(CustomLogger):
    """Pre-call routing steering, DeepSeek structured-output translation,
    on-demand ollama warm-up, and the per-request accounting line."""

    def __init__(self, ensure_timeout: float = 60.0, ensure_ttl: int = 30):
        super().__init__()
        self.ensure_timeout = ensure_timeout
        self.ensure_ttl = ensure_ttl
        self._lock = asyncio.Lock()

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        family = _family(call_type)
        if family is None or not isinstance(data, dict):
            return data

        model = data.get("model")

        # Operator escape hatch: "pin:<deployment>" bypasses the policy and
        # reaches one specific deployment, on every family. For smoke-testing a
        # backend or A/B-comparing models; no consumer uses it in normal operation.
        if isinstance(model, str) and model.startswith(PIN_PREFIX):
            target = model[len(PIN_PREFIX):].strip()
            data["model"] = target
            if _is_ollama_target(target) and not await self._ensure_ollama(cache):
                # warm-up failed/timed out -> serve it rather than erroring.
                data["model"] = DEFAULT_BY_FAMILY[family]
            return data

        group = resolve_group(model, family, data)
        data["model"] = group
        # Remember what the caller asked for, for the accounting line.
        try:
            meta = data.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
                data["metadata"] = meta
            if isinstance(model, str) and model:
                meta.setdefault("caller_model", model)
        except Exception:
            pass
        # Self-heal: if litellm's live map no longer says DeepSeek sees images (see REQUIRED_MODEL_INFO), say so
        # again before this request is transformed, instead of sending a camera frame as text.
        if group == VISION_GROUP and not _capabilities_declared():
            declare_model_capabilities()
        return data

    async def async_pre_call_deployment_hook(self, kwargs, call_type):
        """Per-deployment rewrite: DeepSeek takes no ``json_schema`` (see the
        module docstring). Runs after the router picked a deployment, so
        fallbacks to the CLI gateways / agy still see the caller's original
        ``response_format``. Returning None leaves the request untouched."""
        if not _is_deepseek_deployment(kwargs):
            return None
        return translate_response_format_for_deepseek(kwargs)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._emit(router_call_record(kwargs, "ok", start_time, end_time))

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        rec = router_call_record(kwargs, "error", start_time, end_time)
        exc = _dig(kwargs, "exception")
        if exc is not None:
            rec["error"] = str(exc)[:200]
        self._emit(rec)

    @staticmethod
    def _emit(rec: dict) -> None:
        try:
            print(json.dumps(rec, ensure_ascii=False, default=str), flush=True)
        except Exception:
            pass

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
