"""Shared litellm CustomLLM for the cli-gateways services (opencode-gateway,
claude-gateway) — the LAN HTTP gateways that wrap headless CLIs behind the
agy-gateway API shape.

One parameterized class serves both providers because the gateways share one
API (``POST /run`` -> ``{ok, response, usage, durationMs}``); the thin modules
``opencode_handler.py`` and ``claudecode_handler.py`` instantiate it with their
provider name, env prefix, and default base URL.

Differences from ``agy_handler.py`` (deliberate, per the cli-gateways plan):

- The gateways are text-only. Inline base64 images raise a provider error
  (there is no ``/files`` upload path on these gateways) so the router's
  fallback fires instead of silently dropping caller content.
- No tool-calling: these gateways have no structured-output mode, so a
  request carrying ``tools`` raises a provider error (fallback to ``cloud``,
  which speaks native tool calls).
- The payload is ``{prompt, timeoutMs}`` — model selection lives in each
  gateway's ``GW_DEFAULT_MODEL`` (callers can pass ``optional_params.model``
  via ``extra_body`` to override per request).

HTTP goes through module-level ``_post_run`` / ``_apost_run`` so tests can
monkeypatch them; ``httpx`` is imported lazily inside them.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import litellm


def _provider_error(provider: str, model: Optional[str], message: str) -> Exception:
    """Build a litellm provider exception so the router's fallbacks fire."""
    exceptions = getattr(litellm, "exceptions", None)
    exc_cls = getattr(exceptions, "APIConnectionError", None) if exceptions else None
    if exc_cls is not None:
        try:
            return exc_cls(message=message, llm_provider=provider, model=model or provider)
        except TypeError:
            try:
                return exc_cls(message)
            except Exception:
                pass
    return RuntimeError(message)


# --------------------------------------------------------------------------
# HTTP boundary (monkeypatched in tests)
# --------------------------------------------------------------------------
def _post_run(base: str, token: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """POST /run synchronously. Returns parsed JSON. Raises on non-2xx."""
    import httpx

    resp = httpx.post(
        f"{base}/run",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


async def _apost_run(base: str, token: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """POST /run asynchronously. Returns parsed JSON. Raises on non-2xx."""
    import httpx

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{base}/run",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


# --------------------------------------------------------------------------
# prompt building (text-only flatten; images/tools are hard errors)
# --------------------------------------------------------------------------
class UnsupportedContentError(ValueError):
    """Caller content these gateways cannot serve (images, tools)."""


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    pieces: List[str] = []
    for part in content:
        if not isinstance(part, dict):
            pieces.append(str(part))
            continue
        ptype = part.get("type")
        if ptype == "text":
            pieces.append(part.get("text", ""))
        elif ptype == "image_url":
            image_url = part.get("image_url") or {}
            url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
            if url.startswith("data:"):
                raise UnsupportedContentError(
                    "inline images are not supported by this gateway (no upload path); "
                    "use the agy model for image analysis"
                )
            if url:
                pieces.append(f"[image: {url}]")
        elif "text" in part:
            pieces.append(part.get("text", ""))
    return "\n".join(p for p in pieces if p)


def _build_prompt(messages: List[Dict[str, Any]]) -> str:
    """Flatten messages into one prompt: system turns first, then dialogue.
    Assistant tool_calls / role:"tool" results are rendered as plain text so a
    post-tool-execution transcript still reads coherently."""
    system_parts: List[str] = []
    turns: List[str] = []
    for msg in messages or []:
        role = (msg.get("role") or "user").lower()
        text = _flatten_content(msg.get("content"))
        if role == "tool":
            name = msg.get("name") or msg.get("tool_call_id") or "tool"
            turns.append(f"Tool result ({name}): {text}")
            continue
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
        elif role == "assistant":
            turns.append(f"Assistant: {text}")
        else:
            turns.append(f"User: {text}")

    sections: List[str] = []
    if system_parts:
        sections.append("System:\n" + "\n".join(system_parts))
    if turns:
        sections.append("\n".join(turns))
    return "\n\n".join(sections)


def _map_response(model: Optional[str], body: Dict[str, Any]) -> "litellm.ModelResponse":
    text = body.get("response", "") or ""
    model_response = litellm.ModelResponse()
    message = litellm.Message(content=text, role="assistant")
    choice = litellm.Choices(index=0, message=message, finish_reason="stop")
    model_response.choices = [choice]
    if model:
        model_response.model = model
    usage = body.get("usage")
    if isinstance(usage, dict):
        try:
            model_response.usage = litellm.Usage(
                prompt_tokens=usage.get("inputTokens") or 0,
                completion_tokens=usage.get("outputTokens") or 0,
                total_tokens=usage.get("totalTokens")
                or ((usage.get("inputTokens") or 0) + (usage.get("outputTokens") or 0)),
            )
        except Exception:
            pass
    return model_response


class CliGatewayLLM(litellm.CustomLLM):
    """Routes litellm completions to one cli-gateways instance."""

    def __init__(self, provider: str, env_prefix: str, default_base: str):
        super().__init__()
        self.provider = provider
        self.env_prefix = env_prefix
        self.default_base = default_base

    def _base(self) -> str:
        return os.environ.get(f"{self.env_prefix}_BASE", self.default_base).rstrip("/")

    def _token(self) -> str:
        token = os.environ.get(f"{self.env_prefix}_TOKEN")
        if not token:
            raise ValueError(
                f"{self.env_prefix}_TOKEN is not set — the {self.provider} handler cannot "
                f"authenticate to its gateway. Set it in the litellm proxy environment (.env)."
            )
        return token

    def _prepare(self, kwargs: Dict[str, Any]):
        model = kwargs.get("model")
        messages = kwargs.get("messages") or []
        optional_params = kwargs.get("optional_params") or {}

        if optional_params.get("tools") or kwargs.get("tools"):
            raise _provider_error(
                self.provider,
                model,
                f"tool calling is not supported by the {self.provider} gateway; "
                "route tool-using requests to agy or cloud",
            )
        try:
            prompt = _build_prompt(messages)
        except UnsupportedContentError as exc:
            raise _provider_error(self.provider, model, str(exc)) from exc

        timeout = float(kwargs.get("timeout") or 120.0)
        payload: Dict[str, Any] = {"prompt": prompt, "timeoutMs": int(timeout * 1000)}
        # Per-request model override via extra_body (optional; the gateway's
        # GW_DEFAULT_MODEL rules otherwise).
        requested = optional_params.get("model")
        if isinstance(requested, str) and requested.strip():
            payload["model"] = requested
        return model, payload, timeout

    def _finish(self, model: Optional[str], body: Any) -> "litellm.ModelResponse":
        if not isinstance(body, dict) or not body.get("ok"):
            raise _provider_error(
                self.provider, model, f"{self.provider} gateway /run returned an error body: {str(body)[:200]}"
            )
        return _map_response(model, body)

    def completion(self, *args, **kwargs) -> "litellm.ModelResponse":
        model, payload, timeout = self._prepare(kwargs)
        base, token = self._base(), self._token()
        try:
            body = _post_run(base, token, payload, timeout)
        except Exception as exc:
            raise _provider_error(self.provider, model, f"{self.provider} gateway /run request failed: {exc}") from exc
        return self._finish(model, body)

    async def acompletion(self, *args, **kwargs) -> "litellm.ModelResponse":
        model, payload, timeout = self._prepare(kwargs)
        base, token = self._base(), self._token()
        try:
            body = await _apost_run(base, token, payload, timeout)
        except Exception as exc:
            raise _provider_error(self.provider, model, f"{self.provider} gateway /run request failed: {exc}") from exc
        return self._finish(model, body)
