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

import json
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
class GatewayHTTPError(Exception):
    """A non-2xx from the gateway, WITH the body: the gateway states why it rejected a request (allow-list,
    images, size) in its JSON body, and `raise_for_status()` used to throw that away, so every 400 reached the
    router_call log as a bare "400 Bad Request" (2026-10-20, the night half of Frigate's vision requests were
    rejected and nobody could see why)."""

    def __init__(self, status: int, text: str):
        super().__init__(f"HTTP {status} from the gateway: {text}")
        self.status = status
        self.text = text


def _check_status(resp) -> None:
    status = getattr(resp, "status_code", 200)
    if status < 200 or status >= 300:
        raise GatewayHTTPError(status, (getattr(resp, "text", "") or "")[:300].replace("\n", " "))


def _post_run(base: str, token: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """POST /run synchronously. Returns parsed JSON. Raises GatewayHTTPError on non-2xx."""
    import httpx

    resp = httpx.post(
        f"{base}/run",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    _check_status(resp)
    return resp.json()


async def _apost_run(base: str, token: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """POST /run asynchronously. Returns parsed JSON. Raises GatewayHTTPError on non-2xx."""
    import httpx

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{base}/run",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
        )
        _check_status(resp)
        return resp.json()


# --------------------------------------------------------------------------
# prompt building (text-only flatten; images/tools are hard errors)
# --------------------------------------------------------------------------
class UnsupportedContentError(ValueError):
    """Caller content these gateways cannot serve (images, tools)."""


_IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp", "image/gif"}


def _parse_data_url(url: str) -> Optional[Dict[str, str]]:
    """``data:image/png;base64,....`` -> {"mime", "data"} for the gateway's ``images`` field, else None."""
    header, sep, data = url.partition(",")
    if not sep or not header.startswith("data:") or not data:
        return None
    meta = header[5:].split(";")
    mime = meta[0].strip().lower()
    if "base64" not in meta[1:] or mime not in _IMAGE_MIMES:
        return None
    return {"mime": mime, "data": "".join(data.split())}


def _flatten_content(content: Any, images: Optional[List[Dict[str, str]]] = None) -> str:
    """Flatten one message's content. Inline (data URL) images are appended to
    ``images`` when the gateway can attach them (opencode, 2026-10-20) and
    referenced in the text; a gateway without image support still raises so
    the router falls back to a backend that has it."""
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
                if images is None:
                    raise UnsupportedContentError(
                        "inline images are not supported by this gateway (no upload path); "
                        "use the agy model for image analysis"
                    )
                parsed = _parse_data_url(url)
                if parsed is None:
                    raise UnsupportedContentError(
                        "inline image is not a base64 data URL of a png/jpeg/webp/gif this gateway can attach"
                    )
                images.append(parsed)
                pieces.append("[attached image]")
            elif url:
                pieces.append(f"[image: {url}]")
        elif "text" in part:
            pieces.append(part.get("text", ""))
    return "\n".join(p for p in pieces if p)


# The gateway takes at most 24 images a request (cli-gateways/src/server.js MAX_IMAGES), and Frigate's review
# summaries send one preview frame per second of the review item with no upper cap in 0.18.1
# (review_descriptions.py: min(by_tokens, duration * 1 fps), floor 3) — a 60 s item is 60 frames, which the
# gateway rejected and litellm then re-sent up to six times (the 2026-10-20 router OOM). Frigate's own docs call
# 20 frames the intended maximum, so more than that is thinned to 20 evenly spaced frames, first and last kept.
MAX_GATEWAY_IMAGES = 20


def _subsample(items: List[Any], limit: int) -> List[Any]:
    n = len(items)
    if n <= limit:
        return items
    if limit <= 1:
        return items[:1]
    return [items[round(i * (n - 1) / (limit - 1))] for i in range(limit)]


def _json_instruction(response_format: Any) -> Optional[str]:
    """The CLIs have no ``response_format``; a JSON request gets the ask stated in the prompt instead
    (json_schema -> the schema spelled out, json_object -> a one-liner). None for text/absent."""
    if not isinstance(response_format, dict):
        return None
    rtype = response_format.get("type")
    if rtype == "json_schema":
        spec = response_format.get("json_schema")
        schema = spec.get("schema") if isinstance(spec, dict) else None
        text = "Respond with a single valid JSON object and nothing else (no markdown fences, no prose)."
        if isinstance(schema, dict) and schema:
            text += " The object must conform to this JSON Schema:\n" + json.dumps(schema, ensure_ascii=False)
        return text
    if rtype == "json_object":
        return "Respond with a single valid JSON object and nothing else (no markdown fences, no prose)."
    return None


def _strip_json_fences(text: str) -> str:
    """A reply that is exactly one fenced block whose body parses as JSON becomes the bare JSON
    (models asked for JSON through a CLI like to wrap it in ```json fences; Frigate and HA parse the body)."""
    stripped = text.strip()
    if not (stripped.startswith("```") and stripped.endswith("```")):
        return text
    inner = stripped[3:-3]
    first_nl = inner.find("\n")
    if first_nl != -1 and inner[:first_nl].strip().isalnum():
        inner = inner[first_nl + 1:]
    inner = inner.strip()
    try:
        json.loads(inner)
    except Exception:
        return text
    return inner


def _build_prompt(messages: List[Dict[str, Any]], images: Optional[List[Dict[str, str]]] = None) -> str:
    """Flatten messages into one prompt: system turns first, then dialogue.
    role:"tool" results are rendered as plain text so a post-tool-execution
    transcript still reads coherently. (These gateways reject requests that
    carry `tools`, so no assistant tool_calls turn originates here — but a
    replayed transcript can still contain tool results.)"""
    system_parts: List[str] = []
    turns: List[str] = []
    for msg in messages or []:
        role = (msg.get("role") or "user").lower()
        text = _flatten_content(msg.get("content"), images)
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
    text = _strip_json_fences(body.get("response", "") or "")
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

    def __init__(self, provider: str, env_prefix: str, default_base: str, supports_images: bool = False):
        super().__init__()
        self.provider = provider
        self.env_prefix = env_prefix
        self.default_base = default_base
        # The opencode gateway attaches images to the message (`opencode run -f`, 2026-10-20); the claude
        # gateway cannot, and keeps raising so the router falls back to a backend that can.
        self.supports_images = supports_images

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
        images: Optional[List[Dict[str, str]]] = [] if self.supports_images else None
        try:
            prompt = _build_prompt(messages, images)
        except UnsupportedContentError as exc:
            raise _provider_error(self.provider, model, str(exc)) from exc
        # A JSON request (Frigate's review summaries, HA's AI Task, imageAnalysis) has no response_format on a
        # CLI path: state it in the prompt; _map_response strips the fences models like to add.
        instruction = _json_instruction(kwargs.get("response_format") or optional_params.get("response_format"))
        if instruction:
            prompt = prompt + "\n\n" + instruction

        timeout = float(kwargs.get("timeout") or 120.0)
        # The gateway gets the true budget; the httpx client waits a few seconds
        # LONGER so the gateway's own SIGKILL+504 lands first — otherwise the
        # client always times out before the typed gateway error can return.
        http_timeout = timeout + 5.0
        payload: Dict[str, Any] = {"prompt": prompt, "timeoutMs": int(timeout * 1000)}
        # Model selection (2026-10-20): the deployment's own model string names the CLI model the gateway must
        # run, e.g. litellm_params.model "opencode/opencode-go/glm-5.3-flash" arrives here as
        # "opencode-go/glm-5.3-flash" and is sent as payload.model, so one gateway serves several router
        # deployments (one per subscription model). "default" (the historical entry) leaves the choice to the
        # gateway's GW_DEFAULT_MODEL. A caller's extra_body.model still overrides either.
        cli_model = model.strip() if isinstance(model, str) else ""
        if cli_model.startswith(self.provider + "/"):   # tolerate the un-stripped "opencode/..." form
            cli_model = cli_model[len(self.provider) + 1:]
        if cli_model and cli_model != "default":
            payload["model"] = cli_model
        requested = optional_params.get("model")
        if isinstance(requested, str) and requested.strip():
            payload["model"] = requested
        if images:
            payload["images"] = _subsample(images, MAX_GATEWAY_IMAGES)
        return model, payload, http_timeout

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
