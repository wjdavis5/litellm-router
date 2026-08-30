"""litellm CustomLLM handler that routes chat completions to the home-lab agy
gateway (agy-gateway LXC, http://192.168.0.92:8100).

The handler flattens OpenAI-style ``messages`` into a single prompt string,
uploads any inline base64 images to the gateway's ``POST /files`` endpoint and
references the stored path in the prompt, then calls ``POST /run`` and maps
``agy.response`` back into a ``litellm.ModelResponse``.

HTTP is done through the module-level ``_post_run`` / ``_post_file`` (and async
``_apost_run`` / ``_apost_file``) functions so tests can monkeypatch them and
never touch the network. ``httpx`` is imported lazily inside those functions so
this module imports cleanly without it installed.
"""

from __future__ import annotations

import base64
import os
import uuid
from typing import Any, Callable, Dict, List, Optional

import litellm


DEFAULT_AGY_BASE = "http://192.168.0.92:8100"

# mime -> file extension for uploaded inline images
_MIME_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
}


# --------------------------------------------------------------------------
# config helpers
# --------------------------------------------------------------------------
def _agy_base() -> str:
    return os.environ.get("AGY_BASE", DEFAULT_AGY_BASE).rstrip("/")


def _agy_token() -> str:
    token = os.environ.get("AGY_TOKEN")
    if not token:
        raise ValueError(
            "AGY_TOKEN is not set — the agy handler cannot authenticate to the "
            "agy gateway. Set AGY_TOKEN in the litellm proxy environment (.env)."
        )
    return token


def _provider_error(model: Optional[str], message: str) -> Exception:
    """Build a litellm provider exception so the router's fallbacks fire."""
    exceptions = getattr(litellm, "exceptions", None)
    exc_cls = getattr(exceptions, "APIConnectionError", None) if exceptions else None
    if exc_cls is not None:
        try:
            return exc_cls(message=message, llm_provider="agy", model=model or "agy")
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


def _post_file(base: str, token: str, name: str, data: bytes) -> str:
    """POST /files synchronously. Returns a path/name to reference in a prompt."""
    import httpx

    resp = httpx.post(
        f"{base}/files",
        params={"name": name},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"},
        content=data,
        timeout=60.0,
    )
    resp.raise_for_status()
    return _file_ref_from_response(resp.json(), name)


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


async def _apost_file(base: str, token: str, name: str, data: bytes) -> str:
    """POST /files asynchronously. Returns a path/name to reference in a prompt."""
    import httpx

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{base}/files",
            params={"name": name},
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"},
            content=data,
        )
        resp.raise_for_status()
        return _file_ref_from_response(resp.json(), name)


def _file_ref_from_response(body: Any, fallback_name: str) -> str:
    """Pull the referenceable path out of the gateway's /files response.

    The gateway returns ``{"ok": true, "file": {"name", "containerPath", ...}}``;
    prefer ``containerPath`` (agy reads it under /mnt/agy-share), fall back to the
    stored name, then to the name we sent.
    """
    if isinstance(body, dict):
        file_obj = body.get("file")
        if isinstance(file_obj, dict):
            return file_obj.get("containerPath") or file_obj.get("name") or fallback_name
        # tolerate a flatter shape
        return body.get("containerPath") or body.get("path") or body.get("name") or fallback_name
    return fallback_name


# --------------------------------------------------------------------------
# prompt building
# --------------------------------------------------------------------------
def _decode_data_url(url: str) -> Optional[tuple]:
    """Parse a ``data:<mime>;base64,<b64>`` URL -> (mime, raw_bytes) or None."""
    if not url.startswith("data:"):
        return None
    try:
        header, b64 = url.split(",", 1)
    except ValueError:
        return None
    if ";base64" not in header:
        return None
    mime = header[len("data:"):].split(";", 1)[0] or "application/octet-stream"
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return None
    return mime, raw


def _flatten_content(content: Any, upload: Callable[[bytes, str], str]) -> str:
    """Flatten one message's content (str or list-of-parts) into text.

    ``upload(raw_bytes, ext)`` stores an inline image and returns the path to
    reference. It is called only for base64 ``image_url`` parts.
    """
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
            decoded = _decode_data_url(url)
            if decoded is not None:
                mime, raw = decoded
                ext = _MIME_EXT.get(mime.lower(), "bin")
                ref = upload(raw, ext)
                pieces.append(f"[image: {ref}]")
            elif url and not url.startswith("data:"):
                # remote http(s) image: include the URL in the prompt text,
                # do not download it in this unit
                pieces.append(f"[image: {url}]")
            # a data: URL that failed to decode is DROPPED — never inline the raw
            # base64 blob (it would blow agy's 100k-char prompt cap / inject garbage)
        else:
            # unknown part type: best-effort include any text field
            if "text" in part:
                pieces.append(part.get("text", ""))
    return "\n".join(p for p in pieces if p)


def _build_prompt(messages: List[Dict[str, Any]], upload: Callable[[bytes, str], str]) -> str:
    """Flatten messages into one prompt: system turns first, then the dialogue."""
    system_parts: List[str] = []
    turns: List[str] = []
    for msg in messages or []:
        role = (msg.get("role") or "user").lower()
        text = _flatten_content(msg.get("content"), upload)
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
    """Map a gateway /run success body into a litellm.ModelResponse."""
    agy = body.get("agy") or {}
    text = agy.get("response", "") or ""

    model_response = litellm.ModelResponse()
    message = litellm.Message(content=text, role="assistant")
    choice = litellm.Choices(index=0, message=message, finish_reason="stop")
    model_response.choices = [choice]
    if model:
        model_response.model = model

    usage = agy.get("usage")
    if isinstance(usage, dict):
        try:
            model_response.usage = litellm.Usage(
                prompt_tokens=usage.get("prompt_tokens") or usage.get("input_tokens") or 0,
                completion_tokens=usage.get("completion_tokens") or usage.get("output_tokens") or 0,
                total_tokens=usage.get("total_tokens") or 0,
            )
        except Exception:
            pass
    return model_response


def _resolve_effort(optional_params: Optional[Dict[str, Any]]) -> str:
    if optional_params:
        eff = optional_params.get("effort")
        if eff in ("low", "medium", "high"):
            return eff
    return "high"


# --------------------------------------------------------------------------
# the CustomLLM
# --------------------------------------------------------------------------
class MyAgy(litellm.CustomLLM):
    """Routes litellm completions to the agy gateway."""

    def completion(self, *args, **kwargs) -> "litellm.ModelResponse":
        model = kwargs.get("model")
        messages = kwargs.get("messages") or []
        optional_params = kwargs.get("optional_params") or {}

        base = _agy_base()
        token = _agy_token()  # raises ValueError if missing

        def upload(raw: bytes, ext: str) -> str:
            name = f"{uuid.uuid4().hex}.{ext}"
            return _post_file(base, token, name, raw)

        prompt = _build_prompt(messages, upload)
        timeout = float(kwargs.get("timeout") or 120.0)
        payload = {
            "prompt": prompt,
            "effort": _resolve_effort(optional_params),
            "outputFormat": "json",
            # Bound agy to the same budget as our client: without this agy runs to
            # its own 300s default and keeps a concurrency slot held after we've
            # already timed out (agy has no cancellation).
            "timeoutMs": int(timeout * 1000),
        }

        try:
            body = _post_run(base, token, payload, timeout)
        except Exception as exc:  # network / non-2xx
            raise _provider_error(model, f"agy /run request failed: {exc}") from exc

        if not isinstance(body, dict) or not body.get("ok"):
            raise _provider_error(model, f"agy /run returned an error body: {str(body)[:200]}")

        return _map_response(model, body)

    async def acompletion(self, *args, **kwargs) -> "litellm.ModelResponse":
        model = kwargs.get("model")
        messages = kwargs.get("messages") or []
        optional_params = kwargs.get("optional_params") or {}

        base = _agy_base()
        token = _agy_token()  # raises ValueError if missing

        # Pre-upload any inline base64 images (async), building a queue the
        # synchronous _build_prompt can consume in order.
        uploaded: List[str] = []
        await _preupload_images(base, token, messages, uploaded)
        it = iter(uploaded)

        def upload(raw: bytes, ext: str) -> str:
            try:
                return next(it)
            except StopIteration:
                return "unknown"

        prompt = _build_prompt(messages, upload)
        timeout = float(kwargs.get("timeout") or 120.0)
        payload = {
            "prompt": prompt,
            "effort": _resolve_effort(optional_params),
            "outputFormat": "json",
            "timeoutMs": int(timeout * 1000),  # keep agy's budget == our client budget
        }

        try:
            body = await _apost_run(base, token, payload, timeout)
        except Exception as exc:
            raise _provider_error(model, f"agy /run request failed: {exc}") from exc

        if not isinstance(body, dict) or not body.get("ok"):
            raise _provider_error(model, f"agy /run returned an error body: {str(body)[:200]}")

        return _map_response(model, body)


async def _preupload_images(base: str, token: str, messages: List[Dict[str, Any]], out: List[str]) -> None:
    """Walk messages, upload each inline base64 image via _apost_file, append
    the returned refs to ``out`` in encounter order."""
    for msg in messages or []:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url") or {}
            url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
            decoded = _decode_data_url(url)
            if decoded is None:
                continue
            mime, raw = decoded
            ext = _MIME_EXT.get(mime.lower(), "bin")
            name = f"{uuid.uuid4().hex}.{ext}"
            out.append(await _apost_file(base, token, name, raw))


# instance registered by config.yaml custom_provider_map
agy_llm = MyAgy()
