---
module: litellm-router
date: 2026-08-30
problem_type: architecture_pattern
component: assistant
severity: medium
applies_when: "Exposing an OpenAI-compatible LLM backend to Home Assistant's AI Task or vision features, where HA's native litellm integration only supports conversation subentries."
related_components:
  - home-assistant
  - imageAnalysis
  - agy-gateway
tags:
  - home-assistant
  - ai-task
  - ollama-shim
  - litellm
  - vision
  - llm-proxy
  - integration
---

# Home Assistant AI Task on a custom LLM via an Ollama-API shim

## Context

We wanted Home Assistant to run its **AI Task** feature (e.g. a "Camera Vision"
person check) and camera/vision analysis against our own LLM backend — `agy`
(a headless Antigravity/Claude agent behind `agy-gateway`), fronted by the
`litellm-router` OpenAI-compatible proxy (`:4000`, `model=agy`).

The obvious path — HA's **native `litellm` integration** — does not work for this.
That integration only registers a `conversation` subentry type; it exposes no AI
Task platform. Probing its config entry confirms it:
`supported_subentry_types` is `{"conversation": {...}}`, and starting an
`ai_task`/`ai_task_data` subentry flow against it returns `Invalid handler`.
HA's native **OpenAI Conversation** integration also can't help — its config
flow only accepts an `api_key`, no custom `base_url`, so it can't be pointed at a
local proxy.

Meanwhile HA's native **Ollama** integration *does* support conversation **and**
AI Task **and** vision — but it speaks the Ollama HTTP API, not the OpenAI API our
proxy serves. So neither native OpenAI-shaped path reaches a custom backend for
AI Task.

## Guidance

**Front the OpenAI-compatible proxy with a small Ollama-API shim, and use HA's
native Ollama integration to reach it.** HA's Ollama integration already
implements the AI Task + conversation + vision platforms; the shim only has to
translate the Ollama wire format to the proxy's OpenAI wire format.

The shim (`ollama-shim/server.mjs`, a zero-dependency Node service on `:11435`)
implements the endpoints HA's Ollama client calls and forwards to litellm
(`model=agy`):

- `GET /api/version` — a version string.
- `GET /api/tags` — lists the backing model.
- `POST /api/show` — reports the model's **`capabilities`**. HA gates features on
  these: advertise `["completion", "vision", "tools"]` so HA enables AI Task
  (vision) and tool-calling.
- `POST /api/chat` — Ollama chat → OpenAI `/v1/chat/completions`. Maps message
  `images: [<base64>]` → OpenAI `image_url` data-URL content parts, Ollama
  `format` (a JSON schema or `"json"`) → OpenAI `response_format`, and passes
  `tools` through; the reply (`content`, `tool_calls`) is mapped back to Ollama
  shape.
- `POST /api/generate` — Ollama's single-prompt endpoint (`{prompt, images[]}` →
  `{response}`). Used by consumers that call `/api/generate` directly rather than
  `/api/chat` (our `imageAnalysis` pyproc does).

Two backend-side requirements make this work end to end:

1. **The proxy's model handler must support structured output.** AI Task requests
   carry a JSON schema (Ollama `format`) and expect a schema-matching object
   back. In `handlers/agy_handler.py` this is `response_format` support: an
   OpenAI `response_format` becomes an agy `jsonSchema`, and agy's
   `structured_output` is returned as JSON-string message content.
2. **The handler must accept images the way HA sends them.** HA vision passes
   base64 images; the handler uploads them to the backend and references them so
   the model actually *sees* them.

Wiring on the HA side is then just native config:

- Add a second **Ollama** service pointing at the shim (`http://<host>:11435`) —
  leaving any existing real-Ollama service (`:11434`) untouched.
- Create an **AI Task** (and/or conversation agent) subentry on the shim's model.

Existing consumers that already speak the Ollama API need **no code change** —
only a config repoint. `imageAnalysis`'s Python VLM client
(`imageAnalysis/pyproc/ollama.py` — a separate repo — which POSTs
`/api/generate`) was moved onto agy purely by editing its `.env`:
`OLLAMA_URL=http://127.0.0.1:11435` and `OLLAMA_MODEL=agy`.

## Why This Matters

- **Reuses a mature native integration.** HA's Ollama integration already handles
  AI Task, conversation, vision capability-gating, and multi-service config. A
  shim (~250 lines, one endpoint family) is far less surface than writing a
  custom HA `custom_component` implementing the `ai_task` + `conversation`
  platforms — and it needs no HA restart, no custom-component install, and no
  tracking of HA-internal platform APIs.
- **One backend, every HA surface.** conversation agents, AI Tasks, and any
  Ollama-API consumer (like `imageAnalysis`) all reach the same proxy/model
  through one translator, so backend capabilities (vision, tools, structured
  output) land everywhere at once.
- **Config-only migration for existing Ollama consumers.** Anything already
  built against the Ollama API moves to the new backend by repointing its URL,
  not by rewriting its client.

## When to Apply

- You have an **OpenAI-compatible** LLM endpoint (litellm, vLLM, LM Studio, a
  cloud proxy) and want HA **AI Task** or vision on it, and HA's native `litellm`
  integration's conversation-only limitation blocks you.
- You have existing Ollama-API consumers you want to move onto that backend
  without touching their code.

Do **not** reach for this when a native path already fits: if you only need a
*conversation* agent, HA's native `litellm` integration works directly; if your
backend already speaks the Ollama API, point HA's Ollama integration straight at
it. The shim earns its place specifically for **AI Task / vision against an
OpenAI-shaped backend**.

Caveat: match the shim's advertised `capabilities` to what the backend can
actually do — advertising `vision` for a text-only model makes HA offer a broken
AI Task. Confirm the model is genuinely multimodal first.

## Examples

Verifying the full path without HA in the loop — a structured **vision** request
straight at the shim (the shape an AI Task sends), which returned schema-matched
JSON identifying a red/green test image:

```bash
curl -s -X POST http://127.0.0.1:11435/api/chat -d '{
  "model": "agy", "stream": false,
  "messages": [{
    "role": "user",
    "content": "The image has a left and right half. Report each color.",
    "images": ["<base64-png>"]
  }],
  "format": {"type":"object","properties":{"left":{"type":"string"},"right":{"type":"string"}}}
}'
# -> {"message":{"content":"{\"left\": \"red\", \"right\": \"green\"}"}, "done":true}
```

Repointing an existing Ollama-API consumer onto the backend (no code change):

```bash
# imageAnalysis/.env
OLLAMA_URL=http://127.0.0.1:11435   # was http://127.0.0.1:11434 (real ollama)
OLLAMA_MODEL=agy                    # was a local ollama model
REQUEST_TIMEOUT_MS=120000           # agy is slower than a local model — give headroom
```

**Boundary to respect:** route only the *chat/VLM* work (scene description,
"is there a person", structured extraction). Do **not** try to route local
face-recognition / body re-identification through the chat model — those are
ONNX **embedding** matchers (`imageAnalysis/pyproc/faceid.py`,
`imageAnalysis/pyproc/models/reid.py`, a separate repo), not a
prompt-able model, and a general VLM cannot replace embedding similarity against
enrolled references. In `imageAnalysis` the person-*presence* gate is a local CPU
detector; only the narration/verdict VLM moved to agy.
