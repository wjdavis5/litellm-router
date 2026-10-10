# litellm-router

A **DB-less litellm proxy** (OpenAI-compatible, `:4000`) that fronts the home
lab's LLM backends behind one endpoint and one key. It routes every request to
the **DeepSeek API** by default, spins up a local **ollama** model on-demand
when a caller explicitly asks for it, and falls back across the lab's CLI
gateways (`opencode`, `claude-code`, `agy`) when a backend is unavailable. This
is Phase 2 of the home-lab LLM router: the litellm layer plus its custom Python
handlers.

## Status + deployment

**Deployed** — running as a k3s Deployment (`litellm` + `ollama-shim`) on
`pve01-services` (`192.168.0.140`, fixed DHCP reservation), not this
`docker-compose.yml` anymore. Migrated off Docker Desktop 2026-09-09 — see
`wjdavis5/CommandCenter`'s `repos/k3s-cluster/litellm-router/` for the actual
manifests (this repo's `config.yaml`/`handlers/`/`ollama-shim/` are the
canonical source, copied there) and `docs/node-inventory.md`'s "Docker
Desktop pivot" section for what the move involved. `docker-compose.yml` is
kept for local dev/testing only — it is NOT what's live. LAN-only (no
tunnel), `:4000` + shim `:11435`, listening on `192.168.0.140` now instead of
`192.168.0.34`. HA's `litellm` + second `ollama` integration entries and
every other consumer were repointed at the new address.

## How it routes

- **The router picks the model, not the caller** (2026-09-19).
  `handlers/routing_hook.py` (a litellm pre-call hook) rewrites *every*
  completion request to the cheapest backend that has the capability the
  request needs, **discarding whatever model the caller named**. Cost policy
  lives here, in one place, instead of being re-litigated in every consumer's
  config. Callers describe what they need; the router decides what serves it.
  - plain text → `deepseek`
  - inline images or tools → `deepseek-multimodal`
- **Capability, not preference, is the only thing that changes the answer.**
  `deepseek` and `deepseek-multimodal` are the *same* DeepSeek upstream; they
  differ only in fallback chain. The `opencode`/`claude-code` gateways are
  text-only and raise on images/tools, so the multimodal chain falls back only
  to `agy` — the one other backend that can upload an image or take a tool
  schema. Putting a CLI gateway in that chain would guarantee an error, not a
  recovery.
- **Escape hatch: `pin:<model>`.** `"model": "pin:opencode"` reaches one
  backend directly, bypassing the cost policy. It exists so a fallback backend
  can be smoke-tested or A/B-compared before it is relied on; no consumer uses
  it in normal operation.
- **deepseek backend.** Plain litellm `deepseek/deepseek-flash` against
  `https://api.deepseek.com/v1` — no CLI gateway in the path, so unlike
  `opencode` it honours OpenAI `response_format` and has no prompt-size quirk.
  `deepseek-flash` is the canonical id for **DeepSeek-V4.1-Flash** (natively
  multimodal); the legacy `deepseek-v4-flash` names resolve to retired models.
- **agy backend.** `handlers/agy_handler.py` is a `litellm.CustomLLM` that
  flattens OpenAI `messages` into one prompt, uploads inline base64 images to
  the agy gateway's `POST /files`, references the stored path in the prompt,
  then calls `POST /run` and maps `agy.response` back to an OpenAI response.
- **On-demand ollama.** When an operator pins `pin:ollama-local`, the hook POSTs
  the deployed controller's `ensure-running` endpoint to warm ollama up (one
  coalesced POST across concurrent callers); on timeout/failure it rewrites the
  request to `deepseek`.
- **Fallbacks.** litellm model fallbacks fan out over working backends only —
  `deepseek → opencode → claude-code → agy`, and every other route leads with
  `deepseek`. `cloud` is deliberately absent from every chain: it is keyless and
  dead, and falling back to a dead backend is what turned the agy outage into a
  total outage (CommandCenter#12/#23).

## Build / run

```bash
cp .env.example .env      # fill in the values (see below)
docker compose up -d      # pinned image docker.litellm.ai/berriai/litellm:main-stable
docker compose logs -f
# smoke test once running:
curl -s http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"model":"agy","messages":[{"role":"user","content":"hello"}]}'
```

`config.yaml` and `handlers/` are mounted read-only into the container; edit and
`docker compose restart` to pick up changes. DB-less: no `--database_url`, no
Postgres — model list and settings come entirely from `config.yaml`.

## Tests

Pure-stdlib `unittest`; **no pip installs** — `litellm` is stubbed and the HTTP
boundary is monkeypatched, so tests never hit the network.

```bash
python -m unittest discover -s test
```

Agy handler (message flattening, base64-image upload + prompt reference, agy
error → provider exception, missing token → ValueError), cli-gateway handler,
and the routing hook (default→deepseek, directly-served routes pass through,
coalesced single ensure-running under concurrency, timeout/non-200 → deepseek,
non-completion call types untouched). All passing under Python 3.14.

## Config & credential locations

- **`config.yaml`** — model list, `custom_provider_map`, callbacks, fallbacks,
  router settings. No secrets.
- **`.env`** (git-ignored; template in `.env.example`) — holds the litellm
  master key, `DEEPSEEK_API_KEY`, the agy + cli-gateway bearer tokens, the
  optional cloud API key, the controller bearer token, and the OTLP export
  endpoint. **Never commit real values.** See `.env.example` for the full
  variable list. This file is the source of truth for the k3s Secret
  `litellm-router-secrets` and is itself vaulted in `wjdavis5/CommandCenter`
  (`secrets/repos/litellm-router.sops.env`) — rotating a value means editing it
  here, re-running the Secret apply, and `node tools/secrets.mjs encrypt`.

## Lab context

Part of the home lab whose canonical inventory and docs index live in the lab
root's `CLAUDE.md` (`C:\git` on the lab desktop). The agy backend is the
`agy-gateway` LXC (`192.168.0.92:8100`); the ollama controller and this proxy
are intended to run on the desktop host (`192.168.0.34`).

## Related docs

- [`docs/HA-cutover.md`](docs/HA-cutover.md) — runbook to move Home Assistant's
  Assist from the native Ollama integration onto this proxy (not yet applied).

## Capabilities (live-verified 2026-08-30)

Deployed as the `litellm-router` docker container (`:4000`). End-to-end proven
against live agy: **text**, **vision** (base64 image → `/files` → agy vision),
and **OpenAI tool / function calling** (via agy's `--json-schema` structured
output — `tools` in a request → `tool_calls` back, with the multi-turn tool
result → final-reply loop). See `handlers/agy_handler.py`.

## Ollama-shim — agy as a native HA AI Task backend (`:11435`)

HA's native `litellm` integration is conversation-only (no AI Task). HA's native
**Ollama** integration, by contrast, supports conversation **and** AI Task **and**
vision. So `ollama-shim/` exposes an Ollama-native API (`/api/version`,
`/api/tags`, `/api/show`, `/api/chat`) that translates to litellm (`model=agy`) —
letting HA's built-in Ollama integration use agy as a first-class backend.

- Advertises capabilities `completion, vision, tools` so HA enables all features.
- `/api/chat` maps Ollama → OpenAI: message `images[]` → data-URL image parts,
  `format` (schema or `"json"`) → `response_format`, `tools` pass through; the
  OpenAI reply (content, `tool_calls`) is mapped back to Ollama shape.
- Requires the agy handler's `response_format` support (structured output) — the
  AI Task path. Live-verified: a vision request with a `format` schema returned
  schema-matched JSON describing the image.

**HA setup:** Settings → Devices & Services → **Ollama** → Add service →
URL `http://192.168.0.34:11435` → then Add AI task / conversation agent on the
`agy` model. (Keep the existing `:11434` Ollama service for local qwen3-vl.)

## Known limitations (accepted)

- The `ensure-running` controller endpoint (`:8765`) is assumed deployed; the
  hook falls back to `cloud` if it is unreachable.
- Plaintext HTTP on the trusted LAN, single shared master key.

## Known limitations (verify at the supervised go-live)

- **No streaming to agy.** The agy handler implements `completion`/`acompletion` only; a `stream:true` request routed to `agy` errors and falls back to `cloud`. HA's Extended-OpenAI-Conversation path is non-streaming — confirm at smoke-test.
- **Confirm agy→cloud fallback actually fires** (not a loop): the routing hook rewrites non-ollama models to `agy`, and `cloud` is reached only via litellm's internal `agy→cloud` fallback. litellm runs `async_pre_call_hook` once at the proxy layer before router fallbacks, so this is expected-safe — verify by forcing an agy failure.
- **Coalescing assumes one uvicorn worker + in-memory DualCache** (the shipped docker-compose). With `>1` worker or a Redis-backed cache the "exactly one ensure-running POST" invariant weakens to one-per-worker.
- **DeepSeek takes no `response_format: json_schema`** (2026-10-06). The API accepts only `text` and
  `json_object`; `json_schema` is HTTP 400 "This response_format type is unavailable now", and
  `json_object` is also 400 unless the prompt contains the word "json". `RouterHook.
  async_pre_call_deployment_hook` rewrites such requests, per deployment and only for DeepSeek, into
  `json_object` plus the schema stated in a system message. Fallbacks (agy, which maps `json_schema` to
  `--json-schema`) still receive the caller's original `response_format`. Needs a litellm that has the
  deployment hook (1.100.0 does). DeepSeek's Anthropic-compatible endpoint
  (`https://api.deepseek.com/anthropic`) has no structured output either; the router stays on the
  OpenAI-compatible `https://api.deepseek.com/v1`.
- **DeepSeek can return HTTP 200 with empty `content`.** Observed twice at
  smoke-test, intermittently, on requests that succeeded on an immediate retry
  with the same budget — so it is not purely a `max_tokens` problem. **litellm
  will not fall back on this**, because a 200 is a success. Any consumer that
  matters must treat empty content as a failure and retry, rather than storing
  it as an answer.
- **DeepSeek-V4.1-Flash reasons by default, and reasoning eats `max_tokens`.**
  A small budget can be consumed entirely by `reasoning_tokens`, returning
  **HTTP 200 with empty `content`** — a success-shaped failure. Observed live at
  smoke-test: `max_tokens: 16` → 16 reasoning tokens, 0 content; `max_tokens:
  512` → `PONG` in 0.8s with 0 reasoning tokens. Any consumer that caps
  `max_tokens` tightly must leave reasoning headroom, and must treat empty
  content as a failure rather than an answer.
- **Remote http(s) image URLs are passed as text, not fetched** — agy (headless) won't retrieve them; use base64 data URLs for vision.
- **agy contract** (`/run`, `/files`→`file.containerPath`) verified against the live openapi 2026-08-30; re-verify if agy-gateway changes.
