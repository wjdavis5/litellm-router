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

- **The caller names a use case; the router picks the backend and spreads the
  load across every subscription** (2026-10-20; from 2026-09-19 to then every
  request went to one DeepSeek route). `handlers/routing_hook.py` (a litellm
  pre-call hook) maps the request's `model` onto a **group**, and litellm picks a
  member of that group by weight (`simple-shuffle`), cools a failed member down
  (30 s) and retries another (`enable_weighted_failover`) before the group's
  fallback chain is used. The groups (`config.yaml`):

  | group | members today (weight) | waiting on |
  |---|---|---|
  | `generic` | order 1: **Go DeepSeek-V4.1-Flash through the opencode gateway, alone** (`#GO`); order 2 (spill-over): Gemini Flash (3), Go GLM Flash (2), Z.ai GLM Flash (2), later Claude sonnet; order 3: the DeepSeek API key | the opencode gateway's Go + Z.ai keys, `claudecode/sonnet` (its login) |
  | `generic-api` | the tool-capable subset: DeepSeek Flash (5), Gemini Flash (3) | — |
  | `vision` | order 1 Go DeepSeek Flash through the opencode gateway (images attached with `opencode run -f`); order 2 Gemini Flash (3), Go qwen3.8-max (2); order 3 the API key; fallback `agy` | — |
  | `reasoning` | same shape: order 1 Go DeepSeek Flash; order 2 Gemini Flash, Go Kimi K3; order 3 the API key — its own group so members can differ from `generic` later; the Pro models were taken out the same day (Will: "the Flash models are both darn superior") | the same keys |
  | `code` | order 1 Go DeepSeek Flash; order 2 the other coding subscriptions (Go `kimi-k2.7-code`, `qwen3.8-max`, Z.ai `glm-5.3`, Claude sonnet) and Gemini Flash; order 3 the API key | the same keys |
  | `web` | Perplexity `sonar` | `PERPLEXITYAI_API_KEY` |
  | `embeddings` / `embeddings-gemini` | Gemini `gemini-embedding-2` (3072 dims) | a local `embeddings-local` pod (Ollama cannot serve the EmbeddingGemma 2 GGUF) |
  | `transcription` | the lab whisper server (`openai/whisper-1` at `192.168.0.88:30880/v1`, timeout 900 s) | nothing: no cloud fallback on purpose |

- **What the hook does with the name.** A group name is kept. A legacy or
  foreign name (`deepseek`, `deepseek-multimodal`, `opencode`, `agy`,
  `claude-code`, `gemini`, `gpt-4o`, `any`, `text-embedding-3-small`,
  `whisper-1`, anything unknown) maps onto the group it meant
  (`CALLER_ALIASES`) or the family default (`generic`, `embeddings`,
  `transcription`, `rerank`). A chat request **carrying images always goes to
  `vision`**; one carrying `tools` goes to `generic-api` unless it named a
  tool-capable group (`reasoning`, `vision`). The CLI gateways take neither, so
  `vision` and `generic-api` fall back only to `agy` (the one other backend that
  uploads an image or takes a schema).
- **Escape hatch: `pin:<deployment>`** reaches one deployment directly, on every
  call type (`pin:opencode`, `pin:local-bonsai`, `pin:gemini-pro`,
  `pin:embeddings-gemini`, `pin:transcription`). Smoke-testing and A/B only;
  `log-review` is the one consumer that pins by design (`pin:gemini`).
- **Which subscription served what.** Every request prints one `router_call`
  JSON line (`group`, `deployment`, `provider`, `tokens`, `seconds`,
  `caller_model`) to the pod's stdout, which alloy-pods ships to Loki; count it
  by `deployment`. The response also carries `x-litellm-model-id`.
- **The subscription tools are used only THROUGH their CLIs** (Will, 2026-10-20:
  "the entire reason we containerized opencode and made the litellm proxy was
  to translate requests from our environment into valid requests through those
  tools. It's the only authorized way and it takes full advantage of the plans
  we pay for"). opencode Go and the Z.ai coding plan are reached as
  `opencode/opencode-go/*` and `opencode/zai-coding-plan/*` through the opencode
  gateway, Claude Max as `claudecode/*` through the claude gateway, Antigravity
  through the agy gateway; **no direct Go/Zen/Z.ai deployment exists in this
  proxy** (direct Go HTTP was probed working and deliberately not wired). The
  gateway handler passes the deployment's model name through
  (`opencode/opencode-go/glm-5.3-flash` → `-m opencode-go/glm-5.3-flash`), so one
  gateway serves several router deployments.
- **Order: opencode Go is the default for all calls, on DeepSeek Flash.** (Will,
  2026-10-20.) In `generic`, `reasoning` and `code`, order 1 is
  DeepSeek-V4.1-Flash on the Go subscription through the opencode gateway,
  alone; order 2 is spill-over while it is cooled down (Gemini Flash, the other
  Go / Z.ai models, later Claude sonnet); the paid DeepSeek API key is order 3
  and last. litellm only considers the lowest healthy order. The gateway queues
  beyond its `GW_MAX_CONCURRENT` (4) rather than erroring, so a burst raises
  latency instead of spilling. The `#GO ` lines in `config.yaml` are the
  gateway-backed members, ready to run; they are enabled
  (`sed -i 's/^\(\s*\)#GO /\1/'` on both copies, re-apply) together with the
  gateway rollout that carries the Go and Z.ai keys (done 2026-10-20 evening:
  every text call now lands on `*-go-deepseek-flash`, ~2 s each). `vision`
  rides the same gateway since image attachments landed (`images` in the
  gateway payload, built from OpenAI data-URL parts by
  `handlers/cli_gateway_handler.py`); only `generic-api` (tool calls) bypasses
  it, because the CLI wrapper cannot take `tools`. The gateway handler also
  states a JSON request (`response_format` json_schema / json_object) in the
  prompt — the CLIs have no response_format — and strips the ```json fences
  from the reply when the body parses, so Frigate's and HA's structured
  requests work through it.
- **deepseek backend.** Plain litellm `deepseek/deepseek-flash` against
  `https://api.deepseek.com/v1` — no CLI gateway in the path, so it honours
  OpenAI `response_format` (after the `json_schema` → `json_object` rewrite
  below) and has no prompt-size quirk. `deepseek-flash` is the canonical id for
  **DeepSeek-V4.1-Flash** (natively multimodal). `deepseek-v4-pro` and
  `gemini-pro-latest` are older models and sit in no group (`gemini-pro` stays
  as a pin for A/B only).
- **agy backend.** `handlers/agy_handler.py` is a `litellm.CustomLLM` that
  flattens OpenAI `messages` into one prompt, uploads inline base64 images to
  the agy gateway's `POST /files`, references the stored path in the prompt,
  then calls `POST /run` and maps `agy.response` back to an OpenAI response.
  About a minute a call at the gateway's effort `high`: a fallback, never a
  weighted member.
- **On-demand ollama.** When an operator pins `pin:ollama-local`, the hook POSTs
  the deployed controller's `ensure-running` endpoint to warm ollama up (one
  coalesced POST across concurrent callers); on timeout/failure it rewrites the
  request to `generic`.
- **Fallbacks** (`litellm_settings.fallbacks`) fan out over working backends
  only. `cloud` is deliberately absent from every chain: it is keyless and dead,
  and falling back to a dead backend is what turned the agy outage into a total
  outage (CommandCenter#12/#23). Embeddings and transcription have no fallback
  (one vector space; audio stays on the LAN).

## Gemini (2026-10-10; group member since 2026-10-20)

`gemini` (Flash) and `gemini-pro` (Pro) call Google's Gemini API directly with `GEMINI_API_KEY` (an AI Studio key, in `.env` and the k8s Secret). They use the `*-latest` aliases so a retired model id cannot break the route. Flash is a weighted member of `generic`/`generic-api`/`vision`, Pro of `reasoning`/`code`, and `gemini-embedding-2` is the `embeddings` space; the single names are also pins (`pin:gemini`, `pin:gemini-pro`). Google publishes no numeric free-tier limits any more: a 429 cools the member down and the group carries on. Anything sent to them leaves the lab for Google, so scrub transcripts and names first.

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
