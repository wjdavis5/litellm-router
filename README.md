# litellm-router

A **DB-less litellm proxy** (OpenAI-compatible, `:4000`) that fronts the home
lab's LLM backends behind one endpoint and one key. It routes every request to
the custom **agy** backend by default, spins up a local **ollama** model
on-demand when a caller explicitly asks for it, and falls back to an optional
**cloud** model when a backend is unavailable. This is Phase 2 of the home-lab
LLM router: the litellm layer plus its two custom Python handlers.

## Status + deployment

**Dev project — not deployed.** Intended to run as a Docker container on this
desktop host (`192.168.0.34`), listening on `:4000`, LAN-only (no tunnel).
Nothing is running yet and no live agy/HA calls have been made from this repo.

## How it routes

- **Default → agy.** `handlers/routing_hook.py` (a litellm pre-call hook)
  rewrites any completion request to the `agy` model unless the caller named an
  explicit ollama target.
- **agy backend.** `handlers/agy_handler.py` is a `litellm.CustomLLM` that
  flattens OpenAI `messages` into one prompt, uploads inline base64 images to
  the agy gateway's `POST /files`, references the stored path in the prompt,
  then calls `POST /run` and maps `agy.response` back to an OpenAI response.
- **On-demand ollama.** When a caller asks for `ollama-local`, the hook POSTs
  the deployed controller's `ensure-running` endpoint to warm ollama up (one
  coalesced POST across concurrent callers); on timeout/failure it rewrites the
  request to `cloud`.
- **Fallbacks.** litellm model fallbacks: `agy → cloud`, `ollama-local → agy`.

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

11 tests (agy handler: message flattening, base64-image upload + prompt
reference, agy error → provider exception, missing token → ValueError; routing
hook: default→agy, coalesced single ensure-running under concurrency,
timeout/non-200 → cloud, non-completion call types untouched). All passing under
Python 3.14.

## Config & credential locations

- **`config.yaml`** — model list, `custom_provider_map`, callbacks, fallbacks,
  router settings. No secrets.
- **`.env`** (git-ignored; template in `.env.example`) — holds the litellm
  master key, the agy gateway bearer token, the optional cloud API key, the
  controller bearer token, and the OTLP export endpoint. **Never commit real
  values.** See `.env.example` for the full variable list.

## Lab context

Part of the home lab whose canonical inventory and docs index live in the lab
root's `CLAUDE.md` (`C:\git` on the lab desktop). The agy backend is the
`agy-gateway` LXC (`192.168.0.92:8100`); the ollama controller and this proxy
are intended to run on the desktop host (`192.168.0.34`).

## Related docs

- [`docs/HA-cutover.md`](docs/HA-cutover.md) — runbook to move Home Assistant's
  Assist from the native Ollama integration onto this proxy (not yet applied).

## Known limitations (accepted)

- Not deployed; the agy contract encoded in `handlers/agy_handler.py` matches
  the `agy-gateway` README as of 2026-08-30 but has not been live-verified from
  this repo. Re-check `POST /run` and `POST /files` response shapes before first
  deploy.
- The `ensure-running` controller endpoint (`:8765`) is assumed deployed; the
  hook falls back to `cloud` if it is unreachable.
- Plaintext HTTP on the trusted LAN, single shared master key.
