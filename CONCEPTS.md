# Concepts

Shared domain vocabulary for this project — entities, named processes, and status concepts with project-specific meaning. Seeded with core domain vocabulary, then accretes as ce-compound and ce-compound-refresh process learnings; direct edits are fine. Glossary only, not a spec or catch-all.

## LLM backend and assistant plumbing

### agy
The home lab's LLM backend: a headless Antigravity/Claude agent invoked non-interactively and used as the reasoning-and-vision model behind the lab's assistant surfaces.

One call runs one prompt to completion — no streaming, no mid-run cancellation — and it carries higher latency than a small local model, so callers give it generous timeouts and treat it as request/response rather than interactive.

### agy-gateway
The token-authenticated LAN HTTP service that wraps the headless `agy` command line, turning it into a request/response API with a bounded concurrency queue and file/image upload for analysis.

### litellm-router
The OpenAI-compatible proxy that presents one `/v1` API surface and routes each request to a backing model — `agy` by default, with local and cloud models as alternates and fallbacks. Consumers speak OpenAI; the proxy hides which model actually served the request.

### Ollama-shim
A translator service that presents the Ollama HTTP API and forwards to `litellm-router`, so a consumer that only speaks Ollama — notably Home Assistant's native Ollama integration — can drive an OpenAI-compatible backend. It advertises model capabilities (completion, vision, tools) so the consumer enables the right features, and it is the only path by which `agy` reaches Home Assistant's AI Task and vision surfaces.
