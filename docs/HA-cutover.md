# Home Assistant Assist → litellm cutover (U11 runbook)

Move Home Assistant's Assist conversation agent from the **native Ollama
integration** to this **litellm proxy** (`http://192.168.0.34:4000`), so Assist
talks to agy (via litellm) instead of Ollama directly, while keeping Ollama
configured-but-idle as a safe rollback during a soak period.

**Do not apply this yet** — it is a runbook, not an executed change. litellm must
be running and smoke-tested first (see below). Read
[`homeassistant/SYSADMIN.md`](../../../homeassistant/SYSADMIN.md) before touching
HA.

## Preconditions

1. **litellm is up and healthy** on the desktop host. Verify from a LAN machine:
   ```bash
   curl -s http://192.168.0.34:4000/v1/chat/completions \
     -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
     -d '{"model":"agy","messages":[{"role":"user","content":"say ok"}]}'
   ```
   A 200 with agy's reply is required before proceeding. A failure here means
   Assist would break on cutover.
2. **HA can reach `192.168.0.34:4000`** (same LAN; no tunnel involved).
3. You have the `LITELLM_MASTER_KEY` value from this repo's `.env`.

## Steps

1. **Install HACS "Extended OpenAI Conversation".** In HA: HACS → Integrations →
   search *Extended OpenAI Conversation* → install → restart HA. (HACS itself
   must already be installed.)
2. **Add the integration.** Settings → Devices & Services → Add Integration →
   *Extended OpenAI Conversation*. Configure:
   - **Base URL:** `http://192.168.0.34:4000/v1`
   - **API Key:** the `LITELLM_MASTER_KEY` from this repo's `.env`
   - **Model:** `agy`
   - Leave organization/other fields blank.
3. **Point the Assist pipeline at it.** Settings → Voice assistants → your Assist
   pipeline → **Conversation agent** → select the new Extended OpenAI
   Conversation agent. Save.
4. **Leave the Ollama integration configured but idle.** Do **not** delete it and
   do **not** select it as any pipeline's conversation agent. It stays available
   as an instant rollback target.
5. **Soak.** Exercise Assist (voice + text) across a normal usage window. Watch
   litellm logs (`docker compose logs -f` on the desktop) and HA's Assist debug
   traces. Confirm requests are reaching agy and responses look right.

## Rollback

If Assist misbehaves at any point during the soak:

1. Settings → Voice assistants → the pipeline → **Conversation agent** → switch
   back to the **Ollama** agent. Save.
2. Assist is immediately restored to the pre-cutover path; no restart needed.
3. Investigate litellm/agy separately before re-attempting. Only remove the
   Ollama integration after the litellm path has soaked cleanly and you are
   confident in it.

## Notes

- The Assist model name (`agy`) is a litellm route, not an Ollama model —
  litellm's routing hook keeps sending it to the agy backend, with `cloud`
  fallback if agy is down.
- Keep the litellm container on `restart: unless-stopped` so an HA restart or a
  host reboot doesn't leave Assist pointing at a dead endpoint.
