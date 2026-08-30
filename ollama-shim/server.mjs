// ollama-shim — presents an Ollama-native API that forwards to the litellm
// proxy (model=agy). Lets Home Assistant's built-in Ollama integration use agy
// as a first-class backend for conversation AND AI Task (incl. vision), which
// HA's native `litellm` integration does not support (conversation-only).
//
// Zero-dependency Node (global fetch, node:http). It translates:
//   GET  /api/version -> a version string
//   GET  /api/tags    -> lists the agy model
//   POST /api/show    -> agy model capabilities (completion/vision/tools)
//   POST /api/chat    -> Ollama chat <-> litellm /v1/chat/completions (agy)
// Everything real (routing, vision upload, tools, structured output) happens in
// litellm + the agy handler; this is only a protocol translator.
import http from "node:http";

const PORT = Number(process.env.SHIM_PORT || 11435);
const LITELLM_BASE = (process.env.LITELLM_BASE || "http://litellm:4000").replace(/\/+$/, "");
const MASTER_KEY = process.env.LITELLM_MASTER_KEY || "";
const AGY_MODEL = process.env.AGY_MODEL || "agy";
const VERSION = process.env.SHIM_OLLAMA_VERSION || "0.32.6";
// Advertised so HA enables the right features; agy (Claude) does all three.
const CAPABILITIES = ["completion", "vision", "tools"];

const log = (...a) => console.log(new Date().toISOString(), ...a);

function send(res, code, obj, headers = {}) {
  const body = typeof obj === "string" ? obj : JSON.stringify(obj);
  res.writeHead(code, { "Content-Type": "application/json", ...headers });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve) => {
    let d = "";
    req.on("data", (c) => (d += c));
    req.on("end", () => {
      if (!d) return resolve({});
      try {
        resolve(JSON.parse(d));
      } catch {
        resolve({});
      }
    });
  });
}

// A stable fake digest/size so HA's model list is happy.
function modelEntry() {
  return {
    name: AGY_MODEL,
    model: AGY_MODEL,
    modified_at: "2026-08-30T00:00:00.000000000-00:00",
    size: 0,
    digest: "agy0000000000000000000000000000000000000000000000000000000000000000".slice(0, 64),
    details: {
      parent_model: "",
      format: "api",
      family: "agy",
      families: ["agy"],
      parameter_size: "n/a",
      quantization_level: "none",
    },
    capabilities: CAPABILITIES,
  };
}

// data-URI mime from a raw base64 image prefix (Ollama sends bare base64).
function mimeFromB64(b64) {
  if (b64.startsWith("/9j/")) return "image/jpeg";
  if (b64.startsWith("iVBORw0KGgo")) return "image/png";
  if (b64.startsWith("R0lGOD")) return "image/gif";
  if (b64.startsWith("UklGR")) return "image/webp";
  return "image/png";
}

// Ollama message -> OpenAI message (images -> multimodal parts; tool calls kept).
function toOpenAiMessage(m) {
  const role = m.role || "user";
  const images = Array.isArray(m.images) ? m.images : [];
  if (images.length > 0) {
    const parts = [];
    if (m.content) parts.push({ type: "text", text: m.content });
    for (const img of images) {
      const b64 = typeof img === "string" ? img : "";
      parts.push({ type: "image_url", image_url: { url: `data:${mimeFromB64(b64)};base64,${b64}` } });
    }
    return { role, content: parts };
  }
  const out = { role, content: m.content ?? "" };
  if (Array.isArray(m.tool_calls)) out.tool_calls = m.tool_calls;
  if (m.tool_call_id) out.tool_call_id = m.tool_call_id;
  return out;
}

// Ollama `format` -> OpenAI response_format. Object schema or the string "json".
function toResponseFormat(format) {
  if (!format) return undefined;
  if (format === "json") return { type: "json_object" };
  if (typeof format === "object") {
    return { type: "json_schema", json_schema: { name: "response", schema: format } };
  }
  return undefined;
}

// OpenAI tool_calls (arguments: JSON string) -> Ollama tool_calls (arguments: object).
function toOllamaToolCalls(toolCalls) {
  if (!Array.isArray(toolCalls)) return undefined;
  const out = [];
  for (const tc of toolCalls) {
    const fn = tc.function || {};
    let args = fn.arguments;
    if (typeof args === "string") {
      try {
        args = JSON.parse(args);
      } catch {
        args = {};
      }
    }
    out.push({ function: { name: fn.name, arguments: args || {} } });
  }
  return out.length ? out : undefined;
}

async function callLiteLLM(body) {
  const resp = await fetch(`${LITELLM_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { Authorization: `Bearer ${MASTER_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await resp.text();
  let json;
  try {
    json = JSON.parse(text);
  } catch {
    json = null;
  }
  return { status: resp.status, json, text };
}

async function handleChat(req, res, body) {
  const messages = (body.messages || []).map(toOpenAiMessage);
  const oai = { model: AGY_MODEL, messages, stream: false };
  if (Array.isArray(body.tools) && body.tools.length) oai.tools = body.tools;
  const rf = toResponseFormat(body.format);
  if (rf) oai.response_format = rf;
  const opts = body.options || {};
  if (typeof opts.temperature === "number") oai.temperature = opts.temperature;
  if (typeof opts.num_predict === "number" && opts.num_predict > 0) oai.max_tokens = opts.num_predict;

  const { status, json, text } = await callLiteLLM(oai);
  if (status < 200 || status >= 300 || !json || !json.choices) {
    log("chat: litellm error", status, text.slice(0, 200));
    return send(res, 502, { error: `litellm error ${status}: ${text.slice(0, 200)}` });
  }

  const choice = json.choices[0] || {};
  const msg = choice.message || {};
  const message = { role: "assistant", content: msg.content ?? "" };
  const toolCalls = toOllamaToolCalls(msg.tool_calls);
  if (toolCalls) message.tool_calls = toolCalls;

  const usage = json.usage || {};
  const stats = {
    total_duration: 0,
    load_duration: 0,
    prompt_eval_count: usage.prompt_tokens || 0,
    eval_count: usage.completion_tokens || 0,
  };
  const created_at = new Date().toISOString();
  const doneReason = choice.finish_reason === "tool_calls" ? "stop" : choice.finish_reason || "stop";

  // stream defaults to true in the Ollama API; emit content chunk then a final
  // done chunk (NDJSON). stream:false -> a single aggregated JSON object.
  if (body.stream === false) {
    return send(res, 200, {
      model: AGY_MODEL,
      created_at,
      message,
      done: true,
      done_reason: doneReason,
      ...stats,
    });
  }
  res.writeHead(200, { "Content-Type": "application/x-ndjson" });
  res.write(JSON.stringify({ model: AGY_MODEL, created_at, message, done: false }) + "\n");
  res.end(
    JSON.stringify({
      model: AGY_MODEL,
      created_at,
      message: { role: "assistant", content: "" },
      done: true,
      done_reason: doneReason,
      ...stats,
    }) + "\n"
  );
}

// /api/generate — Ollama's single-prompt endpoint (used by imageAnalysis'
// pyproc VLM: {model, prompt, images:[b64], options} -> {response}).
async function handleGenerate(req, res, body) {
  const prompt = body.prompt || "";
  const images = Array.isArray(body.images) ? body.images : [];
  let content;
  if (images.length > 0) {
    content = [{ type: "text", text: prompt }];
    for (const img of images) {
      const b64 = typeof img === "string" ? img : "";
      content.push({ type: "image_url", image_url: { url: `data:${mimeFromB64(b64)};base64,${b64}` } });
    }
  } else {
    content = prompt;
  }
  const oai = { model: AGY_MODEL, messages: [{ role: "user", content }], stream: false };
  const rf = toResponseFormat(body.format);
  if (rf) oai.response_format = rf;
  const opts = body.options || {};
  if (typeof opts.temperature === "number") oai.temperature = opts.temperature;

  const { status, json, text } = await callLiteLLM(oai);
  if (status < 200 || status >= 300 || !json || !json.choices) {
    log("generate: litellm error", status, text.slice(0, 200));
    return send(res, 502, { error: `litellm error ${status}: ${text.slice(0, 200)}` });
  }
  const responseText = json.choices[0]?.message?.content ?? "";
  const created_at = new Date().toISOString();
  const usage = json.usage || {};
  const stats = {
    total_duration: 0,
    load_duration: 0,
    prompt_eval_count: usage.prompt_tokens || 0,
    eval_count: usage.completion_tokens || 0,
  };
  if (body.stream === false) {
    return send(res, 200, { model: AGY_MODEL, created_at, response: responseText, done: true, done_reason: "stop", ...stats });
  }
  res.writeHead(200, { "Content-Type": "application/x-ndjson" });
  res.write(JSON.stringify({ model: AGY_MODEL, created_at, response: responseText, done: false }) + "\n");
  res.end(JSON.stringify({ model: AGY_MODEL, created_at, response: "", done: true, done_reason: "stop", ...stats }) + "\n");
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  const path = url.pathname;
  const t0 = Date.now();
  res.on("finish", () => log(req.method, path, res.statusCode, `${Date.now() - t0}ms`));
  try {
    if (req.method === "GET" && (path === "/" || path === "/api/version")) {
      return send(res, 200, path === "/" ? "Ollama is running" : { version: VERSION });
    }
    if (req.method === "GET" && path === "/api/tags") {
      return send(res, 200, { models: [modelEntry()] });
    }
    if (req.method === "POST" && (path === "/api/show")) {
      const b = await readBody(req);
      const e = modelEntry();
      return send(res, 200, {
        license: "",
        modelfile: "",
        parameters: "",
        template: "",
        details: e.details,
        model_info: { "general.architecture": "agy", "agy.context_length": 200000 },
        capabilities: CAPABILITIES,
        modified_at: e.modified_at,
      });
    }
    if (req.method === "GET" && path === "/api/ps") {
      return send(res, 200, { models: [] });
    }
    if (req.method === "POST" && path === "/api/chat") {
      const b = await readBody(req);
      return handleChat(req, res, b);
    }
    if (req.method === "POST" && path === "/api/generate") {
      const b = await readBody(req);
      return handleGenerate(req, res, b);
    }
    return send(res, 404, { error: `unsupported: ${req.method} ${path}` });
  } catch (err) {
    log("handler error", err && err.message);
    if (!res.headersSent) send(res, 500, { error: String(err && err.message) });
    else res.end();
  }
});

server.listen(PORT, "0.0.0.0", () => log(`ollama-shim listening on :${PORT} -> ${LITELLM_BASE} (model=${AGY_MODEL})`));
