#!/usr/bin/env python3
"""
gemini_judge_proxy.py

Lightweight proxy implementing the OpenAI /v1/chat/completions endpoint.
Redirects requests to Gemini 3.5 Flash Lite via google.genai with temperature=0.
Allows unmodified benchmark evaluators (evaluate_tool_calls.py, evaluate_pass_rate.py)
to use the Gemini API as their LLM judge with zero code changes.

Supports multiple API keys for automatic rotation on 429 rate-limit errors.
Set GOOGLE_API_KEY, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, ... in .env.
"""

import json
import os
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from google import genai
from google.genai import types

# ── Load API keys from environment / .env ─────────────────────────────────────
def _load_env():
    """Load .env file from project root or script directory."""
    for candidate in [
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).parent / ".env",
        Path(__file__).parent / ".env.local",
    ]:
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    if v.strip():
                        os.environ.setdefault(k.strip(), v.strip())
            break

_load_env()

_raw_keys = []
for _name in ["GOOGLE_API_KEY"] + [f"GOOGLE_API_KEY_{i}" for i in range(2, 10)]:
    _k = os.environ.get(_name, "")
    if _k:
        _raw_keys.append(_k)

if not _raw_keys:
    raise RuntimeError("No GOOGLE_API_KEY found in environment or .env")

print(f"[judge-proxy] Loaded {len(_raw_keys)} API key(s). Model: gemini-3.5-flash-lite")


# ── ApiKeyPool: round-robin with per-key cooldown on 429 ─────────────────────

class ApiKeyPool:
    """Round-robin API key pool with per-key 429 cooldown tracking."""

    def __init__(self, keys: list):
        self._keys = keys
        self._idx = 0
        self._available_at = [0.0] * len(keys)

    def current_client(self):
        return genai.Client(api_key=self._keys[self._idx])

    def rotate(self, cooldown_secs: float = 65.0):
        """Mark current key as rate-limited and return a new genai.Client."""
        self._available_at[self._idx] = time.time() + cooldown_secs
        n = len(self._keys)
        for _ in range(n):
            self._idx = (self._idx + 1) % n
            if time.time() >= self._available_at[self._idx]:
                return self.current_client()
        # All keys in cooldown — wait for the soonest
        soonest = min(range(n), key=lambda i: self._available_at[i])
        wait = max(0.0, self._available_at[soonest] - time.time())
        print(f"[judge-proxy] All keys rate-limited. Waiting {wait:.0f}s...", flush=True)
        time.sleep(wait + 1)
        self._idx = soonest
        return self.current_client()


KEY_POOL = ApiKeyPool(_raw_keys)
MODEL = "gemini-3.5-flash-lite"


# ── Proxy Handler ─────────────────────────────────────────────────────────────

class GeminiProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        data = json.loads(body.decode("utf-8"))

        # Extract prompt from messages
        messages = data.get("messages", [])
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        full_prompt = "\n\n".join(prompt_parts)

        # Call Gemini with automatic key rotation on 429
        response_text = ""
        client = KEY_POOL.current_client()
        for attempt in range(len(_raw_keys) + 3):
            try:
                res = client.models.generate_content(
                    model=MODEL,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                    ),
                )
                response_text = res.text or ""
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    print(f"[judge-proxy] 429 on key ...{KEY_POOL._keys[KEY_POOL._idx][-6:]} — rotating key", flush=True)
                    client = KEY_POOL.rotate(cooldown_secs=65)
                else:
                    err_resp = {"error": {"message": err_str, "type": "gemini_proxy_error"}}
                    err_bytes = json.dumps(err_resp).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_bytes)))
                    self.end_headers()
                    self.wfile.write(err_bytes)
                    return

        openai_resp = {
            "id": f"chatcmpl-gemini-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

        resp_bytes = json.dumps(openai_resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status": "ok", "model": "gemini-3.5-flash-lite"}')

    def log_message(self, format, *args):
        pass  # Silent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gemini Judge Proxy (OpenAI-compat)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    args = parser.parse_args()
    print(f"[judge-proxy] Starting on http://{args.host}:{args.port}/v1")
    server = HTTPServer((args.host, args.port), GeminiProxyHandler)
    server.serve_forever()
