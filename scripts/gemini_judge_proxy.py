#!/usr/bin/env python3
"""
gemini_judge_proxy.py

Lightweight proxy implementing the OpenAI /v1/chat/completions endpoint.
Redirects requests to Gemini 2.5 Flash via google.genai with thinking_budget=0.
Allows unmodified benchmark evaluators (evaluate_tool_calls.py, evaluate_pass_rate.py)
to use the Gemini API as their LLM judge with zero code changes.
"""

import json
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from google import genai
from google.genai import types

# Load Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    env_path = os.path.join(os.path.dirname(__file__), ".env.local")
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith("GOOGLE_API_KEY="):
                GOOGLE_API_KEY = line.strip().split("=", 1)[1]

client = genai.Client(api_key=GOOGLE_API_KEY)


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

        try:
            res = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    temperature=0,
                ),
            )
            response_text = res.text or ""
            
            # Format as standard OpenAI chat completion response
            openai_resp = {
                "id": f"chatcmpl-gemini-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "gemini-2.5-flash",
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
        except Exception as e:
            err_resp = {"error": {"message": str(e), "type": "gemini_proxy_error"}}
            err_bytes = json.dumps(err_resp).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_bytes)))
            self.end_headers()
            self.wfile.write(err_bytes)

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def log_message(self, format, *args):
        pass  # Silent


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8000), GeminiProxyHandler)
    server.serve_forever()
