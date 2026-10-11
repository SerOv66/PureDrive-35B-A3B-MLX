#!/usr/bin/env python3
"""Fake mini server (CPU only, no model) that records what a harness sends to /v1/chat/completions.
Usage: capture_server.py [port] [file.jsonl]. Replies with a fixed message (SSE stream or JSON)."""
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
OUT = sys.argv[2] if len(sys.argv) > 2 else "capture.jsonl"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        b = json.dumps({"object": "list", "data": [{"id": "coder-q6", "object": "model", "max_model_len": 262144}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        with open(OUT, "a") as f:
            f.write(json.dumps({"path": self.path, "body": body}) + "\n")
        msg = {"role": "assistant", "content": "ok"}
        if body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            ch = lambda d, fr=None: "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk", "model": "coder-q6", "choices": [{"index": 0, "delta": d, "finish_reason": fr}]}) + "\n\n"
            self.wfile.write((ch({"role": "assistant", "content": "ok"}) + ch({}, "stop") + "data: [DONE]\n\n").encode())
        else:
            b = json.dumps({"id": "x", "object": "chat.completion", "model": "coder-q6", "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
