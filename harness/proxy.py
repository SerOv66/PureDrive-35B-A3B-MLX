#!/usr/bin/env python3
"""Transparent measuring proxy 127.0.0.1:8002 -> 127.0.0.1:8000 (stdlib only, no model).
For every POST request it appends one JSON line to the current file: t0, ttft (first delta with content/tool_call),
t_end, prompt/completion tokens (stream usage), finish_reason, number of messages, number of tools in the request,
number of tool_calls in the reply, sampling parameters sent by the client, system-prompt bytes, past assistant turns carrying reasoning_content, the first 500 characters of a non-streamed reply and its reasoning length.
The log file is switched with GET /__log?file=<path>. Usage: proxy.py [listen_port] [server_port]."""
import http.client, json, sys, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

LISTEN = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
UP = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
state = {"file": "/dev/null"}
lock = threading.Lock()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass

    def _fwd_headers(self):
        return {k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection", "accept-encoding")}

    def do_GET(self):
        if self.path.startswith("/__log"):
            state["file"] = parse_qs(urlparse(self.path).query)["file"][0]
            b = b"ok"; self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b); return
        c = http.client.HTTPConnection("127.0.0.1", UP, timeout=60); c.request("GET", self.path, headers=self._fwd_headers()); r = c.getresponse(); b = r.read()
        self.send_response(r.status); self.send_header("Content-Type", r.getheader("Content-Type", "application/json")); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try: body = json.loads(raw)
        except Exception: body = {}
        msgs = body.get("messages", [])
        rec = {"path": self.path, "t0": time.time(), "stream": bool(body.get("stream")), "n_msgs": len(msgs),
               "n_tools": len(body.get("tools") or []), "sampling": {k: body[k] for k in ("temperature", "top_p", "top_k", "max_tokens", "max_completion_tokens") if k in body},
               "has_schema": "response_format" in body}
        if msgs and isinstance(msgs[0].get("content"), str): rec["sys_chars"] = len(msgs[0]["content"])
        rec["req_chars"] = len(raw)
        rec["hist_reasoning"] = sum(1 for m in msgs if m.get("role") == "assistant" and m.get("reasoning_content"))
        c = http.client.HTTPConnection("127.0.0.1", UP, timeout=1500)
        c.request("POST", self.path, body=raw, headers=self._fwd_headers())
        r = c.getresponse()
        rec["status"] = r.status
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"): self.send_header(k, v)
        self.send_header("Connection", "close"); self.end_headers()
        ttft = None; txt = ""; usage = None; fin = None; ntc = 0; buf = b""; nchunks = 0; whole = b""
        try:
            while True:
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                if not chunk: break
                self.wfile.write(chunk); self.wfile.flush()
                if rec["stream"]:
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]": continue
                        try: d = json.loads(line[6:])
                        except Exception: continue
                        if d.get("usage"): usage = d["usage"]
                        for ch in d.get("choices", []):
                            dl = ch.get("delta", {})
                            if ch.get("finish_reason"): fin = ch["finish_reason"]
                            if isinstance(dl.get("content"), str) and len(txt) < 20000: txt += dl["content"]
                            if dl.get("tool_calls"):
                                ntc += sum(1 for t in dl["tool_calls"] if t.get("function", {}).get("name"))
                            if (dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning") or dl.get("tool_calls")):
                                nchunks += 1
                                if ttft is None: ttft = time.time()
                else: whole += chunk
        except (BrokenPipeError, ConnectionResetError):
            rec["client_disconnect"] = True
        if not rec["stream"] and whole:
            try:
                d = json.loads(whole); usage = d.get("usage"); ch = d["choices"][0]; fin = ch.get("finish_reason"); ntc = len(ch["message"].get("tool_calls") or [])
                m = ch["message"]; rec["content_head"] = (m.get("content") or "")[:500]; rec["reasoning_chars"] = len(m.get("reasoning_content") or m.get("reasoning") or "")
            except Exception: pass
            ttft = time.time()
        rec["toolcall_leak"] = ("<tool_call>" in txt or "<function=" in txt)
        rec.update(t_end=time.time(), ttft=(ttft - rec["t0"]) if ttft else None, usage=usage, finish=fin, n_tool_calls=ntc, n_chunks=nchunks, out_chars=len(txt))
        with lock, open(state["file"], "a") as f: f.write(json.dumps(rec) + "\n")


ThreadingHTTPServer(("127.0.0.1", LISTEN), H).serve_forever()
