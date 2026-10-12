"""Studio/OpenCode front for the PB-27 vLLM serve.

PseudoBrain Studio and the OpenCode provider talk to http://127.0.0.1:18080 (SSH-tunnelled to the Spark) and expect
the old converted-35B server's contract: GET /health with {ready, model, checkpoint_sha256, context, decoder, ...}
and the OpenAI routes under /v1. This proxy serves that contract in front of the vLLM container on :8000:
  * /health is answered locally from vLLM's /v1/models (ready iff vLLM answers) with the PB-27 identity;
  * every /v1/* request is forwarded unchanged except that the "model" field is rewritten to the served name,
    and streaming responses are passed through chunk by chunk.

    python3 studio_proxy.py --listen 18080 --upstream http://127.0.0.1:8000 --model pb27 \
        --public-model pb27 --sha <core.pt sha256> --context 131072 --decoder "vLLM NVFP4 + DFlash2"
"""
from __future__ import annotations

import argparse
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS: argparse.Namespace
COMPLETED = 0
LOCK = threading.Lock()
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "host"}


def upstream_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{ARGS.upstream}/v1/models", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"ready": upstream_ready(), "model": ARGS.public_model,
                                    "architecture": "PB-27 windowed-hybrid (vLLM plug-in)",
                                    "checkpoint": ARGS.checkpoint, "checkpoint_sha256": ARGS.sha,
                                    "context": ARGS.context, "max_output": ARGS.max_output, "decoder": ARGS.decoder,
                                    "decode_modes": ["baseline"], "prompt_prefix_caching": False,
                                    "completed_requests": COMPLETED})
        if self.path.startswith("/v1/models"):
            return self._json(200, {"object": "list", "data": [{"id": ARGS.public_model, "object": "model",
                                                                "created": 0, "owned_by": "pseudobrain"}]})
        return self._forward(None)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if self.path.startswith("/v1/"):
            try:
                body = json.loads(raw or b"{}")
                if isinstance(body, dict):
                    if "model" in body:
                        body["model"] = ARGS.model
                    for key in ("pseudobrain_mode", "tools", "tool_choice"):  # old-server extras; Studio's tools are
                        body.pop(key, None)                                    # described in its system prompt as XML
                raw = json.dumps(body).encode()
            except ValueError:
                pass
        return self._forward(raw)

    def _forward(self, raw: bytes | None) -> None:
        global COMPLETED
        url = f"{ARGS.upstream}{self.path}"
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        if raw is not None:
            headers["Content-Length"] = str(len(raw))
            headers.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(url, data=raw, headers=headers, method=self.command)
        try:
            resp = urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as e:
            resp = e
        except Exception as e:
            return self._json(502, {"error": {"message": f"upstream unavailable: {e}"}})
        self.send_response(resp.status)
        ctype = resp.headers.get("Content-Type", "application/json")
        self.send_header("Content-Type", ctype)
        streaming = "text/event-stream" in ctype
        if streaming:
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                while True:
                    chunk = resp.read1(8192) if hasattr(resp, "read1") else resp.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            data = resp.read()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        with LOCK:
            COMPLETED += 1


def main() -> None:
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=18080)
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="pb27", help="served-model-name at the upstream")
    ap.add_argument("--public-model", default="pb27", help="model id announced to Studio/OpenCode")
    ap.add_argument("--sha", default="")
    ap.add_argument("--checkpoint", default="pb27a")
    ap.add_argument("--context", type=int, default=131072)
    ap.add_argument("--max-output", type=int, default=16384)
    ap.add_argument("--decoder", default="vLLM NVFP4 + DFlash2")
    ARGS = ap.parse_args()
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer(("127.0.0.1", ARGS.listen), Handler).serve_forever()


if __name__ == "__main__":
    main()
