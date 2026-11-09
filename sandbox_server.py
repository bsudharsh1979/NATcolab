#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Local code-execution sandbox for the ALM agent's ``code_generation_assistant`` (NATcolab).

The DLI lab (and the upstream README) run NeMo Agent Toolkit's ``local_sandbox`` in a Docker
container that mounts the agent's output folder at ``/workspace``. Docker is not available in
Google Colab, so this script provides the same HTTP API without Docker:

    POST /execute   {"generated_code": "...", "timeout": 30, "language": "python"}
    ->  {"process_status": "completed" | "error" | "timeout", "stdout": "...", "stderr": "..."}

Each request runs in a fresh Python subprocess whose working directory is the agent workspace.
Generated code refers to ``/workspace/...`` (e.g. ``/workspace/database/nasa_turbo.db``); if the
``/workspace`` symlink could not be created (non-root environments), those paths are rewritten to
the real workspace path before execution.

NOTE: this is a convenience sandbox for a disposable Colab VM — it is *not* a security boundary.

Usage:
    python sandbox_server.py --workspace /content/NATcolab/agent_workspace --port 6000
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WORKSPACE = Path.cwd()


def _rewrite_paths(code: str) -> str:
    ws = Path("/workspace")
    if ws.exists() and ws.resolve() == WORKSPACE.resolve():
        return code
    return code.replace("/workspace", str(WORKSPACE))


def execute(code: str, timeout: float) -> dict:
    code = _rewrite_paths(code)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=tempfile.gettempdir()) as f:
        f.write(code)
        path = f.name
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONWARNINGS": "ignore", "PYTHONUNBUFFERED": "1"}
    env["PYTHONPATH"] = str(WORKSPACE)  # makes `import utils` work like in the Docker sandbox
    try:
        proc = subprocess.run([sys.executable, path], cwd=str(WORKSPACE), capture_output=True,
                              text=True, timeout=timeout, env=env)
        status = "completed" if proc.returncode == 0 else "error"
        return {"process_status": status, "stdout": proc.stdout, "stderr": proc.stderr}
    except subprocess.TimeoutExpired as e:
        return {"process_status": "timeout", "stdout": (e.stdout or "") if isinstance(e.stdout, str) else "",
                "stderr": "Timed out\n"}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802 - health check
        self._send(200, {"status": "ok", "workspace": str(WORKSPACE)})

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/execute":
            return self._send(404, {"process_status": "error", "stdout": "", "stderr": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._send(400, {"process_status": "error", "stdout": "", "stderr": "Invalid JSON data"})
        for field in ("generated_code", "timeout", "language"):
            if field not in req:
                return self._send(400, {"process_status": "error", "stdout": "",
                                        "stderr": f"Missing required field: {field}"})
        if req["language"] != "python":
            return self._send(400, {"process_status": "error", "stdout": "", "stderr": "Only python execution is supported"})
        self._send(200, execute(req["generated_code"], float(req["timeout"])))

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("[sandbox] " + (fmt % args) + "\n")


def main() -> None:
    global WORKSPACE
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=os.environ.get("ALM_OUTPUT_DIR", "agent_workspace"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6000)
    args = parser.parse_args()
    WORKSPACE = Path(args.workspace).resolve()
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    os.chdir(WORKSPACE)
    print(f"Sandbox listening on http://{args.host}:{args.port}/execute (workspace: {WORKSPACE})", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
