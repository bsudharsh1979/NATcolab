"""NATClient — start, stop, query, and inspect the NAT backend from a notebook.

NATcolab edition: works in Google Colab (and any Linux/macOS Jupyter) instead of the DLI lab image.

Differences from the DLI version (``/dli/task`` + ``/opt/predm-lab``):

* Paths are relative to the NATcolab repo (``ALM_HOME``, default: the folder containing this file)
  and the NAT virtual-env created by ``colab_bootstrap.py`` (``ALM_VENV``, default ``/content/alm-venv``).
* ``start()`` / ``restart()`` wait until the backend is healthy, so "Run all" works top-to-bottom.
* Helpers for the services that the DLI lab ran as Docker containers: Phoenix (``start_phoenix``),
  the code-execution sandbox (``start_sandbox``) and Colab-proxied links to any local port (``link``).
* ``query(..., show_steps=True)`` streams the agent's intermediate steps (tool calls, plans) into the
  notebook — a lightweight replacement for the NAT Toolkit UI's "intermediate steps" panel.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
from IPython.display import HTML, Image, Markdown, display

# --------------------------------------------------------------------------------------- paths
_TASK_DIR = Path(os.environ.get("ALM_HOME", Path(__file__).resolve().parent))
_VENV = Path(os.environ.get("ALM_VENV") or ("/content/alm-venv" if Path("/content").is_dir() else _TASK_DIR / ".alm-venv"))
_BIN = _VENV / ("Scripts" if os.name == "nt" else "bin")
_EXE = ".exe" if os.name == "nt" else ""
_NAT_BIN = str(_BIN / f"nat{_EXE}")
_PYTHON_BIN = str(_BIN / f"python{_EXE}")

_LOG_FILE = _TASK_DIR / "nat_serve.log"
_PHOENIX_LOG = _TASK_DIR / "phoenix_serve.log"
_SANDBOX_LOG = _TASK_DIR / "sandbox.log"
_DEFAULT_CONFIG = "configs/config-reasoning.yml"
_SRC_DIR = _TASK_DIR / "asset_lifecycle_management_agent/src/asset_lifecycle_management_agent"
_BACKUP_DIR = _TASK_DIR / ".asset_lifecycle_management_agent_backup"
_BASE_CONFIG = _BACKUP_DIR / "config-reasoning-base.yml"
_WORKSPACE = _TASK_DIR / "agent_workspace"

PHOENIX_PORT = 6006
SANDBOX_PORT = 6000


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _venv_env() -> dict:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "VIRTUAL_ENV": str(_VENV),
           "ALM_HOME": str(_TASK_DIR), "ALM_OUTPUT_DIR": str(_WORKSPACE)}
    env["PATH"] = f"{_BIN}{os.pathsep}{env.get('PATH', '')}"
    # The kernel's own site-packages must not leak into the NAT venv
    env.pop("PYTHONPATH", None)
    return env


def link(port: int, label: str | None = None, path: str = "/") -> str:
    """Display a clickable link to a server running on ``localhost:<port>``.

    In Colab the link goes through Colab's authenticated port proxy, so it opens in your browser.
    Returns the URL.
    """
    label = label or f"Open localhost:{port}"
    url = f"http://localhost:{port}{path}"
    if _in_colab():
        try:
            from google.colab.output import eval_js
            url = eval_js(f"google.colab.kernel.proxyPort({port})").rstrip("/") + path
        except Exception:  # pragma: no cover - only reachable inside Colab
            pass
    display(HTML(f'<a style="color:#76b900;font-weight:bold;font-size:15px" target="_blank" href="{url}">{label} ↗</a>'))
    return url


def _port_open(port: int, path: str = "/") -> bool:
    try:
        return requests.get(f"http://localhost:{port}{path}", timeout=3).status_code < 500
    except Exception:
        return False


class NATClient:
    """Manage the NAT backend and send queries — all from notebook cells."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self._proc: subprocess.Popen | None = None
        self._phoenix: subprocess.Popen | None = None
        self._sandbox: subprocess.Popen | None = None

    # ---- paths (handy in notebooks) ---------------------------------------
    task_dir = _TASK_DIR
    workspace = _WORKSPACE
    nat_bin = _NAT_BIN
    python_bin = _PYTHON_BIN

    # ---- lifecycle -------------------------------------------------------

    def start(self, config: str = _DEFAULT_CONFIG, wait: bool = True, timeout: int = 600) -> None:
        """Start ``nat serve`` in the background (and by default wait until it is healthy).

        Logs are written to ``nat_serve.log`` in the repo folder — ``client.logs(20)`` shows the tail.
        """
        if self.is_healthy():
            print("⚠️  NAT backend is already running.")
            return
        if not Path(_NAT_BIN).exists():
            raise RuntimeError(f"`nat` not found at {_NAT_BIN}. Run the setup cell (colab_bootstrap.bootstrap()) first.")
        if not os.environ.get("NVIDIA_API_KEY"):
            raise RuntimeError("NVIDIA_API_KEY is not set. Run the setup cell (colab_bootstrap.bootstrap()) first.")

        self._proc = subprocess.Popen(
            [_NAT_BIN, "serve", "--config_file", config, "--host", "0.0.0.0", "--port", "8000"],
            cwd=str(_TASK_DIR),
            stdout=open(_LOG_FILE, "w"),
            stderr=subprocess.STDOUT,
            env=_venv_env(),
            start_new_session=True,
        )
        print(f"🚀 NAT backend starting (PID {self._proc.pid})")
        print(f"   Config: {config}")
        print(f"   Logs  : {_LOG_FILE}")
        print(f"   Tip   : client.logs(20) to view the last 20 lines from here")
        if wait:
            self.wait_until_healthy(timeout=timeout)

    def wait_until_healthy(self, timeout: int = 600, poll: float = 5.0) -> bool:
        """Block until the backend answers (or the process dies / timeout expires)."""
        t0 = time.time()
        last_print = 0.0
        while time.time() - t0 < timeout:
            if self.is_healthy():
                print(f"✅ NAT backend is up after {time.time() - t0:.0f}s")
                return True
            if self._proc is not None and self._proc.poll() is not None:
                print(f"❌ `nat serve` exited with code {self._proc.returncode}. Last log lines:")
                self.logs(40)
                raise RuntimeError("NAT backend failed to start — see the log lines above.")
            if time.time() - last_print > 30:
                print(f"   … waiting for backend ({time.time() - t0:.0f}s) — startup trains the Vanna SQL "
                      f"retriever and loads MOMENT, first start can take a few minutes")
                last_print = time.time()
            time.sleep(poll)
        print(f"❌ Backend not healthy after {timeout}s. Last log lines:")
        self.logs(40)
        raise TimeoutError("NAT backend did not become healthy in time.")

    def stop(self) -> None:
        """Stop the NAT backend."""
        killed = False

        if self._proc and self._proc.poll() is None:
            try:
                if os.name == "nt":
                    raise OSError
                os.killpg(self._proc.pid, signal.SIGTERM)
            except Exception:
                self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            killed = True

        for pid in self._find_pids_on_port(8000):
            try:
                os.kill(pid, signal.SIGTERM)
                killed = True
            except ProcessLookupError:
                pass

        self._proc = None
        for _ in range(20):
            if not self._find_pids_on_port(8000):
                break
            time.sleep(0.5)
        print("🛑 NAT backend stopped." if killed else "ℹ️  Nothing running on port 8000.")

    @staticmethod
    def _find_pids_on_port(port: int) -> list[int]:
        """Find PIDs listening on *port* by reading /proc/net/tcp*."""
        hex_port = f":{port:04X}"
        inodes: set[str] = set()
        for tcp_file in ("/proc/net/tcp", "/proc/net/tcp6"):
            try:
                with open(tcp_file) as f:
                    for line in f:
                        cols = line.split()
                        if len(cols) >= 10 and cols[1].endswith(hex_port) and cols[3] == "0A":
                            inodes.add(cols[9])
            except FileNotFoundError:
                continue

        if not inodes:
            return []

        pids: list[int] = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            fd_dir = entry / "fd"
            try:
                for fd in fd_dir.iterdir():
                    try:
                        link_target = os.readlink(str(fd))
                    except OSError:
                        continue
                    if link_target.startswith("socket:["):
                        inode = link_target[8:-1]
                        if inode in inodes:
                            pids.append(int(entry.name))
                            break
            except (PermissionError, FileNotFoundError):
                continue
        return pids

    def restart(self, config: str = _DEFAULT_CONFIG, wait: bool = True, timeout: int = 600) -> None:
        """Stop then start the NAT backend."""
        self.stop()
        time.sleep(2)
        self.start(config, wait=wait, timeout=timeout)

    # ---- health ----------------------------------------------------------

    def is_healthy(self) -> bool:
        try:
            requests.get(f"{self.base_url}/docs", timeout=3).raise_for_status()
            return True
        except Exception:
            return False

    def check_health(self) -> str:
        status = "✅ NAT backend is running" if self.is_healthy() else "❌ NAT backend is not responding"
        print(status)
        return status

    # ---- logs ------------------------------------------------------------

    def logs(self, n: int = 25, file: str | Path | None = None) -> None:
        """Print the last *n* lines of the NAT serve log (or another log file)."""
        path = Path(file) if file else _LOG_FILE
        if not path.exists():
            print("No log file yet — call .start() first.")
            return
        lines = path.read_text(errors="replace").splitlines()
        for line in lines[-n:]:
            print(line)

    # ---- reset -----------------------------------------------------------

    def reset(self) -> None:
        """Restore source files and config to their original (pre-exercise) state.

        Copies the pristine backup created by ``colab_bootstrap.bootstrap()`` back over
        the working copies, undoing any exercise solution overwrites.
        """
        if not _BACKUP_DIR.exists() or not _BASE_CONFIG.exists():
            print("⚠️  Backup files not found — run the setup cell (colab_bootstrap.bootstrap()) first.")
            return

        if self.is_healthy():
            print("⚠️  Stopping NAT backend before reset…")
            self.stop()

        src_backup = _BACKUP_DIR / "src"
        if _SRC_DIR.exists():
            shutil.rmtree(_SRC_DIR)
        shutil.copytree(str(src_backup), str(_SRC_DIR))
        shutil.copy2(str(_BASE_CONFIG), str(_TASK_DIR / _DEFAULT_CONFIG))

        print("✅ Source files and config restored to base state.")
        print("   Run client.restart() to start the backend with the clean config.")

    # ---- display ---------------------------------------------------------

    def show(self, pattern: str = "*", folder: str = "agent_workspace", latest: int | None = None) -> None:
        """Display images and HTML files generated by the agent.

        Args:
            pattern: Glob pattern to filter files (e.g. ``"line_chart*"``). Defaults to ``"*"`` (show all).
                     If nothing matches exactly, ``*<pattern>*`` is tried as well.
            folder:  Directory under the repo folder where outputs live.
            latest:  Only show the N most recent matching files.
        """
        base = _TASK_DIR / folder
        if not base.exists():
            print(f"⚠️  Output folder not found: {base}")
            return

        files = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files and "*" not in pattern:
            files = sorted(base.glob(f"*{Path(pattern).stem}*"), key=lambda p: p.stat().st_mtime, reverse=True)

        images = [f for f in files if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".svg")]
        htmls = [f for f in files if f.suffix.lower() in (".html", ".htm")]
        if latest:
            images, htmls = images[:latest], htmls[:latest]

        if not images and not htmls:
            print(f"ℹ️  No image/HTML files matching '{pattern}' in {base}")
            return

        for img in images:
            display(Markdown(f"**{img.name}**"))
            display(Image(filename=str(img)))

        for htm in htmls:
            display(Markdown(f"**{htm.name}**"))
            content = htm.read_text(errors="replace")
            display(HTML(content))

    def latest_files(self, n: int = 5, folder: str = "agent_workspace") -> list[str]:
        """Return the names of the *n* most recently modified files in the workspace."""
        base = _TASK_DIR / folder
        files = sorted((p for p in base.glob("*") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.name for p in files[:n]]

    # ---- query -----------------------------------------------------------

    def query(self, message: str, *, render: bool = True, timeout: int = 600, show_steps: bool = False) -> str:
        """Send a query and optionally render the response as Markdown.

        Args:
            show_steps: stream the agent's intermediate steps (plan, tool calls, tool outputs).
        """
        if render:
            display(Markdown(f"---\n**Query:** {message}\n\n---"))

        if show_steps:
            return self._query_stream(message, render=render, timeout=timeout)

        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={"messages": [{"role": "user", "content": message}], "stream": False},
                timeout=timeout,
            )
            resp.raise_for_status()
        except requests.ConnectionError:
            msg = "❌ Cannot connect — is the backend running? Call client.start() first."
            print(msg)
            return msg
        except requests.HTTPError as e:
            msg = f"❌ Backend returned an error: {e}\n{resp.text[:1000]}"
            print(msg)
            return msg

        content = resp.json()["choices"][0]["message"]["content"]
        if render:
            display(Markdown(content))
        return content

    def _query_stream(self, message: str, *, render: bool, timeout: int) -> str:
        answer_parts: list[str] = []
        handles: dict[str, object] = {}  # one collapsible block per step id (LLM token updates are merged)
        try:
            with requests.post(f"{self.base_url}/chat/stream",
                               json={"messages": [{"role": "user", "content": message}], "stream": True},
                               stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    if raw.startswith("intermediate_data:"):
                        try:
                            step = json.loads(raw[len("intermediate_data:"):].strip())
                        except json.JSONDecodeError:
                            continue
                        if not render:
                            continue
                        name = step.get("name", "step")
                        payload = str(step.get("payload", ""))
                        if len(payload) > 3000:
                            payload = payload[:1500] + "\n\n…\n\n" + payload[-1500:]
                        icon = "🧠" if "/" in name else "🔧"
                        block = Markdown(f"<details><summary>{icon} <b>{name}</b></summary>\n\n{payload}\n\n</details>")
                        key = str(step.get("id") or name)
                        if key in handles:
                            handles[key].update(block)  # type: ignore[attr-defined]
                        else:
                            handles[key] = display(block, display_id=True)
                    elif raw.startswith("data:"):
                        body = raw[len("data:"):].strip()
                        if body == "[DONE]":
                            break
                        try:
                            chunk = json.loads(body)
                        except json.JSONDecodeError:
                            continue
                        for choice in chunk.get("choices", []):
                            part = (choice.get("message") or choice.get("delta") or {}).get("content")
                            if part:
                                answer_parts.append(part)
        except requests.ConnectionError:
            msg = "❌ Cannot connect — is the backend running? Call client.start() first."
            print(msg)
            return msg
        content = "".join(answer_parts)
        if render:
            display(Markdown("### ✅ Final answer\n\n" + content))
        return content

    # ---- companion services (Docker containers in the DLI lab) ------------

    def start_phoenix(self, wait: bool = True) -> None:
        """Start the Arize Phoenix observability server on port 6006 (if not already running)."""
        if _port_open(PHOENIX_PORT):
            print(f"ℹ️  Phoenix already running on port {PHOENIX_PORT}")
            return
        env = _venv_env()
        env.setdefault("PHOENIX_WORKING_DIR", str(_TASK_DIR / ".phoenix"))
        self._phoenix = subprocess.Popen([str(_BIN / f"phoenix{_EXE}"), "serve"], cwd=str(_TASK_DIR),
                                         stdout=open(_PHOENIX_LOG, "w"), stderr=subprocess.STDOUT,
                                         env=env, start_new_session=True)
        print(f"🔭 Phoenix starting (PID {self._phoenix.pid}) — logs: {_PHOENIX_LOG}")
        if wait:
            for _ in range(90):
                if _port_open(PHOENIX_PORT):
                    print(f"✅ Phoenix is running at http://localhost:{PHOENIX_PORT}")
                    return
                if self._phoenix.poll() is not None:
                    break
                time.sleep(2)
            print("❌ Phoenix did not start. Last log lines:")
            self.logs(30, _PHOENIX_LOG)

    def phoenix_running(self) -> bool:
        return _port_open(PHOENIX_PORT)

    def start_sandbox(self, wait: bool = True) -> None:
        """Start the local code-execution sandbox (replaces the DLI `code_execution_sandbox` container)."""
        if self.sandbox_running():
            print(f"ℹ️  Sandbox already running on port {SANDBOX_PORT}")
            return
        self._sandbox = subprocess.Popen([_PYTHON_BIN, str(_TASK_DIR / "sandbox_server.py"),
                                          "--workspace", str(_WORKSPACE), "--port", str(SANDBOX_PORT)],
                                         cwd=str(_WORKSPACE), stdout=open(_SANDBOX_LOG, "w"),
                                         stderr=subprocess.STDOUT, env=_venv_env(), start_new_session=True)
        print(f"🧪 Sandbox starting (PID {self._sandbox.pid}) — logs: {_SANDBOX_LOG}")
        if wait:
            for _ in range(30):
                if self.sandbox_running():
                    print(f"✅ Sandbox is running at http://localhost:{SANDBOX_PORT}/execute")
                    return
                if self._sandbox.poll() is not None:
                    break
                time.sleep(1)
            print("❌ Sandbox did not start. Last log lines:")
            self.logs(30, _SANDBOX_LOG)

    def sandbox_running(self) -> bool:
        try:
            r = requests.post(f"http://localhost:{SANDBOX_PORT}/execute",
                              json={"generated_code": "print('ok')", "timeout": 10, "language": "python"}, timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    # ---- CLI helpers -------------------------------------------------------

    def nat(self, *args: str, timeout: int | None = None, stream: bool = True) -> int:
        """Run a ``nat`` CLI command (e.g. ``client.nat("eval", "--config_file", ...)``) from the repo folder."""
        cmd = [_NAT_BIN, *args]
        print("$ nat " + " ".join(args))
        proc = subprocess.Popen(cmd, cwd=str(_TASK_DIR), env=_venv_env(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace")
        t0 = time.time()
        for line in proc.stdout:  # type: ignore[union-attr]
            if stream:
                sys.stdout.write(line)
            if timeout and time.time() - t0 > timeout:
                proc.kill()
                print(f"\n⏱️  Timed out after {timeout}s")
                break
        return proc.wait()
