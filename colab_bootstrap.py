"""
colab_bootstrap.py — one-call environment setup for the NATcolab notebooks.

Recreates, inside a Google Colab VM (or any Linux/macOS machine with Python ≥ 3.10), everything the
DLI "Asset Lifecycle Management with Agentic AI" lab image provided:

  1. NVIDIA_API_KEY            — from Colab Secrets (🔑 sidebar), the environment, or a prompt
  2. ALM agent source           — NVIDIA/GenerativeAIExamples industries/asset_lifecycle_management_agent
                                  (pinned commit) + small Colab patches
  3. NAT virtual environment    — Python 3.12 venv (created with `uv`) with nvidia-nat 1.2.1, the agent
                                  package, MOMENT, Phoenix, … (isolated from Colab's own Python)
  4. Workspace                  — agent_workspace/, utils/, /workspace symlink, Vanna training data
  5. Data + models              — NASA C-MAPSS SQLite database and the XGBoost RUL model
  6. Pristine backups           — used by NATClient.reset()
  7. NIM connectivity check     — verifies the key against the hosted NIM endpoints

Every step is idempotent: re-running `bootstrap()` in the same session takes a few seconds.

Usage (first cell of every notebook):
    import colab_bootstrap; colab_bootstrap.bootstrap()
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------------------- settings
ALM_HOME = Path(os.environ.get("ALM_HOME", Path(__file__).resolve().parent))
ALM_VENV = Path(os.environ.get("ALM_VENV") or ("/content/alm-venv" if Path("/content").is_dir() else ALM_HOME / ".alm-venv"))
VENV_PYTHON_VERSION = os.environ.get("ALM_PYTHON", "3.12")

UPSTREAM_REPO = "https://github.com/NVIDIA/GenerativeAIExamples.git"
UPSTREAM_SHA = os.environ.get("ALM_UPSTREAM_SHA", "203b0de48d1142124842d56d4b40f3733888f5ee")
UPSTREAM_SUBDIR = "industries/asset_lifecycle_management_agent"
AGENT_DIR = ALM_HOME / "asset_lifecycle_management_agent"
SRC_DIR = AGENT_DIR / "src" / "asset_lifecycle_management_agent"
WORKSPACE = ALM_HOME / "agent_workspace"
BACKUP_DIR = ALM_HOME / ".asset_lifecycle_management_agent_backup"
CONFIG = ALM_HOME / "configs" / "config-reasoning.yml"

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODELS = {
    "reasoning / analyst / SQL": "nvidia/nemotron-3-ultra-550b-a55b",
    "code generation / recommendations": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "multimodal judge (eval)": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "embeddings (Vanna)": "nvidia/nemotron-3.5-embed-1b",
}

# Extra packages installed next to the agent package (see README "What gets installed")
EXTRA_PACKAGES = [
    "pandas>=2.1,<3",                 # agent code + vanna are not pandas-3 ready
    "pyarrow<20",                     # newer pyarrow needs NumPy 2 (NAT 1.2.1 pins NumPy 1.26)
    "chardet<6",                      # keeps `requests` from warning about an unsupported chardet
    "arize-phoenix-evals<1",          # arize-phoenix 6.2 (pulled in by nvidia-nat-phoenix 1.2.1) needs the 0.x API
    "transformers>=4.33.3,<5",        # MOMENT backbone (T5)
    "huggingface-hub>=0.24,<1.0",
    "plotly>=5.18",
    "ipython",
    "pyyaml",
]

_BIN = ALM_VENV / ("Scripts" if os.name == "nt" else "bin")
_EXE = ".exe" if os.name == "nt" else ""
VENV_PY = _BIN / f"python{_EXE}"
NAT_BIN = _BIN / f"nat{_EXE}"
# Optional extra arguments for `uv pip install` (e.g. "--only-binary sqlean-py" on machines without a C compiler)
UV_EXTRA_ARGS = os.environ.get("ALM_UV_ARGS", "").split()


# ----------------------------------------------------------------------------------- helpers
def _log(msg: str) -> None:
    print(msg, flush=True)


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None, quiet: bool = True,
         check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env,
                          capture_output=True, text=True, errors="replace")
    if not quiet and proc.stdout:
        print(proc.stdout[-4000:])
    if check and proc.returncode != 0:
        print(proc.stdout[-6000:])
        print(proc.stderr[-6000:])
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}")
    return proc


def _venv_env() -> dict:
    env = {**os.environ, "VIRTUAL_ENV": str(ALM_VENV), "ALM_HOME": str(ALM_HOME),
           "ALM_OUTPUT_DIR": str(WORKSPACE), "PYTHONUNBUFFERED": "1"}
    env["PATH"] = f"{_BIN}{os.pathsep}{env.get('PATH', '')}"
    env.pop("PYTHONPATH", None)
    return env


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _uv() -> str:
    uv = shutil.which("uv")
    if uv:
        return uv
    _log("   installing uv …")
    _run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    uv = shutil.which("uv")
    if uv:
        return uv
    # pip may have installed into a bin dir that is not on PATH
    cand = Path(sys.executable).parent / "uv"
    if cand.exists():
        return str(cand)
    return f"{sys.executable} -m uv"


# ----------------------------------------------------------------------------------- 1. API key
def setup_api_key(interactive: bool = True) -> str:
    """Load NVIDIA_API_KEY from env → Colab Secrets → prompt, export it and return it."""
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    source = "environment"
    if not key and in_colab():
        try:
            from google.colab import userdata
            key = (userdata.get("NVIDIA_API_KEY") or "").strip()
            source = "Colab Secrets"
        except Exception as e:  # SecretNotFoundError / NotebookAccessError
            _log(f"   (Colab secret NVIDIA_API_KEY not available: {type(e).__name__})")
    if not key and interactive:
        from getpass import getpass
        key = getpass("Paste your NVIDIA API key (nvapi-…) from https://build.nvidia.com : ").strip()
        source = "prompt"
    if not key:
        raise RuntimeError("NVIDIA_API_KEY not found. In Colab: 🔑 Secrets → add NVIDIA_API_KEY and enable notebook access.")
    if not key.startswith("nvapi-"):
        _log("⚠️  The key does not start with 'nvapi-' — double-check it if NIM calls fail.")
    os.environ["NVIDIA_API_KEY"] = key
    _log(f"✅ NVIDIA_API_KEY loaded from {source} (nvapi-…{key[-4:]})")
    return key


def check_nims(models: dict | None = None) -> bool:
    """Send a tiny request to every NIM used by the configs."""
    import requests
    headers = {"Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}"}
    ok = True
    for role, model in (models or NIM_MODELS).items():
        t0 = time.time()
        try:
            if "embed" in model:
                r = requests.post(f"{NIM_BASE_URL}/embeddings", headers=headers, timeout=60,
                                  json={"model": model, "input": ["ping"], "input_type": "query"})
            else:
                r = requests.post(f"{NIM_BASE_URL}/chat/completions", headers=headers, timeout=120,
                                  json={"model": model, "max_tokens": 256,
                                        "messages": [{"role": "user", "content": "Reply with the single word OK."}]})
            good = r.status_code == 200
            detail = "" if good else f" → HTTP {r.status_code}: {r.text[:160]}"
        except Exception as e:
            good, detail = False, f" → {type(e).__name__}: {e}"
        ok &= good
        _log(f"   {'✅' if good else '❌'} {role:<36} {model} ({time.time() - t0:.1f}s){detail}")
    return ok


# ----------------------------------------------------------------------------------- 2. source
def fetch_agent_source() -> None:
    if (AGENT_DIR / "pyproject.toml").exists():
        _log("✅ ALM agent source already present")
        return
    _log(f"⬇️  Fetching ALM agent source (GenerativeAIExamples @ {UPSTREAM_SHA[:8]}) …")
    tmp = ALM_HOME / ".upstream"
    if tmp.exists():
        shutil.rmtree(tmp)
    _run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", "--sparse", UPSTREAM_REPO, tmp])
    _run(["git", "-C", tmp, "sparse-checkout", "set", UPSTREAM_SUBDIR])
    _run(["git", "-C", tmp, "checkout", "-q", UPSTREAM_SHA])
    shutil.copytree(tmp / UPSTREAM_SUBDIR, AGENT_DIR)
    shutil.rmtree(tmp, ignore_errors=True)
    _log("✅ ALM agent source fetched")


def _patch(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text and old not in text:
        return  # already applied
    n = text.count(old)
    if n == 0:
        _log(f"   ⚠️  patch '{label}' did not match (upstream changed?) — skipped")
        return
    path.write_text(text.replace(old, new), encoding="utf-8")
    _log(f"   🩹 {label}")


def patch_agent_source() -> None:
    """Small, idempotent Colab patches to the upstream agent package."""
    py = AGENT_DIR / "pyproject.toml"
    text = py.read_text(encoding="utf-8")
    if "momentfm = { path" in text or '"momentfm",' in text:
        # momentfm is installed separately with --no-deps (its PyPI pins numpy==1.25.2, which has no
        # Python 3.12 wheels); the uv path source points at a local clone that we do not need.
        text = re.sub(r'^\s*"momentfm",\s*\n', "", text, flags=re.M)
        text = re.sub(r"^\[tool\.uv\.sources\]\s*\nmomentfm = \{[^\n]*\}\s*\n", "", text, flags=re.M)
        py.write_text(text, encoding="utf-8")
        _log("   🩹 pyproject.toml: install momentfm separately")

    ev = SRC_DIR / "evaluators" / "multimodal_llm_judge_evaluator.py"
    _patch(ev,
           'output_dir = "/Users/vikalluru/Documents/GenerativeAIExamples/industries/manufacturing/asset_lifecycle_management_agent/output_data"',
           'output_dir = os.environ.get("ALM_OUTPUT_DIR", "agent_workspace")',
           "multimodal evaluator: use the agent_workspace output folder")

    vu = SRC_DIR / "retrievers" / "vanna_util.py"
    _patch(vu,
           'api_key=api_key, model_name=model, input_type="query", truncate="NONE"',
           'api_key=api_key, model=model, input_type="query", truncate="NONE"',
           "vanna_util: pass the configured embedding model to NVIDIAEmbeddings (`model=`)")


# ----------------------------------------------------------------------------------- 3. venv
def create_venv(force: bool = False) -> None:
    marker = ALM_VENV / ".natcolab_installed"
    if marker.exists() and NAT_BIN.exists() and not force:
        _log(f"✅ NAT virtual environment ready ({ALM_VENV})")
        return
    uv = _uv().split()
    t0 = time.time()
    _log(f"📦 Creating Python {VENV_PYTHON_VERSION} venv at {ALM_VENV} (uv) …")
    _run([*uv, "venv", "--seed", "--allow-existing", "--python", VENV_PYTHON_VERSION, ALM_VENV])

    torch_flavour = os.environ.get("ALM_TORCH", "cpu").lower()
    _log(f"📦 Installing PyTorch ({torch_flavour}) …")
    if torch_flavour == "cpu":
        _run([*uv, "pip", "install", "--python", VENV_PY, "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
    else:
        _run([*uv, "pip", "install", "--python", VENV_PY, "torch"])

    _log("📦 Installing NeMo Agent Toolkit 1.2.1 + ALM agent package (2–4 min) …")
    _run([*uv, "pip", "install", "--python", VENV_PY, *UV_EXTRA_ARGS, "-e", AGENT_DIR, *EXTRA_PACKAGES])
    _run([*uv, "pip", "install", "--python", VENV_PY, "--no-deps", "momentfm"])

    _log("🔎 Verifying imports …")
    _run([VENV_PY, "-c", "import nat, momentfm, vanna, chromadb, xgboost, torch, phoenix; "
                         "import asset_lifecycle_management_agent; print('imports ok')"], env=_venv_env())
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    _log(f"✅ NAT virtual environment installed in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------------- 4. workspace
def prepare_workspace() -> None:
    for d in (WORKSPACE, ALM_HOME / "database", ALM_HOME / "models", ALM_HOME / "Assessment" / "output"):
        d.mkdir(parents=True, exist_ok=True)

    vtd = ALM_HOME / "vanna_training_data.yaml"
    if not vtd.exists():
        shutil.copy2(AGENT_DIR / "vanna_training_data.yaml", vtd)

    utils_dst = WORKSPACE / "utils"
    if not utils_dst.exists():
        shutil.copytree(AGENT_DIR / "utils_template", utils_dst)

    # Code generated by the coding LLM uses /workspace/database/nasa_turbo.db (Docker sandbox layout)
    db_link = WORKSPACE / "database"
    if not db_link.exists() and not db_link.is_symlink():
        try:
            db_link.symlink_to(Path("..") / "database", target_is_directory=True)
        except OSError:
            pass
    ws = Path("/workspace")
    if not ws.exists() and not ws.is_symlink():
        try:
            ws.symlink_to(WORKSPACE, target_is_directory=True)
        except OSError:
            pass  # not root — sandbox_server.py rewrites /workspace paths instead
    _log(f"✅ Workspace ready ({WORKSPACE})")


def build_data_and_models() -> None:
    db = ALM_HOME / "database" / "nasa_turbo.db"
    if not db.exists():
        _log("🗄️  Building the NASA C-MAPSS SQLite database …")
        _run([VENV_PY, "setup_database.py"], cwd=ALM_HOME, env=_venv_env())
    _log(f"✅ Database ready ({db.stat().st_size / 1e6:.0f} MB)")

    if not (ALM_HOME / "models" / "xgb_model_fd001.pkl").exists():
        _log("🌲 Training the XGBoost RUL model …")
        _run([VENV_PY, "train_rul_model.py"], cwd=ALM_HOME, env=_venv_env())
    _log("✅ RUL model ready (models/xgb_model_fd001.pkl)")


def warm_up_moment() -> None:
    marker = ALM_VENV / ".moment_cached"
    if marker.exists():
        return
    _log("🧠 Downloading MOMENT-1-small weights (one time) …")
    _run([VENV_PY, "-c",
          "from momentfm import MOMENTPipeline\n"
          "m = MOMENTPipeline.from_pretrained('AutonLab/MOMENT-1-small', model_kwargs={'task_name': 'reconstruction'})\n"
          "m.init(); print('MOMENT ok')"], env=_venv_env())
    marker.write_text("ok")
    _log("✅ MOMENT model cached")


def make_backups() -> None:
    """Pristine copies used by NATClient.reset()."""
    if (BACKUP_DIR / "config-reasoning-base.yml").exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if (BACKUP_DIR / "src").exists():
        shutil.rmtree(BACKUP_DIR / "src")
    shutil.copytree(SRC_DIR, BACKUP_DIR / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(CONFIG, BACKUP_DIR / "config-reasoning-base.yml")
    _log("✅ Pristine backups created (used by client.reset())")


# ----------------------------------------------------------------------------------- main
def bootstrap(check_models: bool = True, interactive: bool = True) -> None:
    """Prepare everything the ALM notebooks need. Safe to call repeatedly."""
    t0 = time.time()
    _log("=" * 72)
    _log(f"NATcolab bootstrap — ALM_HOME={ALM_HOME}  ALM_VENV={ALM_VENV}")
    _log("=" * 72)
    os.environ["ALM_HOME"] = str(ALM_HOME)
    os.environ["ALM_VENV"] = str(ALM_VENV)
    os.environ["ALM_OUTPUT_DIR"] = str(WORKSPACE)
    os.environ.setdefault("PHOENIX_WORKING_DIR", str(ALM_HOME / ".phoenix"))
    if str(ALM_HOME) not in sys.path:
        sys.path.insert(0, str(ALM_HOME))
    os.chdir(ALM_HOME)

    setup_api_key(interactive=interactive)
    fetch_agent_source()
    patch_agent_source()
    create_venv()
    prepare_workspace()
    build_data_and_models()
    warm_up_moment()
    make_backups()
    if check_models:
        _log("🌐 Checking hosted NVIDIA NIM endpoints …")
        if not check_nims():
            _log("⚠️  Some NIM checks failed — the agent may not work until this is fixed (key / quota / model name).")
    _log(f"🎉 Setup complete in {time.time() - t0:.0f}s — you can now `from nat_client import NATClient`.")


if __name__ == "__main__":
    bootstrap(interactive=sys.stdin.isatty())
