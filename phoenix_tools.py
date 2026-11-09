"""
phoenix_tools.py — read traces from the Phoenix server from the notebook kernel (NATcolab).

Phoenix (arize-phoenix) is installed in the NAT virtual env, not in the Colab kernel, so the spans are
fetched by a small script that runs with the venv's Python (Phoenix client → JSON). If the client is
unavailable, Phoenix's SQLite database (``$PHOENIX_WORKING_DIR/phoenix.db``) is read directly.
"""

from __future__ import annotations

import glob
import json
import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

ALM_HOME = Path(os.environ.get("ALM_HOME", Path(__file__).resolve().parent))
ALM_VENV = Path(os.environ.get("ALM_VENV") or ("/content/alm-venv" if Path("/content").is_dir() else ALM_HOME / ".alm-venv"))
VENV_PY = ALM_VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PHOENIX_URL = os.environ.get("PHOENIX_URL", "http://localhost:6006")
PHOENIX_WORKING_DIR = Path(os.environ.get("PHOENIX_WORKING_DIR", ALM_HOME / ".phoenix"))
DEFAULT_PROJECT = "asset-lifecycle-management"

_CLIENT_SCRIPT = r'''
import json, sys
import phoenix as px
url, project, out = sys.argv[1], sys.argv[2], sys.argv[3]
client = px.Client(endpoint=url)
df = client.get_spans_dataframe(project_name=project, limit=100000)
if df is None:
    rows = []
else:
    df = df.reset_index()
    rows = json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))
json.dump(rows, open(out, "w"))
print(len(rows))
'''


def _from_client(project: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as td:
        script, out = Path(td) / "export.py", Path(td) / "spans.json"
        script.write_text(_CLIENT_SCRIPT)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        proc = subprocess.run([str(VENV_PY), str(script), PHOENIX_URL, project, str(out)],
                              capture_output=True, text=True, timeout=180, env=env)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-800:])
        return pd.DataFrame(json.load(open(out)))


def _from_sqlite(project: str) -> pd.DataFrame:
    cands = [PHOENIX_WORKING_DIR / "phoenix.db", Path.home() / ".phoenix" / "phoenix.db"]
    cands += [Path(p) for p in glob.glob(str(ALM_HOME / "**" / "phoenix.db"), recursive=True)]
    for db in cands:
        if not db.exists():
            continue
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            df = pd.read_sql_query(
                'SELECT t.trace_id AS "context.trace_id", s.span_id AS "context.span_id", s.parent_id, s.name, '
                's.span_kind, s.start_time, s.end_time, s.status_code, '
                's.cumulative_llm_token_count_prompt AS "attributes.llm.token_count.prompt", '
                's.cumulative_llm_token_count_completion AS "attributes.llm.token_count.completion", '
                'p.name AS project FROM spans s JOIN traces t ON s.trace_rowid = t.id '
                'JOIN projects p ON t.project_rowid = p.id', conn)
        finally:
            conn.close()
        if project:
            df = df[df["project"] == project]
        return df
    return pd.DataFrame()


def load_spans(project: str = DEFAULT_PROJECT, verbose: bool = True) -> pd.DataFrame:
    """Return all spans of *project* as a DataFrame (columns follow Phoenix's span dataframe)."""
    errors = []
    for label, fn in (("Phoenix client", _from_client), ("Phoenix SQLite database", _from_sqlite)):
        try:
            df = fn(project)
            if df is not None and not df.empty:
                if "span_kind" not in df.columns and "kind" in df.columns:
                    df["span_kind"] = df["kind"]
                df["kind"] = df["span_kind"]
                df["project"] = df.get("project", project)
                if verbose:
                    print(f"Spans: {len(df)} loaded via {label}")
                return df
            errors.append(f"{label}: no spans")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{label}: {type(e).__name__}: {str(e)[:200]}")
    if verbose:
        print("No spans found —", "; ".join(errors))
    return pd.DataFrame()


def _to_dt(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, utc=True, format="mixed", errors="coerce")
    except TypeError:
        return pd.to_datetime(s, utc=True, errors="coerce")


def traces_with_span(df: pd.DataFrame, span_name: str) -> list[str]:
    if df.empty:
        return []
    return list(df.loc[df["name"] == span_name, "context.trace_id"].dropna().unique())


def select_trace(df: pd.DataFrame, must_contain: str | None = None) -> pd.DataFrame:
    """The most recent trace (optionally the most recent one containing a span named *must_contain*)."""
    if df.empty:
        return df
    df = df.copy()
    df["_start"] = _to_dt(df["start_time"])
    starts = df.groupby("context.trace_id")["_start"].min()
    candidates = traces_with_span(df, must_contain) if must_contain else []
    chosen = starts.loc[candidates].idxmax() if candidates else starts.idxmax()
    return df[df["context.trace_id"] == chosen].sort_values("_start").drop(columns=["_start"])


def summarize_trace(trace: pd.DataFrame, query: str | None = None) -> dict:
    """num_llm_calls / tools_used / total_latency_ms / num_spans for one trace."""
    if trace.empty:
        print("No trace to summarize.")
        return {}
    kind = trace["span_kind"].astype(str)
    start, end = _to_dt(trace["start_time"]), _to_dt(trace["end_time"])
    root = trace[trace["name"] == "<workflow>"]
    if root.empty and "parent_id" in trace.columns:
        root = trace[trace["parent_id"].isna()]
    if not root.empty:
        i = root.index[0]
        latency = int((end[i] - start[i]).total_seconds() * 1000)
    else:
        latency = 0
    if latency <= 0:
        latency = int((end.max() - start.min()).total_seconds() * 1000)
    out = {
        "num_llm_calls": int((kind == "LLM").sum()),
        "tools_used": sorted(trace.loc[kind == "TOOL", "name"].dropna().unique().tolist()),
        "total_latency_ms": latency,
        "num_spans": int(len(trace)),
    }
    if query is not None:
        out = {"query": query, **out}
    return out


def summarize_latest_trace(df: pd.DataFrame | None = None, must_contain: str | None = None) -> dict:
    df = load_spans(verbose=False) if df is None else df
    trace = select_trace(df, must_contain)
    summary = summarize_trace(trace)
    if summary:
        print(f"Latest trace: {summary['num_spans']} spans, {summary['num_llm_calls']} LLM calls, "
              f"tools={summary['tools_used']}, latency={summary['total_latency_ms'] / 1000:.1f}s")
        by_kind = trace["span_kind"].value_counts().to_dict()
        print("Spans by kind:", by_kind)
    return summary


def export_trace(out_path: str | Path, must_contain: str | None = None, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Write the selected trace to *out_path* (JSON records, same columns as the DLI export)."""
    df = load_spans() if df is None else df
    trace = select_trace(df, must_contain)
    if trace.empty:
        raise RuntimeError("Phoenix holds no spans — is tracing enabled and did a query run after the restart?")
    keep = [c for c in trace.columns
            if c in ("project", "context.trace_id", "context.span_id", "parent_id", "name", "span_kind", "kind",
                     "start_time", "end_time", "status_code") or "token_count" in c]
    trace = trace[keep].copy()
    for c in ("start_time", "end_time"):
        trace[c] = _to_dt(trace[c]).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    trace.to_json(out_path, orient="records", indent=2)
    return trace
