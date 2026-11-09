"""Execute the NATcolab notebooks headlessly (as "Runtime → Run all" would in Colab) and save the outputs.

Usage: python ci/run_notebooks.py [notebook names ...]
Executed copies are written to ci/executed/. Exit code is non-zero if any notebook raised.
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
OUT = ROOT / "ci" / "executed"
ORDER = [
    "0_ALM_Database_Setup.ipynb",
    "1_ALM_Application_Setup.ipynb",
    "2_ALM_Config_Understanding.ipynb",
    "3_ALM_Observability.ipynb",
    "4_ALM_Evaluation.ipynb",
    "5_ALM_AnomalyDetection.ipynb",
    "Optional_ALM_Codegen.ipynb",
    "ALM_assessment.ipynb",
]


def main() -> int:
    names = sys.argv[1:] or ORDER
    OUT.mkdir(parents=True, exist_ok=True)
    failures = []
    for name in names:
        path = NB_DIR / name
        nb = nbformat.read(path, as_version=4)
        t0 = time.time()
        print(f"\n{'=' * 80}\n▶ {name}\n{'=' * 80}", flush=True)
        client = NotebookClient(nb, timeout=3600, kernel_name="python3", allow_errors=False,
                                resources={"metadata": {"path": str(NB_DIR)}})
        try:
            client.execute()
            print(f"✅ {name} finished in {time.time() - t0:.0f}s", flush=True)
        except CellExecutionError as e:
            failures.append(name)
            print(f"❌ {name} failed after {time.time() - t0:.0f}s\n{str(e)[-3000:]}", flush=True)
        finally:
            nbformat.write(nb, OUT / name)
            # echo the text outputs so they are visible in the CI log
            for cell in nb.cells:
                if cell.cell_type != "code":
                    continue
                for o in cell.get("outputs", []):
                    txt = o.get("text") or (o.get("data", {}) or {}).get("text/plain") or ""
                    if isinstance(txt, list):
                        txt = "".join(txt)
                    if txt:
                        print(txt[-2500:], flush=True)
    print("\nSummary:", "all notebooks passed" if not failures else f"failed: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
