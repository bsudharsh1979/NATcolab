"""CI smoke test: bootstrap + start the NAT backend + a few agent queries (mirrors the notebooks)."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import colab_bootstrap  # noqa: E402

colab_bootstrap.bootstrap(interactive=False)

from nat_client import NATClient  # noqa: E402

client = NATClient()
client.start()
results = {}
for q in [
    "What is the ground truth RUL of unit_number 20 in dataset FD001?",
    "Show me a line chart with sensor_measurement_4 values for unit 69 in FD001 training data over time",
    "Detect anomalies in sensor_measurement_4 for unit 78 in FD001 training data and visualize the results",
]:
    t0 = time.time()
    ans = client.query(q, render=False, timeout=900)
    results[q] = {"seconds": round(time.time() - t0), "answer": ans[:1500]}
    print(f"\n### {q}\n({results[q]['seconds']}s)\n{ans[:1500]}\n", flush=True)

print("workspace files:", client.latest_files(20))
client.logs(80)
client.stop()
Path(ROOT / "ci" / "executed").mkdir(parents=True, exist_ok=True)
(ROOT / "ci" / "executed" / "smoke_results.json").write_text(json.dumps(results, indent=2))
bad = [q for q, r in results.items() if r["answer"].startswith("❌")]
sys.exit(1 if bad else 0)
