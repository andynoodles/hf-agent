"""Re-score saved per-model JSON files against the current cases.py
without re-running the models. Used after relaxing or tightening
ground truth on a case — re-running the eval would burn quota
unnecessarily.

Usage:
    uv run evals/rescore.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.cases import CASES  # noqa: E402
from evals.scorer import score  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
CASES_BY_ID = {c.id: c for c in CASES}


def rescore_file(path: Path) -> tuple[int, int]:
    data = json.loads(path.read_text())
    passes = 0
    for r in data["records"]:
        case = CASES_BY_ID.get(r["case_id"])
        if case is None:
            continue
        s = score(case, r["tool"], r["arguments"])
        r["passed"] = s.passed
        r["reason"] = s.reason
        passes += int(s.passed)
    total = len(data["records"])
    data["passes"] = passes
    data["total"] = total
    data["accuracy"] = round(passes / total, 4) if total else 0.0
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return passes, total


def main() -> None:
    results = sorted(p for p in RESULTS.glob("*.json") if not p.name.startswith("_"))
    summaries = []
    for p in results:
        passes, total = rescore_file(p)
        data = json.loads(p.read_text())
        print(f"{data['model']:40s}  {data['accuracy']*100:5.1f}%  {passes:>3}/{total:<3}")
        summaries.append({k: v for k, v in data.items() if k != "records"})
    (RESULTS / "_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
