"""Eval CLI.

    uv run evals/run_eval.py                                # all cases × first configured model
    uv run evals/run_eval.py --models gemma-4-31b-it,gemini-2.5-flash
    uv run evals/run_eval.py --category typos               # subset
    uv run evals/run_eval.py --case b01_textclass_datasets  # one case

Writes:
    evals/results/<provider>__<model>.json   (per-model run records)
    evals/results/_summary.json              (cross-model accuracy table)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Make the repo importable when run as `uv run evals/run_eval.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from hf_agent.config import ModelChoice, available_models  # noqa: E402

from evals.cases import CASES, Case  # noqa: E402
from evals.runner import RunResult, run_case  # noqa: E402
from evals.scorer import Score  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def _resolve_models(
    specs: list[str], default_rpm: int, default_retries: int,
) -> list[tuple[ModelChoice, int, int]]:
    """Parse model specs into (ModelChoice, rpm, max_retries) triples.

    Each spec may carry per-model overrides:
        gemma-4-31b-it@15            → rpm=15, retries=default
        gemma-4-31b-it@15:r0         → rpm=15, retries=0  (e.g. for daily-quota models)
        gemini:gemini-3.1-flash-lite-preview@15
    Without `@N`, falls back to default_rpm. Without `:rN`, default_retries.
    """
    configured = available_models()
    if not specs:
        return [(c, default_rpm, default_retries) for c in (configured[:1] if configured else [])]

    out: list[tuple[ModelChoice, int, int]] = []
    for spec in specs:
        rpm = default_rpm
        retries = default_retries
        # Parse trailing `:rN` retry override
        if ":r" in spec and spec.rsplit(":r", 1)[1].isdigit():
            spec, retries_str = spec.rsplit(":r", 1)
            retries = int(retries_str)
        if "@" in spec:
            spec, rpm_part = spec.rsplit("@", 1)
            try:
                rpm = int(rpm_part)
            except ValueError:
                raise SystemExit(f"bad rpm in spec {spec!r}@{rpm_part!r}")
        match = None
        if ":" in spec:
            provider, model = spec.split(":", 1)
        else:
            provider, model = None, spec
        for c in configured:
            if c.model == model and (provider is None or c.provider == provider):
                match = c
                break
        if match is None:
            inferred_provider = provider or ("openai" if "gpt" in model.lower() else "gemini")
            match = ModelChoice(provider=inferred_provider, model=model)
        out.append((match, rpm, retries))
    return out


def _short_tag(label: str, width: int = 30) -> str:
    """Shorten a model label like '[gemini] gemini-3.1-flash-lite-preview' to
    a left-justified tag for interleaved per-model output."""
    name = label.split("] ", 1)[-1]
    name = name.replace("-preview", "").replace("-it", "")
    if len(name) > width:
        name = name[:width - 1] + "…"
    return name.ljust(width)


def _filter_cases(category: str | None, case_id: str | None) -> list[Case]:
    if case_id:
        cs = [c for c in CASES if c.id == case_id]
        if not cs:
            raise SystemExit(f"no case with id {case_id!r}")
        return cs
    if category:
        cs = [c for c in CASES if c.category == category]
        if not cs:
            raise SystemExit(f"no cases in category {category!r}")
        return cs
    return list(CASES)


async def run_for_model(
    choice: ModelChoice, cases: list[Case], rpm: int, max_retries: int,
) -> dict:
    tag = _short_tag(choice.label)
    print(f"[{tag}] starting — {len(cases)} cases, {rpm} RPM, retries={max_retries}", flush=True)
    records: list[dict] = []
    passes = 0
    started = time.time()
    for i, case in enumerate(cases, 1):
        try:
            result, s = await run_case(case, choice, rpm=rpm, max_retries=max_retries)
        except Exception as e:
            result = RunResult(case.id, choice.label, None, {}, f"{type(e).__name__}: {e}", 0.0)
            s = Score(case.id, False, f"runner crashed: {e}", None, {})
        records.append({
            "case_id": case.id,
            "category": case.category,
            "nl": case.nl,
            "tool": result.tool_name,
            "arguments": result.arguments,
            "elapsed_s": result.elapsed_s,
            "error": result.error,
            "passed": s.passed,
            "reason": s.reason,
        })
        passes += int(s.passed)
        marker = "✔" if s.passed else "✘"
        err_tag = f" [{result.error[:80]}]" if result.error else ""
        print(f"[{tag}] {marker} {i:2d}/{len(cases)}  {case.id:32s}  {s.reason}{err_tag}", flush=True)

    elapsed = round(time.time() - started, 1)
    accuracy = passes / len(cases) if cases else 0.0
    summary = {
        "model": choice.label,
        "provider": choice.provider,
        "model_id": choice.model,
        "total": len(cases),
        "passes": passes,
        "accuracy": round(accuracy, 4),
        "elapsed_s": elapsed,
        "records": records,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{choice.provider}__{choice.model.replace('/', '_')}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[{tag}] DONE  {accuracy*100:.1f}% ({passes}/{len(cases)})  {elapsed}s  → {out_path.relative_to(RESULTS.parent.parent)}", flush=True)
    return summary


async def main_async(args: argparse.Namespace) -> int:
    cases = _filter_cases(args.category, args.case)
    specs = [s.strip() for s in (args.models or "").split(",") if s.strip()]
    model_triples = _resolve_models(
        specs, default_rpm=args.rpm, default_retries=args.max_retries,
    )
    if not model_triples:
        sys.stderr.write(
            "ERROR: no models resolved. Configure GEMINI_API_KEY/GEMINI_MODEL or "
            "OPENAI_API_KEY/OPENAI_MODEL in .env, or pass --models.\n"
        )
        return 2

    print(f"\n— eval: {len(cases)} cases × {len(model_triples)} models, concurrent —\n", flush=True)
    if args.sequential or len(model_triples) == 1:
        summaries = []
        for choice, rpm, retries in model_triples:
            summaries.append(await run_for_model(choice, cases, rpm=rpm, max_retries=retries))
    else:
        summaries = await asyncio.gather(*[
            run_for_model(choice, cases, rpm=rpm, max_retries=retries)
            for choice, rpm, retries in model_triples
        ])

    summary_path = RESULTS / "_summary.json"
    summary_path.write_text(json.dumps([
        {k: v for k, v in s.items() if k != "records"} for s in summaries
    ], indent=2, ensure_ascii=False))

    print("\n=== summary ===", flush=True)
    print(f"{'model':40s}  acc     pass/total   time")
    for s in sorted(summaries, key=lambda s: -s["accuracy"]):
        print(f"{s['model']:40s}  {s['accuracy']*100:5.1f}%  {s['passes']:>3}/{s['total']:<3}    {s['elapsed_s']}s")
    print(f"\n→ {summary_path.relative_to(RESULTS.parent.parent)}", flush=True)

    threshold = 0.85
    failing = [s for s in summaries if s["accuracy"] < threshold]
    if failing:
        print(f"\n⚠ {len(failing)} model(s) below {threshold*100:.0f}% threshold:", flush=True)
        for s in failing:
            print(f"  - {s['model']}: {s['accuracy']*100:.1f}%", flush=True)
        return 1
    return 0


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="Comma-separated model specs, e.g. 'gemma-4-31b-it,gemini-2.5-flash' or 'gemini:gemini-2.5-flash,openai:gpt-4o-mini'.")
    parser.add_argument("--category", help="Run only cases in this category (baseline/ambiguous/conflicting/typos/nonenglish/multistep/edge).")
    parser.add_argument("--case", help="Run a single case by id.")
    parser.add_argument("--rpm", type=int, default=5,
                        help="Default per-model RPM (overridden per-spec via 'model@N').")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Default per-model retry budget on 5xx/429 (overridden per-spec via 'model:rN'). Set to 0 for daily-quota-tight models.")
    parser.add_argument("--sequential", action="store_true",
                        help="Run models sequentially instead of concurrently (debug/diagnostic).")
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
