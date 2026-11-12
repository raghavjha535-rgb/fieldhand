"""Evaluation harness: run every case k times and report pass@1 and pass^k.

    python -m evals.run_eval                 # all 14 cases, k = 3, live Gemini
    python -m evals.run_eval --k 3 --cases C01,C11
    python -m evals.run_eval --resume evals/results/<folder>   # continue after a rate limit
    python -m evals.run_eval --dry-run --cases C01,C14          # plumbing test, no API key

Each trial gets a fresh copy of the seeded database, a fixed date (Wednesday,
October 21, 2026) and the mock forecast, so every case has one known-good answer.
Bookings are checked in the database, not in the agent's words.

Metrics (tau-bench style). For a case with n trials and c passes:
    pass@1 = c / n                               (expected success of one try)
    pass^k = C(c, k) / C(n, k)                   (chance that k tries ALL succeed)
With n = k = 3, pass^3 is 1 only when all three trials pass.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
import uuid
from datetime import datetime
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CASES_FILE = Path(__file__).resolve().parent / "cases.json"


def load_cases(ids: str | None):
    data = json.loads(CASES_FILE.read_text())
    cases = data["cases"]
    if ids:
        wanted = {i.strip() for i in ids.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    return data, cases


def check_trial(case: dict, traces: list[dict], jobs: list[dict]) -> list[str]:
    """Compare one trial with the known-good answer. Returns failed checks (empty means pass)."""
    exp, fails = case["expect"], []
    reply = traces[-1]["reply"] if traces else ""
    called = {s["name"] for t in traces for s in t["steps"] if s["kind"] == "tool_call"}
    gate_reached = any(s["kind"] == "approval_request" for t in traces for s in t["steps"])

    if "booked" in exp:
        if exp["booked"] is False:
            if jobs:
                fails.append(f"expected no booking, got {[(j['service_code'], j['date'], j['time_window']) for j in jobs]}")
        else:
            want = exp["booked"]
            match = [j for j in jobs if all(j[k] == v for k, v in want.items())]
            if not match:
                got = [(j["service_code"], j["date"], j["time_window"]) for j in jobs] or "no booking"
                fails.append(f"expected booking {tuple(want.values())}, got {got}")
            elif len(jobs) > 1:
                fails.append(f"expected one booking, got {len(jobs)}")
    for tool in exp.get("tools_called", []):
        if tool not in called:
            fails.append(f"expected a call to {tool}")
    for tool in exp.get("tools_not_called", []):
        if tool in called:
            fails.append(f"expected no call to {tool}")
    if exp.get("gate_reached") and not gate_reached:
        fails.append("expected the booking to reach the human gate")
    low = reply.lower()
    for s in exp.get("reply_contains_all", []):
        if s.lower() not in low:
            fails.append(f"reply missing {s!r}")
    anys = exp.get("reply_contains_any", [])
    if anys and not any(s.lower() in low for s in anys):
        fails.append(f"reply has none of {anys}")
    for s in exp.get("reply_not_contains", []):
        if s.lower() in low:
            fails.append(f"reply contains forbidden {s!r}")
    for pattern in exp.get("reply_not_regex", []):
        if re.search(pattern, reply):
            fails.append(f"reply matches forbidden pattern {pattern!r}")
    stops = exp.get("stop_reason_in")
    if stops and traces and traces[-1]["stop_reason"] not in stops:
        fails.append(f"stop reason {traces[-1]['stop_reason']!r}, expected {stops}")
    return fails


def run_trial(case: dict, trial: int, make_graph, model_label: str, mode: str, out_dir: Path) -> dict:
    from fieldhand import config, db
    from fieldhand.agent import run_turn

    db_file = Path(tempfile.gettempdir()) / f"fieldhand-eval-{uuid.uuid4().hex[:8]}.db"
    config.use_db(db_file)
    db.reset_db()
    graph = make_graph()
    thread = uuid.uuid4().hex
    approve = case.get("gate", "approve") == "approve"
    traces = []
    for text in case["turns"]:
        traces.append(run_turn(graph, text, case["sender"], thread_id=thread, model=model_label, mode=mode,
                               approver=lambda request: approve))
    with db.connect() as conn:
        jobs = [dict(r) for r in conn.execute(
            "SELECT service_code, date, time_window, price, confirmation FROM jobs WHERE created_by = 'intake_agent'")]
    config.use_db(None)
    db_file.unlink(missing_ok=True)

    fails = check_trial(case, traces, jobs)
    trace_file = out_dir / "traces" / f"{case['id']}-t{trial}.json"
    trace_file.write_text(json.dumps(traces, indent=2, ensure_ascii=False))
    return {
        "case": case["id"], "category": case["category"], "trial": trial, "passed": not fails, "failed_checks": fails,
        "reply": traces[-1]["reply"], "stop_reason": traces[-1]["stop_reason"],
        "tools": sorted({s["name"] for t in traces for s in t["steps"] if s["kind"] == "tool_call"}),
        "bookings": jobs, "reviewer": [r for t in traces for r in t.get("reviews", [])],
        "models_used": sorted({m for t in traces for m in t.get("models_used", [])}),
        "tokens": sum(t["tokens"]["total"] for t in traces), "latency_s": round(sum(t["latency_s"] for t in traces), 2),
        "trace_file": str(trace_file.relative_to(out_dir)),
    }


def pass_hat_k(c: int, n: int, k: int) -> float:
    return comb(c, k) / comb(n, k) if n >= k else float("nan")


def summarize(data: dict, cases: list[dict], results: list[dict], k: int, out_dir: Path, meta: dict) -> str:
    rows, by_cat = [], {}
    for case in cases:
        trials = [r for r in results if r["case"] == case["id"] and not r.get("infra_error")]
        n, c = len(trials), sum(r["passed"] for r in trials)
        p1 = c / n if n else float("nan")
        pk = pass_hat_k(c, n, k) if n else float("nan")
        fails = sorted({f for r in trials for f in r["failed_checks"]})
        rows.append((case, n, c, p1, pk, fails))
        by_cat.setdefault(case["category"], []).append((p1, pk))
    valid = [r for r in rows if r[1] >= k]
    overall_p1 = sum(r[3] for r in valid) / len(valid) if valid else float("nan")
    overall_pk = sum(r[4] for r in valid) / len(valid) if valid else float("nan")
    curve = {j: sum(pass_hat_k(r[2], r[1], j) for r in valid) / len(valid) for j in range(1, k + 1)} if valid else {}

    lines = [f"# Fieldhand evaluation: pass@1 and pass^{k}", "",
             f"Run {meta['run']} · model `{meta['model']}` · {len(cases)} cases × k = {k} · fixed date "
             f"{data['fixed_date']} · {data['weather']} forecast", "",
             f"**Overall: pass@1 = {overall_p1:.2f}, pass^{k} = {overall_pk:.2f}**", "",
             "| Case | Category | Passes | pass@1 | pass^" + str(k) + " | What failed |", "|---|---|---|---|---|---|"]
    for case, n, c, p1, pk, fails in rows:
        lines.append(f"| {case['id']} {case['name']} | {case['category']} | {c}/{n} | {p1:.2f} | {pk:.2f} | "
                     f"{'; '.join(fails) if fails else ''} |")
    lines += ["", "| Category | pass@1 | pass^" + str(k) + " |", "|---|---|---|"]
    for cat, vals in by_cat.items():
        lines.append(f"| {cat} | {sum(v[0] for v in vals) / len(vals):.2f} | {sum(v[1] for v in vals) / len(vals):.2f} |")
    lines += ["", "pass^j as more tries must all succeed: " + ", ".join(f"pass^{j} = {v:.2f}" for j, v in curve.items())]
    infra = [r for r in results if r.get("infra_error")]
    if infra:
        lines += ["", f"Infrastructure errors (not counted as trials): {len(infra)}. "
                      f"Rerun with --resume {out_dir} to fill them in."]
    models = sorted({m for r in results for m in r.get("models_used", [])})
    if models:
        lines += ["", "Models that answered: " + ", ".join(models)]
    md = "\n".join(lines)
    (out_dir / "summary.md").write_text(md)
    (out_dir / "summary.json").write_text(json.dumps({
        "overall": {"pass@1": overall_p1, f"pass^{k}": overall_pk, "curve": curve},
        "cases": [{"id": c["id"], "name": c["name"], "category": c["category"], "n": n, "passes": cc,
                   "pass@1": p1, f"pass^{k}": pk, "failed_checks": f} for c, n, cc, p1, pk, f in rows],
        "meta": meta}, indent=2))
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--cases", help="Comma-separated case ids (default: all)")
    ap.add_argument("--model", help="Gemini model id (default from .env)")
    ap.add_argument("--pause", type=float, default=4.0, help="Seconds between trials, to stay under rate limits")
    ap.add_argument("--resume", help="Results folder to continue")
    ap.add_argument("--dry-run", action="store_true", help="Scripted stand-in model (plumbing test only)")
    args = ap.parse_args()

    data, cases = load_cases(args.cases)
    os.environ["FIELDHAND_WEATHER"] = data["weather"]
    os.environ["FIELDHAND_TODAY"] = data["fixed_date"]

    from fieldhand import config
    from fieldhand.agent import build_graph, make_llms

    if args.dry_run:
        from fieldhand.dryrun import DryRunModel
        model_label, mode = "dry-run (scripted stand-in)", "dry-run"
        make_graph = lambda: build_graph(DryRunModel())
    else:
        if not config.has_api_key():
            print("No GOOGLE_API_KEY found in .env")
            return 1
        model_label, mode = args.model or config.model_name(), "live"
        llm, fallback, reviewer = make_llms(model_label)
        make_graph = lambda: build_graph(llm, fallback_llm=fallback, reviewer_llm=reviewer)

    out_dir = Path(args.resume) if args.resume else ROOT / "evals" / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    log = out_dir / "results.jsonl"
    done = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    finished = {(r["case"], r["trial"]) for r in done if not r.get("infra_error")}
    meta = {"run": out_dir.name, "model": model_label, "mode": mode, "k": args.k,
            "fallback": config.fallback_model_name(), "started": datetime.now().isoformat(timespec="seconds")}

    total = len(cases) * args.k
    i = 0
    for case in cases:
        for trial in range(1, args.k + 1):
            i += 1
            if (case["id"], trial) in finished:
                continue
            print(f"[{i}/{total}] {case['id']} trial {trial} ... ", end="", flush=True)
            result = None
            for attempt in range(3):
                try:
                    result = run_trial(case, trial, make_graph, model_label, mode, out_dir)
                    break
                except Exception as exc:  # API outage or rate limit: wait and retry, never count as a failure
                    msg = f"{type(exc).__name__}: {str(exc)[:160]}"
                    print(f"\n    infrastructure error ({msg}); waiting {20 * (attempt + 1)} s", flush=True)
                    time.sleep(20 * (attempt + 1))
                    result = {"case": case["id"], "category": case["category"], "trial": trial, "passed": False,
                              "failed_checks": [], "infra_error": msg}
            done = [r for r in done if (r["case"], r["trial"]) != (case["id"], trial)] + [result]
            with log.open("a") as fh:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            if result.get("infra_error"):
                print("SKIPPED (infrastructure)")
            else:
                print("PASS" if result["passed"] else "FAIL: " + "; ".join(result["failed_checks"]))
            if not args.dry_run:
                time.sleep(args.pause)

    latest = {}
    for r in done:
        latest[(r["case"], r["trial"])] = r
    results = sorted(latest.values(), key=lambda r: (r["case"], r["trial"]))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print()
    print(summarize(data, cases, results, args.k, out_dir, meta))
    print(f"\nSaved: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
