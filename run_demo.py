"""Run the Fieldhand intake agent once on a real customer text and save the trace.

Examples:
    python run_demo.py                          # Milestone I run: the "ants" scenario
    python run_demo.py --scenario commercial    # other scenarios: existing, out_of_area, injection
    python run_demo.py --text "Do you treat for termites? 30306" --phone "+1 404-555-0100"
    python run_demo.py --dry-run                # no API key: scripted stand-in model, labeled DRY RUN

Traces are saved to traces/<run_id>.json, .md and .html (open the .html and print to PDF).
"""

import argparse
import json
import os
import sys

from fieldhand import config, db
from fieldhand.scenarios import SCENARIOS


def show(tr: dict) -> None:
    print()
    print("=" * 72)
    print(f"FIELDHAND TRACE  {tr['run_id']}   model={tr['model']}   mode={tr['mode']}")
    print("=" * 72)
    print(f"INPUT (SMS from {tr['input']['from']}):\n  {tr['input']['text']}\n")
    current = None
    for s in tr["steps"]:
        if s["iteration"] != current and s["kind"] != "observation":
            current = s["iteration"]
            print(f"--- iteration {current} ---")
        k = s["kind"]
        if k == "thought":
            print(f"  THOUGHT     {s['text'][:400]}")
        elif k == "reasoning":
            print(f"  REASONING   {s['text']}")
        elif k == "tool_call":
            print(f"  TOOL CALL   {s['name']}({json.dumps(s['args'], ensure_ascii=False)})")
        elif k == "observation":
            body = json.dumps(s["content"], ensure_ascii=False) if not isinstance(s["content"], str) else s["content"]
            print(f"  OBSERVATION [{s['status']}] {body[:600]}")
        elif k in ("answer", "guard", "draft"):
            print(f"  {k.upper():11} {s['text']}")
        elif k == "approval_request":
            for c in s["calls"]:
                print(f"  HUMAN GATE  paused before {c['name']}({json.dumps(c['args'], ensure_ascii=False)})")
                print(f"              price book preview: {json.dumps(c['price_preview'], ensure_ascii=False)}")
        elif k == "approval_decision":
            print(f"  DECISION    {'approved' if s['approved'] else 'rejected'} {s['note']}")
        elif k == "review":
            issues = s["code_issues"] + s["llm_issues"]
            print(f"  REVIEWER    {s['outcome']}" + (f": {' | '.join(issues)}" if issues else "") + (f" ({s['note']})" if s["note"] else ""))
    print()
    if tr.get("models_used"):
        print(f"Models that answered: {', '.join(tr['models_used'])}")
    print(f"Stop reason: {tr['stop_reason']}   iterations: {tr['iterations']}/{tr['max_iterations']}   "
          f"tool calls: {tr['tool_call_count']}   tokens: {tr['tokens']['total']}   latency: {tr['latency_s']} s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="ants", choices=sorted(SCENARIOS))
    p.add_argument("--text", help="Custom customer text (overrides --scenario)")
    p.add_argument("--phone", default="+1 404-555-0100", help="Sender phone for --text")
    p.add_argument("--model", help=f"Gemini model ID (default {config.model_name()})")
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--weather", choices=["live", "mock"], help="Forecast source (default live)")
    p.add_argument("--keep-db", action="store_true", help="Keep existing demo data instead of resetting it")
    p.add_argument("--dry-run", action="store_true", help="Use the scripted stand-in model (no API key needed)")
    p.add_argument("--reject", action="store_true", help="Reject the booking at the human gate instead of approving it")
    args = p.parse_args()

    if args.weather:
        os.environ["FIELDHAND_WEATHER"] = args.weather
    if not args.keep_db:
        db.reset_db()

    from fieldhand.agent import build_graph, make_llms, run_turn
    from fieldhand.trace import save_trace

    if args.dry_run:
        from fieldhand.dryrun import DryRunModel
        llm, fallback, reviewer = DryRunModel(), None, None
        model, mode = "dry-run (scripted stand-in)", "dry-run"
    else:
        if not config.has_api_key():
            print("No GOOGLE_API_KEY found. Copy .env.example to .env and paste your key from "
                  "https://aistudio.google.com/apikey, or run with --dry-run to test the plumbing.")
            return 1
        model, mode = args.model or config.model_name(), "live"
        llm, fallback, reviewer = make_llms(model)

    text, phone = (args.text, args.phone) if args.text else (SCENARIOS[args.scenario]["text"], SCENARIOS[args.scenario]["phone"])
    graph = build_graph(llm, args.max_iterations, fallback_llm=fallback, reviewer_llm=reviewer)

    def approver(request):
        decision = not args.reject
        print(f"\n[human gate] {'APPROVING' if decision else 'REJECTING'} {request['action']} "
              f"{json.dumps(request['calls'][0]['args'], ensure_ascii=False)}")
        return decision

    try:
        tr = run_turn(graph, text, phone, model=model, max_iterations=args.max_iterations, mode=mode, approver=approver)
    except Exception as exc:  # surface API problems in plain words
        msg = str(exc)
        print(f"\nThe run failed: {type(exc).__name__}: {msg[:500]}")
        if "404" in msg or "not found" in msg.lower():
            print("Check the model ID. Set FIELDHAND_MODEL in .env to a model listed at "
                  "https://ai.google.dev/gemini-api/docs/models (for example gemini-3.5-flash-lite).")
        elif "429" in msg or "quota" in msg.lower() or "exhausted" in msg.lower():
            print("Free-tier rate limit hit. Wait a minute and run again.")
        elif "api key" in msg.lower() or "403" in msg or "401" in msg:
            print("The request was refused. Check GOOGLE_API_KEY in .env and that your network allows "
                  "generativelanguage.googleapis.com.")
        return 1

    show(tr)
    paths = save_trace(tr)
    print("\nSaved trace:")
    for kind, path in paths.items():
        print(f"  {kind:5} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
