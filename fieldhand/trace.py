"""Turn the agent's streamed updates into an execution trace, and export it.

Trace shape: reasoning -> tool calls -> observations -> ... -> answer, grouped by
iteration (one model call per iteration), with timing and token counts.
Exports: JSON (full data), Markdown (readable), HTML (branded, print to PDF).
"""

import html
import json
from pathlib import Path

from . import config


def split_content(content) -> tuple[list[str], str]:
    """Split a model message into (thought summaries, visible text)."""
    thoughts, texts = [], []
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict):
                kind = block.get("type")
                if kind == "thinking":
                    thoughts.append(block.get("thinking") or "")
                elif kind == "reasoning":
                    thoughts.append(block.get("reasoning") or "")
                elif kind == "text":
                    texts.append(block.get("text") or "")
    return [t.strip() for t in thoughts if t and t.strip()], "\n".join(t for t in texts if t).strip()


def _parse(content):
    if isinstance(content, str):
        try:
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return content
    return content


def build_trace(events: list[dict], *, run_id, started_at, model, mode, max_iterations, sender_phone, text,
                latency) -> dict:
    steps, tokens = [], {"input": 0, "output": 0, "total": 0}
    iteration, stop_reason, reply = 0, "", ""
    models, gate_decisions, reviews = set(), [], []
    for ev in events:
        node, t, delta = ev["node"], ev["t"], ev["delta"]
        if node == "agent":
            iteration = delta.get("iterations", iteration + 1)
            for msg in delta.get("messages", []):
                usage = getattr(msg, "usage_metadata", None) or {}
                tokens["input"] += usage.get("input_tokens", 0)
                tokens["output"] += usage.get("output_tokens", 0)
                tokens["total"] += usage.get("total_tokens", 0)
                thoughts, visible = split_content(msg.content)
                for th in thoughts:
                    steps.append({"kind": "thought", "iteration": iteration, "t": t, "text": th})
                if msg.tool_calls:
                    if visible:
                        steps.append({"kind": "reasoning", "iteration": iteration, "t": t, "text": visible})
                    for tc in msg.tool_calls:
                        steps.append({"kind": "tool_call", "iteration": iteration, "t": t, "name": tc["name"],
                                      "args": tc["args"], "id": tc.get("id")})
                elif visible:
                    reply = visible
                    steps.append({"kind": "answer", "iteration": iteration, "t": t, "text": visible})
                name = (getattr(msg, "response_metadata", None) or {}).get("model_name")
                if name:
                    models.add(name)
        elif node == "__gate_request__":
            steps.append({"kind": "approval_request", "iteration": iteration, "t": t, "calls": delta.get("calls", [])})
        elif node == "__gate_decision__":
            gate_decisions.append(bool(delta.get("approved")))
            steps.append({"kind": "approval_decision", "iteration": iteration, "t": t,
                          "approved": bool(delta.get("approved")), "note": delta.get("note", "")})
        elif node == "gate":
            for msg in delta.get("messages", []):
                steps.append({"kind": "observation", "iteration": iteration, "t": t, "name": msg.name,
                              "status": getattr(msg, "status", "error"), "content": _parse(msg.content)})
        elif node == "reviewer":
            r = delta.get("review", {})
            reviews.append(r.get("outcome"))
            steps.append({"kind": "review", "iteration": iteration, "t": t, "outcome": r.get("outcome"),
                          "code_issues": r.get("code_issues", []), "llm_issues": r.get("llm_issues", []),
                          "note": r.get("note", "")})
            if r.get("outcome") in ("revise", "blocked"):
                reply = ""
                for prev in reversed(steps[:-1]):
                    if prev["kind"] == "answer":
                        prev["kind"] = "draft"
                        break
        elif node == "tools":
            for msg in delta.get("messages", []):
                steps.append({"kind": "observation", "iteration": iteration, "t": t, "name": msg.name,
                              "status": getattr(msg, "status", "success"), "content": _parse(msg.content)})
        elif node == "guard":
            stop_reason = delta.get("stop_reason", "guard")
            for msg in delta.get("messages", []):
                if msg.type == "ai":
                    reply = msg.content
                    steps.append({"kind": "guard", "iteration": iteration, "t": t, "reason": stop_reason,
                                  "text": msg.content})
        elif node == "finish":
            stop_reason = delta.get("stop_reason", "final_answer")

    tool_calls = [s for s in steps if s["kind"] == "tool_call"]
    return {
        "run_id": run_id,
        "started_at": started_at,
        "business": config.BUSINESS_NAME,
        "agent": "intake",
        "model": model,
        "mode": mode,
        "input": {"from": sender_phone, "text": text},
        "reply": reply,
        "stop_reason": stop_reason,
        "iterations": iteration,
        "max_iterations": max_iterations,
        "tool_call_count": len(tool_calls),
        "tools_used": sorted({s["name"] for s in tool_calls}),
        "tokens": tokens,
        "latency_s": latency,
        "models_used": sorted(models),
        "gate_decisions": gate_decisions,
        "reviews": reviews,
        "steps": steps,
    }


STOP_LABELS = {
    "final_answer": "Final answer (model stopped calling tools)",
    "max_iterations": "Max-iteration guard",
    "empty_reply": "Guard (empty model reply)",
    "max_tool_calls": "Guard (tool-call limit)",
    "reviewer_blocked": "Guard (reviewer blocked the reply twice)",
}


def _review_text(s: dict) -> str:
    label = {"sent": "Passed. Reply sent.", "revise": "Sent back for one revision.",
             "blocked": "Blocked. Handed to the guard."}.get(s["outcome"], s["outcome"])
    issues = s["code_issues"] + s["llm_issues"]
    extra = (" Issues: " + " | ".join(issues)) if issues else ""
    note = f" ({s['note']})" if s.get("note") else ""
    return label + extra + note


def _fmt_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) if not isinstance(obj, str) else obj


def to_markdown(tr: dict) -> str:
    lines = [f"# Fieldhand execution trace: {tr['run_id']}", ""]
    if tr["mode"] != "live":
        lines += ["> **DRY RUN.** A scripted stand-in model produced this trace to test the plumbing. "
                  "It is not a run of the real agent.", ""]
    lines += [
        "| | |", "|---|---|",
        f"| Agent | Intake agent for {tr['business']} |",
        f"| Model | `{tr['model']}` |",
        f"| Started | {tr['started_at']} |",
        f"| Stop reason | {STOP_LABELS.get(tr['stop_reason'], tr['stop_reason'])} |",
        f"| Iterations | {tr['iterations']} of {tr['max_iterations']} allowed |",
        f"| Tool calls | {tr['tool_call_count']} ({', '.join(tr['tools_used']) or 'none'}) |",
        f"| Tokens | {tr['tokens']['total']:,} ({tr['tokens']['input']:,} in, {tr['tokens']['output']:,} out) |",
        f"| Latency | {tr['latency_s']} s |",
        "", "## Input", "", f"SMS from {tr['input']['from']}:", "",
        *[f"> {line}" for line in tr["input"]["text"].splitlines()], "", "## Steps", "",
    ]
    current = None
    for s in tr["steps"]:
        if s["iteration"] != current and s["kind"] != "observation":
            current = s["iteration"]
            lines += [f"### Iteration {current}", ""]
        if s["kind"] == "thought":
            lines += [f"**Thought summary** ({s['t']} s)", "", s["text"], ""]
        elif s["kind"] == "reasoning":
            lines += [f"**Reasoning** ({s['t']} s): {s['text']}", ""]
        elif s["kind"] == "tool_call":
            lines += [f"**Tool call** `{s['name']}`", "", "```json", _fmt_json(s["args"]), "```", ""]
        elif s["kind"] == "observation":
            label = "Observation" if s["status"] != "error" else "Observation (error)"
            lines += [f"**{label}** from `{s['name']}` ({s['t']} s)", "", "```json", _fmt_json(s["content"]), "```", ""]
        elif s["kind"] == "answer":
            lines += [f"**Final answer** ({s['t']} s)", "", *[f"> {line}" for line in s["text"].splitlines()], ""]
        elif s["kind"] == "guard":
            lines += [f"**Guard stopped the run** ({s['reason']}). Hand-off reply sent:", "", f"> {s['text']}", ""]
        elif s["kind"] == "draft":
            lines += [f"**Draft reply, not sent** ({s['t']} s)", "", *[f"> {line}" for line in s["text"].splitlines()], ""]
        elif s["kind"] == "approval_request":
            lines += ["**Human gate: owner approval requested** (paused before the tool ran)", "", "```json",
                      _fmt_json(s["calls"]), "```", ""]
        elif s["kind"] == "approval_decision":
            lines += [f"**Human gate decision:** {'Approved' if s['approved'] else 'Rejected'}"
                      + (f" ({s['note']})" if s["note"] else ""), ""]
        elif s["kind"] == "review":
            lines += [f"**Policy reviewer** ({s['t']} s): {_review_text(s)}", ""]
    return "\n".join(lines)


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700&family=Newsreader:ital,wght@0,400;1,400&display=swap');
:root{--forest:#1F3D2B;--amber:#E0A43A;--amber-ink:#96650F;--cream:#F2EBDD;--paper:#FAF6EE;--ink:#1C2A21;--muted:#4A5A4F;--rule:#E3DBCB}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font-family:'Newsreader',Georgia,serif;font-size:15px;line-height:1.55;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.wrap{max-width:820px;margin:0 auto;padding:32px 28px 48px}
.top{display:flex;align-items:center;gap:10px;font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:20px;color:var(--forest)}
.top svg{width:28px;height:28px}.top span.lbl{margin-left:auto;font-weight:500;font-size:11px;letter-spacing:1.6px;text-transform:uppercase;color:var(--muted)}
h1{font-family:'Bricolage Grotesque',sans-serif;font-size:26px;color:var(--forest);margin:18px 0 4px}
.sub{font-family:'Bricolage Grotesque',sans-serif;color:var(--muted);font-size:13px;margin-bottom:16px}
.dry{background:#FBE9D0;border:1px solid var(--amber);border-radius:8px;padding:10px 14px;font-family:'Bricolage Grotesque',sans-serif;font-size:13px;margin:12px 0}
.meta{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:10px 0 18px}
.meta div{background:var(--paper);border:1px solid var(--rule);border-radius:8px;padding:8px 10px}
.meta b{display:block;font-family:'Bricolage Grotesque',sans-serif;font-size:10px;letter-spacing:1.4px;text-transform:uppercase;color:var(--amber-ink)}
.meta span{font-family:'Bricolage Grotesque',sans-serif;font-size:14px;color:var(--forest)}
h2{font-family:'Bricolage Grotesque',sans-serif;font-size:17px;color:var(--forest);margin:22px 0 8px;border-bottom:2px solid var(--amber);display:inline-block;padding-bottom:2px}
.sms{background:var(--cream);border-radius:14px 14px 14px 4px;padding:12px 14px;max-width:560px;white-space:pre-wrap}
.reply{background:var(--forest);color:var(--cream);border-radius:14px 14px 4px 14px;padding:12px 14px;max-width:560px;margin-left:auto;white-space:pre-wrap}
.from{font-family:'Bricolage Grotesque',sans-serif;font-size:12px;color:var(--muted);margin:0 0 4px}
.it{margin:14px 0 6px;font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:12px;letter-spacing:1.5px;text-transform:uppercase;color:var(--amber-ink)}
.step{border-left:3px solid var(--rule);padding:4px 0 6px 12px;margin:0 0 8px;break-inside:avoid}
.step.call{border-color:var(--forest)}.step.obs{border-color:var(--amber)}.step.err{border-color:#B4462B}.step.ans{border-color:var(--forest)}.step.guard{border-color:#B4462B}.step.gate{border-color:#3B6EA8}.step.rev{border-color:#5E8C61}
.k{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:12px;color:var(--forest)}.t{font-family:'Bricolage Grotesque',sans-serif;font-size:11px;color:var(--muted);margin-left:6px}
pre{background:var(--paper);border:1px solid var(--rule);border-radius:6px;padding:8px 10px;font-size:11.5px;line-height:1.45;white-space:pre-wrap;word-break:break-word;margin:4px 0 0;font-family:ui-monospace,Menlo,Consolas,monospace}
.thought{font-style:italic;color:var(--muted)}
.foot{margin-top:28px;font-family:'Bricolage Grotesque',sans-serif;font-size:11px;color:var(--muted);border-top:1px solid var(--rule);padding-top:8px}
@media print{.wrap{padding:0}}
"""

MARK_SVG = ('<svg viewBox="0 0 120 120" aria-hidden="true"><rect width="120" height="120" rx="28" fill="#1F3D2B"/>'
            '<polyline points="28,55 60,27 92,55" fill="none" stroke="#E0A43A" stroke-width="10" '
            'stroke-linecap="round" stroke-linejoin="round"/><g stroke="#F2EBDD" stroke-width="8" stroke-linecap="round">'
            '<line x1="40" y1="67" x2="80" y2="67"/><line x1="40" y1="81" x2="80" y2="81"/>'
            '<line x1="40" y1="95" x2="64" y2="95"/></g></svg>')


def steps_html(tr: dict) -> str:
    """The step list as an HTML fragment (used by the HTML export and the Streamlit app)."""
    e = html.escape
    parts, current = [], None
    for s in tr["steps"]:
        if s["iteration"] != current and s["kind"] != "observation":
            current = s["iteration"]
            parts.append(f"<div class='it'>Iteration {current}</div>")
        k = s["kind"]
        if k == "thought":
            parts.append(f"<div class='step'><span class='k'>Thought summary</span><span class='t'>{s['t']} s</span>"
                         f"<div class='thought'>{e(s['text'])}</div></div>")
        elif k == "reasoning":
            parts.append(f"<div class='step'><span class='k'>Reasoning</span><span class='t'>{s['t']} s</span>"
                         f"<div>{e(s['text'])}</div></div>")
        elif k == "tool_call":
            parts.append(f"<div class='step call'><span class='k'>Tool call · {e(s['name'])}</span>"
                         f"<pre>{e(_fmt_json(s['args']))}</pre></div>")
        elif k == "observation":
            cls = "err" if s["status"] == "error" else "obs"
            parts.append(f"<div class='step {cls}'><span class='k'>Observation · {e(s['name'])}</span>"
                         f"<span class='t'>{s['t']} s</span><pre>{e(_fmt_json(s['content']))}</pre></div>")
        elif k == "answer":
            parts.append(f"<div class='step ans'><span class='k'>Final answer</span><span class='t'>{s['t']} s</span></div>"
                         f"<div class='reply'>{e(s['text'])}</div>")
        elif k == "guard":
            parts.append(f"<div class='step guard'><span class='k'>Guard stopped the run ({e(s['reason'])})</span></div>"
                         f"<div class='reply'>{e(s['text'])}</div>")
        elif k == "draft":
            parts.append(f"<div class='step err'><span class='k'>Draft reply, not sent</span><span class='t'>{s['t']} s</span>"
                         f"<pre>{e(s['text'])}</pre></div>")
        elif k == "approval_request":
            parts.append(f"<div class='step gate'><span class='k'>Human gate · owner approval requested</span>"
                         f"<span class='t'>paused before the tool ran</span><pre>{e(_fmt_json(s['calls']))}</pre></div>")
        elif k == "approval_decision":
            verdict = "Approved" if s["approved"] else "Rejected"
            parts.append(f"<div class='step gate'><span class='k'>Human gate · {verdict}</span>"
                         f"{'<div>' + e(s['note']) + '</div>' if s['note'] else ''}</div>")
        elif k == "review":
            cls = "rev" if s["outcome"] == "sent" else "err"
            parts.append(f"<div class='step {cls}'><span class='k'>Policy reviewer (second agent)</span>"
                         f"<span class='t'>{s['t']} s</span><div>{e(_review_text(s))}</div></div>")
    return "".join(parts)


def to_html(tr: dict) -> str:
    e = html.escape
    meta = [("Model", tr["model"]), ("Stop reason", STOP_LABELS.get(tr["stop_reason"], tr["stop_reason"])),
            ("Iterations", f"{tr['iterations']} of {tr['max_iterations']}"),
            ("Tool calls", str(tr["tool_call_count"])),
            ("Tokens", f"{tr['tokens']['total']:,}"), ("Latency", f"{tr['latency_s']} s"),
            ("Started", tr["started_at"].replace("T", " ")), ("Run", tr["run_id"])]
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Fieldhand trace {e(tr['run_id'])}</title>",
             f"<style>{_CSS}</style></head><body><div class='wrap'>",
             f"<div class='top'>{MARK_SVG}Fieldhand<span class='lbl'>Execution trace</span></div>",
             f"<h1>Intake agent run</h1><div class='sub'>{e(tr['business'])} (fictional demo business) · reasoning → tool calls → observations → answer</div>"]
    if tr["mode"] != "live":
        parts.append("<div class='dry'><b>Dry run.</b> A scripted stand-in model produced this trace to test the "
                     "plumbing. It is not a run of the real agent.</div>")
    parts.append("<div class='meta'>" + "".join(f"<div><b>{e(k)}</b><span>{e(v)}</span></div>" for k, v in meta) + "</div>")
    parts.append(f"<h2>Input</h2><p class='from'>SMS from {e(tr['input']['from'])}</p><div class='sms'>{e(tr['input']['text'])}</div>")
    parts.append("<h2>Steps</h2>")
    parts.append(steps_html(tr))
    parts.append(f"<div class='foot'>Generated by Fieldhand · CIS 4394 Agentic AI · run {e(tr['run_id'])}</div></div></body></html>")
    return "".join(parts)


def save_trace(tr: dict, folder: Path | None = None) -> dict[str, Path]:
    folder = folder or (config.ROOT / "traces")
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / tr["run_id"]
    paths = {"json": stem.with_suffix(".json"), "md": stem.with_suffix(".md"), "html": stem.with_suffix(".html")}
    paths["json"].write_text(json.dumps(tr, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["md"].write_text(to_markdown(tr), encoding="utf-8")
    paths["html"].write_text(to_html(tr), encoding="utf-8")
    return paths
