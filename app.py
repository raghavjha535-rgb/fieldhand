"""Fieldhand public demo.  Run locally:  streamlit run app.py

Left: the customer's text thread (you play the customer) and the owner's approval card.
Right: the execution trace for the last message, the calendar, and the owner's inbox.
Every visitor gets a private sandbox copy of the demo database.
"""

import json
import os
import re
import time
import uuid
from datetime import date
from pathlib import Path

import streamlit as st

# Hosted app: copy Streamlit secrets into the environment before the settings are read.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float)):
            os.environ.setdefault(_k, str(_v))
except Exception:
    pass

import pandas as pd  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from fieldhand import config, db  # noqa: E402
from fieldhand.scenarios import SCENARIOS  # noqa: E402
from fieldhand.trace import MARK_SVG, STOP_LABELS, build_trace, steps_html, to_html, to_markdown  # noqa: E402

st.set_page_config(page_title="Fieldhand · Intake agent", page_icon=":material/pest_control:", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;600;700&display=swap');
:root{--forest:#1F3D2B;--amber:#E0A43A;--amber-ink:#96650F;--cream:#F2EBDD;--muted:#4A5A4F}
h1,h2,h3,.fh-word{font-family:'Bricolage Grotesque',sans-serif!important;color:var(--forest)}
.block-container{padding-top:3.2rem}
[data-testid='stMetricValue']{font-size:24px;font-family:'Bricolage Grotesque',sans-serif;color:var(--forest)}
.fh-head{display:flex;align-items:center;gap:14px;margin-bottom:6px}
.fh-head svg{width:46px;height:46px;flex:none}
.fh-word{font-size:30px;font-weight:700;line-height:1;letter-spacing:-0.5px}
.fh-sub{font-family:'Bricolage Grotesque',sans-serif;color:var(--muted);font-size:14px;margin-top:4px}
.fh-label{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:11px;letter-spacing:1.6px;text-transform:uppercase;color:var(--amber-ink);margin:4px 0 6px}
.fh-banner{background:#FBF3E2;border:1px solid #E0A43A;border-radius:8px;padding:8px 12px;font-size:13px;margin:4px 0 12px;color:#1C2A21}
.fh-gate{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;color:#3B6EA8;font-size:15px}
</style>
""", unsafe_allow_html=True)

TRACE_CSS = """
.fh-trace{font-family:'Source Sans Pro','Source Sans 3',sans-serif;color:#1C2A21;font-size:15px;line-height:1.5}
.fh-trace .it{margin:14px 0 6px;font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:#96650F}
.fh-trace .step{border-left:3px solid #E3DBCB;padding:3px 0 6px 12px;margin:0 0 8px}
.fh-trace .step.call{border-color:#1F3D2B}.fh-trace .step.obs{border-color:#E0A43A}.fh-trace .step.gate{border-color:#3B6EA8}
.fh-trace .step.rev{border-color:#5E8C61}.fh-trace .step.err,.fh-trace .step.guard{border-color:#B4462B}.fh-trace .step.ans{border-color:#1F3D2B}
.fh-trace .k{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:13px;color:#1F3D2B}
.fh-trace .t{font-size:11px;color:#4A5A4F;margin-left:6px}
.fh-trace pre{background:#FAF6EE;border:1px solid #E3DBCB;border-radius:6px;padding:8px 10px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.45;white-space:pre-wrap;word-break:break-word;margin:4px 0 0;max-height:260px;overflow:auto;color:#1C2A21}
.fh-trace .thought{font-style:italic;color:#4A5A4F;font-size:14px}
.fh-trace .reply{background:#1F3D2B;color:#F2EBDD;border-radius:14px 14px 4px 14px;padding:10px 14px;margin:2px 0 8px;white-space:pre-wrap}
"""

PERSONAS = {
    "Maria Gonzales (new lead)": "+1 404-555-0142",
    "Dana Whitfield (existing customer)": "+1 404-555-0117",
    "Restaurant manager (new lead)": "+1 678-555-0190",
    "Peachtree Corners resident": "+1 770-555-0108",
    "Unknown number": "+1 404-555-0199",
}
SAMPLES = [("New lead: ants, book it", "ants", 0), ("Existing customer: ants came back", "existing", 1),
           ("Restaurant wants a quote", "commercial", 2), ("Outside the service area", "out_of_area", 3),
           ("Red team: prompt injection", "injection", 4)]
SESSIONS_DIR = config.ROOT / "data" / "sessions"
USAGE_FILE = config.ROOT / "data" / "usage.json"


def sms_md(text: str) -> str:
    """Show SMS text literally: no bullet points, headings, or math from dashes, hashes, or dollar signs."""
    text = text.replace("\\", "\\\\").replace("$", "\\$")
    return re.sub(r"(?m)^(\s*)([-*+#>]|\d+\.)", r"\1\\\2", text)


# ---------------------------------------------------------------- session setup
ss = st.session_state
ss.setdefault("sid", uuid.uuid4().hex[:12])
ss.setdefault("thread_id", uuid.uuid4().hex)
ss.setdefault("chat", [])
ss.setdefault("traces", [])
ss.setdefault("checkpointer", InMemorySaver())
ss.setdefault("persona", list(PERSONAS)[0])
ss.setdefault("turns", 0)
ss.setdefault("pending", None)

SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
if "cleaned" not in ss:  # drop sandboxes older than a day
    ss.cleaned = True
    for old in SESSIONS_DIR.glob("*.db"):
        try:
            if time.time() - old.stat().st_mtime > 86400:
                old.unlink()
        except OSError:
            pass
config.use_db(SESSIONS_DIR / f"{ss.sid}.db")  # this visitor's private sandbox
db.ensure_db()


def new_conversation():
    ss.thread_id = uuid.uuid4().hex
    ss.chat = []
    ss.pending = None
    ss.checkpointer = InMemorySaver()


def daily_usage(increment: bool = False) -> int:
    today = date.today().isoformat()
    try:
        data = json.loads(USAGE_FILE.read_text())
    except Exception:
        data = {}
    count = data.get("count", 0) if data.get("date") == today else 0
    if increment:
        count += 1
        USAGE_FILE.write_text(json.dumps({"date": today, "count": count}))
    return count


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("<div class='fh-label'>Demo controls</div>", unsafe_allow_html=True)
    persona = st.selectbox("Who is texting?", list(PERSONAS), index=list(PERSONAS).index(ss.persona))
    if persona != ss.persona:
        ss.persona = persona
        new_conversation()
    has_key = config.has_api_key()
    engine = st.radio("Model", ["Gemini (live)", "Dry run (scripted, no key)"], index=0 if has_key else 1,
                      help="Dry run replays a fixed booking script to test the plumbing. It is not the agent.")
    st.caption(f"Agent and reviewer: {config.model_name()} · fallback: {config.fallback_model_name() or 'none'}")
    max_iter = st.slider("Max model calls per message (guard)", 2, 8, min(8, config.max_iterations()),
                         help="Lower it to watch the guard stop a run.")
    weather_mode = st.radio("Forecast", ["live", "mock"], horizontal=True,
                            help="live calls api.weather.gov; mock uses a fixed forecast")
    st.markdown("<div class='fh-label'>Sample texts</div>", unsafe_allow_html=True)
    clicked = None
    for label, key, idx in SAMPLES:
        if st.button(label, key=f"s_{key}", width="stretch", disabled=ss.pending is not None):
            clicked = (key, idx)
    st.markdown("<div class='fh-label'>Your sandbox</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("New chat", width="stretch"):
        new_conversation()
        st.rerun()
    if c2.button("Reset data", width="stretch", help="Restore the seeded calendar in your sandbox"):
        db.reset_db()
        new_conversation()
        ss.traces = []
        st.rerun()
    st.caption(f"Messages this session: {ss.turns} of {config.max_turns_per_session()}. "
               f"Limits per message: {config.max_message_chars()} characters, {max_iter} model calls, "
               f"{config.max_tool_calls()} tool calls, {config.max_output_tokens()} output tokens per answer.")


# ---------------------------------------------------------------- running the agent
def make_graph():
    from fieldhand.agent import build_graph, make_llms

    os.environ["FIELDHAND_WEATHER"] = weather_mode
    if engine != "Gemini (live)":
        from fieldhand.dryrun import DryRunModel
        return build_graph(DryRunModel(), max_iter, checkpointer=ss.checkpointer), "dry-run (scripted stand-in)", "dry-run"
    llm, fallback, reviewer = make_llms()
    graph = build_graph(llm, max_iter, checkpointer=ss.checkpointer, fallback_llm=fallback, reviewer_llm=reviewer)
    return graph, config.model_name(), "live"


def finish_turn(turn):
    if turn["status"] == "awaiting_approval":
        ss.pending = turn
        return
    ss.pending = None
    ss.chat.append({"role": "assistant", "text": turn["trace"]["reply"]})
    ss.traces.append(turn["trace"])


def send(text: str, phone: str):
    from fieldhand.agent import start_turn

    if ss.turns >= config.max_turns_per_session():
        ss.chat.append({"role": "error", "text": "This session reached its message limit. Reload the page to start over."})
        return
    if daily_usage() >= config.daily_run_cap():
        ss.chat.append({"role": "error", "text": "The demo reached today's usage cap. Please try again tomorrow."})
        return
    text = text[: config.max_message_chars()]
    ss.turns += 1
    daily_usage(increment=True)
    ss.chat.append({"role": "user", "text": text})
    graph, label, mode = make_graph()
    try:
        with st.spinner("Agent working..."):
            finish_turn(start_turn(graph, text, phone, thread_id=ss.thread_id, model=label,
                                   max_iterations=max_iter, mode=mode))
    except Exception as exc:
        ss.chat.append({"role": "error", "text": f"The model service returned an error ({type(exc).__name__}). "
                                                 "Wait a minute and try again."})


def decide(approved: bool, note: str):
    from fieldhand.agent import resume_turn

    graph, _, _ = make_graph()
    try:
        with st.spinner("Resuming..."):
            finish_turn(resume_turn(graph, ss.pending, approved, note))
    except Exception as exc:
        ss.pending = None
        ss.chat.append({"role": "error", "text": f"The model service returned an error ({type(exc).__name__})."})


# ---------------------------------------------------------------- header
st.markdown(f"<div class='fh-head'>{MARK_SVG}<div><div class='fh-word'>Fieldhand</div>"
            f"<div class='fh-sub'>Intake agent + policy reviewer for {config.BUSINESS_NAME} (fictional) · "
            f"owner approval before any booking</div></div></div>", unsafe_allow_html=True)
st.markdown("<div class='fh-banner'><b>Public demo.</b> The business, customers, and calendar are fictional, and "
            "your bookings stay in a private sandbox. Messages are processed by Google's Gemini API, so don't type "
            "real personal information. No texts, emails, or payments are ever sent.</div>", unsafe_allow_html=True)

left, right = st.columns([5, 7], gap="large")

with left:
    st.markdown(f"<div class='fh-label'>Customer texts · {ss.persona}</div>", unsafe_allow_html=True)
    if clicked:
        key, idx = clicked
        target = list(PERSONAS)[idx]
        if target != ss.persona:
            ss.persona = target
            new_conversation()
        send(SCENARIOS[key]["text"], PERSONAS[target])
        st.rerun()
    thread = st.container(height=520, border=True)
    with thread:
        if not ss.chat:
            st.caption("Text the business as the customer, or pick a sample in the sidebar.")
        for m in ss.chat:
            if m["role"] == "user":
                st.chat_message("user", avatar=":material/smartphone:").markdown(sms_md(m["text"]))
            elif m["role"] == "assistant":
                st.chat_message("assistant", avatar=":material/pest_control:").markdown(sms_md(m["text"]))
            else:
                st.error(m["text"])
        if ss.pending:
            req = ss.pending["pending"]
            with st.chat_message("assistant", avatar=":material/gavel:"):
                st.markdown("<div class='fh-gate'>Owner approval needed before this runs</div>", unsafe_allow_html=True)
                st.caption("You are playing the owner. The agent proposed this call; nothing is written to the "
                           "calendar until you approve. Arguments exactly as the agent sent them:")
                for call in req["calls"]:
                    st.code(f"{call['name']}(\n" + "\n".join(f"  {k} = {json.dumps(v, ensure_ascii=False)}"
                                                         for k, v in call["args"].items()) + "\n)", language="python",
                            wrap_lines=True)
                    prev = call.get("price_preview") or {}
                    if prev.get("price") is not None:
                        st.caption(f"Price book will charge: ${prev['price']} ({prev.get('service', '')}, "
                                   f"{prev.get('pricing_basis', '')}). {prev.get('note', '')}")
                note = st.text_input("Note to the agent (optional)", key=f"note_{len(ss.traces)}")
                a, r = st.columns(2)
                if a.button("Approve booking", type="primary", width="stretch"):
                    decide(True, note)
                    st.rerun()
                if r.button("Reject", width="stretch"):
                    decide(False, note)
                    st.rerun()
    limit_hit = ss.turns >= config.max_turns_per_session()
    prompt = st.chat_input(f"Text {config.BUSINESS_NAME} as the customer...", max_chars=config.max_message_chars(),
                           disabled=(engine == "Gemini (live)" and not has_key) or ss.pending is not None or limit_hit)
    if prompt:
        send(prompt, PERSONAS[ss.persona])
        st.rerun()

with right:
    tab_trace, tab_sched, tab_inbox, tab_hist = st.tabs(["Agent trace", "Schedule", "Owner inbox", "Run history"])

    with tab_trace:
        if ss.pending:
            p = ss.pending
            tr = build_trace(p["events"], latency=round(p["clock"]["active"], 2), **p["meta"])
            st.info("Paused at the human gate. Approve or reject in the chat to continue.")
        elif ss.traces:
            tr = ss.traces[-1]
        else:
            tr = None
            st.info("Send a message to see the agent's reasoning, tool calls, observations, the reviewer's check, "
                    "and the answer.")
        if tr:
            if tr["mode"] != "live":
                st.warning("Dry run: scripted stand-in model, not the real agent.")
            m = st.columns(5)
            stop = tr["stop_reason"] or ("Paused" if ss.pending else "")
            m[0].metric("Stop reason", "Answer" if stop == "final_answer" else (stop and "Guard" if stop != "Paused" else "Paused"),
                        help=STOP_LABELS.get(tr["stop_reason"], tr["stop_reason"]))
            m[1].metric("Model calls", f"{tr['iterations']} / {tr['max_iterations']}")
            m[2].metric("Tool calls", tr["tool_call_count"])
            m[3].metric("Tokens", f"{tr['tokens']['total']:,}")
            m[4].metric("Latency", f"{tr['latency_s']} s")
            st.html(f"<style>{TRACE_CSS}</style><div class='fh-trace'>{steps_html(tr)}</div>")
            if not ss.pending:
                d = st.columns(3)
                d[0].download_button("Trace (HTML)", to_html(tr), f"{tr['run_id']}.html", "text/html", width="stretch")
                d[1].download_button("Trace (Markdown)", to_markdown(tr), f"{tr['run_id']}.md", "text/markdown",
                                     width="stretch")
                d[2].download_button("Trace (JSON)", json.dumps(tr, indent=2), f"{tr['run_id']}.json",
                                     "application/json", width="stretch")

    with tab_sched:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT j.date, j.time_window AS window, c.name AS customer, s.name AS service, j.price, "
                "j.confirmation, j.created_by FROM jobs j JOIN customers c ON c.id = j.customer_id "
                "JOIN services s ON s.code = j.service_code WHERE j.status = 'scheduled' AND j.date >= ? "
                "ORDER BY j.date, j.time_window", (config.today().isoformat(),)).fetchall()
        df = pd.DataFrame([dict(r) for r in rows])
        if df.empty:
            st.caption("No upcoming jobs.")
        else:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%a %b %d")
            df["booked by"] = df.pop("created_by").map({"intake_agent": "Agent", "seed": "Existing"})
            df["price"] = df["price"].map(lambda p: f"${p}" if p else "Included")
            st.dataframe(df, hide_index=True, width="stretch")
            st.caption("Your private sandbox. Rows marked Agent were booked by the agent after owner approval.")

    with tab_inbox:
        with db.connect() as conn:
            rows = conn.execute("SELECT ticket, created_at, customer_name AS customer, phone, reason, details, status "
                                "FROM approvals ORDER BY id DESC").fetchall()
        if not rows:
            st.caption("Nothing waiting for the owner.")
        else:
            idf = pd.DataFrame([dict(r) for r in rows])
            idf["created_at"] = pd.to_datetime(idf["created_at"]).dt.strftime("%b %d %I:%M %p")
            st.dataframe(idf, hide_index=True, width="stretch")

    with tab_hist:
        if not ss.traces:
            st.caption("No runs yet.")
        else:
            hist = pd.DataFrame([{"run": t["run_id"], "from": t["input"]["from"], "stop": t["stop_reason"],
                                  "model calls": t["iterations"], "tools": ", ".join(t["tools_used"]),
                                  "gate": ", ".join("approved" if g else "rejected" for g in t.get("gate_decisions", [])),
                                  "reviewer": ", ".join(t.get("reviews", [])), "tokens": t["tokens"]["total"],
                                  "latency (s)": t["latency_s"]} for t in reversed(ss.traces)])
            st.dataframe(hist, hide_index=True, width="stretch")
