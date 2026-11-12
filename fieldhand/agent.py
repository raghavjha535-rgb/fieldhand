"""Fieldhand's agent graph (Milestone II).

    START -> agent --(tool calls)----------------> tools -> agent
                   --(book_appointment proposed)--> gate --(approved)--> tools -> agent
                                                         --(rejected)--> agent
                   --(draft reply)----------------> reviewer --(pass)----> finish -> END
                                                             --(revise)--> agent      (once)
                                                             --(blocked)-> guard  -> END
                   --(any cap reached)-------------> guard -> END

Two agents: the intake agent (Gemini with six tools) writes; the policy reviewer
(critic.py, no tools) checks every draft before it is sent.

Human gate: book_appointment is the only action that writes to the calendar. The
graph pauses with interrupt() after the agent proposes the call and before the
tool runs. The app shows the exact arguments with Approve and Reject buttons.

Stop conditions: a reviewed final answer, or the guard. The guard fires when the
agent reaches the model-call limit or the tool-call limit, returns an empty
reply, or the reviewer blocks a reply twice. It files an owner ticket and sends
a hand-off reply. LangGraph's recursion_limit is a further backstop.
"""

import json
import time
import uuid
from datetime import datetime
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from . import config, critic, db, prompts
from .tools import ALL_TOOLS, booking_preview
from .trace import build_trace, split_content

MAX_REVISIONS = 1


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    sender_phone: str
    iterations: int
    tool_calls_turn: int
    revisions: int
    stop_reason: str
    gate: str
    review: dict


# ---------------------------------------------------------------- models
def make_llms(model: str | None = None):
    """(agent model, fallback model or None, reviewer model). Reads GOOGLE_API_KEY from the environment."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    def chat(name, thoughts):
        kw = {"model": name, "temperature": config.temperature(), "max_retries": 2,
              "max_output_tokens": config.max_output_tokens(), "timeout": 60}
        if thoughts:
            kw["include_thoughts"] = True
        return ChatGoogleGenerativeAI(**kw)

    primary = model or config.model_name()
    fb = config.fallback_model_name()
    agent_llm = chat(primary, config.show_thoughts())
    fallback = chat(fb, config.show_thoughts()) if fb and fb != primary else None
    reviewer = chat(primary, False)
    if fb and fb != primary:
        reviewer = reviewer.with_fallbacks([chat(fb, False)])
    return agent_llm, fallback, reviewer


# ---------------------------------------------------------------- graph
def build_graph(llm, max_iterations: int | None = None, checkpointer=None, fallback_llm=None, reviewer_llm=None):
    limit = max_iterations or config.max_iterations()
    tool_cap = config.max_tool_calls()
    model = llm.bind_tools(ALL_TOOLS)
    if fallback_llm is not None:
        model = model.with_fallbacks([fallback_llm.bind_tools(ALL_TOOLS)])
    use_reviewer = config.critic_enabled()

    def agent(state: AgentState) -> dict:
        messages = [SystemMessage(prompts.intake_system_prompt())] + state["messages"]
        response = model.invoke(messages)
        calls = len(getattr(response, "tool_calls", None) or [])
        return {"messages": [response], "iterations": state.get("iterations", 0) + 1,
                "tool_calls_turn": state.get("tool_calls_turn", 0) + calls}

    def route_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        if calls:
            if state["iterations"] >= limit or state.get("tool_calls_turn", 0) > tool_cap:
                return "guard"
            if any(tc["name"] in config.GATED_TOOLS for tc in calls):
                return "gate"
            return "tools"
        _, text = split_content(last.content)
        if not text:
            return "guard"
        return "reviewer" if use_reviewer else "finish"

    def gate(state: AgentState) -> dict:
        pending = state["messages"][-1].tool_calls
        gated = [tc for tc in pending if tc["name"] in config.GATED_TOOLS]
        request = {"action": gated[0]["name"],
                   "calls": [{"tool_call_id": tc["id"], "name": tc["name"], "args": tc["args"],
                              "price_preview": booking_preview(tc["args"])} for tc in gated],
                   "between": "agent proposed the call -> tool writes to the calendar"}
        decision = interrupt(request)  # pauses here until a person approves or rejects
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        note = (decision.get("note") or "") if isinstance(decision, dict) else ""
        if approved:
            return {"gate": "approved"}
        msgs = []
        for tc in pending:
            if tc["name"] in config.GATED_TOOLS:
                body = {"booked": False, "rejected_by_owner": True, "owner_note": note or "Rejected at the approval gate.",
                        "next_step": f"Do not say anything was booked. Tell the customer {config.OWNER_NAME} will "
                                     "text them to arrange a time."}
            else:
                body = {"skipped": True, "reason": "Sent together with a rejected booking. Call again if still needed."}
            msgs.append(ToolMessage(content=json.dumps(body), tool_call_id=tc["id"], name=tc["name"], status="error"))
        return {"messages": msgs, "gate": "rejected"}

    def route_gate(state: AgentState) -> str:
        return "tools" if state.get("gate") == "approved" else "agent"

    def reviewer(state: AgentState) -> dict:
        last = state["messages"][-1]
        _, draft = split_content(last.content)
        customer_text = next((m.content for m in reversed(state["messages"])
                              if isinstance(m, HumanMessage) and not str(m.content).startswith("[Policy reviewer")), "")
        code_issues = critic.code_checks(draft, state["messages"], state.get("sender_phone", ""))
        llm_issues, note, verdict = [], "", "pass"
        if reviewer_llm is not None:
            verdict, llm_issues, note = critic.llm_review(reviewer_llm, draft, state["messages"], customer_text)
        if code_issues:
            verdict = "revise"
        revisions = state.get("revisions", 0)
        review = {"verdict": verdict, "code_issues": code_issues, "llm_issues": llm_issues, "note": note,
                  "revision": revisions, "draft": draft}
        if verdict == "pass":
            return {"review": {**review, "outcome": "sent"}}
        if revisions < MAX_REVISIONS and state["iterations"] < limit:
            feedback = HumanMessage(content="[Policy reviewer, not the customer] Your draft was not sent. Fix these "
                                    "problems and write a new reply:\n- " + "\n- ".join(code_issues + llm_issues))
            return {"messages": [feedback], "revisions": revisions + 1, "review": {**review, "outcome": "revise"}}
        return {"review": {**review, "outcome": "blocked"}}

    def route_reviewer(state: AgentState) -> str:
        return {"sent": "finish", "revise": "agent"}.get(state["review"].get("outcome"), "guard")

    def finish(state: AgentState) -> dict:
        return {"stop_reason": "final_answer"}

    def guard(state: AgentState) -> dict:
        last = state["messages"][-1]
        pending = getattr(last, "tool_calls", None) or []
        if state.get("review", {}).get("outcome") == "blocked":
            reason = "reviewer_blocked"
        elif pending and state.get("tool_calls_turn", 0) > tool_cap:
            reason = "max_tool_calls"
        elif pending:
            reason = "max_iterations"
        else:
            reason = "empty_reply"
        skipped = [ToolMessage(content='{"skipped": true, "reason": "limit reached"}', tool_call_id=tc["id"],
                               name=tc["name"], status="error") for tc in pending]
        db.file_ticket(state.get("sender_phone", ""), "agent stopped by guard",
                       f"Stop reason: {reason} after {state['iterations']} model calls "
                       f"and {state.get('tool_calls_turn', 0)} tool calls.")
        return {"messages": skipped + [AIMessage(content=prompts.GUARD_REPLY, name="guard")], "stop_reason": reason}

    g = StateGraph(AgentState)
    for name, fn in [("agent", agent), ("gate", gate), ("reviewer", reviewer), ("finish", finish), ("guard", guard)]:
        g.add_node(name, fn)
    g.add_node("tools", ToolNode(ALL_TOOLS, handle_tool_errors=True))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_agent, {"tools": "tools", "gate": "gate", "reviewer": "reviewer",
                                                   "finish": "finish", "guard": "guard"})
    g.add_conditional_edges("gate", route_gate, {"tools": "tools", "agent": "agent"})
    g.add_conditional_edges("reviewer", route_reviewer, {"finish": "finish", "agent": "agent", "guard": "guard"})
    g.add_edge("tools", "agent")
    g.add_edge("finish", END)
    g.add_edge("guard", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


# ---------------------------------------------------------------- running a turn
def _stream(graph, inputs, cfg, events: list, clock: dict) -> dict | None:
    """Run until the graph ends or pauses. Returns the gate request if it paused."""
    t0 = time.perf_counter()
    pending = None
    for update in graph.stream(inputs, cfg, stream_mode="updates"):
        t = round(clock["active"] + time.perf_counter() - t0, 2)
        for node, delta in update.items():
            if node == "__interrupt__":
                pending = delta[0].value
                events.append({"node": "__gate_request__", "t": t, "delta": pending})
            else:
                events.append({"node": node, "t": t, "delta": delta or {}})
    clock["active"] += time.perf_counter() - t0
    return pending


def start_turn(graph, text: str, sender_phone: str, thread_id: str | None = None, model: str | None = None,
               max_iterations: int | None = None, mode: str = "live") -> dict:
    """Send one customer text. Returns a turn dict with status 'done' (and a trace) or 'awaiting_approval'."""
    db.ensure_db()
    limit = max_iterations or config.max_iterations()
    received = config.now_local()
    thread_id = thread_id or uuid.uuid4().hex
    turn = {
        "thread_id": thread_id, "events": [], "clock": {"active": 0.0},
        "meta": {"run_id": f"run-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}",
                 "started_at": received.isoformat(timespec="seconds"), "model": model or config.model_name(),
                 "mode": mode, "max_iterations": limit, "sender_phone": sender_phone, "text": text},
    }
    human = HumanMessage(content=f"[Incoming SMS from {sender_phone} at {received:%I:%M %p}]\n{text}")
    inputs = {"messages": [human], "sender_phone": sender_phone, "iterations": 0, "tool_calls_turn": 0,
              "revisions": 0, "stop_reason": "", "gate": "", "review": {}}
    return _advance(graph, turn, inputs)


def resume_turn(graph, turn: dict, approved: bool, note: str = "") -> dict:
    """Continue a paused turn with the person's decision at the gate."""
    decision = {"approved": bool(approved), "note": note}
    turn["events"].append({"node": "__gate_decision__", "t": round(turn["clock"]["active"], 2), "delta": decision})
    return _advance(graph, turn, Command(resume=decision))


def _advance(graph, turn: dict, inputs) -> dict:
    meta = turn["meta"]
    cfg = {"configurable": {"thread_id": turn["thread_id"]}, "recursion_limit": 3 * meta["max_iterations"] + 12,
           "run_name": "fieldhand-intake", "tags": ["fieldhand", "milestone-2"],
           "metadata": {"sender": meta["sender_phone"], "model": meta["model"], "mode": meta["mode"]}}
    pending = _stream(graph, inputs, cfg, turn["events"], turn["clock"])
    if pending is not None:
        turn.update(status="awaiting_approval", pending=pending)
        return turn
    turn.update(status="done", pending=None,
                trace=build_trace(turn["events"], latency=round(turn["clock"]["active"], 2), **meta))
    return turn


def run_turn(graph, text: str, sender_phone: str, thread_id: str | None = None, model: str | None = None,
             max_iterations: int | None = None, mode: str = "live", approver=None) -> dict:
    """Run a whole turn without a person in the loop. approver(request) -> bool decides at the gate
    (default: approve). Returns the trace."""
    approver = approver or (lambda request: True)
    turn = start_turn(graph, text, sender_phone, thread_id, model, max_iterations, mode)
    while turn["status"] == "awaiting_approval":
        turn = resume_turn(graph, turn, approver(turn["pending"]))
    return turn["trace"]
