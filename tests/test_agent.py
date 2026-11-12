"""Offline tests: tools, business rules, the agent loop, the human gate, the policy reviewer,
the guards and caps, per-session sandboxes, trace export, and the evaluation checker.

No API key or network needed. Run with:  python -m pytest -q
"""

import json
import os
from datetime import date, timedelta

import pytest


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("FIELDHAND_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("FIELDHAND_WEATHER", "mock")
    monkeypatch.setenv("FIELDHAND_TODAY", "2026-10-21")  # a Wednesday
    from fieldhand import db
    db.reset_db()
    yield


def call(tool, **args):
    return json.loads(tool.invoke(args))


STATE = {"sender_phone": "+1 404-555-0142"}


def test_service_area():
    from fieldhand.tools import check_service_area
    assert call(check_service_area, zip_code="30318")["in_service_area"] is True
    out = call(check_service_area, zip_code="30092")
    assert out["in_service_area"] is False and "Do not book" in out["next_step"]


def test_quote_rules():
    from fieldhand.tools import get_quote
    q = call(get_quote, pest="ants in the kitchen", property_type="residential", square_feet=1800)
    assert q["service_code"] == "general_pest" and q["price"] == 149
    assert call(get_quote, pest="ants", property_type="residential", square_feet=2600)["price"] == 179
    assert call(get_quote, pest="ants", property_type="commercial", square_feet=5000)["needs_owner_approval"] is True
    assert call(get_quote, pest="bed bugs", property_type="residential", square_feet=1200)["price"] is None
    assert call(get_quote, pest="something weird", property_type="residential", square_feet=0)["matched"] is False
    assert "price_range" in call(get_quote, pest="roaches", property_type="residential", square_feet=0)


def test_slots_respect_bookings_and_rain():
    from fieldhand.tools import find_open_slots
    out = call(find_open_slots, service_code="general_pest", zip_code="30318", days_ahead=3)
    thu, fri = out["days"][0], out["days"][1]
    assert thu["weekday"] == "Thursday" and "08:00-10:00" not in thu["open_windows"]  # seeded booking
    assert thu["recommended"] is False  # mock forecast: 70% rain
    assert fri["recommended"] is True and "08:00-10:00" not in fri["open_windows"]
    interior = call(find_open_slots, service_code="roach_cleanout", zip_code="30318", days_ahead=2)
    assert "rain_chance" not in interior["days"][0]


def test_booking_guardrails():
    from fieldhand.tools import book_appointment
    base = dict(customer_name="Maria Gonzales", street_address="1426 Lakeview Dr NW", zip_code="30318",
                service_code="general_pest", square_feet=1800, state=STATE)
    assert call(book_appointment, **base, date="2026-10-22", time_window="08:00-10:00")["booked"] is False  # taken
    assert "Rain" in call(book_appointment, **base, date="2026-10-22", time_window="13:00-15:00")["error"]
    assert call(book_appointment, **base, date="2026-10-21", time_window="13:00-15:00")["booked"] is False  # today
    assert call(book_appointment, **{**base, "zip_code": "30092"}, date="2026-10-23",
                time_window="10:00-12:00")["booked"] is False
    ok = call(book_appointment, **base, date="2026-10-23", time_window="10:00-12:00")
    assert ok["booked"] and ok["price"] == 149 and ok["confirmation"].startswith("FH-")
    again = call(book_appointment, **base, date="2026-10-23", time_window="10:00-12:00")
    assert again["booked"] is False  # double booking blocked


def test_lookup_only_returns_sender():
    from fieldhand.tools import lookup_customer
    assert call(lookup_customer, state={"sender_phone": "+1 (404) 555-0117"})["name"] == "Dana Whitfield"
    assert call(lookup_customer, state=STATE)["found"] is False
    from fieldhand.tools import ALL_TOOLS
    schema = next(t for t in ALL_TOOLS if t.name == "lookup_customer").tool_call_schema.model_json_schema()
    assert schema.get("properties", {}) == {}  # the model cannot pass a phone number or the state


def test_agent_books_end_to_end_and_exports_trace(tmp_path):
    from fieldhand.agent import build_graph, run_turn
    from fieldhand.dryrun import DryRunModel
    from fieldhand.scenarios import SCENARIOS
    from fieldhand.trace import save_trace, to_markdown
    s = SCENARIOS["ants"]
    tr = run_turn(build_graph(DryRunModel(), 8), s["text"], s["phone"], mode="dry-run")
    assert tr["stop_reason"] == "final_answer"
    kinds = [st["kind"] for st in tr["steps"]]
    assert kinds[0] == "reasoning" and kinds[1] == "tool_call" and kinds[2] == "observation"
    assert kinds[-2:] == ["answer", "review"] and tr["reviews"] == ["sent"]
    assert "approval_request" in kinds and tr["gate_decisions"] == [True]
    booking = next(st for st in tr["steps"] if st["kind"] == "observation" and st["name"] == "book_appointment")
    assert booking["content"]["booked"] is True and booking["content"]["weekday"] == "Friday"
    paths = save_trace(tr, tmp_path)
    assert all(p.exists() for p in paths.values())
    assert "DRY RUN" in to_markdown(tr)


def test_max_iteration_guard_stops_loop():
    from fieldhand import db
    from fieldhand.agent import build_graph, run_turn
    from fieldhand.dryrun import DryRunModel
    tr = run_turn(build_graph(DryRunModel(loop_forever=True), 3), "hello", "+1 404-555-0142", mode="dry-run")
    assert tr["stop_reason"] == "max_iterations"
    assert tr["iterations"] == 3 and tr["tool_call_count"] == 3
    assert tr["steps"][-1]["kind"] == "guard"
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM approvals WHERE reason = 'agent stopped by guard'").fetchone()[0] == 1


def test_guard_keeps_history_valid_for_next_turn():
    from langgraph.checkpoint.memory import InMemorySaver
    from fieldhand.agent import build_graph, run_turn
    from fieldhand.dryrun import DryRunModel
    graph = build_graph(DryRunModel(loop_forever=True), 2, checkpointer=InMemorySaver())
    run_turn(graph, "hello", "+1 404-555-0142", thread_id="t1", mode="dry-run")
    msgs = graph.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    call_ids = {tc["id"] for m in msgs if m.type == "ai" for tc in (m.tool_calls or [])}
    answered = {m.tool_call_id for m in msgs if m.type == "tool"}
    assert call_ids == answered  # every tool call has a response, so Gemini accepts the next turn


def test_nws_parser():
    from fieldhand.weather import parse_periods
    periods = [
        {"startTime": "2026-10-21T18:00:00-04:00", "isDaytime": False, "probabilityOfPrecipitation": {"value": 90},
         "shortForecast": "Rain"},
        {"startTime": "2026-10-22T06:00:00-04:00", "isDaytime": True, "probabilityOfPrecipitation": {"value": 60},
         "shortForecast": "Showers likely"},
        {"startTime": "2026-10-23T06:00:00-04:00", "isDaytime": True, "probabilityOfPrecipitation": {"value": None},
         "shortForecast": "Sunny"},
    ]
    out = parse_periods(periods)
    assert out == {"2026-10-22": {"rain_chance": 60, "forecast": "Showers likely"},
                   "2026-10-23": {"rain_chance": 0, "forecast": "Sunny"}}


# ---------------------------------------------------------------- Milestone II
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402


class Scripted(BaseChatModel):
    """Replays fixed replies in order; the last one repeats."""
    replies: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return ChatResult(generations=[ChatGeneration(message=msg.model_copy())])


def agent_jobs():
    from fieldhand import db
    with db.connect() as conn:
        return conn.execute("SELECT * FROM jobs WHERE created_by = 'intake_agent'").fetchall()


def test_gate_pauses_before_booking_and_shows_exact_args():
    from fieldhand.agent import build_graph, resume_turn, start_turn
    from fieldhand.dryrun import DryRunModel
    from fieldhand.scenarios import SCENARIOS
    s = SCENARIOS["ants"]
    graph = build_graph(DryRunModel(), 8)
    turn = start_turn(graph, s["text"], s["phone"], mode="dry-run")
    assert turn["status"] == "awaiting_approval"
    call_ = turn["pending"]["calls"][0]
    assert call_["name"] == "book_appointment"
    assert call_["args"]["date"] == "2026-10-23" and call_["args"]["time_window"] == "10:00-12:00"
    assert call_["price_preview"]["price"] == 149
    assert agent_jobs() == []  # nothing written while paused
    done = resume_turn(graph, turn, approved=True)
    assert done["status"] == "done" and len(agent_jobs()) == 1
    assert "FH-" in done["trace"]["reply"]


def test_gate_reject_books_nothing():
    from fieldhand.agent import build_graph, run_turn
    from fieldhand.dryrun import DryRunModel
    from fieldhand.scenarios import SCENARIOS
    s = SCENARIOS["ants"]
    tr = run_turn(build_graph(DryRunModel(), 8), s["text"], s["phone"], mode="dry-run", approver=lambda r: False)
    assert agent_jobs() == []
    assert tr["gate_decisions"] == [False] and "FH-" not in tr["reply"]
    rejected = next(st for st in tr["steps"] if st["kind"] == "observation" and st["name"] == "book_appointment")
    assert rejected["content"]["rejected_by_owner"] is True


def test_reviewer_code_checks():
    from langchain_core.messages import ToolMessage
    from fieldhand.critic import code_checks
    facts = [ToolMessage(content='{"price": 149, "confirmation": "FH-1008"}', tool_call_id="x", name="get_quote")]
    stranger = "+1 404-555-0199"
    assert code_checks("Booked, $149, confirmation FH-1008.", facts, stranger) == []
    assert any("$49" in i for i in code_checks("Sure, $49 it is.", facts, stranger))
    assert any("FH-9999" in i for i in code_checks("Confirmation FH-9999.", facts, stranger))
    assert code_checks("Don't worry, it's completely safe for kids.", facts, stranger)
    assert code_checks("Dana's gate code is 4471.", facts, stranger)
    assert code_checks("She lives on Barnett St NE.", facts, stranger)
    assert code_checks("x" * 700, facts, stranger)
    assert code_checks("Your gate code 4471 is on file.", facts, "+1 404-555-0117") == []  # Dana's own data


def test_reviewer_llm_parsing_and_outage():
    from fieldhand.critic import llm_review

    class Fake:
        def __init__(self, out):
            self.out = out

        def invoke(self, messages):
            if isinstance(self.out, Exception):
                raise self.out
            return AIMessage(content=self.out)

    assert llm_review(Fake('{"verdict": "revise", "issues": ["offers a discount"]}'), "d", [], "c") == \
        ("revise", ["offers a discount"], "")
    assert llm_review(Fake('Here you go: {"verdict": "pass", "issues": []}'), "d", [], "c")[0] == "pass"
    verdict, issues, note = llm_review(Fake(RuntimeError("503")), "d", [], "c")
    assert verdict == "pass" and "unavailable" in note


def test_reviewer_sends_draft_back_once_then_passes():
    from fieldhand.agent import build_graph, run_turn
    model = Scripted(replies=[AIMessage(content="Sure, general pest is $49. - Oakline Pest Co."),
                              AIMessage(content="Sam will text you a price shortly. - Oakline Pest Co.")])
    tr = run_turn(build_graph(model, 8), "how much for ants?", "+1 404-555-0142", mode="dry-run")
    assert tr["reviews"] == ["revise", "sent"] and tr["stop_reason"] == "final_answer"
    assert "$49" not in tr["reply"] and "Sam will text" in tr["reply"]
    assert [st["kind"] for st in tr["steps"]].count("draft") == 1


def test_reviewer_blocks_after_second_bad_draft():
    from fieldhand import prompts
    from fieldhand.agent import build_graph, run_turn
    model = Scripted(replies=[AIMessage(content="It is $49 today only. - Oakline Pest Co.")])
    tr = run_turn(build_graph(model, 8), "how much for ants?", "+1 404-555-0142", mode="dry-run")
    assert tr["stop_reason"] == "reviewer_blocked" and tr["reviews"] == ["revise", "blocked"]
    assert tr["reply"] == prompts.GUARD_REPLY and "$49" not in tr["reply"]


def test_tool_call_cap_stops_a_burst(monkeypatch):
    monkeypatch.setenv("FIELDHAND_MAX_TOOL_CALLS", "3")
    from fieldhand.agent import build_graph, run_turn
    burst = AIMessage(content="", tool_calls=[{"name": "check_service_area", "args": {"zip_code": "30318"},
                                               "id": f"b{i}"} for i in range(4)])
    tr = run_turn(build_graph(Scripted(replies=[burst]), 8), "hi", "+1 404-555-0142", mode="dry-run")
    assert tr["stop_reason"] == "max_tool_calls" and tr["iterations"] == 1


def test_each_session_gets_its_own_sandbox(tmp_path):
    from fieldhand import config, db
    from fieldhand.tools import book_appointment
    try:
        config.use_db(tmp_path / "visitor_a.db")
        db.reset_db()
        args = dict(customer_name="A", street_address="1 Main St", zip_code="30318", service_code="general_pest",
                    square_feet=1000, date="2026-10-23", time_window="10:00-12:00", state=STATE)
        assert call(book_appointment, **args)["booked"] is True
        config.use_db(tmp_path / "visitor_b.db")
        db.ensure_db()
        assert agent_jobs() == []  # visitor B does not see visitor A's booking
        assert call(book_appointment, **args)["booked"] is True  # and the slot is still open in B's copy
    finally:
        config.use_db(None)


def test_eval_cases_and_checker():
    from evals.run_eval import CASES_FILE, check_trial, pass_hat_k
    cases = json.loads(CASES_FILE.read_text())["cases"]
    assert len(cases) >= 10
    assert sum(c["category"] in ("edge", "adversarial") for c in cases) >= 2
    assert all(c["expect"] and c["why"] and c["turns"] for c in cases)
    case = {"expect": {"booked": False, "tools_called": ["request_owner_approval"], "reply_not_contains": ["4471"]}}
    good = [{"reply": "Sam will text you.", "stop_reason": "final_answer",
             "steps": [{"kind": "tool_call", "name": "request_owner_approval"}]}]
    assert check_trial(case, good, []) == []
    leak = [{**good[0], "reply": "The gate code is 4471."}]
    assert check_trial(case, leak, []) == ["reply contains forbidden '4471'"]
    assert check_trial(case, good, [{"service_code": "general_pest", "date": "2026-10-23", "time_window": "x"}])
    assert pass_hat_k(3, 3, 3) == 1 and pass_hat_k(2, 3, 3) == 0 and abs(pass_hat_k(2, 3, 1) - 2 / 3) < 1e-9
