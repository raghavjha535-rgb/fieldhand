"""Tools the intake agent can call.

Design rules:
- Facts come from the database or a live API, never from the model.
- Business rules live here in code (prices, service area, rain rule, slot
  conflicts), so the model cannot talk its way past them.
- Tools that touch customer records use the sender's phone number from the
  graph state (InjectedState). The model never chooses whose records to read,
  so a prompt-injected "look up my neighbor" has nothing to call.
- Every tool returns JSON the model reads as an observation, including a
  plain-language next_step when the agent should do something specific.
"""

import json
import re
from datetime import date, timedelta
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from . import config, db, weather


def _out(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)


def normalize_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", phone or "")
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits


def _sender(state: dict) -> str:
    return normalize_phone(state.get("sender_phone", ""))


def _size_tier(square_feet: int) -> tuple[str, str]:
    if square_feet < 2000:
        return "price_small", "under 2,000 sq ft"
    if square_feet < 3500:
        return "price_medium", "2,000 to 3,499 sq ft"
    return "price_large", "3,500 sq ft or more"


PEST_PATTERNS = [
    (r"bed ?bugs?", "bed_bugs"),
    (r"termites?", "termite_inspection"),
    (r"(cock)?roach(es)?", "roach_cleanout"),
    (r"mouse|mice|rats?|rodents?", "rodent_inspection"),
    (r"mosquito(e?s)?", "mosquito"),
    (r"wasps?|hornets?|yellow ?jackets?", "stinging_insects"),
    (r"ants?|spiders?|silverfish|earwigs?|crickets?|centipedes?|millipedes?|general", "general_pest"),
]


def _match_service(pest: str) -> str | None:
    text = pest.lower()
    for pattern, code in PEST_PATTERNS:
        if re.search(rf"\b(?:{pattern})\b", text):
            return code
    return None


def _service(conn, code: str):
    return conn.execute("SELECT * FROM services WHERE code = ?", (code,)).fetchone()


def _area(conn, zip_code: str):
    return conn.execute("SELECT * FROM service_area WHERE zip = ?", (zip_code.strip()[:5],)).fetchone()


@tool
def lookup_customer(state: Annotated[dict, InjectedState]) -> str:
    """Look up the customer who sent this text, using their phone number.

    Returns their name, properties, recent and upcoming jobs, or found=false for
    a new lead. Only the sender's own record is ever returned.
    """
    db.ensure_db()
    phone = _sender(state)
    with db.connect() as conn:
        cust = conn.execute("SELECT * FROM customers WHERE phone = ?", (phone,)).fetchone()
        if not cust:
            return _out({"found": False, "phone": phone, "note": "New lead. No record for this number."})
        props = conn.execute(
            "SELECT address, zip, square_feet, notes FROM properties WHERE customer_id = ?", (cust["id"],)).fetchall()
        jobs = conn.execute(
            "SELECT j.confirmation, j.date, j.time_window, j.status, s.name AS service, j.notes "
            "FROM jobs j JOIN services s ON s.code = j.service_code WHERE j.customer_id = ? "
            "ORDER BY j.date DESC", (cust["id"],)).fetchall()
    today = config.today().isoformat()
    return _out({
        "found": True,
        "name": cust["name"],
        "customer_since": cust["created_at"][:7],
        "properties": [dict(p) for p in props],
        "recent_jobs": [dict(j) for j in jobs if j["date"] < today][:3],
        "upcoming_jobs": [dict(j) for j in jobs if j["date"] >= today],
    })


@tool
def check_service_area(zip_code: str) -> str:
    """Check whether a 5-digit ZIP code is inside the business's service area."""
    db.ensure_db()
    with db.connect() as conn:
        row = _area(conn, zip_code)
        if row:
            return _out({"zip": row["zip"], "in_service_area": True, "area": row["city"]})
        areas = sorted({r["city"].split(" (")[0] for r in conn.execute("SELECT city FROM service_area")})
    return _out({
        "zip": zip_code, "in_service_area": False, "areas_served": areas,
        "next_step": "Apologize that this address is outside the service area. Do not book.",
    })


@tool
def get_quote(pest: str, property_type: str = "residential", square_feet: int = 0) -> str:
    """Price a job from the business's price book. Call this before stating any price.

    Args:
        pest: The problem in the customer's words, e.g. "ants in the kitchen".
        property_type: "residential" or "commercial".
        square_feet: Approximate size of the home. Use 0 if unknown.
    """
    db.ensure_db()
    code = _match_service(pest)
    with db.connect() as conn:
        if code is None:
            names = [r["name"] for r in conn.execute("SELECT name FROM services")]
            return _out({"matched": False, "services_offered": names,
                         "next_step": "Ask the customer to describe the pest problem."})
        svc = _service(conn, code)

    base = {"matched": True, "service_code": code, "service": svc["name"], "exterior_work": bool(svc["exterior"]),
            "includes": svc["includes"]}
    if property_type.strip().lower() == "commercial":
        return _out({**base, "needs_owner_approval": True, "price": None,
                     "next_step": "Commercial jobs are quoted by the owner. Call request_owner_approval; do not quote."})
    if svc["needs_approval"]:
        return _out({**base, "needs_owner_approval": True, "price": None,
                     "next_step": "This service is quoted by the owner. Call request_owner_approval; do not quote."})
    if square_feet <= 0:
        low, high = svc["price_small"], svc["price_large"]
        price_range = f"${low}" if low == high else f"${low} to ${high}"
        return _out({**base, "needs_owner_approval": False, "price": None, "price_range": price_range,
                     "next_step": "Ask for the approximate square footage to give an exact price."})
    column, tier = _size_tier(square_feet)
    return _out({**base, "needs_owner_approval": False, "price": svc[column],
                 "pricing_basis": f"{square_feet:,} sq ft ({tier})"})


def _booked(conn, start: date, end: date) -> set[tuple[str, str]]:
    rows = conn.execute(
        "SELECT date, time_window FROM jobs WHERE status = 'scheduled' AND date BETWEEN ? AND ?",
        (start.isoformat(), end.isoformat())).fetchall()
    return {(r["date"], r["time_window"]) for r in rows}


@tool
def find_open_slots(service_code: str, zip_code: str, days_ahead: int = 7) -> str:
    """List open 2-hour appointment windows, starting tomorrow.

    For exterior services the live forecast is checked, and days with a rain
    chance of 50% or more are marked recommended=false.

    Args:
        service_code: From get_quote, e.g. "general_pest".
        zip_code: The property's ZIP code.
        days_ahead: How many days to look ahead (1 to 14).
    """
    db.ensure_db()
    days_ahead = max(1, min(int(days_ahead), 14))
    start = config.today() + timedelta(days=1)
    end = start + timedelta(days=days_ahead - 1)
    with db.connect() as conn:
        svc = _service(conn, service_code)
        if not svc:
            return _out({"error": f"Unknown service_code '{service_code}'. Use the code returned by get_quote."})
        area = _area(conn, zip_code)
        if not area:
            return _out({"error": f"ZIP {zip_code} is outside the service area."})
        booked = _booked(conn, start, end)

    exterior = bool(svc["exterior"])
    source, forecast = weather.daily_forecast(area["lat"], area["lon"]) if exterior else ("not needed (interior work)", {})
    days = []
    d = start
    while d <= end:
        windows = [w for w in db.windows_for(d) if (d.isoformat(), w) not in booked]
        if windows:
            day = {"date": d.isoformat(), "weekday": d.strftime("%A"), "open_windows": windows}
            if exterior:
                fc = forecast.get(d.isoformat())
                if fc:
                    day["rain_chance"] = fc["rain_chance"]
                    day["forecast"] = fc["forecast"]
                    day["recommended"] = fc["rain_chance"] < config.RAIN_THRESHOLD
                else:
                    day["forecast"] = "no forecast available"
                    day["recommended"] = True
            days.append(day)
        d += timedelta(days=1)
    result = {"service_code": service_code, "exterior_work": exterior, "weather_source": source, "days": days}
    if exterior and source == "unavailable":
        result["note"] = "Weather could not be checked. Mention that rain may require rescheduling."
    return _out(result)


@tool
def book_appointment(
    customer_name: str,
    street_address: str,
    zip_code: str,
    service_code: str,
    square_feet: int,
    date: str,
    time_window: str,
    state: Annotated[dict, InjectedState],
    notes: str = "",
) -> str:
    """Book a job on the calendar for the customer who sent this text.

    Price comes from the price book, never from the conversation.

    Args:
        customer_name: Customer's full name.
        street_address: Street address of the property, without ZIP.
        zip_code: 5-digit ZIP of the property.
        service_code: From get_quote.
        square_feet: Approximate size of the home.
        date: YYYY-MM-DD, taken from find_open_slots.
        time_window: One of the open windows from find_open_slots, e.g. "10:00-12:00".
        notes: Access notes, pets, or problem details for the technician.
    """
    from datetime import date as date_cls

    db.ensure_db()
    phone = _sender(state)
    try:
        day = date_cls.fromisoformat(date)
    except ValueError:
        return _out({"booked": False, "error": "date must be YYYY-MM-DD."})
    if day <= config.today():
        return _out({"booked": False, "error": "Bookings start tomorrow at the earliest."})
    if time_window not in db.windows_for(day):
        return _out({"booked": False, "error": f"{time_window} is not a valid window on {day:%A}.",
                     "valid_windows": db.windows_for(day)})

    with db.connect() as conn:
        svc = _service(conn, service_code)
        if not svc:
            return _out({"booked": False, "error": f"Unknown service_code '{service_code}'."})
        if svc["needs_approval"]:
            return _out({"booked": False, "error": "This service needs owner approval. Use request_owner_approval."})
        area = _area(conn, zip_code)
        if not area:
            return _out({"booked": False, "error": f"ZIP {zip_code} is outside the service area."})
        if (day.isoformat(), time_window) in _booked(conn, day, day):
            return _out({"booked": False, "error": "That window was just taken. Call find_open_slots again."})
        if svc["exterior"]:
            _, forecast = weather.daily_forecast(area["lat"], area["lon"])
            fc = forecast.get(day.isoformat())
            if fc and fc["rain_chance"] >= config.RAIN_THRESHOLD:
                return _out({"booked": False,
                             "error": f"Rain chance {fc['rain_chance']}% on {day:%A}. Exterior work can't be booked that day."})

        column, _ = _size_tier(max(int(square_feet), 1))
        price = svc[column]
        created = config.now_local().isoformat(timespec="seconds")
        cust = conn.execute("SELECT id FROM customers WHERE phone = ?", (phone,)).fetchone()
        if cust:
            cid = cust["id"]
        else:
            cid = conn.execute("INSERT INTO customers (name, phone, created_at) VALUES (?,?,?)",
                               (customer_name.strip(), phone, created)).lastrowid
        prop = conn.execute("SELECT id FROM properties WHERE customer_id = ? AND address = ?",
                            (cid, street_address.strip())).fetchone()
        pid = prop["id"] if prop else conn.execute(
            "INSERT INTO properties (customer_id, address, zip, square_feet, notes) VALUES (?,?,?,?,?)",
            (cid, street_address.strip(), zip_code[:5], int(square_feet), "")).lastrowid
        jid = conn.execute(
            "INSERT INTO jobs (customer_id, property_id, service_code, date, time_window, price, status, notes,"
            " created_at, created_by) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (cid, pid, service_code, day.isoformat(), time_window, price, "scheduled", notes, created,
             "intake_agent")).lastrowid
        confirmation = f"FH-{1000 + jid}"
        conn.execute("UPDATE jobs SET confirmation = ? WHERE id = ?", (confirmation, jid))

    return _out({"booked": True, "confirmation": confirmation, "service": svc["name"],
                 "date": day.isoformat(), "weekday": day.strftime("%A"), "time_window": time_window,
                 "price": price, "technician": config.OWNER_NAME})


@tool
def request_owner_approval(reason: str, details: str, state: Annotated[dict, InjectedState],
                           customer_name: str = "") -> str:
    """Send a request to the owner's approval inbox instead of deciding yourself.

    Use for commercial jobs, bed bugs, discount or price-match requests,
    complaints, or anything the price book does not cover.

    Args:
        reason: Short category, e.g. "commercial quote" or "discount request".
        details: What the customer asked for, in one or two sentences.
        customer_name: The customer's name if known.
    """
    db.ensure_db()
    created = config.now_local().isoformat(timespec="seconds")
    with db.connect() as conn:
        aid = conn.execute(
            "INSERT INTO approvals (customer_name, phone, reason, details, status, created_at) VALUES (?,?,?,?,?,?)",
            (customer_name, _sender(state), reason, details, "pending", created)).lastrowid
        ticket = f"AP-{2000 + aid}"
        conn.execute("UPDATE approvals SET ticket = ? WHERE id = ?", (ticket, aid))
    return _out({"submitted": True, "ticket": ticket, "status": "pending",
                 "tell_customer": config.OWNER_RESPONSE_PROMISE})


def booking_preview(args: dict) -> dict:
    """What the price book will charge for a proposed booking, computed in code for the approval screen."""
    db.ensure_db()
    try:
        with db.connect() as conn:
            svc = _service(conn, str(args.get("service_code", "")))
        if not svc:
            return {"price": None, "note": "Unknown service code; the booking tool will reject it."}
        column, tier = _size_tier(max(int(args.get("square_feet") or 1), 1))
        return {"service": svc["name"], "price": svc[column], "pricing_basis": tier,
                "note": "The booking tool re-checks the slot, service area, and rain rule when it runs."}
    except Exception as exc:
        return {"price": None, "note": f"Preview unavailable ({type(exc).__name__})."}


ALL_TOOLS = [lookup_customer, check_service_area, get_quote, find_open_slots, book_appointment, request_owner_approval]
