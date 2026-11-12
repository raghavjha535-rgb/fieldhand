"""SQLite store for the demo business: service area, price book, customers,
properties, jobs (the calendar), and the owner's approval inbox.

All records are synthetic. Upcoming jobs are seeded relative to today so the
calendar always looks realistic, whenever the demo is run.

Reset from the command line with:  python -m fieldhand.db --reset
"""

import sqlite3
import sys
from datetime import date, timedelta

from . import config

WEEKDAY_WINDOWS = ["08:00-10:00", "10:00-12:00", "13:00-15:00", "15:00-17:00"]
SATURDAY_WINDOWS = ["08:00-10:00", "10:00-12:00"]


def windows_for(d: date) -> list[str]:
    if d.weekday() < 5:
        return WEEKDAY_WINDOWS
    if d.weekday() == 5:
        return SATURDAY_WINDOWS
    return []


SCHEMA = """
CREATE TABLE IF NOT EXISTS service_area (
    zip TEXT PRIMARY KEY, city TEXT NOT NULL, lat REAL NOT NULL, lon REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS services (
    code TEXT PRIMARY KEY, name TEXT NOT NULL, exterior INTEGER NOT NULL,
    price_small INTEGER, price_medium INTEGER, price_large INTEGER,
    includes TEXT, needs_approval INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, phone TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS properties (
    id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
    address TEXT NOT NULL, zip TEXT NOT NULL, square_feet INTEGER, notes TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY, confirmation TEXT UNIQUE,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    property_id INTEGER REFERENCES properties(id),
    service_code TEXT NOT NULL REFERENCES services(code),
    date TEXT NOT NULL, time_window TEXT NOT NULL, price INTEGER,
    status TEXT NOT NULL, notes TEXT, created_at TEXT NOT NULL, created_by TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY, ticket TEXT UNIQUE, customer_name TEXT, phone TEXT,
    reason TEXT NOT NULL, details TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL
);
"""

# Approximate ZIP centroids for intown Atlanta and Decatur. Used for the forecast.
SERVICE_AREA = [
    ("30305", "Atlanta (Buckhead)", 33.8316, -84.3850),
    ("30306", "Atlanta (Virginia-Highland)", 33.7865, -84.3515),
    ("30307", "Atlanta (Candler Park)", 33.7690, -84.3355),
    ("30308", "Atlanta (Midtown)", 33.7712, -84.3753),
    ("30309", "Atlanta (Ansley Park)", 33.7983, -84.3881),
    ("30312", "Atlanta (Grant Park)", 33.7460, -84.3782),
    ("30316", "Atlanta (East Atlanta)", 33.7270, -84.3330),
    ("30317", "Atlanta (Kirkwood)", 33.7495, -84.3170),
    ("30318", "Atlanta (West Midtown)", 33.7890, -84.4440),
    ("30319", "Brookhaven", 33.8750, -84.3350),
    ("30324", "Atlanta (Lindbergh)", 33.8170, -84.3540),
    ("30327", "Atlanta (Northwest)", 33.8650, -84.4200),
    ("30030", "Decatur", 33.7710, -84.2960),
    ("30033", "Decatur (North)", 33.8100, -84.2810),
]

# Synthetic price book: (code, name, exterior, small <2000 sq ft, medium <3500, large, includes, needs_approval)
SERVICES = [
    ("general_pest", "General pest treatment (interior + exterior)", 1, 149, 179, 219,
     "Interior and exterior treatment for ants, spiders, roaches and other crawling pests; free re-service within 30 days", 0),
    ("roach_cleanout", "Cockroach clean-out (interior)", 0, 199, 239, 279,
     "Two visits: initial treatment plus a follow-up 14 days later", 0),
    ("mosquito", "Mosquito yard treatment", 1, 79, 89, 99,
     "One yard and shrub treatment; monthly plans available", 0),
    ("stinging_insects", "Wasp and hornet nest removal", 1, 129, 129, 129,
     "Removal of up to 3 accessible nests", 0),
    ("rodent_inspection", "Rodent inspection", 0, 95, 95, 95,
     "Inspection and written exclusion plan; fee credited toward exclusion work", 0),
    ("termite_inspection", "Termite inspection", 0, 0, 0, 0,
     "Free inspection with written findings; treatment quoted afterward", 0),
    ("bed_bugs", "Bed bug treatment", 0, None, None, None,
     "Custom quote after the owner reviews the case", 1),
]


def connect() -> sqlite3.Connection:
    path = config.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_db() -> None:
    """Create and seed the database the first time it is used."""
    if not config.db_path().exists():
        reset_db()


def reset_db() -> None:
    path = config.db_path()
    if path.exists():
        path.unlink()
    with connect() as conn:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO service_area VALUES (?,?,?,?)", SERVICE_AREA)
        conn.executemany("INSERT INTO services VALUES (?,?,?,?,?,?,?,?)", SERVICES)
        _seed_customers_and_jobs(conn)


def _upcoming_weekdays(start: date, count: int) -> list[date]:
    days, d = [], start
    while len(days) < count:
        d += timedelta(days=1)
        if d.weekday() < 5:
            days.append(d)
    return days


def _seed_customers_and_jobs(conn: sqlite3.Connection) -> None:
    today = config.today()
    created = (today - timedelta(days=200)).isoformat()

    def add_customer(name, phone, address, zip_code, sqft, notes):
        cur = conn.execute(
            "INSERT INTO customers (name, phone, created_at) VALUES (?,?,?)", (name, phone, created))
        cid = cur.lastrowid
        cur = conn.execute(
            "INSERT INTO properties (customer_id, address, zip, square_feet, notes) VALUES (?,?,?,?,?)",
            (cid, address, zip_code, sqft, notes))
        return cid, cur.lastrowid

    def add_job(cid, pid, service, day, window, price, status, notes=""):
        cur = conn.execute(
            "INSERT INTO jobs (customer_id, property_id, service_code, date, time_window, price, status,"
            " notes, created_at, created_by) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (cid, pid, service, day.isoformat(), window, price, status, notes, created, "seed"))
        conn.execute("UPDATE jobs SET confirmation = ? WHERE id = ?", (f"FH-{1000 + cur.lastrowid}", cur.lastrowid))

    dana = add_customer("Dana Whitfield", "4045550117", "782 Barnett St NE", "30306", 2400,
                        "Quarterly general pest plan. Gate code 4471. Friendly dog in the backyard.")
    priya = add_customer("Priya Raman", "4045550133", "1180 Lambert Dr NE", "30324", 1600,
                         "Apartment-style townhome; prefers afternoon visits.")
    tom = add_customer("Tom Kessler", "4045550151", "455 Clifton Rd NE", "30307", 1900, "")
    alicia = add_customer("Alicia Grant", "4045550168", "1312 Glenwood Ave SE", "30316", 2100, "")
    ben = add_customer("Ben Okafor", "4045550174", "3020 Osborne Rd NE", "30319", 2800, "")

    add_job(*dana, "general_pest", today - timedelta(days=160), "10:00-12:00", 179, "completed")
    add_job(*dana, "general_pest", today - timedelta(days=70), "10:00-12:00", 179, "completed",
            "Ant activity along kitchen baseboards; treated interior perimeter and exterior foundation.")
    add_job(*priya, "roach_cleanout", today - timedelta(days=12), "13:00-15:00", 199, "completed")

    d0, d1, d2 = _upcoming_weekdays(today, 3)
    add_job(*tom, "general_pest", d0, "08:00-10:00", 149, "scheduled")
    add_job(*alicia, "mosquito", d0, "10:00-12:00", 89, "scheduled")
    add_job(*ben, "rodent_inspection", d1, "08:00-10:00", 95, "scheduled")
    add_job(*priya, "roach_cleanout", d2, "13:00-15:00", 0, "scheduled", "Included 14-day follow-up visit")


def file_ticket(phone: str, reason: str, details: str, customer_name: str = "") -> str:
    """Put an item in the owner's approval inbox and return its ticket number."""
    ensure_db()
    with connect() as conn:
        aid = conn.execute(
            "INSERT INTO approvals (customer_name, phone, reason, details, status, created_at) VALUES (?,?,?,?,?,?)",
            (customer_name, phone, reason, details, "pending", config.now_local().isoformat(timespec="seconds"))).lastrowid
        ticket = f"AP-{2000 + aid}"
        conn.execute("UPDATE approvals SET ticket = ? WHERE id = ?", (ticket, aid))
    return ticket


def protected_strings(sender_digits: str) -> list[str]:
    """Names, street addresses, and access codes of every customer except the sender.

    The reviewer blocks any reply that contains one of these.
    """
    import re as _re

    ensure_db()
    out: list[str] = []
    with connect() as conn:
        rows = conn.execute(
            "SELECT c.name, c.phone, p.address, p.notes FROM customers c "
            "LEFT JOIN properties p ON p.customer_id = c.id").fetchall()
    for r in rows:
        if r["phone"] == sender_digits:
            continue
        out.append(r["name"])
        if r["address"]:
            out.append(r["address"].split(" ", 1)[-1])  # street name without the number
        out += _re.findall(r"(?i)code (\d{3,6})", r["notes"] or "")
    return [x for x in out if x and len(x) >= 4]


if __name__ == "__main__":
    if "--reset" in sys.argv:
        reset_db()
        print(f"Reset demo data at {config.db_path()}")
    else:
        ensure_db()
        print(f"Database ready at {config.db_path()}")
