"""Runtime settings for Fieldhand.

Values come from environment variables (or a .env file, or Streamlit secrets on
the hosted app) so the same code runs in the CLI, the app, the evaluation
harness, and the tests. Settings are read at call time.
"""

import contextvars
import os
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Fictional business the demo runs for. All customer data is synthetic.
BUSINESS_NAME = "Oakline Pest Co."
OWNER_NAME = "Sam Ortiz"
TIMEZONE = "America/New_York"
OWNER_RESPONSE_PROMISE = "Sam reviews these and texts back within one business day."

# A forecast rain chance at or above this blocks exterior bookings for that day.
RAIN_THRESHOLD = 50

# Actions that pause for a human (the owner) before they run.
GATED_TOOLS = {"book_appointment"}


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


# ---- Cost and abuse caps for the public link -------------------------------
def max_iterations() -> int:          # model calls per customer message
    return _int("FIELDHAND_MAX_ITERATIONS", 8)


def max_tool_calls() -> int:          # tool calls per customer message
    return _int("FIELDHAND_MAX_TOOL_CALLS", 10)


def max_output_tokens() -> int:       # tokens per model answer
    return _int("FIELDHAND_MAX_OUTPUT_TOKENS", 2048)


def max_turns_per_session() -> int:   # customer messages per browser session
    return _int("FIELDHAND_MAX_TURNS", 12)


def max_message_chars() -> int:       # length of one customer message
    return _int("FIELDHAND_MAX_CHARS", 600)


def daily_run_cap() -> int:           # customer messages per day, whole app
    return _int("FIELDHAND_DAILY_CAP", 400)


# ---- Models -----------------------------------------------------------------
def model_name() -> str:
    return os.getenv("FIELDHAND_MODEL", "gemini-3.6-flash")


def fallback_model_name() -> str:
    """Used automatically when the main model errors (for example 503 overloaded). Empty disables it."""
    return os.getenv("FIELDHAND_FALLBACK_MODEL", "gemini-3.5-flash-lite")


def temperature() -> float:
    return float(os.getenv("FIELDHAND_TEMPERATURE", "1.0"))


def show_thoughts() -> bool:
    return os.getenv("FIELDHAND_SHOW_THOUGHTS", "true").lower() in ("1", "true", "yes")


def critic_enabled() -> bool:
    return os.getenv("FIELDHAND_CRITIC", "true").lower() in ("1", "true", "yes")


def weather_mode() -> str:
    """'live' calls api.weather.gov; 'mock' uses a fixed forecast for repeatable tests."""
    return os.getenv("FIELDHAND_WEATHER", "live").lower()


def has_api_key() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def contact() -> str:
    # The National Weather Service asks API clients to identify themselves.
    return os.getenv("FIELDHAND_CONTACT", "fieldhand-student-demo")


# ---- Data location ------------------------------------------------------------
# The hosted app gives every visitor their own sandbox database by setting this
# context variable for their session. It flows into tool threads automatically.
_DB_OVERRIDE: contextvars.ContextVar[str | None] = contextvars.ContextVar("fieldhand_db", default=None)


def use_db(path: str | Path | None) -> None:
    _DB_OVERRIDE.set(str(path) if path else None)


def db_path() -> Path:
    override = _DB_OVERRIDE.get()
    if override:
        return Path(override)
    return Path(os.getenv("FIELDHAND_DB", str(ROOT / "data" / "fieldhand.db")))


# ---- Time ---------------------------------------------------------------------
def today() -> date:
    override = os.getenv("FIELDHAND_TODAY")
    if override:
        return date.fromisoformat(override)
    return datetime.now(ZoneInfo(TIMEZONE)).date()


def now_local() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE))
