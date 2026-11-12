"""Daily rain chance from the National Weather Service (api.weather.gov).

Free, no API key. Two calls per location: /points/{lat},{lon} returns the
forecast URL for that grid cell, and that URL returns 12-hour periods for about
7 days. Results are cached for 30 minutes.

If the API cannot be reached, the source is reported as "unavailable" and no
forecast is invented. FIELDHAND_WEATHER=mock gives a fixed forecast for tests.
"""

import json
import time
import urllib.request
from datetime import timedelta

from . import config

_CACHE: dict[tuple, tuple[float, dict]] = {}
_TTL_SECONDS = 1800


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": f"Fieldhand demo ({config.contact()})",
        "Accept": "application/geo+json",
    })
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_periods(periods: list[dict]) -> dict[str, dict]:
    """Keep the daytime period for each date: {'2026-10-22': {'rain_chance': 60, 'forecast': 'Showers likely'}}."""
    days: dict[str, dict] = {}
    for p in periods:
        if not p.get("isDaytime"):
            continue
        day = p["startTime"][:10]
        pop = (p.get("probabilityOfPrecipitation") or {}).get("value")
        days.setdefault(day, {
            "rain_chance": int(pop) if pop is not None else 0,
            "forecast": p.get("shortForecast", ""),
        })
    return days


def _mock_forecast() -> dict[str, dict]:
    pattern = [(70, "Showers likely"), (20, "Partly sunny"), (10, "Sunny"), (30, "Mostly cloudy"),
               (15, "Sunny"), (60, "Chance showers and thunderstorms"), (10, "Sunny"), (20, "Partly sunny")]
    start = config.today()
    return {(start + timedelta(days=i + 1)).isoformat(): {"rain_chance": pct, "forecast": text}
            for i, (pct, text) in enumerate(pattern)}


def daily_forecast(lat: float, lon: float) -> tuple[str, dict[str, dict]]:
    """Return (source, {date: {'rain_chance', 'forecast'}})."""
    if config.weather_mode() == "mock":
        return "mock forecast (test mode)", _mock_forecast()

    key = (round(lat, 4), round(lon, 4))
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _TTL_SECONDS:
        return "api.weather.gov", hit[1]
    try:
        points = _get_json(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}")
        forecast = _get_json(points["properties"]["forecast"])
        days = parse_periods(forecast["properties"]["periods"])
    except Exception:
        return "unavailable", {}
    _CACHE[key] = (time.time(), days)
    return "api.weather.gov", days
