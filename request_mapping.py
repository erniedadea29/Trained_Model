"""
Turn a budget request from the app's Supabase tables into model input rows.

budget_requests       -> College (department_code), Semester (semester_id), Date & Event_Title (event_id)
budget_request_items  -> Category (mapped), Item (description), Qty (quantity), requested amount (amount)
"""
from datetime import date, datetime

import pandas as pd

# App category -> category name used in the training data.
# None = no equivalent in training data; the model still predicts, but with low confidence.
CATEGORY_MAP = {
    "Honorarium / Speakers": "Honorarium/Speakers",
    "Materials & Supplies": "Materials & Supplies",
    "Venue & Logistics": "Venue/Facilities",
    "Equipment / Rentals": "Sound & Lighting Equipment",
    "Food & Catering": "Food & Catering",
    "Printing & Documentation": "Printing & Documentation",
    "Transportation": "Transportation",
    "Prizes / Tokens": "Prizes/Tokens",
    "Souvenirs / Giveaways": "Souvenirs/Giveaways",
    "Decorations & Layout": "Decorations & Layout",
    "Photography / Documentation": "Photography/Documentation",
    "Sound & Lighting Equipment": "Sound & Lighting Equipment",
    "Internet & Connectivity": "Internet & Connectivity",
    "Registration & Online Platform Fees": "Registration & Online Platform Fees",
    "Security & First Aid": "Security & First Aid",
    # Older app category names (requests created before the app switched to the training categories)
    "Food & Refreshments": "Food & Catering",
    "Prizes & Awards": "Prizes/Tokens",
    "Events": None,
    "General": None,
    "Contingency": None,
}

OVER_PCT, UNDER_PCT = 25.0, -25.0   # flag items requested >25% above / below the prediction


def _norm(s: str) -> str:
    return " ".join((s or "").replace(" / ", "/").replace("/", " / ").split()).lower()


_LOOKUP = {_norm(k): v for k, v in CATEGORY_MAP.items()}


def map_category(app_category: str):
    """Return (model_category, low_confidence)."""
    model_cat = _LOOKUP.get(_norm(app_category))
    if model_cat:
        return model_cat, False
    return (app_category or "General").strip(), True


def semester_from_date(d: date) -> str:
    # Training data: Jan-May = 2nd Semester, Jun-Dec = 1st Semester
    return "2nd Semester" if d.month <= 5 else "1st Semester"


def fetch_request(sb, request_id: str) -> tuple[dict, pd.DataFrame]:
    """Load a budget request and its items, returning (context, model-input DataFrame)."""
    reqs = sb.table("budget_requests").select("*").eq("id", request_id).limit(1).execute().data
    if not reqs:
        raise LookupError(f"budget_request {request_id} not found")
    req = reqs[0]
    items = sb.table("budget_request_items").select("*").eq("request_id", request_id).execute().data
    if not items:
        raise LookupError(f"budget_request {request_id} has no items")

    event = None
    if req.get("event_id"):
        ev = sb.table("school_events").select("event_title,event_date").eq("id", req["event_id"]).limit(1).execute().data
        event = ev[0] if ev else None

    when = (date.fromisoformat(event["event_date"]) if event and event.get("event_date")
            else datetime.fromisoformat(req["created_at"]).date())

    semester = None
    if req.get("semester_id"):
        s = sb.table("semesters").select("name").eq("id", req["semester_id"]).limit(1).execute().data
        semester = s[0]["name"] if s else None
    semester = semester or semester_from_date(when)

    rows = []
    for it in items:
        model_cat, low = map_category(it.get("category"))
        rows.append({
            "request_item_id": it["id"],
            "Date": when.isoformat(),
            "Semester": semester,
            "College": req["department_code"],
            "Event_Title": event["event_title"] if event else req["title"],
            "Category": model_cat,
            "Item": (it.get("description") or "").strip() or model_cat,
            "Qty": max(float(it.get("quantity") or 1), 1.0),
            "requested_amount": float(it.get("amount") or 0),
            "low_confidence": low,
        })
    return req, pd.DataFrame(rows)


def flag_for(variance_pct: float) -> str:
    if variance_pct > OVER_PCT:
        return "OVER"
    if variance_pct < UNDER_PCT:
        return "UNDER"
    return "OK"
