"""
Budget prediction API backed by Supabase.

On startup it downloads the active model from Supabase Storage. POST /predict
returns estimates and saves them to the budget_predictions table.

Run:   uvicorn api:app --reload
Docs:  http://127.0.0.1:8000/docs
"""
import tempfile
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predict import predict
from request_mapping import fetch_request, flag_for
from supabase_client import BUCKET, MODEL_NAME, get_client
from train import load_h5

STATE = {}
LOCAL_MODEL = Path(__file__).parent / "models" / "best_model.h5"


def load_active_model():
    """Load the active model from Supabase; fall back to the bundled .h5 so the API never fails to start."""
    sb = get_client()
    try:
        rows = (sb.table("ml_models").select("*").eq("name", MODEL_NAME).eq("is_active", True)
                  .order("created_at", desc=True).limit(1).execute().data)
        if not rows:
            raise RuntimeError("no active model in ml_models (run `python upload_model.py`)")
        blob = sb.storage.from_(BUCKET).download(rows[0]["storage_path"])
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp.write(blob)
        model, meta = load_h5(tmp.name)
        STATE.update(sb=sb, model=model, meta=meta, row=rows[0], source="supabase", warning=None)
        print(f"Loaded model id={rows[0]['id']} ({meta['algorithm']}) from Supabase")
    except Exception as e:
        model, meta = load_h5(LOCAL_MODEL)
        row = {"id": None, "algorithm": meta["algorithm"], "metrics": meta["cv_metrics"]}
        STATE.update(sb=sb, model=model, meta=meta, row=row, source="local file", warning=str(e)[:300])
        print(f"WARNING: Supabase model unavailable ({str(e)[:150]}). Using bundled {LOCAL_MODEL.name}")


@asynccontextmanager
async def lifespan(_):
    load_active_model()
    yield


app = FastAPI(title="NCF Budget Estimator", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class LineItem(BaseModel):
    Date: date
    Semester: str = Field(examples=["2nd Semester"])
    College: str = Field(examples=["CCS"])
    Event_Title: str = Field(examples=["Web Dev Workshop"])
    Category: str = Field(examples=["Food & Catering"])
    Item: str = Field(examples=["Snacks for Participants"])
    Qty: int = Field(gt=0, examples=[100])


@app.get("/health")
def health():
    r = STATE["row"]
    return {"status": "ok", "model_source": STATE["source"], "model_id": r["id"],
            "algorithm": r["algorithm"], "metrics": r["metrics"], "warning": STATE["warning"]}


@app.post("/reload")
def reload_model():
    """Pick up a newly uploaded model without restarting."""
    load_active_model()
    return health()


@app.post("/predict")
def predict_budget(items: list[LineItem], save: bool = True):
    if not items:
        raise HTTPException(400, "Send at least one line item.")
    out = predict(pd.DataFrame([i.model_dump() for i in items]), model=STATE["model"])
    rows = [{
        "model_id": STATE["row"]["id"],
        "event_title": r.Event_Title, "event_date": r.Date.date().isoformat(),
        "semester": r.Semester, "college": r.College, "category": r.Category, "item": r.Item,
        "qty": int(r.Qty), "pred_unit_cost": float(r.Pred_Unit_Cost), "pred_subtotal": float(r.Pred_Subtotal),
    } for r in out.itertuples()]
    saved, save_error = False, None
    if save:
        try:
            STATE["sb"].table("budget_predictions").insert(rows).execute()
            saved = True
        except Exception as e:  # still return predictions if the table is missing
            save_error = str(e)[:300]
    totals = out.groupby("Event_Title")["Pred_Subtotal"].sum().round(2).to_dict()
    return {"model_id": STATE["row"]["id"], "saved": saved, "save_error": save_error,
            "items": rows, "event_totals": totals}


@app.post("/predict/request/{request_id}")
def predict_budget_request(request_id: str, save: bool = True):
    """
    Predict every item of an existing budget request (budget_requests + budget_request_items),
    compare with the requested amounts, and save to budget_predictions (re-running replaces old rows).
    """
    sb = STATE["sb"]
    try:
        req, df = fetch_request(sb, request_id)
    except LookupError as e:
        raise HTTPException(404, str(e))

    out = predict(df, model=STATE["model"])
    items = []
    for r in out.itertuples():
        pred = float(r.Pred_Subtotal)
        variance = r.requested_amount - pred
        pct = round(variance / pred * 100, 2) if pred else 0.0
        items.append({
            "model_id": STATE["row"]["id"], "request_id": request_id, "request_item_id": r.request_item_id,
            "event_title": r.Event_Title, "event_date": r.Date.date().isoformat(), "semester": r.Semester,
            "college": r.College, "category": r.Category, "model_category": r.Category, "item": r.Item,
            "qty": int(round(r.Qty)), "pred_unit_cost": float(r.Pred_Unit_Cost), "pred_subtotal": pred,
            "requested_amount": round(float(r.requested_amount), 2), "variance_amount": round(variance, 2),
            "variance_pct": pct, "flag": flag_for(pct), "low_confidence": bool(r.low_confidence),
        })

    saved, save_error = False, None
    if save:
        try:
            sb.table("budget_predictions").delete().eq("request_id", request_id).execute()
            sb.table("budget_predictions").insert(items).execute()
            saved = True
        except Exception as e:
            save_error = str(e)[:300]

    requested = round(sum(i["requested_amount"] for i in items), 2)
    predicted = round(sum(i["pred_subtotal"] for i in items), 2)
    pct = round((requested - predicted) / predicted * 100, 2) if predicted else 0.0
    return {
        "request_id": request_id, "title": req["title"], "department_code": req["department_code"],
        "model_id": STATE["row"]["id"], "saved": saved, "save_error": save_error,
        "requested_total": requested, "predicted_total": predicted,
        "variance_amount": round(requested - predicted, 2), "variance_pct": pct, "flag": flag_for(pct),
        "items_over": sum(i["flag"] == "OVER" for i in items),
        "low_confidence_items": sum(i["low_confidence"] for i in items),
        "items": items,
    }
