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

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predict import predict
from supabase_client import BUCKET, MODEL_NAME, get_client
from train import load_h5

STATE = {}


def load_active_model():
    sb = get_client()
    rows = (sb.table("ml_models").select("*").eq("name", MODEL_NAME).eq("is_active", True)
              .order("created_at", desc=True).limit(1).execute().data)
    if not rows:
        raise RuntimeError("No active model in ml_models. Run `python upload_model.py` first.")
    blob = sb.storage.from_(BUCKET).download(rows[0]["storage_path"])
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp.write(blob)
    model, meta = load_h5(tmp.name)
    STATE.update(sb=sb, model=model, meta=meta, row=rows[0])
    print(f"Loaded model id={rows[0]['id']} ({meta['algorithm']}) from Supabase")


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
    return {"status": "ok", "model_id": r["id"], "algorithm": r["algorithm"], "metrics": r["metrics"]}


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
    if save:
        STATE["sb"].table("budget_predictions").insert(rows).execute()
    totals = out.groupby("Event_Title")["Pred_Subtotal"].sum().round(2).to_dict()
    return {"model_id": STATE["row"]["id"], "saved": save, "items": rows, "event_totals": totals}
