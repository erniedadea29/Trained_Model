"""
Prove the trained model is connected to Supabase, step by step.

    python verify_connection.py              # checks Supabase
    python verify_connection.py --api URL    # also checks the running API, e.g. http://127.0.0.1:8000
"""
import hashlib
import sys
import tempfile
from pathlib import Path

import httpx
import numpy as np

from predict import EXAMPLE, predict
from supabase_client import BUCKET, MODEL_NAME, get_client
from train import load_h5

LOCAL = Path(__file__).parent / "models" / "best_model.h5"
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append(True)
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
        return detail
    except Exception as e:
        results.append(False)
        print(f"  FAIL  {name}\n        -> {str(e)[:200]}")
        return None


def main():
    api = sys.argv[sys.argv.index("--api") + 1] if "--api" in sys.argv else None
    print("Verifying trained model <-> Supabase connection\n")
    sb = get_client()
    ctx = {}

    check("1. Supabase credentials accepted",
          lambda: f"{len(sb.storage.list_buckets())} buckets visible")

    def tables():
        for t in ("ml_models", "budget_predictions"):
            sb.table(t).select("id").limit(1).execute()
        return "ml_models, budget_predictions"
    check("2. Tables exist", tables)

    def active():
        rows = (sb.table("ml_models").select("*").eq("name", MODEL_NAME).eq("is_active", True)
                  .order("created_at", desc=True).limit(1).execute().data)
        assert rows, "no active model registered - run `python upload_model.py`"
        ctx["row"] = rows[0]
        return f"id={rows[0]['id']}, {rows[0]['algorithm']}, file={rows[0]['storage_path']}"
    check("3. Active model registered in ml_models", active)

    def download():
        blob = sb.storage.from_(BUCKET).download(ctx["row"]["storage_path"])
        ctx["blob"] = blob
        return f"{len(blob)/1e6:.2f} MB downloaded from storage://{BUCKET}"
    check("4. Model file downloads from Supabase Storage", download)

    def same_file():
        remote = hashlib.sha256(ctx["blob"]).hexdigest()
        local = hashlib.sha256(LOCAL.read_bytes()).hexdigest()
        assert remote == local, "Supabase copy differs from local best_model.h5 (re-run upload_model.py)"
        return f"sha256 {remote[:12]}... matches local file"
    check("5. Supabase model is identical to local best_model.h5", same_file)

    def remote_predicts():
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp.write(ctx["blob"])
        model, _ = load_h5(tmp.name)
        a = predict(EXAMPLE, model=model)["Pred_Subtotal"].values
        b = predict(EXAMPLE, model=load_h5(LOCAL)[0])["Pred_Subtotal"].values
        assert np.allclose(a, b), "predictions differ"
        return f"example event total PHP {a.sum():,.2f}"
    check("6. Downloaded model makes the same predictions", remote_predicts)

    def write_read():
        row = sb.table("budget_predictions").insert({
            "model_id": ctx["row"]["id"], "event_title": "__verify_connection_test__",
            "college": "TEST", "category": "TEST", "item": "TEST", "qty": 1,
            "pred_unit_cost": 1, "pred_subtotal": 1}).execute().data[0]
        got = sb.table("budget_predictions").select("id").eq("id", row["id"]).execute().data
        sb.table("budget_predictions").delete().eq("id", row["id"]).execute()
        assert got, "inserted row not readable"
        return "test row inserted, read back, deleted"
    check("7. Predictions can be saved to budget_predictions", write_read)

    if api:
        def api_health():
            h = httpx.get(f"{api}/health", timeout=10).raise_for_status().json()
            assert h["model_id"] == ctx["row"]["id"], f"API serves model {h['model_id']}, active is {ctx['row']['id']}"
            return f"API serving model id={h['model_id']} ({h['algorithm']})"
        check("8. API is running the active Supabase model", api_health)

        def api_predict():
            body = EXAMPLE.assign(Date=EXAMPLE["Date"].astype(str)).to_dict("records")
            r = httpx.post(f"{api}/predict?save=false", json=body, timeout=30).raise_for_status().json()
            return f"POST /predict -> {r['event_totals']}"
        check("9. API /predict responds", api_predict)

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed -> "
          + ("CONNECTED" if passed == len(results) else "NOT fully connected yet"))
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
