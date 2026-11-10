"""
Upload models/best_model.h5 to Supabase Storage and register it as the active model.
Run after every `python train.py`.
"""
from datetime import datetime
from pathlib import Path

from supabase_client import BUCKET, MODEL_NAME, get_client
from train import load_h5

LOCAL = Path(__file__).parent / "models" / "best_model.h5"


def main():
    sb = get_client()
    _, meta = load_h5(LOCAL)

    object_name = f"best_model_{datetime.now():%Y%m%d_%H%M%S}.h5"
    sb.storage.from_(BUCKET).upload(object_name, LOCAL.read_bytes(),
                                    {"content-type": "application/x-hdf5"})
    print(f"Uploaded {LOCAL.name} -> storage://{BUCKET}/{object_name}")

    sb.table("ml_models").update({"is_active": False}).eq("name", MODEL_NAME).execute()
    row = sb.table("ml_models").insert({
        "name": MODEL_NAME,
        "algorithm": meta["algorithm"],
        "params": meta["params"],
        "metrics": meta["cv_metrics"],
        "storage_path": object_name,
        "is_active": True,
    }).execute().data[0]
    print(f"Registered model id={row['id']} ({row['algorithm']}) as active")


if __name__ == "__main__":
    main()
