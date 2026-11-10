"""
Estimate the budget of a planned event using the trained model.

Usage:
    python predict.py                      # runs the built-in example
    python predict.py planned_items.csv    # CSV with columns:
        Date,Semester,College,Event_Title,Category,Item,Qty
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from train import FEATURES, add_features, load_h5

MODEL = Path(__file__).parent / "models" / "best_model.h5"

EXAMPLE = pd.DataFrame([
    {"Date": "2027-02-15", "Semester": "2nd Semester", "College": "CCS", "Event_Title": "Web Dev Workshop",
     "Category": "Honorarium/Speakers", "Item": "Guest Speaker Honorarium", "Qty": 2},
    {"Date": "2027-02-15", "Semester": "2nd Semester", "College": "CCS", "Event_Title": "Web Dev Workshop",
     "Category": "Food & Catering", "Item": "Snacks for Participants", "Qty": 100},
    {"Date": "2027-02-15", "Semester": "2nd Semester", "College": "CCS", "Event_Title": "Web Dev Workshop",
     "Category": "Printing & Documentation", "Item": "Certificates & Programs Printing", "Qty": 100},
    {"Date": "2027-02-15", "Semester": "2nd Semester", "College": "CCS", "Event_Title": "Web Dev Workshop",
     "Category": "Venue/Facilities", "Item": "Function Hall Rental", "Qty": 1},
])


def predict(items: pd.DataFrame, model=None) -> pd.DataFrame:
    """Add Pred_Unit_Cost and Pred_Subtotal columns. Loads the local .h5 if no model is given."""
    if model is None:
        model, meta = load_h5(MODEL)
        print(f"Model: {meta['algorithm']} {meta['params']}\n")
    items = items.copy()
    items["Date"] = pd.to_datetime(items["Date"])
    feats = add_features(items)
    items["Pred_Unit_Cost"] = np.exp(model.predict(feats[FEATURES])).round(2)
    items["Pred_Subtotal"] = (items["Pred_Unit_Cost"] * items["Qty"]).round(2)
    return items


if __name__ == "__main__":
    items = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else EXAMPLE
    out = predict(items)
    print(out[["Event_Title", "Category", "Item", "Qty", "Pred_Unit_Cost", "Pred_Subtotal"]].to_string(index=False))
    print("\nEstimated total per event:")
    print(out.groupby("Event_Title")["Pred_Subtotal"].sum().map("PHP {:,.2f}".format).to_string())
