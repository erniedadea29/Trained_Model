"""
Train and compare Random Forest, Gradient Boosting and Linear Regression on the
NCF event budget dataset, then save the best model as models/best_model.h5.

Target: log(Unit_Cost) of a line item. Subtotal = Qty * predicted Unit_Cost,
event total = sum of its line items.

Each algorithm is tuned with a hyperparameter grid. Every configuration is scored
with repeated GroupKFold grouped by event (an event never appears in both train
and test), so the scores reflect budgeting a NEW event.
"""
import io
import itertools
import json
import pickle
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
DATA = ROOT / "ncf_synthetic_budget_data_expanded.csv"
OUT = ROOT / "models"

CAT_COLS = ["College", "Semester", "Category", "Item"]
NUM_COLS = ["Qty_log", "Year", "Month", "Is_Bulk"]
FEATURES = CAT_COLS + NUM_COLS
N_SPLITS, N_REPEATS = 5, 3


# ---------------------------------------------------------------- data
def load(path=DATA) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    bad = df[df["Qty"] * df["Unit_Cost"] != df["Subtotal"]]
    if len(bad):
        print(f"WARNING: {len(bad)} rows where Qty*Unit_Cost != Subtotal")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Qty_log"] = np.log1p(df["Qty"])
    df["Is_Bulk"] = (df["Qty"] == 1).astype(int)  # lump-sum items (rentals, kits)
    return df


# ---------------------------------------------------------------- models
def preprocessor(scale: bool, min_freq: int) -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_freq), CAT_COLS),
        ("num", StandardScaler() if scale else "passthrough", NUM_COLS),
    ])


def grid(**params):
    keys = list(params)
    return [dict(zip(keys, vals)) for vals in itertools.product(*params.values())]


ALGORITHMS = {
    "Linear Regression": {
        # Ridge = linear regression with L2 regularisation; alpha=0.001 is ~plain OLS.
        "build": lambda p: Pipeline([("pre", preprocessor(True, p["min_freq"])),
                                     ("m", Ridge(alpha=p["alpha"]))]),
        "grid": grid(alpha=[0.001, 0.1, 0.3, 1.0, 3.0, 10.0], min_freq=[1, 2, 3]),
    },
    "Random Forest": {
        "build": lambda p: Pipeline([("pre", preprocessor(False, p["min_freq"])),
                                     ("m", RandomForestRegressor(n_estimators=300,
                                                                 max_features=p["max_features"],
                                                                 min_samples_leaf=p["min_samples_leaf"],
                                                                 random_state=42, n_jobs=-1))]),
        "grid": grid(max_features=[0.3, 0.6, 1.0], min_samples_leaf=[1, 2, 4], min_freq=[1, 2]),
    },
    "Gradient Boosting": {
        "build": lambda p: Pipeline([("pre", preprocessor(False, p["min_freq"])),
                                     ("m", GradientBoostingRegressor(n_estimators=p["n_estimators"],
                                                                     learning_rate=p["learning_rate"],
                                                                     max_depth=p["max_depth"],
                                                                     subsample=0.8, random_state=42))]),
        "grid": grid(n_estimators=[200, 500], learning_rate=[0.03, 0.1],
                     max_depth=[2, 3, 4], min_freq=[1, 2]),
    },
}


# ---------------------------------------------------------------- evaluation
def cv_predict(build, params, df):
    """Out-of-fold unit-cost predictions, averaged over repeated grouped splits."""
    X, y, groups = df[FEATURES], np.log(df["Unit_Cost"]), df["Event_Title"]
    preds = np.zeros((N_REPEATS, len(df)))
    for r in range(N_REPEATS):
        for tr, te in GroupKFold(N_SPLITS, shuffle=True, random_state=r).split(X, y, groups):
            m = build(params).fit(X.iloc[tr], y.iloc[tr])
            preds[r, te] = np.exp(m.predict(X.iloc[te]))
    return preds


def metrics(df, unit_preds):
    """Average each metric over the repeats."""
    rows = []
    for pu in unit_preds:
        ps = pu * df["Qty"].values
        ev = df.assign(pred=ps).groupby("Event_Title")[["Subtotal", "pred"]].sum()
        rows.append({
            "R2 (line item)": r2_score(df["Subtotal"], ps),
            "R2 (event total)": r2_score(ev["Subtotal"], ev["pred"]),
            "R2 (log unit cost)": r2_score(np.log(df["Unit_Cost"]), np.log(pu)),
            "MAE line (PHP)": mean_absolute_error(df["Subtotal"], ps),
            "RMSE line (PHP)": mean_squared_error(df["Subtotal"], ps) ** 0.5,
            "MAE event (PHP)": mean_absolute_error(ev["Subtotal"], ev["pred"]),
            "MAPE event (%)": float(np.mean(np.abs(ev["pred"] - ev["Subtotal"]) / ev["Subtotal"]) * 100),
            "Accuracy event (100-MAPE %)": 0.0,
        })
        rows[-1]["Accuracy event (100-MAPE %)"] = 100 - rows[-1]["MAPE event (%)"]
    return pd.DataFrame(rows).mean().to_dict()


# ---------------------------------------------------------------- .h5 I/O
def save_h5(model, path, meta: dict):
    buf = io.BytesIO()
    joblib.dump(model, buf)
    with h5py.File(path, "w") as f:
        f.create_dataset("model", data=np.frombuffer(buf.getvalue(), dtype=np.uint8))
        for k, v in meta.items():
            f.attrs[k] = json.dumps(v)


def load_h5(path):
    with h5py.File(path, "r") as f:
        model = joblib.load(io.BytesIO(f["model"][()].tobytes()))
        meta = {k: json.loads(v) for k, v in f.attrs.items()}
    return model, meta


# ---------------------------------------------------------------- main
def main():
    df = add_features(load())
    print(f"Rows: {len(df)} | Events: {df['Event_Title'].nunique()} | "
          f"CV: GroupKFold {N_SPLITS} folds x {N_REPEATS} repeats\n")

    summary, tuning_log = {}, []
    for name, algo in ALGORITHMS.items():
        best = None
        for params in algo["grid"]:
            m = metrics(df, cv_predict(algo["build"], params, df))
            tuning_log.append({"algorithm": name, **params, **m})
            if best is None or m["R2 (line item)"] > best[1]["R2 (line item)"]:
                best = (params, m)
        summary[name] = {"params": best[0], **best[1]}
        print(f"{name:18s} tried {len(algo['grid']):2d} configs | best R2(line)={best[1]['R2 (line item)']:.3f} "
              f"| params={best[0]}")

    table = pd.DataFrame({k: {m: v for m, v in s.items() if m != "params"} for k, s in summary.items()})
    table = table[sorted(table.columns, key=lambda c: -table.loc["R2 (line item)", c])]
    winner = table.columns[0]
    # rank each algorithm on every metric (1 = best); lower is better for errors
    higher_better = [r for r in table.index if r.startswith("R2") or r.startswith("Accuracy")]
    ranks = table.apply(lambda row: row.rank(ascending=row.name not in higher_better).astype(int), axis=1)

    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "comparison_matrix.csv", float_format="%.4f")
    ranks.to_csv(OUT / "comparison_ranks.csv")
    pd.DataFrame(tuning_log).to_csv(OUT / "tuning_log.csv", index=False, float_format="%.4f")

    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print("\n=== COMPARISON MATRIX (cross-validated, unseen events) ===")
    print(table.to_string())
    print("\n=== RANK PER METRIC (1 = best) ===")
    print(ranks.to_string())

    params = summary[winner]["params"]
    print(f"\nBest model: {winner} {params} -> refitting on all {len(df)} rows")
    model = ALGORITHMS[winner]["build"](params).fit(df[FEATURES], np.log(df["Unit_Cost"]))
    save_h5(model, OUT / "best_model.h5", {
        "algorithm": winner, "params": params, "features": FEATURES,
        "target": "log(Unit_Cost)", "cv_metrics": {k: float(v) for k, v in table[winner].items()},
    })
    for old in ("unit_cost_model.joblib", "metrics.json"):
        (OUT / old).unlink(missing_ok=True)
    print(f"Saved: {OUT / 'best_model.h5'}")


if __name__ == "__main__":
    main()
