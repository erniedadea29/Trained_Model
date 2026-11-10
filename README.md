# NCF Event Budget Estimator

Predicts the cost of planned college event line items (unit cost → subtotal → event total)
from the NCF event budget dataset, and serves predictions through a FastAPI backend
connected to Supabase.

## Model comparison

Each algorithm was tuned with a hyperparameter grid and evaluated with repeated
GroupKFold by event (5 folds × 3 repeats), so scores reflect budgeting events the model has never seen.

| Metric | **Random Forest** | Gradient Boosting | Linear Regression |
|---|---|---|---|
| R² (line item) | **0.506** | 0.481 | 0.400 |
| R² (event total) | **0.272** | 0.252 | 0.094 |
| R² (log unit cost) | **0.921** | 0.916 | 0.892 |
| MAE line item (PHP) | **2,649** | 2,760 | 2,995 |
| MAE event total (PHP) | **6,524** | 6,682 | 7,865 |
| Event accuracy (100 − MAPE) | **77.3%** | 76.2% | 71.5% |

Best model: **Random Forest**, saved as [`models/best_model.h5`](models/best_model.h5)
(scikit-learn model stored in HDF5; load with `train.load_h5`, not `keras.models.load_model`).
Full results: [`models/comparison_matrix.csv`](models/comparison_matrix.csv),
[`models/tuning_log.csv`](models/tuning_log.csv).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in SUPABASE_URL and SUPABASE_SERVICE_KEY
```

Run [`supabase_setup.sql`](supabase_setup.sql) once in the Supabase SQL Editor.

## Usage

| Command | What it does |
|---|---|
| `python train.py` | Tune + compare the 3 algorithms, save best to `models/best_model.h5` |
| `python predict.py [items.csv]` | Estimate budgets locally |
| `python upload_model.py` | Upload the `.h5` to Supabase Storage and mark it active |
| `python -m uvicorn api:app` | Start the API (docs at http://127.0.0.1:8000/docs) |
| `python verify_connection.py [--api URL]` | Prove the model ↔ Supabase ↔ API connection |

### API

- `POST /predict`: list of `{Date, Semester, College, Event_Title, Category, Item, Qty}` → predicted costs, saved to `budget_predictions`
- `POST /predict/request/{request_id}`: predicts every item of an existing `budget_requests` row
  (from `budget_request_items`), compares requested vs predicted, flags items >25% over/under, and saves
  to `budget_predictions` (re-running replaces old rows). Per-request summary: view `budget_request_predictions`.
  Requires [`supabase_link_requests.sql`](supabase_link_requests.sql); app categories are mapped in
  [`request_mapping.py`](request_mapping.py).
- `GET /health`: which model is loaded
- `POST /reload`: switch to a newly uploaded model

## Security

Never commit `.env`. The `service_role` key is server-side only; frontends use the `anon` key.
