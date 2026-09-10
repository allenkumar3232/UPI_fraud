
# Real-Time UPI Fraud & Anomaly Detection Pipeline

A synthetic-data pipeline that detects suspicious UPI transactions using
velocity, location/IP, and amount-deviation signals, with a two-stage
Isolation Forest → Logistic Regression model and a Streamlit dashboard.

## Pipeline

```
src/generate_data.py   ->  data/upi_transactions.csv     (124k+ synthetic transactions)
src/sql_features.py    ->  data/features.csv             (SQL-engineered risk features)
src/train_model.py     ->  outputs/model_metrics.json     (Isolation Forest + Logistic Regression)
                            outputs/flagged_transactions.csv
                            outputs/roc_pr_curves.png
                            outputs/feature_importance.png
src/streamlit_app.py   ->  interactive dashboard (run locally)
```

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib streamlit plotly
python src/generate_data.py
python src/sql_features.py
python src/train_model.py
streamlit run src/streamlit_app.py
```

## How it works

**1. Synthetic data (`generate_data.py`)**
6,000 users with realistic home cities and spend profiles generate ~120k
baseline transactions, into which three fraud patterns are injected
(~2.8% fraud rate):
- **Velocity fraud** — 6–15 transactions bursting within ~2.5 minutes
- **Location/IP fraud** — a transaction from a city hundreds of km away,
  minutes after a home-city transaction (impossible travel)
- **Amount fraud** — a single transaction 8–25x the user's typical amount

**2. SQL feature engineering (`sql_features.py`)**
Loads transactions into SQLite and computes real-time-style risk metrics
using window functions and self-joins — the same pattern you'd deploy
against a streaming table in Postgres/ClickHouse with a sliding window:
- `txns_last_1min` / `txns_last_5min` — velocity per user
- `distinct_ips_last_1hr` — IP-hopping signal
- `implied_speed_kmh` — haversine distance from previous transaction ÷
  time elapsed (impossible-travel detector)
- `amount_zscore` — deviation from the user's rolling 30-transaction
  mean/std amount

**3. Modeling (`train_model.py`)**
- **Stage 1 — Isolation Forest** (unsupervised): scores every
  transaction on anomalousness using only engineered features, no
  labels required — mirrors production, where confirmed fraud labels
  often arrive late via chargebacks.
- **Stage 2 — Logistic Regression** (supervised): trained on the
  engineered features + the Isolation Forest score, producing a
  calibrated fraud probability. This is what actually reduces false
  positives: single hard-coded rules (e.g. "flag if >3 txns/min")
  over-trigger on legitimate bursts like bill-splitting, whereas
  the model learns to weigh multiple weak signals together.
- **Threshold tuning**: rather than a naive 0.5 cutoff, the decision
  threshold is chosen to match the recall of a naive rule-based
  baseline, then compared on false positives at that same recall —
  an apples-to-apples comparison. In this run it cuts false positives
  by **~69%** relative to the rule-based baseline (your resume bullet's
  25% figure is a conservative floor — see `outputs/model_metrics.json`
  for the exact run).

## Notes on the resume bullets

- "100k+ multi-dimensional data logs" → 124,016 rows × 13 raw fields,
  generated in `generate_data.py`.
- "reducing false-positive alerts by 25%" → the pipeline as built
  actually achieves ~69% FP reduction at matched recall in this run;
  25% is a safe, defensible number to quote if you want to be
  conservative, or you can cite the measured number from
  `model_metrics.json` and re-run with different `random_state` seeds
  to see how stable it is.
- "query-optimized complex SQL aggregations" → `sql_features.py` uses
  window functions (`LAG`, rolling `AVG` with `ROWS BETWEEN`) and
  time-bounded self-joins, indexed on `(user_id, timestamp)`.

## Files

| File | Purpose |
|---|---|
| `data/upi_transactions.csv` | Raw synthetic transactions |
| `data/features.csv` | Post-SQL engineered features |
| `outputs/model_metrics.json` | All evaluation numbers |
| `outputs/flagged_transactions.csv` | Model-flagged transactions, ranked by probability |
| `outputs/roc_pr_curves.png` | ROC + Precision-Recall curves |
| `outputs/feature_importance.png` | Logistic Regression coefficients |
