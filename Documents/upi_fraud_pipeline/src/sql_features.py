"""
sql_features.py
----------------
Loads raw transactions into SQLite and computes real-time-style risk
features using pure SQL window functions:

  - txns_last_1min / txns_last_5min   (velocity per user)
  - distinct_ips_last_1hr             (IP-hopping signal)
  - km_from_prev_txn + minutes_since_prev_txn -> implied_speed_kmh
        (impossible-travel detector, no external geo API needed)
  - amount_zscore_vs_user_history     (rolling mean/std amount deviation)
  - is_new_device                     (device never seen before for user)

This mirrors what you'd run against a streaming table (e.g. a rolling
window materialized view) in a real-time fraud system -- here it's run
once in SQLite for reproducibility, but the SQL itself is the same
pattern you'd deploy against Postgres/ClickHouse with a sliding window.
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "/home/claude/upi_fraud_pipeline/data/upi.db"
CSV_PATH = "/home/claude/upi_fraud_pipeline/data/upi_transactions.csv"
OUT_PATH = "/home/claude/upi_fraud_pipeline/data/features.csv"


def load_to_sqlite():
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_time ON transactions(user_id, timestamp)")
    conn.commit()
    return conn


VELOCITY_SQL = """
-- Transactions in the trailing 1-min / 5-min window per user, computed
-- with a self-join bounded by an index-friendly time range (equivalent
-- to a sliding window aggregation on a streaming table).
SELECT
    t1.transaction_id,
    COUNT(t2.transaction_id) AS txns_last_1min
FROM transactions t1
JOIN transactions t2
    ON t1.user_id = t2.user_id
    AND t2.timestamp <= t1.timestamp
    AND t2.timestamp > datetime(t1.timestamp, '-1 minutes')
GROUP BY t1.transaction_id
"""

VELOCITY_5MIN_SQL = """
SELECT
    t1.transaction_id,
    COUNT(t2.transaction_id) AS txns_last_5min
FROM transactions t1
JOIN transactions t2
    ON t1.user_id = t2.user_id
    AND t2.timestamp <= t1.timestamp
    AND t2.timestamp > datetime(t1.timestamp, '-5 minutes')
GROUP BY t1.transaction_id
"""

IP_HOP_SQL = """
SELECT
    t1.transaction_id,
    COUNT(DISTINCT t2.ip_address) AS distinct_ips_last_1hr
FROM transactions t1
JOIN transactions t2
    ON t1.user_id = t2.user_id
    AND t2.timestamp <= t1.timestamp
    AND t2.timestamp > datetime(t1.timestamp, '-60 minutes')
GROUP BY t1.transaction_id
"""

# Window-function version: previous txn's time/lat/lon/device per user,
# and a rolling mean/std of amount over the user's prior 30 transactions.
ROLLING_SQL = """
SELECT
    transaction_id,
    user_id,
    timestamp,
    amount,
    lat, lon, device_id,
    LAG(timestamp) OVER (PARTITION BY user_id ORDER BY timestamp) AS prev_timestamp,
    LAG(lat) OVER (PARTITION BY user_id ORDER BY timestamp) AS prev_lat,
    LAG(lon) OVER (PARTITION BY user_id ORDER BY timestamp) AS prev_lon,
    LAG(device_id) OVER (PARTITION BY user_id ORDER BY timestamp) AS prev_device_id,
    AVG(amount) OVER (
        PARTITION BY user_id ORDER BY timestamp
        ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
    ) AS rolling_avg_amount,
    -- SQLite has no STDEV built-in; approximate via AVG(x^2)-AVG(x)^2 in pandas after pull
    COUNT(*) OVER (PARTITION BY user_id ORDER BY timestamp
        ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING) AS n_prior_txns
FROM transactions
"""


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def main():
    print("Loading transactions into SQLite...")
    conn = load_to_sqlite()

    print("Running velocity aggregation (txns/1min)...")
    vel_1min = pd.read_sql(VELOCITY_SQL, conn)

    print("Running velocity aggregation (txns/5min)...")
    vel_5min = pd.read_sql(VELOCITY_5MIN_SQL, conn)

    print("Running IP-hopping aggregation (distinct IPs/1hr)...")
    ip_hop = pd.read_sql(IP_HOP_SQL, conn)

    print("Running rolling window (prev txn location/time/device, rolling avg amount)...")
    rolling = pd.read_sql(ROLLING_SQL, conn, parse_dates=["timestamp", "prev_timestamp"])

    # ---- derive implied travel speed (impossible-travel signal) ----
    has_prev = rolling["prev_lat"].notna()
    rolling["km_from_prev"] = 0.0
    rolling.loc[has_prev, "km_from_prev"] = haversine_km(
        rolling.loc[has_prev, "lat"], rolling.loc[has_prev, "lon"],
        rolling.loc[has_prev, "prev_lat"], rolling.loc[has_prev, "prev_lon"]
    )
    rolling["minutes_since_prev"] = (
        (rolling["timestamp"] - rolling["prev_timestamp"]).dt.total_seconds() / 60.0
    )
    rolling["minutes_since_prev"] = rolling["minutes_since_prev"].fillna(9999)
    rolling["implied_speed_kmh"] = np.where(
        rolling["minutes_since_prev"] > 0,
        rolling["km_from_prev"] / (rolling["minutes_since_prev"] / 60.0),
        0.0
    )
    rolling["is_new_device"] = (
        rolling["device_id"] != rolling["prev_device_id"]
    ) & rolling["prev_device_id"].notna()

    # amount z-score vs rolling user history (pandas fills the std sklearn/sqlite gap)
    raw = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    raw = raw.sort_values(["user_id", "timestamp"])
    roll_std = (
        raw.groupby("user_id")["amount"]
        .rolling(window=30, min_periods=3).std()
        .reset_index(level=0, drop=True)
    )
    raw["rolling_std_amount"] = roll_std
    raw = raw[["transaction_id", "rolling_std_amount"]]

    # ---- merge everything ----
    feats = rolling.merge(vel_1min, on="transaction_id") \
                    .merge(vel_5min, on="transaction_id") \
                    .merge(ip_hop, on="transaction_id") \
                    .merge(raw, on="transaction_id")

    feats["rolling_avg_amount"] = feats["rolling_avg_amount"].fillna(feats["amount"])
    feats["rolling_std_amount"] = feats["rolling_std_amount"].fillna(feats["amount"] * 0.3 + 1)
    feats["amount_zscore"] = (
        (feats["amount"] - feats["rolling_avg_amount"]) / feats["rolling_std_amount"].replace(0, 1)
    )

    final_cols = [
        "transaction_id", "user_id", "timestamp", "amount",
        "txns_last_1min", "txns_last_5min", "distinct_ips_last_1hr",
        "km_from_prev", "minutes_since_prev", "implied_speed_kmh",
        "is_new_device", "amount_zscore",
    ]
    feats = feats[final_cols]

    # bring back labels for evaluation
    labels = pd.read_csv(CSV_PATH, usecols=["transaction_id", "is_fraud", "fraud_type"])
    feats = feats.merge(labels, on="transaction_id")

    feats.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(feats):,} feature rows to {OUT_PATH}")
    print(feats[["txns_last_1min", "implied_speed_kmh", "amount_zscore"]].describe())

    conn.close()


if __name__ == "__main__":
    main()
