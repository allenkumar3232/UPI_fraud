"""
generate_data.py
-----------------
Generates a synthetic UPI (Unified Payments Interface) transaction dataset
with realistic multi-dimensional fields and three injected fraud patterns:

  1. VELOCITY FRAUD   -> bursts of many transactions in a short window
  2. LOCATION/IP FRAUD -> "impossible travel" - location/IP jumps that are
                          geographically implausible given the time elapsed
  3. AMOUNT FRAUD      -> transaction amounts far outside a user's normal
                          spending pattern (e.g. account takeover, mule txns)

Output: data/upi_transactions.csv  (100k+ rows)
Ground-truth fraud labels are kept ONLY for evaluation purposes -- the
detection pipeline itself is designed to work without needing them.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

N_USERS = 6000
N_TRANSACTIONS = 120_000
FRAUD_RATE = 0.025  # ~2.5% of transactions are fraudulent

CITIES = [
    ("Mumbai", 19.0760, 72.8777), ("Delhi", 28.7041, 77.1025),
    ("Bengaluru", 12.9716, 77.5946), ("Hyderabad", 17.3850, 78.4867),
    ("Chennai", 13.0827, 80.2707), ("Kolkata", 22.5726, 88.3639),
    ("Pune", 18.5204, 73.8567), ("Ahmedabad", 23.0225, 72.5714),
    ("Jaipur", 26.9124, 75.7873), ("Lucknow", 26.8467, 80.9462),
    ("Guwahati", 26.1445, 91.7362), ("Silchar", 24.8333, 92.7789),
]

MERCHANT_CATEGORIES = [
    "grocery", "food_delivery", "utility_bill", "peer_transfer",
    "ecommerce", "fuel", "entertainment", "healthcare", "travel", "rent",
]

BANKS = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB", "BOB", "Paytm_Bank"]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def make_users(n_users):
    """Each user gets a home city, a typical spend profile, and a device/IP base."""
    users = []
    for uid in range(1, n_users + 1):
        city = CITIES[rng.integers(0, len(CITIES))]
        avg_amount = np.round(rng.lognormal(mean=6.0, sigma=0.8), 2)  # skewed, INR
        avg_amount = float(np.clip(avg_amount, 50, 20000))
        std_amount = avg_amount * rng.uniform(0.2, 0.5)
        device_id = f"DEV{uid:06d}"
        base_ip_octets = rng.integers(1, 255, size=2)
        users.append({
            "user_id": f"U{uid:06d}",
            "home_city": city[0], "home_lat": city[1], "home_lon": city[2],
            "avg_amount": avg_amount, "std_amount": std_amount,
            "device_id": device_id,
            "base_ip": f"49.{base_ip_octets[0]}.{base_ip_octets[1]}",
            "bank": BANKS[rng.integers(0, len(BANKS))],
        })
    return pd.DataFrame(users)


def generate_normal_transactions(users_df, n_txns):
    """Baseline, non-fraudulent transaction behavior."""
    start_time = datetime(2025, 1, 1)
    user_idx = rng.integers(0, len(users_df), size=n_txns)
    sampled_users = users_df.iloc[user_idx].reset_index(drop=True)

    # Spread transactions across 90 days, with realistic hour-of-day skew
    day_offsets = rng.integers(0, 90, size=n_txns)
    hour_weights = np.array([
        0.5,0.3,0.2,0.2,0.3,0.5,1,2,3,3,2.5,2.5,3,2.5,2,2,2.5,3,4,4,3,2,1.5,1
    ])
    hour_weights = hour_weights / hour_weights.sum()
    hours = rng.choice(24, size=n_txns, p=hour_weights)
    minutes = rng.integers(0, 60, size=n_txns)
    seconds = rng.integers(0, 60, size=n_txns)

    timestamps = [
        start_time + timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s))
        for d, h, m, s in zip(day_offsets, hours, minutes, seconds)
    ]

    amounts = np.clip(
        rng.normal(sampled_users["avg_amount"], sampled_users["std_amount"] + 1e-6),
        10, None
    ).round(2)

    # Small realistic jitter in location/IP for a normal user (rarely travels)
    lat_jitter = rng.normal(0, 0.05, size=n_txns)
    lon_jitter = rng.normal(0, 0.05, size=n_txns)
    ip_last_octet = rng.integers(1, 255, size=n_txns)

    df = pd.DataFrame({
        "user_id": sampled_users["user_id"],
        "timestamp": timestamps,
        "amount": amounts,
        "location_city": sampled_users["home_city"],
        "lat": sampled_users["home_lat"].values + lat_jitter,
        "lon": sampled_users["home_lon"].values + lon_jitter,
        "ip_address": sampled_users["base_ip"] + "." + ip_last_octet.astype(str),
        "device_id": sampled_users["device_id"],
        "bank": sampled_users["bank"],
        "merchant_category": rng.choice(MERCHANT_CATEGORIES, size=n_txns),
        "is_fraud": 0,
        "fraud_type": "none",
    })
    return df


def inject_velocity_fraud(df, users_df, n_events):
    """Pick random users and burst 6-15 transactions within 1-3 minutes."""
    injected = []
    target_users = users_df.sample(n=n_events, random_state=1, replace=True).reset_index(drop=True)
    for i in range(n_events):
        u = target_users.iloc[i]
        base_time = datetime(2025, 1, 1) + timedelta(
            days=int(rng.integers(0, 90)), hours=int(rng.integers(0, 24))
        )
        n_burst = int(rng.integers(6, 16))
        for _ in range(n_burst):
            offset_sec = int(rng.integers(0, 150))  # all within 2.5 minutes
            amt = float(np.clip(rng.normal(u["avg_amount"] * 0.6, 50), 10, None))
            injected.append({
                "user_id": u["user_id"],
                "timestamp": base_time + timedelta(seconds=offset_sec),
                "amount": round(amt, 2),
                "location_city": u["home_city"],
                "lat": u["home_lat"] + rng.normal(0, 0.02),
                "lon": u["home_lon"] + rng.normal(0, 0.02),
                "ip_address": u["base_ip"] + f".{int(rng.integers(1,255))}",
                "device_id": u["device_id"],
                "bank": u["bank"],
                "merchant_category": "peer_transfer",
                "is_fraud": 1,
                "fraud_type": "velocity",
            })
    return pd.DataFrame(injected)


def inject_location_fraud(df, users_df, n_events):
    """A transaction from a city far from home, minutes after a home-city transaction
    -> physically impossible travel speed = classic account-takeover signature."""
    injected = []
    target_users = users_df.sample(n=n_events, random_state=2, replace=True).reset_index(drop=True)
    for i in range(n_events):
        u = target_users.iloc[i]
        base_time = datetime(2025, 1, 1) + timedelta(
            days=int(rng.integers(0, 90)), hours=int(rng.integers(0, 24))
        )
        # legit txn at home
        injected.append({
            "user_id": u["user_id"], "timestamp": base_time,
            "amount": round(float(np.clip(rng.normal(u["avg_amount"], u["std_amount"]), 10, None)), 2),
            "location_city": u["home_city"], "lat": u["home_lat"], "lon": u["home_lon"],
            "ip_address": u["base_ip"] + f".{int(rng.integers(1,255))}",
            "device_id": u["device_id"], "bank": u["bank"],
            "merchant_category": rng.choice(MERCHANT_CATEGORIES),
            "is_fraud": 0, "fraud_type": "none",
        })
        # fraudulent txn from a far city, only 2-8 minutes later
        far_city = CITIES[rng.integers(0, len(CITIES))]
        while far_city[0] == u["home_city"]:
            far_city = CITIES[rng.integers(0, len(CITIES))]
        injected.append({
            "user_id": u["user_id"],
            "timestamp": base_time + timedelta(minutes=int(rng.integers(2, 8))),
            "amount": round(float(np.clip(rng.normal(u["avg_amount"] * 1.8, 200), 10, None)), 2),
            "location_city": far_city[0], "lat": far_city[1], "lon": far_city[2],
            "ip_address": f"{int(rng.integers(1,255))}.{int(rng.integers(1,255))}.{int(rng.integers(1,255))}.{int(rng.integers(1,255))}",
            "device_id": f"DEV{int(rng.integers(900000,999999))}",  # new/unknown device
            "bank": u["bank"],
            "merchant_category": "peer_transfer",
            "is_fraud": 1, "fraud_type": "location_ip",
        })
    return pd.DataFrame(injected)


def inject_amount_fraud(df, users_df, n_events):
    """A single transaction amount wildly outside the user's normal range."""
    injected = []
    target_users = users_df.sample(n=n_events, random_state=3, replace=True).reset_index(drop=True)
    for i in range(n_events):
        u = target_users.iloc[i]
        ts = datetime(2025, 1, 1) + timedelta(
            days=int(rng.integers(0, 90)), hours=int(rng.integers(0, 24)),
            minutes=int(rng.integers(0, 60))
        )
        spike_amt = u["avg_amount"] * rng.uniform(8, 25) + 5000
        injected.append({
            "user_id": u["user_id"], "timestamp": ts,
            "amount": round(float(spike_amt), 2),
            "location_city": u["home_city"], "lat": u["home_lat"], "lon": u["home_lon"],
            "ip_address": u["base_ip"] + f".{int(rng.integers(1,255))}",
            "device_id": u["device_id"], "bank": u["bank"],
            "merchant_category": rng.choice(["ecommerce", "peer_transfer", "travel"]),
            "is_fraud": 1, "fraud_type": "amount_spike",
        })
    return pd.DataFrame(injected)


def main():
    print("Generating user base...")
    users_df = make_users(N_USERS)

    n_fraud_total = int(N_TRANSACTIONS * FRAUD_RATE)
    n_velocity_events = n_fraud_total // 20   # each event spawns ~10 txns
    n_location_events = n_fraud_total // 6    # each event spawns 2 txns (1 fraud)
    n_amount_events = n_fraud_total // 2

    n_normal = N_TRANSACTIONS  # generate full normal volume, fraud is additive

    print(f"Generating {n_normal:,} baseline transactions...")
    normal_df = generate_normal_transactions(users_df, n_normal)

    print("Injecting velocity fraud bursts...")
    velocity_df = inject_velocity_fraud(normal_df, users_df, n_velocity_events)

    print("Injecting location/IP impossible-travel fraud...")
    location_df = inject_location_fraud(normal_df, users_df, n_location_events)

    print("Injecting amount-spike fraud...")
    amount_df = inject_amount_fraud(normal_df, users_df, n_amount_events)

    full_df = pd.concat([normal_df, velocity_df, location_df, amount_df], ignore_index=True)
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)
    full_df["transaction_id"] = [f"TXN{100000+i}" for i in range(len(full_df))]

    cols = ["transaction_id", "user_id", "timestamp", "amount", "location_city",
            "lat", "lon", "ip_address", "device_id", "bank", "merchant_category",
            "is_fraud", "fraud_type"]
    full_df = full_df[cols]

    out_path = "/home/claude/upi_fraud_pipeline/data/upi_transactions.csv"
    full_df.to_csv(out_path, index=False)

    print(f"\nDone. Wrote {len(full_df):,} rows to {out_path}")
    print(f"Fraud rate: {full_df['is_fraud'].mean()*100:.2f}%")
    print(full_df['fraud_type'].value_counts())


if __name__ == "__main__":
    main()
