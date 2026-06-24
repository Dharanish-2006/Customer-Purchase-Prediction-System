import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_sample_data(n_customers=500, n_transactions=3000, seed=42):
    """Generate realistic synthetic e-commerce transaction data."""
    np.random.seed(seed)
    random.seed(seed)

    # Product catalog
    products = [
        {"id": "P001", "name": "Wireless Headphones", "category": "Electronics", "price": 79.99},
        {"id": "P002", "name": "Smart Watch", "category": "Electronics", "price": 199.99},
        {"id": "P003", "name": "Running Shoes", "category": "Sports", "price": 89.99},
        {"id": "P004", "name": "Yoga Mat", "category": "Sports", "price": 29.99},
        {"id": "P005", "name": "Coffee Maker", "category": "Kitchen", "price": 49.99},
        {"id": "P006", "name": "Air Fryer", "category": "Kitchen", "price": 69.99},
        {"id": "P007", "name": "Novel - Bestseller", "category": "Books", "price": 14.99},
        {"id": "P008", "name": "Cookbook", "category": "Books", "price": 24.99},
        {"id": "P009", "name": "Face Moisturizer", "category": "Beauty", "price": 34.99},
        {"id": "P010", "name": "Vitamin C Serum", "category": "Beauty", "price": 44.99},
        {"id": "P011", "name": "Desk Lamp", "category": "Home", "price": 39.99},
        {"id": "P012", "name": "Throw Pillow Set", "category": "Home", "price": 54.99},
        {"id": "P013", "name": "Gaming Mouse", "category": "Electronics", "price": 59.99},
        {"id": "P014", "name": "Protein Powder", "category": "Sports", "price": 44.99},
        {"id": "P015", "name": "Scented Candle", "category": "Home", "price": 19.99},
    ]

    # Customer segments for realistic data
    segment_profiles = {
        "high_value": {"freq": (10, 25), "spend": (150, 500), "recency": (1, 30)},
        "regular":    {"freq": (4, 12),  "spend": (50, 150),  "recency": (15, 90)},
        "occasional": {"freq": (1, 5),   "spend": (20, 80),   "recency": (60, 180)},
        "churned":    {"freq": (1, 3),   "spend": (10, 50),   "recency": (150, 365)},
    }

    segments = list(segment_profiles.keys())
    seg_weights = [0.15, 0.35, 0.30, 0.20]

    transactions = []
    customer_data = []
    end_date = datetime(2024, 12, 31)

    for cust_id in range(1, n_customers + 1):
        cid = f"C{cust_id:04d}"
        seg = np.random.choice(segments, p=seg_weights)
        prof = segment_profiles[seg]

        freq = np.random.randint(*prof["freq"])
        recency_days = np.random.randint(*prof["recency"])
        last_purchase = end_date - timedelta(days=recency_days)

        age = np.random.randint(18, 70)
        location = np.random.choice(["North", "South", "East", "West", "Central"])
        browsing_sessions = freq * np.random.randint(2, 6)

        # Assign preferred categories
        preferred_cats = np.random.choice(
            ["Electronics", "Sports", "Kitchen", "Books", "Beauty", "Home"],
            size=np.random.randint(1, 3), replace=False
        ).tolist()

        customer_data.append({
            "customer_id": cid,
            "age": age,
            "location": location,
            "segment_true": seg,
            "preferred_categories": "|".join(preferred_cats),
            "browsing_sessions": browsing_sessions,
        })

        for _ in range(freq):
            days_ago = np.random.randint(recency_days, min(365, recency_days + 300))
            t_date = end_date - timedelta(days=days_ago)
            product = random.choice(products)
            qty = np.random.randint(1, 4)
            spend = round(product["price"] * qty * np.random.uniform(0.85, 1.15), 2)

            transactions.append({
                "transaction_id": f"T{len(transactions)+1:06d}",
                "customer_id": cid,
                "product_id": product["id"],
                "product_name": product["name"],
                "category": product["category"],
                "quantity": qty,
                "unit_price": product["price"],
                "total_amount": spend,
                "purchase_date": t_date.strftime("%Y-%m-%d"),
                "browsing_time_mins": np.random.randint(2, 45),
            })

    df_transactions = pd.DataFrame(transactions)
    df_customers = pd.DataFrame(customer_data)
    product_df = pd.DataFrame(products)

    return df_transactions, df_customers, product_df


def load_or_generate_data():
    """Return generated datasets."""
    return generate_sample_data()
