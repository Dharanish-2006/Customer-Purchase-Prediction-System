import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_rfm(df_transactions: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """Compute RFM (Recency, Frequency, Monetary) metrics per customer."""
    df = df_transactions.copy()
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])

    if reference_date is None:
        reference_date = df["purchase_date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        Recency=("purchase_date", lambda x: (reference_date - x.max()).days),
        Frequency=("transaction_id", "count"),
        Monetary=("total_amount", "sum"),
    ).reset_index()

    # RFM Scores (1-5, higher is better)
    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    # Segment labels
    def rfm_segment(row):
        score = row["RFM_Score"]
        if score >= 13:
            return "Champions"
        elif score >= 10:
            return "Loyal Customers"
        elif score >= 7:
            return "Potential Loyalists"
        elif score >= 5:
            return "At Risk"
        else:
            return "Lost Customers"

    rfm["RFM_Segment"] = rfm.apply(rfm_segment, axis=1)
    return rfm


def kmeans_segmentation(rfm: pd.DataFrame, n_clusters=4) -> pd.DataFrame:
    """Perform K-Means clustering on RFM features."""
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # Find optimal k via silhouette
    best_k, best_score = n_clusters, -1
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        sc = silhouette_score(scaled, labels)
        if sc > best_score:
            best_score = sc
            best_k = k

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm["Cluster"] = km_final.fit_predict(scaled)

    # Label clusters by average monetary value
    cluster_means = rfm.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False)
    label_map = {cid: lbl for cid, lbl in zip(
        cluster_means.index,
        ["High Value", "Mid Value", "Low Value", "Occasional"][:best_k]
    )}
    rfm["Cluster_Label"] = rfm["Cluster"].map(label_map)
    return rfm, best_k, best_score


def eda_summary(df_transactions: pd.DataFrame) -> dict:
    """Compute key EDA metrics."""
    df = df_transactions.copy()
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    df["month"] = df["purchase_date"].dt.to_period("M").astype(str)

    summary = {
        "total_revenue": round(df["total_amount"].sum(), 2),
        "total_transactions": len(df),
        "unique_customers": df["customer_id"].nunique(),
        "unique_products": df["product_id"].nunique(),
        "avg_order_value": round(df["total_amount"].mean(), 2),
        "monthly_revenue": df.groupby("month")["total_amount"].sum().reset_index().rename(
            columns={"month": "Month", "total_amount": "Revenue"}
        ),
        "category_revenue": df.groupby("category")["total_amount"].sum().reset_index().rename(
            columns={"category": "Category", "total_amount": "Revenue"}
        ).sort_values("Revenue", ascending=False),
        "top_products": df.groupby("product_name")["total_amount"].sum().reset_index().rename(
            columns={"product_name": "Product", "total_amount": "Revenue"}
        ).sort_values("Revenue", ascending=False).head(10),
        "daily_transactions": df.groupby("purchase_date").size().reset_index(name="Transactions"),
    }
    return summary
