import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


def build_user_item_matrix(df_transactions: pd.DataFrame) -> pd.DataFrame:
    """Build user-item interaction matrix."""
    matrix = df_transactions.pivot_table(
        index="customer_id",
        columns="product_id",
        values="total_amount",
        aggfunc="sum",
        fill_value=0
    )
    return matrix


def collaborative_filtering(df_transactions: pd.DataFrame, customer_id: str, top_n: int = 5):
    """User-based collaborative filtering recommendations."""
    matrix = build_user_item_matrix(df_transactions)

    if customer_id not in matrix.index:
        return pd.DataFrame()

    # Compute cosine similarity between users
    sim_matrix = cosine_similarity(matrix)
    sim_df = pd.DataFrame(sim_matrix, index=matrix.index, columns=matrix.index)

    # Get most similar users
    similar_users = sim_df[customer_id].drop(customer_id).nlargest(10).index.tolist()

    # Products the target customer has NOT bought
    bought = set(matrix.loc[customer_id][matrix.loc[customer_id] > 0].index)
    not_bought = [p for p in matrix.columns if p not in bought]

    if not not_bought:
        return pd.DataFrame()

    # Score products based on similar users' purchases
    scores = {}
    for product in not_bought:
        score = 0
        for sim_user in similar_users:
            score += matrix.loc[sim_user, product] * sim_df.loc[customer_id, sim_user]
        scores[product] = score

    recs = pd.Series(scores).sort_values(ascending=False).head(top_n)
    return recs.reset_index().rename(columns={"index": "product_id", 0: "score"})


def content_based_filtering(df_transactions: pd.DataFrame, product_df: pd.DataFrame,
                             customer_id: str, top_n: int = 5):
    """Content-based filtering using product features."""
    # Build product feature matrix (category one-hot)
    prod_features = pd.get_dummies(product_df, columns=["category"])
    feature_cols = [c for c in prod_features.columns if c.startswith("category_")]

    # Normalize price
    scaler = MinMaxScaler()
    prod_features["price_norm"] = scaler.fit_transform(prod_features[["price"]])
    feature_cols.append("price_norm")

    prod_matrix = prod_features.set_index("id")[feature_cols]

    # Get products bought by customer
    cust_tx = df_transactions[df_transactions["customer_id"] == customer_id]
    if cust_tx.empty:
        return pd.DataFrame()

    bought_products = cust_tx["product_id"].unique()
    bought_in_matrix = [p for p in bought_products if p in prod_matrix.index]

    if not bought_in_matrix:
        return pd.DataFrame()

    # Customer profile = mean of bought products' features
    cust_profile = prod_matrix.loc[bought_in_matrix].mean().values.reshape(1, -1)

    # Similarity with all products
    sim_scores = cosine_similarity(cust_profile, prod_matrix.values)[0]
    sim_series = pd.Series(sim_scores, index=prod_matrix.index)

    # Filter out already bought
    not_bought = [p for p in prod_matrix.index if p not in bought_products]
    recs = sim_series[not_bought].nlargest(top_n).reset_index()
    recs.columns = ["product_id", "similarity_score"]

    # Merge with product details
    recs = recs.merge(product_df[["id", "name", "category", "price"]], left_on="product_id", right_on="id")
    recs["similarity_score"] = (recs["similarity_score"] * 100).round(1)
    return recs.drop(columns=["id"])


def hybrid_recommendations(df_transactions: pd.DataFrame, product_df: pd.DataFrame,
                            customer_id: str, top_n: int = 5):
    """Combine content-based and collaborative filtering."""
    cb = content_based_filtering(df_transactions, product_df, customer_id, top_n * 2)
    cf = collaborative_filtering(df_transactions, customer_id, top_n * 2)

    if cb.empty and cf.empty:
        # Fallback: popular products
        popular = df_transactions.groupby("product_id")["total_amount"].sum().nlargest(top_n)
        recs = product_df[product_df["id"].isin(popular.index)].copy()
        recs["score"] = 50.0
        recs["method"] = "Popular"
        return recs.rename(columns={"id": "product_id", "name": "product_name"})

    # Combine scores
    combined = {}
    if not cb.empty:
        for _, row in cb.iterrows():
            pid = row["product_id"]
            combined[pid] = combined.get(pid, 0) + row["similarity_score"] * 0.6

    if not cf.empty:
        cf_max = cf["score"].max() if cf["score"].max() > 0 else 1
        for _, row in cf.iterrows():
            pid = row["product_id"]
            combined[pid] = combined.get(pid, 0) + (row["score"] / cf_max * 100) * 0.4

    top_pids = sorted(combined, key=combined.get, reverse=True)[:top_n]
    recs = product_df[product_df["id"].isin(top_pids)].copy()
    recs["score"] = recs["id"].map(combined).round(1)
    recs["method"] = "Hybrid"
    return recs.rename(columns={"id": "product_id", "name": "product_name"}).sort_values("score", ascending=False)
