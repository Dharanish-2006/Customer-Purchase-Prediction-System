import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def build_features(df_transactions: pd.DataFrame, df_customers: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for purchase prediction."""
    df = df_transactions.copy()
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    ref = df["purchase_date"].max()

    # RFM features
    rfm = df.groupby("customer_id").agg(
        recency=("purchase_date", lambda x: (ref - x.max()).days),
        frequency=("transaction_id", "count"),
        monetary=("total_amount", "sum"),
        avg_order_value=("total_amount", "mean"),
        num_categories=("category", "nunique"),
        avg_browsing=("browsing_time_mins", "mean"),
        last_qty=("quantity", "last"),
    ).reset_index()

    # Merge customer demographics
    cust = df_customers[["customer_id", "age", "location", "browsing_sessions"]].copy()
    features = rfm.merge(cust, on="customer_id", how="left")

    # Encode location
    le = LabelEncoder()
    features["location_enc"] = le.fit_transform(features["location"].fillna("Unknown"))

    # Target: will purchase again (recency < 60 days = likely)
    features["will_purchase"] = (features["recency"] < 60).astype(int)

    features = features.fillna(0)
    return features


def train_model(features: pd.DataFrame, model_type: str = "XGBoost"):
    """Train classification model and return model + metrics."""
    feature_cols = [
        "recency", "frequency", "monetary", "avg_order_value",
        "num_categories", "avg_browsing", "last_qty",
        "age", "browsing_sessions", "location_enc"
    ]
    X = features[feature_cols]
    y = features["will_purchase"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    if model_type == "XGBoost" and HAS_XGB:
        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42
        )
    elif model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    else:
        model = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)

    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importance": dict(zip(feature_cols, model.feature_importances_.tolist())),
        "cv_scores": cross_val_score(model, X_train_sc, y_train, cv=5, scoring="roc_auc").tolist(),
    }

    return model, scaler, feature_cols, metrics


def predict_customers(model, scaler, feature_cols, features: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for all customers."""
    X = features[feature_cols].fillna(0)
    X_sc = scaler.transform(X)
    probs = model.predict_proba(X_sc)[:, 1]
    preds = model.predict(X_sc)

    result = features[["customer_id", "recency", "frequency", "monetary"]].copy()
    result["purchase_probability"] = np.round(probs * 100, 1)
    result["will_purchase_pred"] = preds
    result["risk_level"] = pd.cut(
        probs, bins=[0, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"]
    )
    return result.sort_values("purchase_probability", ascending=False)
