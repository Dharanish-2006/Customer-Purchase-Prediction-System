import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings("ignore")

from utils.data_generator import generate_sample_data
from utils.analytics import compute_rfm, kmeans_segmentation, eda_summary
from utils.ml_model import build_features, train_model, predict_customers
from utils.recommender import hybrid_recommendations, content_based_filtering, collaborative_filtering

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #252d3f 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 4px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #60a5fa; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
    .section-header {
        font-size: 1.4rem; font-weight: 600; color: #e2e8f0;
        border-left: 4px solid #3b82f6;
        padding-left: 12px; margin: 24px 0 16px 0;
    }
    .insight-box {
        background: #1a2744; border: 1px solid #2563eb;
        border-radius: 8px; padding: 14px; margin: 8px 0;
        color: #bfdbfe; font-size: 0.9rem;
    }
    .tag {
        display: inline-block; padding: 2px 10px;
        border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white; border: none; border-radius: 8px;
        padding: 8px 24px; font-weight: 600;
    }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "df_tx": None, "df_cust": None, "product_df": None,
        "rfm": None, "rfm_clustered": None, "eda": None,
        "features": None, "model": None, "scaler": None,
        "feature_cols": None, "metrics": None, "predictions": None,
        "data_loaded": False, "model_trained": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Customer Intelligence")
    st.markdown("---")

    st.markdown("### 📂 Data Source")
    data_source = st.radio("", ["Use Sample Data", "Upload CSV"], label_visibility="collapsed")

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload Transactions CSV", type=["csv"])
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                required = ["customer_id", "transaction_id", "product_id", "product_name",
                            "category", "total_amount", "purchase_date", "quantity"]
                missing = [c for c in required if c not in df_up.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    st.session_state.df_tx = df_up
                    st.success(f"✅ Loaded {len(df_up)} rows")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        if st.button("🔄 Load Sample Data"):
            with st.spinner("Generating data..."):
                df_tx, df_cust, prod_df = generate_sample_data()
                st.session_state.df_tx = df_tx
                st.session_state.df_cust = df_cust
                st.session_state.product_df = prod_df
                st.session_state.data_loaded = True
            st.success("✅ Sample data loaded!")

    if st.session_state.df_tx is not None:
        st.markdown("---")
        st.markdown("### ⚙️ Model Settings")
        model_type = st.selectbox("Algorithm", ["XGBoost", "Random Forest", "Gradient Boosting"])

        if st.button("🚀 Train Model"):
            with st.spinner("Training..."):
                try:
                    if st.session_state.df_cust is None:
                        st.session_state.df_cust = pd.DataFrame({
                            "customer_id": st.session_state.df_tx["customer_id"].unique(),
                            "age": 35, "location": "Central", "browsing_sessions": 10
                        })
                    if st.session_state.product_df is None:
                        prods = st.session_state.df_tx[["product_id", "product_name", "category"]].drop_duplicates()
                        prods["price"] = st.session_state.df_tx.groupby("product_id")["unit_price"].mean().values[:len(prods)] if "unit_price" in st.session_state.df_tx.columns else 50.0
                        st.session_state.product_df = prods.rename(columns={"product_id": "id", "product_name": "name"})

                    features = build_features(st.session_state.df_tx, st.session_state.df_cust)
                    model, scaler, feat_cols, metrics = train_model(features, model_type)
                    predictions = predict_customers(model, scaler, feat_cols, features)
                    rfm = compute_rfm(st.session_state.df_tx)
                    rfm_c, _, _ = kmeans_segmentation(rfm)
                    eda = eda_summary(st.session_state.df_tx)

                    st.session_state.update({
                        "features": features, "model": model, "scaler": scaler,
                        "feature_cols": feat_cols, "metrics": metrics,
                        "predictions": predictions, "rfm": rfm,
                        "rfm_clustered": rfm_c, "eda": eda,
                        "model_trained": True,
                    })
                    st.success(f"✅ Trained! AUC: {metrics['roc_auc']:.3f}")
                except Exception as e:
                    st.error(f"Training error: {e}")
                    raise

    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    page = st.selectbox("Go to", [
        "📊 Overview Dashboard",
        "🔍 EDA & Analytics",
        "👥 Customer Segments",
        "🤖 Purchase Prediction",
        "🎯 Recommendations",
        "💡 Business Insights",
    ])


# ── Helper: plot config ───────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1"),
    margin=dict(l=20, r=20, t=40, b=20),
)

COLOR_SEQ = px.colors.qualitative.Bold


def metric_card(label, value, delta=None, color="#60a5fa"):
    delta_html = f'<div style="color:#34d399;font-size:0.8rem">▲ {delta}</div>' if delta else ""
    return f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview Dashboard":
    st.markdown("# 📊 Customer Intelligence Platform")
    st.markdown("*AI-powered analytics · Purchase prediction · Product recommendations*")

    if not st.session_state.data_loaded and st.session_state.df_tx is None:
        st.info("👈 Load sample data or upload a CSV to get started.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size:2.5rem">🔍</div>
                <div class="metric-label" style="font-size:1rem;color:#60a5fa">EDA & Analytics</div>
                <div class="metric-label">Explore trends, revenue, and patterns</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size:2.5rem">🤖</div>
                <div class="metric-label" style="font-size:1rem;color:#a78bfa">ML Predictions</div>
                <div class="metric-label">Predict future customer purchases</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size:2.5rem">🎯</div>
                <div class="metric-label" style="font-size:1rem;color:#34d399">Recommendations</div>
                <div class="metric-label">AI-powered product suggestions</div>
            </div>""", unsafe_allow_html=True)
        st.stop()

    eda = st.session_state.eda
    if eda is None and st.session_state.df_tx is not None:
        eda = eda_summary(st.session_state.df_tx)
        st.session_state.eda = eda

    if eda:
        cols = st.columns(5)
        cards = [
            ("Total Revenue", f"${eda['total_revenue']:,.0f}", "#60a5fa"),
            ("Transactions", f"{eda['total_transactions']:,}", "#a78bfa"),
            ("Customers", f"{eda['unique_customers']:,}", "#34d399"),
            ("Products", f"{eda['unique_products']}", "#f59e0b"),
            ("Avg Order", f"${eda['avg_order_value']:.2f}", "#f43f5e"),
        ]
        for col, (label, val, color) in zip(cols, cards):
            col.markdown(metric_card(label, val, color=color), unsafe_allow_html=True)

        st.markdown('<div class="section-header">📈 Revenue Trend</div>', unsafe_allow_html=True)
        fig = px.area(
            eda["monthly_revenue"], x="Month", y="Revenue",
            color_discrete_sequence=["#3b82f6"],
            title="Monthly Revenue"
        )
        fig.update_traces(fill="tozeroy", fillcolor="rgba(59,130,246,0.15)")
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.pie(
                eda["category_revenue"], values="Revenue", names="Category",
                color_discrete_sequence=COLOR_SEQ, title="Revenue by Category"
            )
            fig2.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            fig3 = px.bar(
                eda["top_products"].head(8), x="Revenue", y="Product",
                orientation="h", color="Revenue", color_continuous_scale="Blues",
                title="Top Products by Revenue"
            )
            fig3.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)

    if st.session_state.model_trained:
        st.markdown('<div class="section-header">🤖 Model Performance Snapshot</div>', unsafe_allow_html=True)
        m = st.session_state.metrics
        cols = st.columns(4)
        perf = [
            ("Accuracy", f"{m['accuracy']*100:.1f}%", "#60a5fa"),
            ("ROC-AUC", f"{m['roc_auc']:.3f}", "#a78bfa"),
            ("F1 Score", f"{m['f1']:.3f}", "#34d399"),
            ("Precision", f"{m['precision']*100:.1f}%", "#f59e0b"),
        ]
        for col, (lbl, val, color) in zip(cols, perf):
            col.markdown(metric_card(lbl, val, color=color), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA & Analytics":
    st.markdown("# 🔍 Exploratory Data Analysis")

    if st.session_state.df_tx is None:
        st.info("👈 Please load data first.")
    else:
        df = st.session_state.df_tx
        eda = st.session_state.eda or eda_summary(df)
        st.session_state.eda = eda

        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Customers", f"{df['customer_id'].nunique():,}")
        c3.metric("Products", f"{df['product_id'].nunique():,}")
        c4.metric("Date Range", f"{df['purchase_date'].min()} → {df['purchase_date'].max()}")

        with st.expander("📋 Raw Data Sample"):
            st.dataframe(df.head(50), use_container_width=True)

        st.markdown('<div class="section-header">Revenue Analysis</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(eda["monthly_revenue"], x="Month", y="Revenue",
                         color="Revenue", color_continuous_scale="Blues", title="Monthly Revenue")
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(eda["category_revenue"], x="Category", y="Revenue",
                          color="Category", color_discrete_sequence=COLOR_SEQ, title="Revenue by Category")
            fig2.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">Purchase Distribution</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig3 = px.histogram(df, x="total_amount", nbins=40, color_discrete_sequence=["#3b82f6"],
                                title="Order Value Distribution")
            fig3.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            tx_per_cust = df.groupby("customer_id").size().reset_index(name="tx_count")
            fig4 = px.histogram(tx_per_cust, x="tx_count", nbins=30, color_discrete_sequence=["#a78bfa"],
                                title="Transactions per Customer")
            fig4.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown('<div class="section-header">Category Deep Dive</div>', unsafe_allow_html=True)
        cat_month = df.copy()
        cat_month["purchase_date"] = pd.to_datetime(cat_month["purchase_date"])
        cat_month["month"] = cat_month["purchase_date"].dt.to_period("M").astype(str)
        cat_trend = cat_month.groupby(["month", "category"])["total_amount"].sum().reset_index()
        fig5 = px.line(cat_trend, x="month", y="total_amount", color="category",
                       color_discrete_sequence=COLOR_SEQ, title="Category Revenue Over Time")
        fig5.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segments":
    st.markdown("# 👥 Customer Segmentation")

    if st.session_state.df_tx is None:
        st.info("👈 Please load data first.")
    else:
        if st.session_state.rfm is None:
            with st.spinner("Computing RFM..."):
                rfm = compute_rfm(st.session_state.df_tx)
                rfm_c, n_k, sil = kmeans_segmentation(rfm)
                st.session_state.rfm = rfm
                st.session_state.rfm_clustered = rfm_c

        rfm = st.session_state.rfm
        rfm_c = st.session_state.rfm_clustered

        tab1, tab2 = st.tabs(["📊 RFM Analysis", "🔵 K-Means Clustering"])

        with tab1:
            st.markdown('<div class="section-header">RFM Score Distribution</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                seg_counts = rfm["RFM_Segment"].value_counts().reset_index()
                seg_counts.columns = ["Segment", "Count"]
                fig = px.pie(seg_counts, values="Count", names="Segment",
                             color_discrete_sequence=COLOR_SEQ, title="Customer Segments (RFM)")
                fig.update_layout(**PLOT_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                seg_rev = rfm.groupby("RFM_Segment")["Monetary"].mean().reset_index()
                fig2 = px.bar(seg_rev, x="RFM_Segment", y="Monetary",
                              color="RFM_Segment", color_discrete_sequence=COLOR_SEQ,
                              title="Avg Monetary Value by Segment")
                fig2.update_layout(**PLOT_LAYOUT)
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown('<div class="section-header">RFM 3D Scatter</div>', unsafe_allow_html=True)
            fig3 = px.scatter_3d(rfm, x="Recency", y="Frequency", z="Monetary",
                                  color="RFM_Segment", color_discrete_sequence=COLOR_SEQ,
                                  opacity=0.7, title="RFM 3D Visualization")
            fig3.update_layout(**PLOT_LAYOUT, height=500)
            st.plotly_chart(fig3, use_container_width=True)

            with st.expander("📋 RFM Table"):
                st.dataframe(rfm.sort_values("RFM_Score", ascending=False), use_container_width=True)

        with tab2:
            st.markdown('<div class="section-header">K-Means Cluster Analysis</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                fig4 = px.scatter(rfm_c, x="Recency", y="Monetary",
                                   color="Cluster_Label", size="Frequency",
                                   color_discrete_sequence=COLOR_SEQ,
                                   title="Clusters: Recency vs Monetary")
                fig4.update_layout(**PLOT_LAYOUT)
                st.plotly_chart(fig4, use_container_width=True)
            with col2:
                fig5 = px.scatter(rfm_c, x="Frequency", y="Monetary",
                                   color="Cluster_Label", size="Recency",
                                   color_discrete_sequence=COLOR_SEQ,
                                   title="Clusters: Frequency vs Monetary")
                fig5.update_layout(**PLOT_LAYOUT)
                st.plotly_chart(fig5, use_container_width=True)

            cluster_stats = rfm_c.groupby("Cluster_Label").agg(
                Count=("customer_id", "count"),
                Avg_Recency=("Recency", "mean"),
                Avg_Frequency=("Frequency", "mean"),
                Avg_Monetary=("Monetary", "mean"),
            ).round(1).reset_index()
            st.markdown('<div class="section-header">Cluster Statistics</div>', unsafe_allow_html=True)
            st.dataframe(cluster_stats, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PURCHASE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Purchase Prediction":
    st.markdown("# 🤖 Purchase Prediction Engine")

    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model using the sidebar first.")
    else:
        m = st.session_state.metrics
        preds = st.session_state.predictions

        st.markdown('<div class="section-header">Model Performance Metrics</div>', unsafe_allow_html=True)
        cols = st.columns(5)
        for col, (lbl, val) in zip(cols, [
            ("Accuracy", f"{m['accuracy']*100:.1f}%"),
            ("ROC-AUC", f"{m['roc_auc']:.4f}"),
            ("F1 Score", f"{m['f1']:.4f}"),
            ("Precision", f"{m['precision']*100:.1f}%"),
            ("Recall", f"{m['recall']*100:.1f}%"),
        ]):
            col.metric(lbl, val)

        tab1, tab2, tab3 = st.tabs(["📊 Model Analytics", "👤 Customer Predictions", "🔬 Feature Importance"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                cm = np.array(m["confusion_matrix"])
                fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                labels=dict(x="Predicted", y="Actual"),
                                x=["No Purchase", "Will Purchase"],
                                y=["No Purchase", "Will Purchase"],
                                title="Confusion Matrix")
                fig.update_layout(**PLOT_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                cv = m["cv_scores"]
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(y=cv, x=[f"Fold {i+1}" for i in range(len(cv))],
                                       marker_color="#3b82f6", name="CV AUC"))
                fig2.add_hline(y=np.mean(cv), line_dash="dash", line_color="#f59e0b",
                               annotation_text=f"Mean: {np.mean(cv):.3f}")
                fig2.update_layout(**PLOT_LAYOUT, title="Cross-Validation AUC Scores")
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.markdown('<div class="section-header">All Customer Predictions</div>', unsafe_allow_html=True)
            risk_filter = st.multiselect("Filter by Risk Level", ["High", "Medium", "Low"],
                                          default=["High", "Medium", "Low"])
            filtered = preds[preds["risk_level"].isin(risk_filter)]

            col1, col2, col3 = st.columns(3)
            col1.metric("High Risk (Likely to Buy)", f"{(preds['risk_level']=='High').sum()}")
            col2.metric("Medium Risk", f"{(preds['risk_level']=='Medium').sum()}")
            col3.metric("Low Risk", f"{(preds['risk_level']=='Low').sum()}")

            fig3 = px.histogram(preds, x="purchase_probability", nbins=30,
                                color_discrete_sequence=["#a78bfa"],
                                title="Purchase Probability Distribution")
            fig3.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)

            st.dataframe(
                filtered[["customer_id", "recency", "frequency", "monetary",
                           "purchase_probability", "risk_level"]].head(100),
                use_container_width=True
            )

            # Download
            csv = filtered.to_csv(index=False)
            st.download_button("📥 Download Predictions CSV", csv,
                               "predictions.csv", "text/csv")

        with tab3:
            fi = m["feature_importance"]
            fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).sort_values("Importance")
            fig4 = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                          color="Importance", color_continuous_scale="Viridis",
                          title="Feature Importance")
            fig4.update_layout(**PLOT_LAYOUT, height=450)
            st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Recommendations":
    st.markdown("# 🎯 AI-Powered Recommendations")

    if st.session_state.df_tx is None or st.session_state.product_df is None:
        st.info("👈 Please load data first.")
    else:
        df_tx = st.session_state.df_tx
        product_df = st.session_state.product_df
        customers = sorted(df_tx["customer_id"].unique())

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            selected_cust = st.selectbox("Select Customer", customers)
        with col2:
            method = st.selectbox("Method", ["Hybrid", "Content-Based", "Collaborative"])
        with col3:
            top_n = st.slider("Recommendations", 3, 10, 5)

        if st.button("🎯 Generate Recommendations"):
            with st.spinner("Generating..."):
                if method == "Hybrid":
                    recs = hybrid_recommendations(df_tx, product_df, selected_cust, top_n)
                elif method == "Content-Based":
                    recs = content_based_filtering(df_tx, product_df, selected_cust, top_n)
                    if not recs.empty:
                        recs = recs.rename(columns={"product_name": "product_name",
                                                     "similarity_score": "score"})
                else:
                    recs = collaborative_filtering(df_tx, selected_cust, top_n)

            st.markdown('<div class="section-header">Recommended Products</div>', unsafe_allow_html=True)

            if recs is None or recs.empty:
                st.warning("Not enough data to generate recommendations for this customer.")
            else:
                # Show customer purchase history
                cust_history = df_tx[df_tx["customer_id"] == selected_cust]
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**Purchase History**")
                    hist_summary = cust_history.groupby("category")["total_amount"].sum().reset_index()
                    fig_hist = px.bar(hist_summary, x="category", y="total_amount",
                                      color="category", color_discrete_sequence=COLOR_SEQ,
                                      title="Spending by Category")
                    fig_hist.update_layout(**PLOT_LAYOUT, height=300)
                    st.plotly_chart(fig_hist, use_container_width=True)
                with col2:
                    st.markdown("**Recommendations**")
                    for _, row in recs.iterrows():
                        score_col = "score" if "score" in recs.columns else "similarity_score"
                        score = row.get(score_col, row.get("similarity_score", 0))
                        name = row.get("product_name", row.get("name", row.get("product_id", "")))
                        cat = row.get("category", "")
                        price = row.get("price", "")
                        price_str = f" · ${price:.2f}" if price else ""
                        st.markdown(f"""
                        <div class="insight-box">
                            <strong>{name}</strong>{price_str}<br>
                            <span style="color:#94a3b8">{cat}</span>
                            <span style="float:right;color:#34d399">Score: {score:.1f}</span>
                        </div>""", unsafe_allow_html=True)

                # Bar chart of recommendations
                if "product_name" in recs.columns or "name" in recs.columns:
                    name_col = "product_name" if "product_name" in recs.columns else "name"
                    score_col = "score" if "score" in recs.columns else "similarity_score"
                    if score_col in recs.columns:
                        fig_rec = px.bar(recs.head(top_n), x=score_col, y=name_col,
                                         orientation="h", color=score_col,
                                         color_continuous_scale="Greens",
                                         title="Recommendation Scores")
                        fig_rec.update_layout(**PLOT_LAYOUT)
                        st.plotly_chart(fig_rec, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Business Insights":
    st.markdown("# 💡 Business Insights & Strategy")

    if st.session_state.df_tx is None:
        st.info("👈 Please load data first.")
    else:
        df = st.session_state.df_tx
        eda = st.session_state.eda or eda_summary(df)

        st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        kpis = [
            ("💰 Total Revenue", f"${eda['total_revenue']:,.0f}"),
            ("🛒 Avg Order Value", f"${eda['avg_order_value']:.2f}"),
            ("👥 Active Customers", f"{eda['unique_customers']:,}"),
            ("📦 Total Orders", f"{eda['total_transactions']:,}"),
        ]
        for col, (lbl, val) in zip(cols, kpis):
            col.metric(lbl, val)

        st.markdown('<div class="section-header">Strategic Insights</div>', unsafe_allow_html=True)

        insights = []
        top_cat = eda["category_revenue"].iloc[0]
        insights.append(f"🏆 **Top Revenue Category**: {top_cat['Category']} (${top_cat['Revenue']:,.0f}) drives the most revenue. Consider expanding product lines here.")

        top_prod = eda["top_products"].iloc[0]
        insights.append(f"⭐ **Best-Selling Product**: {top_prod['Product']} is your top performer — ensure it stays well-stocked and promoted.")

        monthly_rev = eda["monthly_revenue"]
        if len(monthly_rev) >= 2:
            growth = (monthly_rev["Revenue"].iloc[-1] / monthly_rev["Revenue"].iloc[-2] - 1) * 100
            trend = "📈 Growing" if growth > 0 else "📉 Declining"
            insights.append(f"{trend} **Revenue Trend**: {abs(growth):.1f}% change month-over-month.")

        if st.session_state.model_trained:
            preds = st.session_state.predictions
            high_prob = (preds["purchase_probability"] >= 66).sum()
            insights.append(f"🎯 **High-Intent Customers**: {high_prob} customers show >66% purchase probability. Target them with personalized campaigns.")

        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

        # Revenue heatmap
        st.markdown('<div class="section-header">Sales Patterns</div>', unsafe_allow_html=True)
        df_heat = df.copy()
        df_heat["purchase_date"] = pd.to_datetime(df_heat["purchase_date"])
        df_heat["dow"] = df_heat["purchase_date"].dt.day_name()
        df_heat["month"] = df_heat["purchase_date"].dt.month_name()

        heatmap_data = df_heat.groupby(["dow", "month"])["total_amount"].sum().unstack(fill_value=0)
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heatmap_data = heatmap_data.reindex([d for d in days_order if d in heatmap_data.index])

        fig_heat = px.imshow(heatmap_data, color_continuous_scale="Blues",
                              title="Revenue Heatmap: Day of Week × Month")
        fig_heat.update_layout(**PLOT_LAYOUT, height=350)
        st.plotly_chart(fig_heat, use_container_width=True)

        # Customer lifetime value distribution
        clv = df.groupby("customer_id")["total_amount"].sum().reset_index()
        clv.columns = ["customer_id", "CLV"]
        fig_clv = px.histogram(clv, x="CLV", nbins=40, color_discrete_sequence=["#f59e0b"],
                                title="Customer Lifetime Value Distribution")
        fig_clv.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig_clv, use_container_width=True)

        st.markdown('<div class="section-header">Actionable Recommendations</div>', unsafe_allow_html=True)
        actions = [
            ("🔴 Re-engagement Campaign", "Target 'Lost Customers' (RFM Score <5) with win-back offers and discounts."),
            ("🟡 Loyalty Program", "Reward 'Champions' and 'Loyal Customers' with exclusive perks to maintain retention."),
            ("🟢 Upsell Opportunity", "Suggest premium products to 'Mid Value' cluster customers who show potential."),
            ("🔵 Email Drip Campaigns", "Automate follow-up emails for customers with medium purchase probability (33–66%)."),
        ]
        col1, col2 = st.columns(2)
        for i, (title, desc) in enumerate(actions):
            with (col1 if i % 2 == 0 else col2):
                st.markdown(f"""
                <div class="insight-box">
                    <strong>{title}</strong><br>
                    <span style="color:#94a3b8">{desc}</span>
                </div>""", unsafe_allow_html=True)
