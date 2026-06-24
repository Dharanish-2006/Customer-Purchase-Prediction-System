# 🛒 Customer Purchase Prediction & Intelligent Recommendation System

An end-to-end AI/ML platform for e-commerce customer analytics, purchase prediction, and personalized product recommendations — built with Python, Streamlit, Scikit-Learn, and Plotly.

---

## 🚀 Live Demo

Deploy on **Streamlit Community Cloud** (free) — see deployment instructions below.

---

## ✨ Features

| Module | Description |
|---|---|
| 📊 **EDA Dashboard** | Interactive charts: revenue trends, category analysis, order distributions |
| 👥 **RFM Segmentation** | Recency/Frequency/Monetary scoring with 5 behavioral segments |
| 🔵 **K-Means Clustering** | Auto-optimal cluster count via silhouette analysis |
| 🤖 **Purchase Prediction** | XGBoost / Random Forest / Gradient Boosting classifier |
| 🎯 **Recommendations** | Hybrid (content-based + collaborative filtering) engine |
| 💡 **Business Insights** | KPIs, heatmaps, CLV distribution, actionable strategies |

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive web dashboard
- **Pandas / NumPy** — data processing
- **Scikit-Learn** — ML models, clustering, preprocessing
- **XGBoost** — gradient boosted trees (optional)
- **Plotly** — interactive charts
- **SQLite** — lightweight data persistence (optional)

---

## 📦 Installation (Local)

```bash
git clone https://github.com/YOUR_USERNAME/customer-prediction.git
cd customer-prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Community Cloud (Free)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/customer-prediction.git
   git push -u origin main
   ```

2. **Go to** [share.streamlit.io](https://share.streamlit.io)

3. **Sign in** with your GitHub account

4. Click **"New app"** → select your repo → set **Main file: `app.py`**

5. Click **"Deploy"** — your app will be live at:
   `https://YOUR_USERNAME-customer-prediction-app-XXXX.streamlit.app`

---

## 📂 Project Structure

```
customer-prediction/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Python dependencies
├── README.md
├── .streamlit/
│   └── config.toml           # Dark theme config
└── utils/
    ├── __init__.py
    ├── data_generator.py     # Synthetic data generator
    ├── analytics.py          # RFM, K-Means, EDA
    ├── ml_model.py           # Feature engineering + ML
    └── recommender.py        # Recommendation engine
```

---

## 📊 CSV Upload Format

If uploading your own data, the CSV must include these columns:

| Column | Type | Example |
|---|---|---|
| `customer_id` | string | C0001 |
| `transaction_id` | string | T000001 |
| `product_id` | string | P001 |
| `product_name` | string | Wireless Headphones |
| `category` | string | Electronics |
| `quantity` | int | 2 |
| `total_amount` | float | 159.98 |
| `purchase_date` | date | 2024-03-15 |
| `browsing_time_mins` | int | 12 |

---

## 🔬 ML Model Details

**Features used for prediction:**
- `recency` — days since last purchase
- `frequency` — total number of orders
- `monetary` — total spend
- `avg_order_value` — mean order amount
- `num_categories` — product diversity
- `avg_browsing` — average browsing time
- `age`, `browsing_sessions`, `location_enc`

**Target:** Binary classification — will customer purchase within next 60 days?

---

## 📄 License

MIT License — free to use, modify, and deploy.
