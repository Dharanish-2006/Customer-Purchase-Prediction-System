# 📘 Customer Purchase Prediction & Intelligent Recommendation System
### Complete Project Documentation — For Beginners & Stakeholders

---

## 🧭 Table of Contents

1. [What Is This Project?](#what-is-this-project)
2. [What Problem Does It Solve?](#what-problem-does-it-solve)
3. [How the App Works (Step by Step)](#how-the-app-works)
4. [Features Explained in Detail](#features-explained-in-detail)
5. [Tech Stack — Every Tool Explained](#tech-stack)
6. [Project File Structure](#project-file-structure)
7. [Data — What It Looks Like](#data)
8. [Machine Learning — How It Works](#machine-learning)
9. [Recommendation Engine — How It Works](#recommendation-engine)
10. [Who Is This For?](#who-is-this-for)
11. [How to Run It](#how-to-run-it)

---

## 🤔 What Is This Project?

This is an **AI and Machine Learning web application** built for e-commerce businesses. It takes customer transaction data (who bought what, when, and for how much) and does three powerful things automatically:

1. **Understands your customers** — groups them by behavior using data science
2. **Predicts future purchases** — tells you which customers are likely to buy again soon
3. **Recommends products** — suggests the right products to the right customer

Think of it as your own data science team packed into a single web dashboard that anyone can use without writing a single line of code.

---

## 💡 What Problem Does It Solve?

Most e-commerce businesses face these common challenges:

| Problem | How This App Solves It |
|---|---|
| "I don't know which customers will come back" | ML model predicts purchase probability per customer |
| "I don't know what to recommend to each customer" | AI recommendation engine gives personalized suggestions |
| "I have sales data but no insights from it" | EDA dashboard visualizes all patterns automatically |
| "I don't know who my best customers are" | RFM + K-Means segments customers into clear groups |
| "I need a data scientist to analyze my data" | The entire pipeline runs in-browser — no coding needed |

---

## 🖥️ How the App Works

Here is the complete user journey from start to finish:

```
Step 1: Load Data
      ↓
   Upload your own CSV file   OR   Click "Load Sample Data"
      ↓
Step 2: Click "Train Model" in the sidebar
      ↓
   The app processes your data, runs analytics, and trains the ML model
      ↓
Step 3: Explore the 6 dashboard pages
      ↓
   Overview → EDA → Segments → Predictions → Recommendations → Insights
```

Everything happens in real time in the browser. No installation needed if deployed on Streamlit Cloud.

---

## 🗂️ Features Explained in Detail

### 1. 📊 Overview Dashboard

The first page you see. It gives you a quick snapshot of the entire business.

**What it shows:**
- **Total Revenue** — sum of all sales across all time
- **Total Transactions** — number of orders placed
- **Unique Customers** — how many individual customers exist
- **Unique Products** — how many different products were sold
- **Average Order Value** — what a typical customer spends per order
- **Monthly Revenue Area Chart** — a visual timeline showing how revenue grows or drops month by month
- **Revenue by Category (Pie Chart)** — which product category earns the most
- **Top Products Bar Chart** — your 8 best-selling products by revenue
- **Model Performance Snapshot** — if the model is trained, shows Accuracy, ROC-AUC, F1 Score, Precision

**Why it matters:** You get a complete business health check in seconds without opening Excel or writing queries.

---

### 2. 🔍 EDA & Analytics (Exploratory Data Analysis)

EDA means "exploring the data to understand it." This page digs deeper into patterns.

**What it shows:**
- **Dataset Overview** — total rows, customers, products, and the date range of your data
- **Raw Data Sample** — expandable table showing the first 50 rows of actual data
- **Monthly Revenue Bar Chart** — same as overview but with more detail and color
- **Revenue by Category Bar Chart** — side-by-side comparison of categories
- **Order Value Distribution (Histogram)** — how spread out order amounts are (do most people spend $20 or $200?)
- **Transactions per Customer (Histogram)** — how often customers buy (1 time? 10 times?)
- **Category Revenue Over Time (Line Chart)** — how each category's performance changes month by month

**Why it matters:** Before making any business decision, you need to understand your data. EDA reveals hidden patterns — like which months are slow, which categories are growing, and what a "typical" order looks like.

---

### 3. 👥 Customer Segmentation

This page uses two different scientific methods to group customers into meaningful categories.

#### Tab 1: RFM Analysis

**RFM** stands for **Recency, Frequency, Monetary** — three of the most powerful metrics in customer analytics.

| Metric | What It Measures | Example |
|---|---|---|
| **Recency (R)** | How many days since the customer last bought something | Bought 5 days ago = great; 300 days ago = concerning |
| **Frequency (F)** | How many times the customer has bought in total | Bought 15 times = loyal; bought 1 time = risky |
| **Monetary (M)** | How much the customer has spent in total | Spent $2,000 = high value; spent $20 = low value |

Each customer gets a score from 1–5 in each category (5 = best). These scores are combined into an **RFM Total Score** (3–15), and customers are placed into one of 5 segments:

| Segment | Score Range | Who They Are |
|---|---|---|
| **Champions** | 13–15 | Your best customers. Buy often, recently, and spend a lot |
| **Loyal Customers** | 10–12 | Reliable buyers, high total spend |
| **Potential Loyalists** | 7–9 | Recent buyers with mid-range frequency |
| **At Risk** | 5–6 | Used to buy but haven't recently — losing them |
| **Lost Customers** | 3–4 | Haven't bought in a very long time |

**Visuals on this page:**
- Pie chart of how many customers fall in each segment
- Bar chart of average spending by segment
- 3D Scatter Plot — a three-dimensional view of all customers plotted by R, F, and M values, colored by segment

#### Tab 2: K-Means Clustering

K-Means is a **machine learning algorithm** that automatically groups customers based on mathematical similarity — without being told what the groups should be.

**How it works:**
1. The algorithm tries different numbers of groups (2 through 6)
2. For each option, it calculates a **Silhouette Score** — a measure of how well-separated the groups are
3. It picks the number of groups with the highest score
4. Customers are labeled: High Value, Mid Value, Low Value, Occasional

**Visuals:**
- Scatter plot: Recency vs Monetary (colored by cluster)
- Scatter plot: Frequency vs Monetary (sized by recency)
- Cluster Statistics Table — count, averages for each cluster

**Why this matters:** Instead of guessing which customers are valuable, the algorithm finds the natural groupings that exist in your actual data.

---

### 4. 🤖 Purchase Prediction

This is the core machine learning feature. The model predicts whether each customer is **likely to make a purchase in the next 60 days**.

#### Three Model Tabs:

**Tab 1: Model Analytics**
- **Confusion Matrix** — a grid showing how many predictions were correct vs wrong
  - True Positive: predicted "will buy" and they did
  - True Negative: predicted "won't buy" and they didn't
  - False Positive: predicted "will buy" but they didn't (false alarm)
  - False Negative: predicted "won't buy" but they did (missed opportunity)
- **Cross-Validation AUC Scores** — the model was tested 5 different ways (K-Fold CV) to make sure it's not just memorizing data. Each fold's AUC score is shown as a bar, with the mean as a dashed line.

**Tab 2: Customer Predictions**
- Filter customers by Risk Level: High / Medium / Low
- Summary counts of how many customers fall in each risk level
- **Purchase Probability Distribution** — histogram showing spread of probabilities across all customers
- Full sortable table with: Customer ID, Recency, Frequency, Monetary, Purchase Probability %, Risk Level
- **Download as CSV button** — export the predictions for use in email marketing, CRM tools, etc.

**Tab 3: Feature Importance**
A horizontal bar chart showing which input variables the model found most useful for making predictions. Higher bar = more influential feature.

#### What gets measured for each customer (the 10 features):

| Feature | What It Is |
|---|---|
| `recency` | Days since last purchase |
| `frequency` | Number of total orders |
| `monetary` | Total money spent |
| `avg_order_value` | Average spend per order |
| `num_categories` | How many different product categories purchased |
| `avg_browsing` | Average minutes spent browsing per session |
| `last_qty` | Quantity in their most recent order |
| `age` | Customer age |
| `browsing_sessions` | Total number of browsing sessions |
| `location_enc` | Geographic region (encoded as a number) |

#### Model Performance Metrics (shown at the top):

| Metric | What It Means | Good Value |
|---|---|---|
| **Accuracy** | % of predictions that were correct overall | > 80% |
| **ROC-AUC** | How well the model separates buyers from non-buyers (0.5 = random, 1.0 = perfect) | > 0.85 |
| **F1 Score** | Balance between precision and recall | > 0.75 |
| **Precision** | Of all "will buy" predictions, how many actually bought | > 75% |
| **Recall** | Of all actual buyers, how many did we catch | > 70% |

---

### 5. 🎯 AI-Powered Recommendations

This page lets you select any customer and generate a personalized list of products they are most likely to enjoy and buy.

**Three recommendation methods available:**

#### Method 1: Content-Based Filtering
"Recommend products similar to what this customer already likes."

How it works:
1. Look at all products the customer has bought before
2. Build a "taste profile" from the categories and price ranges they prefer
3. Compare every product in the catalog to this profile using **cosine similarity** (a math formula for measuring how similar two things are)
4. Recommend the products that most closely match the profile, excluding ones already purchased

#### Method 2: Collaborative Filtering
"Find customers similar to this one and recommend what they liked."

How it works:
1. Build a matrix of every customer vs every product (how much each person spent on each item)
2. Calculate similarity between all pairs of customers using **cosine similarity**
3. Find the 10 most similar customers
4. Recommend products those similar customers bought, that the target customer hasn't tried yet
5. Score each recommendation by how many similar users liked it and how similar they are

#### Method 3: Hybrid (Default — Best Results)
Combines both methods with weighted scoring:
- Content-based score × 0.6 (60% weight)
- Collaborative score × 0.4 (40% weight)
- Merge and rank — best of both worlds

**What the page shows:**
- Customer's spending by category (bar chart showing their purchase history)
- Recommendation cards showing: product name, category, price, and score
- Recommendation scores bar chart (horizontal) for visual comparison

---

### 6. 💡 Business Insights & Strategy

The final page translates all the data into plain-English business guidance.

**What it shows:**

**KPI Cards (4 metrics):**
- Total Revenue, Average Order Value, Active Customers, Total Orders

**Auto-Generated Strategic Insights:**
The app reads your data and writes insights automatically, such as:
- Which category drives the most revenue and should be expanded
- Which product is the best-seller and should stay in stock
- Month-over-month revenue growth or decline percentage
- How many high-intent customers (>66% probability) exist right now

**Revenue Heatmap:**
A color grid showing revenue by Day of Week × Month. Darker blue = more revenue. Instantly reveals your busiest shopping days and months.

**Customer Lifetime Value (CLV) Distribution:**
A histogram showing how total spend is distributed across all customers — are most customers low-spend or high-spend?

**Actionable Campaign Recommendations (4 strategies):**
1. Re-engagement Campaign — target Lost Customers with discounts
2. Loyalty Program — reward Champions and Loyal Customers
3. Upsell Opportunity — push premium products to Mid Value cluster
4. Email Drip Campaigns — nurture Medium probability customers automatically

---

## 🛠️ Tech Stack

Every tool used in this project, explained simply:

### 🐍 Python
**What it is:** The programming language everything is written in.
**Why it's used:** Python is the world's most popular language for data science and AI. It has libraries for everything — data processing, machine learning, web apps, and visualization.
**Version:** Python 3.10+

---

### 🌐 Streamlit
**What it is:** A Python library that turns Python scripts into interactive web applications.
**Why it's used:** Without Streamlit, you'd need to build a separate frontend in HTML/CSS/JavaScript. Streamlit lets a data scientist build a full web dashboard using only Python — buttons, sliders, file uploaders, charts, all in one file.
**Key features used:**
- `st.sidebar` — the left panel with controls
- `st.selectbox`, `st.slider`, `st.button` — interactive widgets
- `st.file_uploader` — CSV upload functionality
- `st.session_state` — stores data between page interactions
- `st.plotly_chart` — renders interactive Plotly charts
- `st.stop()` — halts page rendering early when needed
- `st.download_button` — lets users download predictions as CSV

---

### 🐼 Pandas
**What it is:** The standard Python library for working with tables of data (like Excel, but in code).
**Why it's used:** All transaction data, customer data, RFM tables, and feature tables are stored and manipulated as Pandas DataFrames. Used for grouping, filtering, aggregating, merging, and reshaping data.
**Key operations used:**
- `groupby()` — group transactions by customer to compute totals
- `pivot_table()` — reshape data for the recommendation matrix
- `merge()` — join customer demographics with transaction features
- `pd.to_datetime()` — convert date strings into date objects
- `pd.cut()` / `pd.qcut()` — bin continuous values into categories

---

### 🔢 NumPy
**What it is:** Python's numerical computing library — fast math on arrays and matrices.
**Why it's used:** Used under the hood for all mathematical computations — probability calculations, array operations, random number generation for synthetic data, and matrix math in the recommendation engine.
**Key uses:**
- `np.random` — generating realistic synthetic customer data
- `np.round()` — rounding probabilities to one decimal place
- Array operations in cosine similarity calculations

---

### 🤖 Scikit-Learn
**What it is:** The most widely used machine learning library in Python.
**Why it's used:** Provides ready-to-use implementations of every ML algorithm and tool used in this project. No need to code algorithms from scratch.

**Specific components used:**

| Component | What It Does |
|---|---|
| `RandomForestClassifier` | ML model — ensemble of decision trees that vote on predictions |
| `GradientBoostingClassifier` | ML model — trees that learn from each other's mistakes sequentially |
| `KMeans` | Clustering algorithm — groups customers by mathematical similarity |
| `StandardScaler` | Scales all features to the same range so no feature dominates the model |
| `LabelEncoder` | Converts text labels (like location names) into numbers the model can use |
| `train_test_split` | Splits data into training set (80%) and test set (20%) for honest evaluation |
| `cross_val_score` | Tests the model 5 times on different data splits to verify consistency |
| `cosine_similarity` | Measures similarity between customer profiles and product features |
| `silhouette_score` | Measures how well-separated K-Means clusters are (used to pick optimal k) |
| `confusion_matrix` | Matrix showing correct vs incorrect predictions |
| `roc_auc_score`, `accuracy_score`, `f1_score`, `precision_score`, `recall_score` | All model evaluation metrics |
| `MinMaxScaler` | Scales product prices to 0–1 range for content-based filtering |

---

### ⚡ XGBoost
**What it is:** "Extreme Gradient Boosting" — one of the most powerful and widely used ML algorithms in industry.
**Why it's used:** XGBoost wins most machine learning competitions. It's faster and often more accurate than standard Gradient Boosting. Used as the default model for purchase prediction.
**Key settings:**
- `n_estimators=200` — builds 200 trees
- `max_depth=5` — each tree can make up to 5 decisions
- `learning_rate=0.1` — how much each tree corrects the previous one
- Falls back to Gradient Boosting if XGBoost is not installed

---

### 📊 Plotly
**What it is:** A Python library for creating beautiful, interactive charts.
**Why it's used:** Unlike static charts (like Matplotlib), Plotly charts are interactive — users can hover for exact values, zoom in, click to filter, and download the chart as an image. All charts in the dashboard use Plotly.

**Chart types used:**

| Chart Type | Where It's Used |
|---|---|
| `px.area` | Monthly revenue trend (filled line) |
| `px.pie` | Revenue by category, RFM segment distribution |
| `px.bar` | Top products, category comparison, feature importance |
| `px.histogram` | Order value distribution, transactions per customer, probability distribution, CLV |
| `px.scatter` | K-Means cluster visualization (2D) |
| `px.scatter_3d` | RFM 3D visualization |
| `px.line` | Category revenue over time |
| `px.imshow` | Confusion matrix, revenue heatmap |
| `go.Figure` + `go.Bar` | Cross-validation AUC scores with mean line |

---

### 🗄️ SQLite (Architecture-ready)
**What it is:** A lightweight, file-based database built into Python.
**Why it's included:** The project architecture supports SQLite for persisting model results, customer predictions, and session data. In the current version, data is held in Streamlit session state (in memory). SQLite is the intended persistence layer for production use — storing trained model outputs and user uploads between sessions without needing a full database server.

---

### 🎨 Custom CSS (HTML/CSS inside Streamlit)
**What it is:** Standard web styling language, injected into Streamlit via `st.markdown(unsafe_allow_html=True)`.
**Why it's used:** Streamlit's default styling is plain. Custom CSS gives the dashboard a professional dark-themed look with:
- Gradient metric cards (`.metric-card`)
- Blue-left-border section headers (`.section-header`)
- Dark blue insight boxes (`.insight-box`)
- Gradient purple-blue buttons
- Hover effects on buttons

---

## 📁 Project File Structure

```
customer_prediction/
│
├── app.py                        # Main application — all 6 dashboard pages
│
├── requirements.txt              # List of Python packages to install
│
├── README.md                     # Quick-start guide and deployment instructions
│
├── DETAILS.md                    # This file — full project documentation
│
├── .streamlit/
│   └── config.toml               # Dark theme + server configuration
│
├── utils/                        # All backend logic (the "brain" of the app)
│   ├── __init__.py               # Makes utils a Python package
│   │
│   ├── data_generator.py         # Generates 500 synthetic customers + 3,000 transactions
│   │                               with realistic behavior patterns and product catalog
│   │
│   ├── analytics.py              # Three functions:
│   │                               - compute_rfm()     → RFM scores + segment labels
│   │                               - kmeans_segmentation() → K-Means with auto optimal k
│   │                               - eda_summary()     → Revenue, category, product stats
│   │
│   ├── ml_model.py               # Three functions:
│   │                               - build_features()  → Feature engineering from raw data
│   │                               - train_model()     → Train + evaluate classifier
│   │                               - predict_customers() → Score all customers
│   │
│   └── recommender.py            # Three functions:
│                                   - content_based_filtering() → cosine similarity on products
│                                   - collaborative_filtering() → user-user similarity
│                                   - hybrid_recommendations()  → weighted combination
│
└── data/
    └── sample_transactions.csv   # Pre-generated sample dataset (1,886 rows)
```

---

## 📋 Data — What It Looks Like

### Transaction Data (the main dataset)

Each row = one purchase. Required columns:

| Column | Type | Example | Description |
|---|---|---|---|
| `transaction_id` | string | T000001 | Unique ID for each order |
| `customer_id` | string | C0001 | Which customer placed the order |
| `product_id` | string | P001 | Which product was bought |
| `product_name` | string | Wireless Headphones | Product display name |
| `category` | string | Electronics | Product category |
| `quantity` | integer | 2 | How many units were bought |
| `unit_price` | float | 79.99 | Price per item |
| `total_amount` | float | 159.98 | Total paid (quantity × price ± variation) |
| `purchase_date` | date | 2024-03-15 | When the purchase happened |
| `browsing_time_mins` | integer | 12 | Minutes spent browsing before buying |

### Sample Data Generation

The app can generate 500 realistic synthetic customers with 3,000 transactions. Customers are drawn from 4 behavioral profiles:

| Profile | % of Customers | Purchase Frequency | Avg Spend | Recency |
|---|---|---|---|---|
| High Value | 15% | 10–25 orders | $150–$500 | 1–30 days ago |
| Regular | 35% | 4–12 orders | $50–$150 | 15–90 days ago |
| Occasional | 30% | 1–5 orders | $20–$80 | 60–180 days ago |
| Churned | 20% | 1–3 orders | $10–$50 | 150–365 days ago |

Products span 6 categories: Electronics, Sports, Kitchen, Books, Beauty, Home — 15 products total.

---

## 🧠 Machine Learning — How It Works

### The Problem Being Solved

**Binary Classification:** For each customer, predict whether they will make a purchase within the next 60 days.
- Output: 1 = "Will Purchase", 0 = "Will Not Purchase"
- Plus: a probability score from 0% to 100%

### The Pipeline

```
Raw Transactions
      ↓
Feature Engineering (build_features)
      ↓
10 numerical features per customer
      ↓
StandardScaler (normalize all features to same scale)
      ↓
80% Training Data → Train Model → Learn patterns
20% Test Data → Evaluate → Measure accuracy
      ↓
5-Fold Cross Validation → Verify model is reliable
      ↓
Predict probability for ALL customers
      ↓
Classify into: High (>66%) / Medium (33–66%) / Low (<33%) risk
```

### Why StandardScaler?

Without scaling, a feature like `monetary` (values in the hundreds) would mathematically overpower a feature like `num_categories` (values of 1–6), even if both are equally important. Scaling puts everything on the same 0–1 playing field.

### Model Comparison

| Model | Speed | Accuracy | Best For |
|---|---|---|---|
| XGBoost | Fast | Highest | Large datasets, competitions |
| Random Forest | Medium | High | Balanced, interpretable |
| Gradient Boosting | Slower | High | When XGBoost not available |

---

## 🎯 Recommendation Engine — How It Works

### Content-Based Filtering (Solo Method)

```
Customer C001 bought: Wireless Headphones (Electronics), Gaming Mouse (Electronics), Coffee Maker (Kitchen)
      ↓
Build profile: [Electronics=high, Kitchen=medium, price_range=mid]
      ↓
Compare to every product using cosine similarity
      ↓
Recommend: Smart Watch (Electronics, $199) → high similarity
           Air Fryer (Kitchen, $69) → medium similarity
```

Cosine similarity measures the angle between two vectors — if they point in the same direction (same categories, same price range), the angle is small and similarity is high (close to 1.0).

### Collaborative Filtering (Solo Method)

```
Customer C001 → Find 10 most similar customers based on purchase history
      ↓
Customer C042 is very similar and bought: Protein Powder, Yoga Mat
Customer C078 is similar and bought: Running Shoes, Protein Powder
      ↓
Protein Powder appears in both → weighted high score
Recommend: Protein Powder (Sports, $44.99)
```

### Hybrid Method (Default)

```
Content-Based Score × 0.6
+
Collaborative Score × 0.4 (normalized to 100)
= Combined Score → Rank → Return Top N
```

The hybrid approach is more robust: content-based works even with few users, collaborative works even with few product features. Together they cover each other's weaknesses.

---

## 👥 Who Is This For?

| Role | How They Use It |
|---|---|
| **E-commerce Business Owner** | Load real sales data → get customer segments + purchase predictions → plan marketing |
| **Marketing Team** | Download high-probability customer list → run targeted campaigns |
| **Data Science Student** | Study the code to learn RFM, K-Means, classification, and recommendation systems end-to-end |
| **Product Manager** | Use recommendation engine to plan which products to promote to which segments |
| **Business Analyst** | Use EDA and insights pages to build reports and presentations |

---

## 🚀 How to Run It

### Option 1: Local (on your computer)

```bash
# 1. Download the project zip and unzip it
# 2. Open terminal in the project folder
# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py

# 5. Open browser at: http://localhost:8501
```

### Option 2: Streamlit Community Cloud (Free, Public URL)

```bash
# 1. Push the project to GitHub
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USERNAME/customer-prediction.git
git push -u origin main

# 2. Go to https://share.streamlit.io
# 3. Click "New app" → select your repo → Main file: app.py → Deploy
# 4. Get a public URL like: https://username-customer-prediction-xxxx.streamlit.app
```

### First Time Using the App

```
1. Open the app in browser
2. In the LEFT SIDEBAR → click "Load Sample Data"
3. Wait for "✅ Sample data loaded!" message
4. In the sidebar → click "🚀 Train Model"
5. Wait for "✅ Trained! AUC: X.XXX" message
6. Now explore all 6 pages from the navigation dropdown
```

---

## 📦 Requirements (requirements.txt)

```
streamlit==1.35.0        # Web dashboard framework
pandas==2.1.4            # Data manipulation
numpy==1.26.4            # Numerical computing
scikit-learn==1.4.2      # Machine learning algorithms
plotly==5.22.0           # Interactive charts
xgboost==2.0.3           # XGBoost ML algorithm
scipy==1.13.0            # Scientific computing (used by sklearn)
```

---

## 📐 Key Algorithms Summary

| Algorithm | Type | Where Used | What It Produces |
|---|---|---|---|
| RFM Scoring | Statistical | Customer Segments | 5 behavioral segments |
| K-Means | Unsupervised ML | Customer Segments | 2–6 auto-detected clusters |
| Silhouette Analysis | Evaluation | K-Means tuning | Optimal number of clusters |
| Random Forest | Supervised ML | Purchase Prediction | Probability + classification |
| XGBoost | Supervised ML | Purchase Prediction | Probability + classification |
| Gradient Boosting | Supervised ML | Purchase Prediction | Probability + classification |
| StandardScaler | Preprocessing | Feature normalization | Scaled feature matrix |
| Cosine Similarity | Math/Geometry | Recommendations | Similarity scores 0–1 |
| Content-Based Filtering | Recommendation | Product suggestions | Top-N similar products |
| Collaborative Filtering | Recommendation | Product suggestions | Top-N peer-based products |
| Hybrid Filtering | Recommendation | Product suggestions | Weighted combined ranking |

---

*Built with Python · Streamlit · Scikit-Learn · XGBoost · Plotly · Pandas*
