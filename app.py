import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(
    page_title="Customer Behavior Mining", layout="wide"
)
st.title("🛒 Customer Behavior Mining & Segmentation")

# --- 1. USER INPUTS / PROMPTS IN SIDEBAR ---
st.sidebar.header("Customer & Order Details")

# Order Date & Time
order_date = st.sidebar.date_input("Date of Order")
order_time = st.sidebar.time_input("Time of Order")
order_datetime = pd.to_datetime(f"{order_date} {order_time}")

# Financial Metrics & Salary Prompt
monthly_income = st.sidebar.number_input(
    "Monthly Income (₹)", min_value=1000.0, value=50000.0, step=1000.0
)
total_spend = st.sidebar.number_input(
    "Total Order / Spend Amount (₹)",
    min_value=0.0,
    value=15000.0,
    step=500.0,
)

# Derived Feature: Salary Exhausted in %
salary_exhausted_pct = (total_spend / monthly_income) * 100
st.sidebar.metric(
    label="Salary Exhausted", value=f"{salary_exhausted_pct:.2f}%"
)

# Additional behavioral features
purchase_frequency = st.sidebar.slider(
    "Purchases per Month", min_value=1, max_value=30, value=5
)

# --- 2. SAMPLE DATA ENGINE ---
# Synthesizing or loading dataset including the new derived features
np.random.seed(42)
n_samples = 300

sample_incomes = np.random.uniform(20000, 120000, n_samples)
sample_spends = sample_incomes * np.random.uniform(0.05, 0.70, n_samples)
sample_exhausted = (sample_spends / sample_incomes) * 100
sample_freq = np.random.randint(1, 25, n_samples)

df = pd.DataFrame(
    {
        "Monthly_Income": sample_incomes,
        "Total_Spend": sample_spends,
        "Salary_Exhausted_Pct": sample_exhausted,
        "Purchase_Frequency": sample_freq,
    }
)

# --- 3. MODEL TRAINING & VALIDATION METRICS ---
features = [
    "Monthly_Income",
    "Total_Spend",
    "Salary_Exhausted_Pct",
    "Purchase_Frequency",
]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Clustering Quality Metrics ("Accuracy" substitutes for unsupervised models)
sil_score = silhouette_score(X_scaled, df["Cluster"])
db_index = davies_bouldin_score(X_scaled, df["Cluster"])

# --- 4. DISPLAY DASHBOARD METRICS ---
col1, col2, col3 = st.columns(3)
col1.metric("Order Timestamp", order_datetime.strftime("%d %b %Y, %I:%M %p"))
col2.metric("Salary Exhausted (%)", f"{salary_exhausted_pct:.2f}%")
col3.metric("Silhouette Score (Quality)", f"{sil_score:.3f}")

st.info(
    f"**Model Evaluation Note:** Clustering algorithms use **Silhouette Score** ({sil_score:.3f}) "
    f"and **Davies-Bouldin Index** ({db_index:.3f}) to assess grouping separation and cluster tightness."
)

# --- 5. PREDICTING THE NEW CUSTOMER ---
input_df = pd.DataFrame(
    [
        {
            "Monthly_Income": monthly_income,
            "Total_Spend": total_spend,
            "Salary_Exhausted_Pct": salary_exhausted_pct,
            "Purchase_Frequency": purchase_frequency,
        }
    ]
)

input_scaled = scaler.transform(input_df[features])
predicted_cluster = kmeans.predict(input_scaled)[0]

st.subheader(f"Assigned Segment: **Cluster {predicted_cluster}**")

# Contextual interpretation based on salary exhaustion
if salary_exhausted_pct > 50:
    st.warning(
        "High Spender / High Exhaustion: Customer spends a substantial portion of their income."
    )
elif salary_exhausted_pct < 20:
    st.success(
        "Conservative Spender: Low exhaustion rate, high potential for targeted upselling."
    )
else:
    st.info(
        "Balanced Spender: Moderate expenditure relative to overall income."
    )

st.dataframe(df.head())
