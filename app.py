import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation & Behavior Mining", layout="wide")
st.title("Customer Behavior Mining & Segmentation Dashboard")
st.caption("Data Mining & Warehousing: Unit II (Pre-processing/Visualization) & Unit IV (Partitioning Clustering)")

# Function to generate customer dataset
@st.cache_data
def get_customer_data():
    np.random.seed(42)
    n = 5000
    return pd.DataFrame({
        "Customer_ID": [f"CUST_{i+1:05d}" for i in range(n)],
        "Age": np.random.randint(18, 70, size=n),
        "Annual_Income_k$": np.random.randint(15, 140, size=n),
        "Spending_Score_1_to_100": np.random.randint(1, 100, size=n),
        "Annual_Purchases": np.random.randint(1, 50, size=n)
    })

# Sidebar Controls
st.sidebar.header("Data Configuration")
data_source = st.sidebar.radio(
    "Data Source:",
    ("Use Built-in 5,000 Records", "Upload Custom CSV File")
)

df = None
if data_source == "Upload Custom CSV File":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.sidebar.success(f"Loaded {len(df):,} records successfully!")
    else:
        st.info("Awaiting CSV file...")
else:
    df = get_customer_data()

if df is not None:
    # 1. Dataset Overview
    st.markdown("### 1. Data Exploration & Overview (Unit II)")
    with st.expander("View Raw Data Preview"):
        st.dataframe(df.head(100), use_container_width=True)
        st.caption("Displaying initial 100 rows preview for browser speed optimization.")
        
        # One-click download button for full 5,000 dataset
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full 5,000 Records Dataset (.csv)",
            data=csv_bytes,
            file_name="customer_segmentation_5000_records.csv",
            mime="text/csv",
            help="Click to download the complete 5,000 dataset for offline inspection or WEKA analysis."
        )


    # 2. Pre-processing & Feature Selection
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if len(numeric_cols) >= 2:
        st.sidebar.markdown("---")
        st.sidebar.header("Clustering Parameters (Unit IV)")
        
        feat_x = st.sidebar.selectbox("Feature X-Axis:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        feat_y = st.sidebar.selectbox("Feature Y-Axis:", numeric_cols, index=2 if len(numeric_cols) > 2 else 0)
        k_clusters = st.sidebar.slider("Number of Clusters (K):", min_value=2, max_value=8, value=4)

        # Standard Scaler (Data Pre-processing: Normalization)
        X = df[[feat_x, feat_y]].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-Means Clustering Algorithm
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        df_clustered = df.loc[X.index].copy()
        df_clustered["Cluster_ID"] = [f"Cluster {c}" for c in clusters]

        # 3. Interactive Visualization
        st.markdown("### 2. K-Means Cluster Distribution (Unit IV)")
        fig = px.scatter(
            df_clustered,
            x=feat_x,
            y=feat_y,
            color="Cluster_ID",
            hover_data=[df_clustered.columns[0]],
            title=f"Partitioned Clustering Analysis (K={k_clusters}) on 5,000 Records",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # 4. Cluster Profile Insights
        st.markdown("### 3. Discovered Cluster Profiles & Behavioral Insights")
        summary = df_clustered.groupby("Cluster_ID")[[feat_x, feat_y]].mean().reset_index()
        summary["Customer Count"] = df_clustered["Cluster_ID"].value_counts().values
        st.dataframe(summary, use_container_width=True)

        # 5. Live Prediction for Single Customer
        st.markdown("---")
        st.subheader("Predict Segment for a New Data Instance")
        col_a, col_b = st.columns(2)
        val_x = col_a.number_input(f"Enter {feat_x}:", value=float(df[feat_x].mean()))
        val_y = col_b.number_input(f"Enter {feat_y}:", value=float(df[feat_y].mean()))

        new_point_scaled = scaler.transform([[val_x, val_y]])
        predicted_cluster = kmeans.predict(new_point_scaled)[0]
        st.success(f"This record is classified into: **Cluster {predicted_cluster}**")
    else:
        st.error("Please provide a dataset with at least 2 numerical columns.")
