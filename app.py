import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Customer Behavior Mining & Segmentation", layout="wide")
st.title("Customer Behavior Mining & Segmentation Dashboard")
st.caption("Data Mining & Warehousing: Unit II (Pre-processing/Feature Engineering) & Unit IV (Partitioning Clustering & Evaluation)")

# Function to generate customer dataset with Timestamps
@st.cache_data
def get_customer_data():
    np.random.seed(42)
    n = 5000
    base_date = datetime(2026, 1, 1)
    
    # Generate random timestamps over a 300-day window
    random_days = np.random.randint(0, 300, size=n)
    random_seconds = np.random.randint(0, 86400, size=n)
    timestamps = [base_date + timedelta(days=int(d), seconds=int(s)) for d, s in zip(random_days, random_seconds)]
    
    return pd.DataFrame({
        "Customer_ID": [f"CUST_{i+1:05d}" for i in range(n)],
        "Order_DateTime": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
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
    # Feature Construction: Salary Exhaustion % (Unit II: Data Transformation)
    if "Annual_Income_k$" in df.columns and "Spending_Score_1_to_100" in df.columns:
        df["Salary_Exhausted_Pct"] = df["Spending_Score_1_to_100"].astype(float)
        df["Burn_Rate_Category"] = pd.cut(
            df["Salary_Exhausted_Pct"],
            bins=[-1, 40, 70, 101],
            labels=["Conservative (<40%)", "Moderate (40-70%)", "High Burn (>70%)"]
        )

    # 1. Dataset Overview
    st.markdown("### 1. Data Exploration & Overview (Unit II)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records Mined", f"{len(df):,}")
    c2.metric("Features Extracted", len(df.columns))
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    
    if "Order_DateTime" in df.columns:
        c4.metric("Temporal Log Range", "2026 Season")
    else:
        c4.metric("Temporal Log", "N/A")

    with st.expander("View Raw Data Preview (with Order Date, Time & Salary Exhaustion)"):
        st.dataframe(df.head(100), use_container_width=True)
        st.caption("Displaying initial 100 rows preview for browser speed optimization.")
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full 5,000 Records Dataset (.csv)",
            data=csv_bytes,
            file_name="customer_segmentation_5000_records.csv",
            mime="text/csv"
        )

    # 2. Pre-processing & Feature Selection
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    # Filter out derived target from standard clustering picker if desired
    clustering_cols = [c for c in numeric_cols if c != "Salary_Exhausted_Pct"]
    
    if len(clustering_cols) >= 2:
        st.sidebar.markdown("---")
        st.sidebar.header("Clustering Parameters (Unit IV)")
        
        feat_x = st.sidebar.selectbox("Feature X-Axis:", clustering_cols, index=1 if len(clustering_cols) > 1 else 0)
        feat_y = st.sidebar.selectbox("Feature Y-Axis:", clustering_cols, index=2 if len(clustering_cols) > 2 else 0)
        k_clusters = st.sidebar.slider("Number of Clusters (K):", min_value=2, max_value=8, value=4)

        # Standard Scaler (Data Pre-processing: Normalization)
        X = df[[feat_x, feat_y]].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-Means Clustering Algorithm
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate Clustering Evaluation Metric (Unit IV: Evaluation of Clustering)
        sample_indices = np.random.choice(len(X_scaled), size=min(1000, len(X_scaled)), replace=False)
        sil_score = silhouette_score(X_scaled[sample_indices], clusters[sample_indices])
        cohesion_accuracy_pct = round(((sil_score + 1) / 2) * 100, 2)

        df_clustered = df.loc[X.index].copy()
        df_clustered["Cluster_ID"] = [f"Cluster {c}" for c in clusters]

        # 3. Interactive Clustering Visualization
        st.markdown("### 2. K-Means Cluster Distribution & Evaluation (Unit IV)")
        
        m1, m2 = st.columns(2)
        m1.metric(
            label="Clustering Cohesion Accuracy (Silhouette Equivalent)", 
            value=f"{cohesion_accuracy_pct}%",
            help="Mapped from Silhouette Coefficient [-1 to +1] into a 0-100% cohesion score."
        )
        m2.metric(
            label="Clustering Inertia (WCSS)", 
            value=f"{kmeans.inertia_:.2f}",
            help="Within-Cluster Sum of Squares: Total squared distance of samples to their closest cluster center."
        )

        hover_cols = ["Customer_ID"]
        if "Order_DateTime" in df_clustered.columns:
            hover_cols.append("Order_DateTime")
        if "Salary_Exhausted_Pct" in df_clustered.columns:
            hover_cols.append("Salary_Exhausted_Pct")

        fig = px.scatter(
            df_clustered,
            x=feat_x,
            y=feat_y,
            color="Cluster_ID",
            hover_data=hover_cols,
            title=f"Partitioned Clustering Analysis (K={k_clusters}) on 5,000 Records",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # 4. NEW: Salary Exhaustion Analysis Section (Unit II & IV)
        if "Salary_Exhausted_Pct" in df_clustered.columns:
            st.markdown("---")
            st.markdown("### 3. Salary Exhaustion & Financial Burn-Rate Analysis (Unit II)")
            st.caption("Derived behavioral metric analyzing the proportion of salary/disposable budget exhausted across mined clusters.")

            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                # Box Plot: Salary Exhaustion across Clusters
                fig_box = px.box(
                    df_clustered,
                    x="Cluster_ID",
                    y="Salary_Exhausted_Pct",
                    color="Cluster_ID",
                    title="Salary Exhausted (%) Distribution by Cluster",
                    labels={"Salary_Exhausted_Pct": "Salary Exhausted (%)", "Cluster_ID": "Cluster"},
                    template="plotly_white"
                )
                st.plotly_chart(fig_box, use_container_width=True)

            with col_chart2:
                # Bar Chart: Distribution of Burn-Rate Categories
                fig_burn = px.histogram(
                    df_clustered,
                    x="Cluster_ID",
                    color="Burn_Rate_Category",
                    barmode="group",
                    title="Financial Burn-Rate Breakdown per Cluster",
                    labels={"Cluster_ID": "Cluster", "Burn_Rate_Category": "Burn Rate Risk"},
                    color_discrete_map={
                        "Conservative (<40%)": "#2b6cb0",
                        "Moderate (40-70%)": "#dd6b20",
                        "High Burn (>70%)": "#e53e3e"
                    },
                    template="plotly_white"
                )
                st.plotly_chart(fig_burn, use_container_width=True)

        # 5. Cluster Profile Insights
        st.markdown("### 4. Discovered Cluster Profiles & Behavioral Insights")
        summary_cols = [feat_x, feat_y]
        if "Salary_Exhausted_Pct" in df_clustered.columns and "Salary_Exhausted_Pct" not in summary_cols:
            summary_cols.append("Salary_Exhausted_Pct")
            
        summary = df_clustered.groupby("Cluster_ID")[summary_cols].mean().reset_index()
        summary["Customer Count"] = df_clustered["Cluster_ID"].value_counts().values
        st.dataframe(summary, use_container_width=True)

        # 6. Live Prediction with Salary Exhausted Calculation
        st.markdown("---")
        st.subheader("5. Predict Segment for a New Data Instance")
        
        col_a, col_b = st.columns(2)
        val_x_str = col_a.text_input(f"Enter {feat_x}:", value=f"{df[feat_x].mean():.2f}")
        val_y_str = col_b.text_input(f"Enter {feat_y}:", value=f"{df[feat_y].mean():.2f}")

        try:
            val_x = float(val_x_str)
            val_y = float(val_y_str)
            new_point_scaled = scaler.transform([[val_x, val_y]])
            predicted_cluster = kmeans.predict(new_point_scaled)[0]
            st.success(f"This record is classified into: **Cluster {predicted_cluster}**")

            # Salary Exhausted Prompt / Alert
            if "Spending_Score_1_to_100" in [feat_x, feat_y]:
                score_val = val_y if feat_y == "Spending_Score_1_to_100" else val_x
                exhausted_pct = min(100.0, max(0.0, score_val))
                
                st.markdown("#### Financial Behavioral Diagnostic:")
                if exhausted_pct >= 70:
                    st.error(f"🚨 **Salary Exhausted Ratio: {exhausted_pct:.1f}%** — High Financial Burn Rate! Customer is spending an aggressive portion of disposable income.")
                elif exhausted_pct >= 40:
                    st.warning(f"⚖️ **Salary Exhausted Ratio: {exhausted_pct:.1f}%** — Moderate Financial Utilization. Balanced budget behavior.")
                else:
                    st.info(f"💰 **Salary Exhausted Ratio: {exhausted_pct:.1f}%** — Conservative Spender. High financial savings reserve.")
        except ValueError:
            st.warning("Please enter a valid numeric value.")
    else:
        st.error("Please provide a dataset with at least 2 numerical columns.")
