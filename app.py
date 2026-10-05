import os
import re
import pandas as pd
import streamlit as st

# Page setup
st.set_page_config(page_title="TOC Regex Text Scanner", layout="wide")
st.title("Automata & Regex Token Scanner (Unit II & VI)")
st.caption("Batch scanning 5,000 records using Deterministic Regular Expressions")

# Ensure dataset exists
if not os.path.exists("dataset_5000.csv"):
    import generate_dataset

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("dataset_5000.csv")

df = load_data()

# Unit II / VI: Regular Expression Patterns
PATTERNS = {
    "Financial Urgency": r"(\$\d+|\bFREE\b|\blottery\b|\bprize\b|\bearn\b)",
    "Suspicious URL": r"https?://[a-zA-Z0-9.-]+\.[a-z]{2,}",
    "Phone / Contact Mask": r"(\bCall\s\d+|\+\d{10,12})",
    "Account Threat": r"(\bsuspended\b|\bverify\b|\balert\b|\burgent\b)"
}

def regex_scanner(text):
    for pattern_name, regex in PATTERNS.items():
        if re.search(regex, text, re.IGNORECASE):
            return "Flagged (Spam)", pattern_name
    return "Clean (Ham)", "None"

# Process all 5,000 rows
results = [regex_scanner(msg) for msg in df["Text"]]
df["Predicted_Status"] = [r[0] for r in results]
df["Triggered_Rule"] = [r[1] for r in results]

# Metrics
total_count = len(df)
flagged_count = len(df[df["Predicted_Status"] == "Flagged (Spam)"])
clean_count = total_count - flagged_count

c1, c2, c3 = st.columns(3)
c1.metric("Total Records Processed", f"{total_count:,}")
c2.metric("Flagged as Suspicious", f"{flagged_count:,}")
c3.metric("Clean Records", f"{clean_count:,}")

st.markdown("---")

# Data Table with Filters
filter_option = st.selectbox("Filter records by:", ["All Records", "Flagged (Spam)", "Clean (Ham)"])

if filter_option == "Flagged (Spam)":
    display_df = df[df["Predicted_Status"] == "Flagged (Spam)"]
elif filter_option == "Clean (Ham)":
    display_df = df[df["Predicted_Status"] == "Clean (Ham)"]
else:
    display_df = df

st.dataframe(display_df[["Message_ID", "Text", "Predicted_Status", "Triggered_Rule"]], use_container_width=True)

# Single String Live Test
st.markdown("---")
st.subheader("Live String Validator (Unit I / II Testing)")
user_input = st.text_input("Enter custom text to test regex matcher:", "Urgent: Click http://verify.xyz to claim prize")
if user_input:
    status, rule = regex_scanner(user_input)
    if "Flagged" in status:
        st.error(f"Status: {status} | Triggered Regex Token: {rule}")
    else:
        st.success(f"Status: {status} (No threat patterns detected)")
