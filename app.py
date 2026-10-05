import os
import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TOC Regex Text Scanner", layout="wide")
st.title("Automata & Regex Token Scanner (Unit II & VI)")
st.caption("Batch scanning records using Deterministic Regular Expressions")

# Unit II / VI: Regular Expression Patterns
PATTERNS = {
    "Financial Urgency": r"(\$\d+|\bFREE\b|\blottery\b|\bprize\b|\bearn\b)",
    "Suspicious URL": r"https?://[a-zA-Z0-9.-]+\.[a-z]{2,}",
    "Phone / Contact Mask": r"(\bCall\s\d+|\+\d{10,12})",
    "Account Threat": r"(\bsuspended\b|\bverify\b|\balert\b|\burgent\b)"
}

def regex_scanner(text):
    text_str = str(text)
    for pattern_name, regex in PATTERNS.items():
        if re.search(regex, text_str, re.IGNORECASE):
            return "Flagged (Spam)", pattern_name
    return "Clean (Ham)", "None"

# Sidebar: Dataset selection & Custom Teacher Upload
st.sidebar.header("Dataset Options")
upload_choice = st.sidebar.radio(
    "Choose Data Source:",
    ("Use Built-in 5,000 Records", "Upload Teacher's Custom CSV")
)

df = None

if upload_choice == "Upload Teacher's Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        # Teacher might name text column 'Text', 'Message', or 'Content'
        possible_cols = [c for c in raw_df.columns if c.lower() in ["text", "message", "content", "msg", "sms"]]
        selected_col = possible_cols[0] if possible_cols else raw_df.columns[0]
        
        df = pd.DataFrame({
            "Message_ID": range(1, len(raw_df) + 1),
            "Text": raw_df[selected_col].astype(str)
        })
        st.sidebar.success(f"Loaded {len(df):,} records from column: '{selected_col}'")
    else:
        st.info("Awaiting CSV upload. You can upload any CSV file containing text messages.")
else:
    if not os.path.exists("dataset_5000.csv"):
        import generate_dataset
    df = pd.read_csv("dataset_5000.csv")

# If dataset is loaded, run evaluation
if df is not None:
    results = [regex_scanner(msg) for msg in df["Text"]]
    df["Predicted_Status"] = [r[0] for r in results]
    df["Triggered_Rule"] = [r[1] for r in results]

    total_count = len(df)
    flagged_count = len(df[df["Predicted_Status"] == "Flagged (Spam)"])
    clean_count = total_count - flagged_count

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records Processed", f"{total_count:,}")
    c2.metric("Flagged as Suspicious", f"{flagged_count:,}")
    c3.metric("Clean Records", f"{clean_count:,}")

    st.markdown("---")

    filter_option = st.selectbox("Filter records:", ["All Records", "Flagged (Spam)", "Clean (Ham)"])
    if filter_option == "Flagged (Spam)":
        display_df = df[df["Predicted_Status"] == "Flagged (Spam)"]
    elif filter_option == "Clean (Ham)":
        display_df = df[df["Predicted_Status"] == "Clean (Ham)"]
    else:
        display_df = df

    st.dataframe(display_df[["Message_ID", "Text", "Predicted_Status", "Triggered_Rule"]], use_container_width=True)

# Live single-string testing area
st.markdown("---")
st.subheader("Live Single String Tester")
user_input = st.text_input("Test a custom message live:", "Urgent: Call +919876543210 to claim free reward")
if user_input:
    status, rule = regex_scanner(user_input)
    if "Flagged" in status:
        st.error(f"Status: {status} | Triggered Token Rule: {rule}")
    else:
        st.success(f"Status: {status} (Clean string)")
