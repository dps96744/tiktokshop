################################################################################
# (1) COHORT ANALYSIS TOOL - PARTIAL GATING WITH STRIPE & AWS S3
#     Complete codebase to handle automatic data retrieval post-payment,
#     with requested modifications.
################################################################################

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import stripe
import boto3
from datetime import datetime, timedelta
import math
import random
import itertools
import functools
import logging
import uuid
import sys
import threading
import io
from typing import Dict, Any, List, Optional

# ------------------------------------------------------------------------------
# (A) PAGE CONFIG MUST BE FIRST
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Cohort Analysis Tool (AWS S3 + Stripe, Full Automation)",
    layout="wide",
    initial_sidebar_state="expanded"
)

################################################################################
# (B) STRIPE & AWS S3 SETUP (No Hardcoded Keys)
################################################################################

os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
os.environ["AWS_DEFAULT_REGION"] = st.secrets["AWS_DEFAULT_REGION"]

stripe.api_key = st.secrets["STRIPE_SECRET_KEY"]
STRIPE_PUBLISHABLE_KEY = st.secrets["STRIPE_PUBLISHABLE_KEY"]

# 2) Initialize S3
s3 = boto3.client("s3")
BUCKET_NAME = st.secrets["S3_BUCKET_NAME"]  # e.g., "cohort-analysis-ai-uploads"

################################################################################
# (C) ITEMS_FOR_SALE
################################################################################
# Modified: 
#   - Single chart = $5
#   - "Unlock All Charts" = $20
#
# Reordered as per user instructions.

ITEMS_FOR_SALE = [
    {"name": "Average Lifetime Value (LTV) by Cohort (Heatmap)", "price": 5},
    {"name": "Average Retention by Cohort (Heatmap)", "price": 5},
    {"name": "LTV:CAC Ratio by Cohort", "price": 5},
    {"name": "CAC & New Customers vs. Repeat Customers", "price": 5},
    {"name": "Percent of Customers Making Multiple Orders", "price": 5},
    {"name": "Time Between First and Second Purchase", "price": 5},
    {"name": "Cohort Size vs. Average Lifetime Value (LTV) by Cohort", "price": 5},
    {"name": "Average Lifetime Value (LTV) by Cohort (Line)", "price": 5},
    {"name": "Average Retention by Cohort (Line)", "price": 5},
    {"name": "Unlock All Charts", "price": 20},
]

################################################################################
# (D) SESSION STATE INITIALIZATION
################################################################################

if "purchased_items" not in st.session_state:
    st.session_state["purchased_items"] = []

# Store filenames to retrieve from S3 after payment
if "order_data_filename" not in st.session_state:
    st.session_state["order_data_filename"] = None
if "marketing_spend_filename" not in st.session_state:
    st.session_state["marketing_spend_filename"] = None

# DataFrames
if "order_data_df" not in st.session_state:
    st.session_state["order_data_df"] = pd.DataFrame()
if "marketing_spend_df" not in st.session_state:
    st.session_state["marketing_spend_df"] = pd.DataFrame()

# Upload ID
if "upload_id" not in st.session_state:
    st.session_state["upload_id"] = str(uuid.uuid4())

################################################################################
# (E) STRIPE CHECKOUT CREATION & PAYMENT HANDLING
################################################################################

# Set BASE_URL to your exact Streamlit app URL
BASE_URL = "https://amzn-cohort-analysis-e2kytjfwa7weeo3c7hp42z.streamlit.app"  # Replace with your actual URL

SUCCESS_URL_TEMPLATE = f"{BASE_URL}/?success=true&session_id={{CHECKOUT_SESSION_ID}}&upload_id="
CANCEL_URL_TEMPLATE  = f"{BASE_URL}/?canceled=true&session_id={{CHECKOUT_SESSION_ID}}&upload_id="

def create_checkout_session(item_list: List[Dict[str, Any]], upload_id: str,
                            order_filename: Optional[str], spend_filename: Optional[str]) -> Optional[str]:
    """
    Creates a Stripe checkout session and returns the session URL.
    Passes filenames in the success URL for automatic data retrieval.
    """
    success_url = SUCCESS_URL_TEMPLATE + upload_id
    cancel_url  = CANCEL_URL_TEMPLATE + upload_id

    # Append filenames if available
    if order_filename:
        success_url += f"&odf={order_filename}"
        cancel_url  += f"&odf={order_filename}"
    if spend_filename:
        success_url += f"&msf={spend_filename}"
        cancel_url  += f"&msf={spend_filename}"

    line_items = []
    for it in item_list:
        line_items.append({
            "price_data": {
                "currency": "usd",
                "unit_amount": it["price"] * 100,  # Stripe expects amount in cents
                "product_data": {"name": it["name"]},
            },
            "quantity": 1
        })
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=line_items,
            mode="payment",
            success_url=success_url,
            cancel_url=cancel_url
        )
        return session.url
    except Exception as exc:
        st.error(f"Error creating Stripe checkout session: {exc}")
        return None

def handle_payment():
    """
    Handles Stripe payment by parsing query parameters and updating session state.
    """
    qp = st.query_params

    success = qp.get("success", ["false"])
    canceled = qp.get("canceled", ["false"])
    session_id = qp.get("session_id", [""])
    upload_id = qp.get("upload_id", [""])
    odf = qp.get("odf", [None])
    msf = qp.get("msf", [None])

    # Convert any lists to single values
    if isinstance(success, list):
        success = success[0] if success else "false"
    if isinstance(canceled, list):
        canceled = canceled[0] if canceled else "false"
    if isinstance(session_id, list):
        session_id = session_id[0] if session_id else ""
    if isinstance(upload_id, list):
        upload_id = upload_id[0] if upload_id else ""
    if isinstance(odf, list):
        odf = odf[0] if odf else None
    if isinstance(msf, list):
        msf = msf[0] if msf else None

    # Update filenames in session state
    if odf:
        st.session_state["order_data_filename"] = odf
    if msf:
        st.session_state["marketing_spend_filename"] = msf

    if upload_id:
        st.session_state["upload_id"] = upload_id

    if success == "true" and session_id:
        st.session_state["show_refresh_spinner"] = True
        try:
            stripe_session = stripe.checkout.Session.retrieve(session_id)
            if stripe_session.payment_status == "paid":
                line_items = stripe.checkout.Session.list_line_items(session_id)
                purchased_names = [li.description for li in line_items.data]
                for item_name in purchased_names:
                    for it in ITEMS_FOR_SALE:
                        if it["name"] == item_name:
                            if it["name"] not in st.session_state["purchased_items"]:
                                st.session_state["purchased_items"].append(it["name"])
        except Exception as e:
            st.error(f"Error retrieving Stripe checkout session: {e}")
    elif canceled == "true":
        st.warning("Payment canceled.")

################################################################################
# (F) DATA PREPROCESSING FUNCTIONS
################################################################################

@st.cache_data
def preprocess_order_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the order data DataFrame.
    
    We now require these 5 columns EXACTLY:
       1) purchase-date
       2) buyer-email
       3) item-price
       4) sku
       5) amazon-order-id

    After verifying them, we rename:
       purchase-date   -> Date
       item-price      -> Order Total
       sku             -> SKU
       amazon-order-id -> Order ID

    We drop rows that are missing ANY of these required columns,
    but we keep any extra columns the user may have.
    """
    needed = ["purchase-date", "buyer-email", "item-price", "sku", "amazon-order-id"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Order Data: {missing}")

    # Keep any extra columns, only dropping rows that are missing ANY required column
    df = df.dropna(subset=needed, how="any")

    # Rename them to our internal columns
    rename_map = {
        "purchase-date": "Date",
        "item-price": "Order Total",
        "sku": "SKU",
        "amazon-order-id": "Order ID"
    }
    df.rename(columns=rename_map, inplace=True)

    # Convert 'buyer-email' -> 'Customer ID'
    df["buyer-email"] = df["buyer-email"].astype(str).str.strip()
    df.rename(columns={"buyer-email": "Customer ID"}, inplace=True)

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Date"] = df["Date"].dt.tz_localize(None)

    # Clean up
    df["Customer ID"] = df["Customer ID"].astype(str).str.strip()
    df = df[df["Customer ID"] != ""]
    df["Order Total"] = pd.to_numeric(df["Order Total"], errors="coerce").round(2)
    df = df.dropna(subset=["Order Total"])
    df["SKU"] = df["SKU"].astype(str).str.strip()
    # We'll keep "Order ID" for reference

    return df

@st.cache_data
def preprocess_marketing_spend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the marketing spend DataFrame.
    """
    needed = ["Date", "Marketing Spend"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame(columns=needed)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Date"] = df["Date"].dt.floor("D").dt.tz_localize(None)
    df["Marketing Spend"] = pd.to_numeric(df["Marketing Spend"], errors="coerce").fillna(0)
    df = df.groupby("Date")["Marketing Spend"].sum().reset_index()
    return df

################################################################################
# (G) FREQUENCY / INDEX HELPERS
################################################################################

def unify_cohort_label_for_date(d: datetime, freq: str) -> str:
    """
    Converts a date to a cohort label based on frequency.
    (We now fix freq='Monthly')
    """
    return d.strftime("%Y-%m")

def date_to_period_index(row, freq: str) -> int:
    """
    Calculates the period index based on frequency.
    (We now fix freq='Monthly')
    """
    diff_days = (row["Date"] - row["Cohort Date"]).days
    yd = row["Date"].year - row["Cohort Date"].year
    md = row["Date"].month - row["Cohort Date"].month
    return yd * 12 + md + 1

@st.cache_data
def group_spend_by_freq(spend_df: pd.DataFrame, freq: str) -> Dict[str, float]:
    """
    Groups marketing spend by the specified frequency.
    freq='Monthly' is forced.
    """
    if spend_df.empty:
        return {}
    df = spend_df.copy()
    df["Label"] = df["Date"].apply(lambda x: x.strftime("%Y-%m"))
    grouped = df.groupby("Label")["Marketing Spend"].sum()
    return grouped.to_dict()

################################################################################
# (H) LTV & RETENTION CALCULATIONS
################################################################################

@st.cache_data
def calculate_avg_ltv(order_df: pd.DataFrame, freq: str, marketing_spend_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Average Lifetime Value (LTV) by Cohort (Monthly only).
    """
    if order_df.empty:
        return pd.DataFrame()
    df = order_df.copy()
    df["Date"] = df["Date"].dt.floor("D")
    df["Cohort Date"] = df.groupby("Customer ID")["Date"].transform("min")
    df["Cohort Index"] = df.apply(lambda r: date_to_period_index(r, freq), axis=1)
    df["Cohort"] = df["Cohort Date"].apply(lambda d: unify_cohort_label_for_date(d, freq))

    pivot = df.pivot_table(index="Cohort", columns="Cohort Index", values="Order Total", aggfunc="sum")
    if pivot.empty:
        return pd.DataFrame()

    pivot.columns = [f"Month {c}" for c in pivot.columns]
    csize = df.groupby("Cohort")["Customer ID"].nunique()
    cumsums = pivot.cumsum(axis=1)
    avg_ltv = cumsums.div(csize, axis=0)

    spend_dict = group_spend_by_freq(marketing_spend_df, freq)
    mk_vals = [spend_dict.get(idx, 0.0) for idx in avg_ltv.index]
    avg_ltv.insert(0, "Marketing Spend", mk_vals)
    avg_ltv.insert(1, "Cohort Size", csize)

    avg_ltv["CAC"] = avg_ltv.apply(
        lambda row: row["Marketing Spend"] / row["Cohort Size"] if row["Cohort Size"] else np.nan,
        axis=1
    )
    for col in avg_ltv.columns:
        avg_ltv[col] = pd.to_numeric(avg_ltv[col], errors="coerce").round(2)
    avg_ltv = avg_ltv.reset_index()
    return avg_ltv

@st.cache_data
def calculate_percent_retained(order_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Calculates the percentage of customers retained by Cohort (Monthly only).
    We'll STILL generate 'Month 1' in the pivot, but the chart function will remove Month 1.
    """
    if order_df.empty:
        return pd.DataFrame()
    df = order_df.copy()
    df["Cohort Date"] = df.groupby("Customer ID")["Date"].transform("min")
    df["Cohort Index"] = df.apply(lambda r: date_to_period_index(r, freq), axis=1)
    df["Cohort"] = df["Cohort Date"].apply(lambda d: unify_cohort_label_for_date(d, freq))

    pivot = df.groupby(["Cohort", "Cohort Index"])["Customer ID"].nunique().unstack()
    if pivot.empty:
        return pd.DataFrame()

    pivot.columns = [f"Month {c}" for c in pivot.columns]

    # If there's a "Month 1" => that's the initial # of customers in that cohort
    if "Month 1" in pivot.columns:
        csize = pivot["Month 1"]
    else:
        csize = pivot.iloc[:, 0]

    ret = pivot.div(csize, axis=0) * 100
    ret = ret.reset_index().round(2)
    return ret

################################################################################
# (I) AWS S3 UPLOAD & DOWNLOAD FUNCTIONS
################################################################################

def upload_file_to_s3(upload_id: str, file_bytes: bytes, filename: str) -> bool:
    """
    Uploads a file to AWS S3 under a specific upload_id.
    """
    key = f"uploads/{upload_id}/{filename}"
    try:
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=file_bytes)
        return True
    except Exception as ex:
        st.error(f"Failed to upload file to S3: {ex}")
        return False

def download_file_from_s3(upload_id: str, filename: str) -> Optional[bytes]:
    """
    Downloads a file from AWS S3 based on upload_id and filename.
    """
    key = f"uploads/{upload_id}/{filename}"
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return obj["Body"].read()
    except Exception as ex:
        st.error(f"Failed to download file from S3: {ex}")
        return None

################################################################################
# (J) 6-MONTH PARTIAL GATING FUNCTIONS
################################################################################

def limit_to_6_months_ltv(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Limits the LTV DataFrame to 6 months.
    """
    if full_df.empty or "Cohort" not in full_df.columns:
        return full_df
    month_cols = [c for c in full_df.columns if c.startswith("Month ")]
    keep_months = month_cols[:6]
    keep_cols = ["Cohort", "Marketing Spend", "Cohort Size", "CAC"] + keep_months
    keep_cols = [x for x in keep_cols if x in full_df.columns]
    df_limited = full_df[keep_cols].copy()
    for i, old_col in enumerate(keep_months, start=1):
        if old_col in df_limited.columns:
            df_limited.rename(columns={old_col: f"Month {i}"}, inplace=True)
    return df_limited

def limit_to_6_months_ret(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Limits the Retention DataFrame to 6 months 
    and removes Month 1 from the partial gating version.
    """
    if full_df.empty or "Cohort" not in full_df.columns:
        return full_df
    # remove "Month 1" column entirely for partial gating:
    if "Month 1" in full_df.columns:
        full_df = full_df.drop(columns=["Month 1"], errors="ignore")

    month_cols = [c for c in full_df.columns if c.startswith("Month ")]
    keep_months = month_cols[:6]
    keep_cols = ["Cohort"] + keep_months
    keep_cols = [x for x in keep_cols if x in full_df.columns]
    df_limited = full_df[keep_cols].copy()
    # We re-label them so that the first displayed column is "Month 2" => "Month 2", etc.
    for i, old_col in enumerate(keep_months, start=2):  
        if old_col in df_limited.columns:
            df_limited.rename(columns={old_col: f"Month {i}"}, inplace=True)
    return df_limited

################################################################################
# (K) CHART PLOTTING FUNCTIONS (with optional "purchased" param for download)
################################################################################

def plot_ltv_heatmap(avg_ltv: pd.DataFrame, purchased: bool = False):
    """
    Plots the LTV Heatmap using Plotly, now includes 'CAC' column.
    If purchased=True, display a 'Download CSV' button with full data.
    """
    if avg_ltv.empty or "Cohort" not in avg_ltv.columns:
        st.write("No data available for the LTV Heatmap.")
        return
    if "Cohort Size" not in avg_ltv.columns:
        st.write("No 'Cohort Size' column in LTV data.")
        return

    month_cols = [col for col in avg_ltv.columns if col.startswith("Month ")]
    reorder = ["Cohort", "Cohort Size", "CAC"] + month_cols
    reorder = [c for c in reorder if c in avg_ltv.columns]
    df_heat = avg_ltv[reorder].copy()
    df_heat.set_index("Cohort", inplace=True)

    z_vals = df_heat.values.astype(float)

    # Exclude "Cohort Size" & "CAC" columns from the color scale
    if df_heat.shape[1] >= 2:
        z_vals[:, 0:2] = np.nan

    df_heat_for_text = df_heat.copy().fillna("")
    text_vals = df_heat_for_text.astype(str).values

    x_labels = df_heat.columns.tolist()
    y_labels = df_heat.index.tolist()

    valid_z = z_vals[~np.isnan(z_vals)]
    if len(valid_z) == 0:
        st.write("All LTV values are NaN. Nothing to display.")
        return

    zmin = 0
    zmax = np.nanmax(valid_z)
    row_count = len(y_labels)
    row_height = 40
    fig_height = row_count * row_height if row_count > 0 else 400

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=x_labels,
        y=y_labels,
        xgap=3,
        ygap=3,
        zmin=zmin,
        zmax=zmax,
        colorscale="RdBu",
        text=text_vals,
        texttemplate="%{text}",
        hovertemplate="Cohort: %{y}<br>%{x}: %{z}<extra></extra>"
    ))
    fig.update_layout(
        height=fig_height,
        uniformtext=dict(minsize=14, mode="hide"),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=14),
        xaxis=dict(type="category", side="top", showgrid=False, zeroline=False),
        yaxis=dict(type="category", showgrid=False, zeroline=False, autorange="reversed", title="Cohort")
    )

    # Annotate the first two columns
    for i, row_name in enumerate(y_labels):
        size_str = text_vals[i, 0]  # Cohort Size
        cac_str = text_vals[i, 1]   # CAC
        fig.add_annotation(
            x="Cohort Size",
            y=row_name,
            text=size_str,
            showarrow=False,
            font=dict(color="white"),
            xref="x",
            yref="y"
        )
        if len(df_heat.columns) > 2:
            fig.add_annotation(
                x="CAC",
                y=row_name,
                text=cac_str,
                showarrow=False,
                font=dict(color="white"),
                xref="x",
                yref="y"
            )

    st.plotly_chart(fig, use_container_width=True)

    # Subtitle placeholder text
    st.markdown("<p style='color:white; font-size:12px;'>subtitle placeholder text</p>", unsafe_allow_html=True)

    # Download button if purchased
    if purchased:
        csv_data = avg_ltv.to_csv(index=False)
        st.download_button("Download CSV", data=csv_data, file_name="avg_ltv_heatmap.csv", mime="text/csv")

def plot_cohort_lifetime_value_line_chart(avg_ltv: pd.DataFrame, purchased: bool = False):
    """
    Plots the LTV Line Chart using Plotly. 
    """
    if avg_ltv.empty or "Cohort" not in avg_ltv.columns:
        st.write("No data available for the LTV line chart.")
        return
    mon_cols = [c for c in avg_ltv.columns if c.startswith("Month ")]
    if not mon_cols:
        st.write("No month columns found in LTV data.")
        return

    fig = go.Figure()
    for _, row in avg_ltv.iterrows():
        cohort = row["Cohort"]
        vals = row[mon_cols].dropna().values
        if len(vals) == 0:
            continue
        xvals = range(1, len(vals) + 1)
        fig.add_trace(go.Scatter(
            x=list(xvals),
            y=vals,
            mode="lines+markers",
            name=str(cohort),
            hovertemplate=f"Cohort: {cohort}<br>Month: %{{x}}<br>LTV: %{{y:.2f}}<extra></extra>"
        ))

    # Average line
    df_sub = avg_ltv[mon_cols].copy()
    means = []
    for c in df_sub.columns:
        col_vals = df_sub[c].dropna()
        if len(col_vals) > 0:
            means.append(col_vals.mean())
        else:
            means.append(np.nan)

    xinds = []
    yvals = []
    for i, val in enumerate(means):
        if not np.isnan(val):
            xinds.append(i + 1)
            yvals.append(val)

    if yvals:
        fig.add_trace(go.Scatter(
            x=xinds,
            y=yvals,
            mode="lines+markers",
            name="Average (All Cohorts)",
            line=dict(dash="dash", color="white"),
            hovertemplate="Month: %{x}<br>Avg LTV: %{y:.2f}<extra></extra>"
        ))

    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=14),
        height=600,
        xaxis=dict(title="Months Since First Purchase", dtick=1),
        yaxis=dict(title="Lifetime Value (LTV)"),
        legend=dict(x=1.02, y=1.0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Subtitle placeholder text
    st.markdown("<p style='color:white; font-size:12px;'>subtitle placeholder text</p>", unsafe_allow_html=True)

    # Download button if purchased
    if purchased:
        csv_data = avg_ltv.to_csv(index=False)
        st.download_button("Download CSV", data=csv_data, file_name="avg_ltv_line.csv", mime="text/csv")

def plot_retention_heatmap_with_size(ret_df: pd.DataFrame, avg_ltv: pd.DataFrame, purchased: bool = False):
    """
    Plots the Retention Heatmap with 'Cohort Size' & 'CAC' columns.
    As requested, the heat map should begin in Month 2, so we drop "Month 1" entirely, 
    even for the full/unlocked dataset as well.
    """
    # Remove Month 1 no matter what
    if "Month 1" in ret_df.columns:
        ret_df = ret_df.drop(columns=["Month 1"], errors="ignore")

    if ret_df.empty or "Cohort" not in ret_df.columns:
        st.write("No data available for the Retention Heatmap.")
        return
    if avg_ltv.empty or "Cohort" not in avg_ltv.columns or "Cohort Size" not in avg_ltv.columns:
        st.write("Insufficient data to merge 'Cohort Size' & 'CAC' for Retention Heatmap.")
        return

    merged = pd.merge(ret_df, avg_ltv[["Cohort", "Cohort Size", "CAC"]], on="Cohort", how="left")
    if merged.empty:
        st.write("No data after merging Cohort Size & CAC into Retention Heatmap.")
        return

    month_cols = [col for col in merged.columns if col.startswith("Month ")]
    reorder = ["Cohort", "Cohort Size", "CAC"] + month_cols
    reorder = [c for c in reorder if c in merged.columns]
    merged = merged[reorder].copy()
    merged.set_index("Cohort", inplace=True)

    z_vals = merged.values.astype(float)
    # Exclude the first 2 columns from color scale
    if merged.shape[1] >= 2:
        z_vals[:, 0:2] = np.nan

    merged_for_text = merged.copy().fillna("")
    text_vals = merged_for_text.astype(str).values

    x_labels = merged.columns.tolist()
    y_labels = merged.index.tolist()

    valid_z = z_vals[~np.isnan(z_vals)]
    if len(valid_z) == 0:
        st.write("No numeric data for retention heatmap.")
        return

    zmin = 0
    zmax = np.nanmax(valid_z)
    row_count = len(y_labels)
    row_height = 40
    fig_height = row_count * row_height if row_count > 0 else 400

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=x_labels,
        y=y_labels,
        xgap=3,
        ygap=3,
        zmin=zmin,
        zmax=zmax,
        colorscale="RdBu",
        text=text_vals,
        texttemplate="%{text}",
        hovertemplate="Cohort: %{y}<br>%{x}: %{z:.2f} %<extra></extra>"
    ))
    fig.update_layout(
        height=fig_height,
        uniformtext=dict(minsize=14, mode="hide"),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=14),
        xaxis=dict(side="top", showgrid=False, zeroline=False, type="category"),
        yaxis=dict(showgrid=False, zeroline=False, autorange="reversed", type="category", title="Cohort")
    )

    for i, row_name in enumerate(y_labels):
        size_str = text_vals[i, 0]
        cac_str = text_vals[i, 1]
        fig.add_annotation(
            x="Cohort Size",
            y=row_name,
            text=size_str,
            showarrow=False,
            font=dict(color="white"),
            xref="x",
            yref="y"
        )
        if len(merged.columns) > 2:
            fig.add_annotation(
                x="CAC",
                y=row_name,
                text=cac_str,
                showarrow=False,
                font=dict(color="white"),
                xref="x",
                yref="y"
            )

    st.plotly_chart(fig, use_container_width=True)

    # Subtitle placeholder text
    st.markdown("<p style='color:white; font-size:12px;'>subtitle placeholder text</p>", unsafe_allow_html=True)

    # Download button if purchased
    if purchased:
        csv_data = merged.reset_index().to_csv(index=False)
        st.download_button("Download CSV", data=csv_data, file_name="retention_heatmap.csv", mime="text/csv")

def plot_percent_retained_line_chart(ret_df: pd.DataFrame, purchased: bool = False):
    """
    Plots the Retention Line Chart using Plotly, with thinner lines
    """
    # We do NOT remove "Month 1" for the line chart by default, 
    # but we already handled "Month 1" removal for partial gating in limit_to_6_months_ret.
    if ret_df.empty or "Cohort" not in ret_df.columns:
        st.write("No data available for the Retention line chart.")
        return
    mon_cols = [c for c in ret_df.columns if c.startswith("Month ")]
    if not mon_cols:
        st.write("No month columns found in Retention data.")
        return

    fig = go.Figure()
    for _, row in ret_df.iterrows():
        cohort = row["Cohort"]
        vals = row[mon_cols].dropna().values
        if len(vals) == 0:
            continue
        xvals = list(range(1, len(vals) + 1))
        fig.add_trace(go.Scatter(
            x=xvals,
            y=vals,
            mode="lines+markers",
            line=dict(width=1),
            name=str(cohort),
            hovertemplate=f"Cohort: {cohort}<br>Month: %{{x}}<br>Retention: %{{y:.2f}}%<extra></extra>"
        ))

    # Average line
    df_sub = ret_df[mon_cols].copy()
    means = []
    for c in df_sub.columns:
        col_vals = df_sub[c].dropna()
        if len(col_vals) > 0:
            means.append(col_vals.mean())
        else:
            means.append(np.nan)

    xinds = []
    yvals = []
    for i, val in enumerate(means):
        if not np.isnan(val):
            xinds.append(i + 1)
            yvals.append(val)

    if yvals:
        fig.add_trace(go.Scatter(
            x=xinds,
            y=yvals,
            mode="lines+markers",
            line=dict(dash="dash", color="white", width=1),
            name="Average (All Cohorts)",
            hovertemplate="Month: %{x}<br>Avg Retention: %{y:.2f}%<extra></extra>"
        ))

    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=14),
        height=600,
        xaxis=dict(title="Months Since First Purchase", dtick=1),
        yaxis=dict(title="Retention (%)", range=[0, 100]),
        legend=dict(x=1.02, y=1.0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Subtitle placeholder text
    st.markdown("<p style='color:white; font-size:12px;'>subtitle placeholder text</p>", unsafe_allow_html=True)

    # Download button if purchased
    if purchased:
        csv_data = ret_df.to_csv(index=False)
        st.download_button("Download CSV", data=csv_data, file_name="retention_line.csv", mime="text/csv")

def plot_cohort_size_and_ltv_analysis_chart(avg_ltv: pd.DataFrame, purchased: bool = False):
    """
    Plots Cohort Size vs. LTV Analysis Chart using Plotly.
    """
    if avg_ltv.empty or "Cohort" not in avg_ltv.columns:
        st.write("No data for Cohort Size vs. LTV chart.")
        return
    if "Cohort Size" not in avg_ltv.columns:
        st.write("Missing 'Cohort Size' in LTV data.")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=avg_ltv["Cohort"],
        y=avg_ltv["Cohort Size"],
        name="Cohort Size",
        yaxis="y1",
        marker=dict(color="lightblue"),
        hovertemplate="Cohort: %{x}<br>Cohort Size: %{y}<extra></extra>"
    ))
    mon_cols = [c for c in avg_ltv.columns if c.startswith("Month ")]
    for col in mon_cols:
        if avg_ltv[col].notna().any():
            fig.add_trace(go.Scatter(
                x=avg_ltv["Cohort"],
                y=avg_ltv[col],
                mode="lines+markers",
                name=col,
                yaxis="y2",
                hovertemplate="Cohort: %{x}<br>LTV: %{y:.2f}<extra></extra>"
            ))
    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=14),
        xaxis=dict(type="category", title="Cohort"),
        yaxis=dict(title="Cohort Size"),
        yaxis2=dict(title="Lifetime Value (LTV)", overlaying="y", side="right"),
        legend=dict(x=1.02, y=1.0),
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)

    # Subtitle placeholder text
    st.markdown("<p style='color:white; font-size:12px;'>subtitle placeholder text</p>", unsafe_allow_html=True)

    # Download button if purchased
    if purchased:
        csv_data = avg_ltv.to_csv(index=False)
        st.download_button("Download CSV", data=csv_data, file_name="cohort_size_vs_ltv.csv", mime="text/csv")

def plot_ltv_cac_by_cohort(avg_ltv: pd.DataFrame, purchased: bool = False):
    """
    Plots LTV:CAC Ratio by Cohort using Plotly.
    """
    if avg_ltv.empty or "Cohort" not in avg_ltv.columns:
        st.write("No data for LTV:CAC chart.")
        return
    if "CAC" not in avg_ltv.columns:
        st.write("Missing 'CAC' in LTV data.")
        return
    mon_cols = [c for c in avg_ltv.columns if c.startswith("Month ")]
    if not mon_cols:
        st.write("No month columns found to compute LTV:CAC.")
        return

    fig = go.Figure()
    all_ratios = []
    trace_count = 0
    for _, row in avg_ltv.iterrows():
        cohort = row["Cohort"]
        cac = row["CAC"]
        if pd.isna(cac) or cac == 0:
            continue
        vals = row[mon_cols].dropna().values
        if len(vals) == 0:
            continue
        ratio_vals = [v / cac for v in vals]
        xvals = list(range(1, len(ratio_vals) + 1))
        fig.add_trace(go.Scatter(
            x=xvals,
            y=ratio_vals,
            mode="lines+markers",
            name=cohort,
            hovertemplate=f"Cohort: {cohort}<br>Month: %{{x}}<br>LTV:CAC Ratio: %{{y:.2f}}<extra></extra>"
        ))
        all_ratios.extend(ratio_vals)
        trace_count += 1
    if trace_count == 0:
        st.write("No valid data for LTV:CAC by Cohort chart.")
        return

    rmin = min(all_ratios)
    rmax = max(all_ratios)
    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=14),
        xaxis=dict(title="Months Since First Purchase", dtick=1),
        yaxis=dict(title="LTV:CAC Ratio",
                   range=[rmin * 0.9 if rmin > 0 else rmin * 1.1, rmax * 1.1]),
        legend=dict(x=1.02, y=1.0),
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)

    # Subtitle placeholder text
    st.markdown("<p style='color:white; font-size:12px;'>subtitle placeholder text</p>", unsafe_allow_html=True)

    # Download button if purchased
    if purchased:
        csv_data = avg_ltv.to_csv(index=False)
        st.download_button("Download CSV", data=csv_data, file_name="ltv_cac_ratio.csv", mime="text/csv")

################################################################################
# (NEW) CHARTS REQUESTED
################################################################################

def plot_cac_new_vs_repeat_all_data(order_df: pd.DataFrame, spend_df: pd.DataFrame, purchased: bool = False):
    """
    (Chart) Plots a combo chart: 
      - Bars: New vs. Repeat customers 
      - Line: CAC
    """
    if order_df.empty:
        st.write("No data available for CAC & New vs. Repeat chart.")
        return

    df = order_df.copy()
    df["Month"] = df["Date"].dt.to_period("M")
    monthly_spend = spend_df.copy()
    if not spend_df.empty:
        monthly_spend["Month"] = monthly_spend["Date"].dt.to_period("M")

    df.sort_values(["Customer ID", "Date"], inplace=True)
    df["CustomerOrderIndex"] = df.groupby("Customer ID").cumcount() + 1

    new_df = df[df["CustomerOrderIndex"] == 1].groupby("Month")["Customer ID"].nunique().reset_index()
    new_df.rename(columns={"Customer ID": "NewCustomers"}, inplace=True)

    rep_df = df[df["CustomerOrderIndex"] > 1].groupby("Month")["Customer ID"].nunique().reset_index()
    rep_df.rename(columns={"Customer ID": "RepeatCustomers"}, inplace=True)

    merged = pd.merge(new_df, rep_df, on="Month", how="outer").fillna(0)
    if not monthly_spend.empty:
        monthly_agg_spend = monthly_spend.groupby("Month")["Marketing Spend"].sum().reset_index()
    else:
        monthly_agg_spend = pd.DataFrame(columns=["Month", "Marketing Spend"])

    merged = pd.merge(merged, monthly_agg_spend, on="Month", how="left").fillna(0)

    def calc_cac(row):
        if row["NewCustomers"] > 0:
            return row["Marketing Spend"] / row["NewCustomers"]
        return 0
    merged["CAC"] = merged.apply(calc_cac, axis=1).round(2)
    merged["Month"] = merged["Month"].astype(str)

    fig = go.Figure()

    # new
    fig.add_trace(go.Bar(
        x=merged["Month"],
        y=merged["NewCustomers"],
        name="New Customers",
        marker=dict(color="#386092"),
        yaxis="y",
        hovertemplate="Month: %{x}<br>New Customers: %{y:,}<extra></extra>"
    ))

    # repeat
    fig.add_trace(go.Bar(
        x=merged["Month"],
        y=merged["RepeatCustomers"],
        name="Repeat Customers",
        marker=dict(color="#9a3936"),
        yaxis="y",
        hovertemplate="Month: %{x}<br>Repeat Customers: %{y:,}<extra></extra>"
    ))

    # CAC line
    fig.add_trace(go.Scatter(
        x=merged["Month"],
        y=merged["CAC"],
        name="CAC",
        mode="lines+markers",
        line=dict(color="white"),
        yaxis="y2",
        hovertemplate="Month: %{x}<br>CAC: %{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        barmode="group",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=14),
        height=600,
        xaxis=dict(type="category", title="Month"),
        yaxis=dict(title="Number of Customers"),
        yaxis2=dict(title="CAC (USD)", overlaying="y", side="right"),
        legend=dict(x=1.02, y=1.0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Subtitle placeholder text
    st.markdown("<p style='color:white; font-size:12px;'>subtitle placeholder text</p>", unsafe_allow_html=True)

    # Download button if purchased
    if purchased:
        csv_data = merged.to_csv(index=False)
        st.download_button("Download CSV", data=csv_data, file_name="new_vs_repeat_cac.csv", mime="text/csv")

def plot_percent_multiple_orders(order_df: pd.DataFrame, purchased: bool = False):
    """
    (Chart) Horizontal bar chart for # of orders: 1, 2, 3 (cap at 3).
    """
    if order_df.empty:
        st.write("No data for 'Percent of Customers Making Multiple Orders'.")
        return

    st.markdown(
        "<p style='color:white;font-size:14px;margin-top:-10px;'>"
        "% increases b/c this is calculated as the percentage of users who made two orders "
        "that make three orders, etc. This tells you how likely someone is to stick around "
        "as they make more orders."
        "</p>", unsafe_allow_html=True
    )

    df = order_df.copy()
    df["CustOrderRank"] = df.groupby("Customer ID")["Date"].rank(method="first")
    max_orders_by_cust = df.groupby("Customer ID")["CustOrderRank"].max().reset_index()
    max_orders_by_cust["TotalOrders"] = max_orders_by_cust["CustOrderRank"].astype(int)

    # Cap at 3
    max_orders_by_cust.loc[max_orders_by_cust["TotalOrders"] > 3, "TotalOrders"] = 3

    dist = max_orders_by_cust["TotalOrders"].value_counts().sort_index()

    total_customers = len(max_orders_by_cust)
    results = []

    # 1 order => force 100% of total
    c1 = total_customers
    p1 = 100.0
    results.append((1, c1, p1))

    c2 = dist.get(2, 0)
    p2 = (100.0 * c2 / c1) if c1 > 0 else 0
    results.append((2, c2, p2))

    c3 = dist.get(3, 0)
    p3 = (100.0 * c3 / c2) if c2 > 0 else 0
    results.append((3, c3, p3))

    df_plot = pd.DataFrame(results, columns=["OrderN", "Count", "PctRelative"]).sort_values("OrderN", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot["Count"],
        y=df_plot["OrderN"].astype(str),
        orientation="h",
        marker=dict(color="#9a3936"),
        text="",
        name="Order Count",
        customdata=df_plot["PctRelative"],
        hovertemplate=(
            "Number of customers: %{x:,}<br>"
            "Percent of customers: %{customdata:.1f}%<extra></extra>"
        )
    ))

    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=14),
        height=500,
        xaxis=dict(visible=False),
        yaxis=dict(title="Number of Orders", tickmode="array", tickvals=[1,2,3], autorange="reversed"),
        showlegend=False
    )

    # Annotate each bar
    for i, row in df_plot.iterrows():
        y_val = str(row["OrderN"])
        x_val = row["Count"]
        x_str = f"{x_val:,.0f}"
        pct_val = row["PctRelative"]
        annotation_text = f"{x_str} | {pct_val:.1f}%"
        fig.add_annotation(
            x=x_val,
            y=y_val,
            text=annotation_text,
            xanchor="left",
            showarrow=False,
            font=dict(color="white")
        )

    st.plotly_chart(fig, use_container_width=True)

    # Subtitle placeholder text
    st.markdown("<p style='color:white; font-size:12px;'>subtitle placeholder text</p>", unsafe_allow_html=True)

    # Download button if purchased
    if purchased:
        csv_data = df_plot.to_csv(index=False)
        st.download_button("Download CSV", data=csv_data, file_name="percent_multiple_orders.csv", mime="text/csv")

def plot_time_between_first_and_second_purchase(order_df: pd.DataFrame, purchased: bool = False):
    """
    (Chart) Column chart for % of repeat orders that occur by X day, plus 
    cumulative line. 
    """
    if order_df.empty:
        st.write("No data for 'Time Between First and Second Purchase'.")
        return

    df = order_df.copy()
    df.sort_values(["Customer ID", "Date"], inplace=True)
    df["OrderIndex"] = df.groupby("Customer ID").cumcount() + 1
    second_orders_df = df[df["OrderIndex"] == 2].copy()

    if second_orders_df.empty:
        st.write("No customers have a 2nd purchase in this dataset.")
        return

    df_first = df[df["OrderIndex"] == 1].copy()
    df_first.rename(columns={"Date": "FirstPurchaseDate"}, inplace=True)
    second_orders_df.rename(columns={"Date": "SecondPurchaseDate"}, inplace=True)

    merged = pd.merge(
        second_orders_df[["Customer ID", "SecondPurchaseDate"]],
        df_first[["Customer ID", "FirstPurchaseDate"]],
        on="Customer ID",
        how="inner"
    )
    merged["DiffDays"] = (merged["SecondPurchaseDate"] - merged["FirstPurchaseDate"]).dt.days
    merged = merged[merged["DiffDays"] >= 0]

    bins = [0, 7, 15, 30, 60, 90, 120, 150, 179, 365]
    labels = ["0-7", "8-15", "16-30", "31-60", "61-90", "91-120", "121-150", "151-179", "180-365"]
    merged["DayBin"] = pd.cut(merged["DiffDays"], bins=bins, labels=labels, right=True)
    bin_counts = merged["DayBin"].value_counts().reindex(labels, fill_value=0)
    total_2nd_orders = len(merged)

    percent_per_bin = (bin_counts / total_2nd_orders) * 100.0
    percent_per_bin = percent_per_bin.round(2)
    cumulative_percent = percent_per_bin.cumsum().round(2)

    df_plot = pd.DataFrame({
        "DayRange": labels,
        "% of repeat orders": percent_per_bin.values,
        "Cumulative % of repeat orders": cumulative_percent.values
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot["DayRange"],
        y=df_plot["% of repeat orders"],
        name="% of repeat orders",
        marker=dict(color="#386092"),
        yaxis="y1",
        hovertemplate=(
            "Days since first purchase: %{x}<br>"
            "% of repeat orders occurring: %{y:.2f}%<extra></extra>"
        )
    ))
    fig.add_trace(go.Scatter(
        x=df_plot["DayRange"],
        y=df_plot["Cumulative % of repeat orders"],
        name="Cumulative % of repeat orders",
        mode="lines+markers",
        line=dict(color="white"),
        yaxis="y2",
        hovertemplate=(
            "Days since first purchase: %{x}<br>"
            "Cumulative % of repeat orders by this point: %{y:.2f}%<extra></extra>"
        )
    ))

    fig.update_layout(
        barmode="group",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=14),
        height=600,
        xaxis=dict(title="Days Since First Purchase"),
        yaxis=dict(title="% of repeat orders", range=[0, 110]),
        yaxis2=dict(title="Cumulative % of repeat orders", overlaying="y", side="right", range=[0, 110]),
        legend=dict(x=1.02, y=1.0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Subtitle placeholder text
    st.markdown("<p style='color:white; font-size:12px;'>subtitle placeholder text</p>", unsafe_allow_html=True)

    # Download button if purchased
    if purchased:
        csv_data = df_plot.to_csv(index=False)
        st.download_button("Download CSV", data=csv_data, file_name="time_between_first_second.csv", mime="text/csv")

################################################################################
# (L) PURCHASE LINKS FUNCTIONS
################################################################################
# [No direct text shown here, replaced in partial gating code below]

################################################################################
# (M) RENDER CHARTS WITH PARTIAL GATING
################################################################################

def plot_charts_with_partial_gating(avg_ltv_full: pd.DataFrame,
                                    ret_df_full: pd.DataFrame,
                                    order_data: pd.DataFrame):
    """
    Renders charts based on purchased items. freq is fixed at Monthly now.
    The new partial gating text is:
       "Only displaying 6 months of data. Get this chart with your full dataset for $5. [Unlock Full Chart]"
       "Get all 9 charts with your full dataset for $20 (SAVE 55%). [Unlock All Charts]"
    """
    avg_ltv_limited = limit_to_6_months_ltv(avg_ltv_full)
    ret_df_limited  = limit_to_6_months_ret(ret_df_full)

    # Check if user has "Unlock All Charts"
    full_unlock_purchased = ("Unlock All Charts" in st.session_state["purchased_items"])

    # The 9 charts we care about (see user-provided language)
    for item in ITEMS_FOR_SALE:
        # Skip the "Unlock All Charts" iteration
        if item["name"] == "Unlock All Charts":
            continue

        st.write(f"### {item['name']}")
        has_item = (item["name"] in st.session_state["purchased_items"])

        # Add subtitle placeholder text
        st.markdown("<p style='color:white; font-size:12px;'>subtitle placeholder text</p>", unsafe_allow_html=True)

        # Decide which dataset to plot (full or partial)
        purchased = (full_unlock_purchased or has_item)
        if purchased:
            df_ltv = avg_ltv_full
            df_ret = ret_df_full
        else:
            df_ltv = avg_ltv_limited
            df_ret = ret_df_limited

        # Dispatch each chart
        if item["name"] == "Average Lifetime Value (LTV) by Cohort (Heatmap)":
            plot_ltv_heatmap(df_ltv, purchased=purchased)

        elif item["name"] == "Average Retention by Cohort (Heatmap)":
            plot_retention_heatmap_with_size(df_ret, df_ltv, purchased=purchased)

        elif item["name"] == "LTV:CAC Ratio by Cohort":
            plot_ltv_cac_by_cohort(df_ltv, purchased=purchased)

        elif item["name"] == "CAC & New Customers vs. Repeat Customers":
            # For partial gating, we limit to first 3 months
            if purchased:
                plot_cac_new_vs_repeat_all_data(order_data, st.session_state["marketing_spend_df"], purchased=True)
            else:
                tmp_orders = order_data.copy()
                tmp_orders["Month"] = tmp_orders["Date"].dt.to_period("M")
                unique_months = sorted(tmp_orders["Month"].unique())
                keep_months = unique_months[:3]  # first 3 months
                tmp_orders = tmp_orders[tmp_orders["Month"].isin(keep_months)]
                tmp_spend = st.session_state["marketing_spend_df"].copy()
                if not tmp_spend.empty:
                    tmp_spend["Month"] = tmp_spend["Date"].dt.to_period("M")
                    tmp_spend = tmp_spend[tmp_spend["Month"].isin(keep_months)]
                plot_cac_new_vs_repeat_all_data(tmp_orders, tmp_spend, purchased=False)

        elif item["name"] == "Percent of Customers Making Multiple Orders":
            if purchased:
                plot_percent_multiple_orders(order_data, purchased=True)
            else:
                tmp = order_data.copy()
                tmp["OrderIndex"] = tmp.groupby("Customer ID").cumcount() + 1
                tmp.loc[tmp["OrderIndex"] > 3, "OrderIndex"] = 3
                plot_percent_multiple_orders(tmp, purchased=False)

        elif item["name"] == "Time Between First and Second Purchase":
            if purchased:
                plot_time_between_first_and_second_purchase(order_data, purchased=True)
            else:
                tmp = order_data.copy()
                tmp.sort_values(["Customer ID", "Date"], inplace=True)
                tmp["OrderIndex"] = tmp.groupby("Customer ID").cumcount() + 1
                df_first = tmp[tmp["OrderIndex"] == 1][["Customer ID", "Date"]].rename(columns={"Date": "FirstPurchaseDate"})
                df_second = tmp[tmp["OrderIndex"] == 2][["Customer ID", "Date"]].rename(columns={"Date": "SecondPurchaseDate"})
                merged = pd.merge(df_second, df_first, on="Customer ID", how="inner")
                merged["DiffDays"] = (merged["SecondPurchaseDate"] - merged["FirstPurchaseDate"]).dt.days
                allowed_customers = merged[merged["DiffDays"] <= 30]["Customer ID"].unique()
                tmp = tmp[tmp["Customer ID"].isin(allowed_customers)]
                plot_time_between_first_and_second_purchase(tmp, purchased=False)

        elif item["name"] == "Cohort Size vs. Average Lifetime Value (LTV) by Cohort":
            plot_cohort_size_and_ltv_analysis_chart(df_ltv, purchased=purchased)

        elif item["name"] == "Average Lifetime Value (LTV) by Cohort (Line)":
            plot_cohort_lifetime_value_line_chart(df_ltv, purchased=purchased)

        elif item["name"] == "Average Retention by Cohort (Line)":
            plot_percent_retained_line_chart(df_ret, purchased=purchased)

        else:
            # If it's something we removed or doesn't match, skip
            pass

        # Show partial gating text if user does NOT have the item & hasn't purchased all charts
        if not purchased:
            single_chart_link = create_checkout_session(
                [item],
                st.session_state["upload_id"],
                st.session_state["order_data_filename"],
                st.session_state["marketing_spend_filename"]
            )
            full_item = next((x for x in ITEMS_FOR_SALE if x["name"] == "Unlock All Charts"), None)
            full_link = create_checkout_session(
                [full_item],
                st.session_state["upload_id"],
                st.session_state["order_data_filename"],
                st.session_state["marketing_spend_filename"]
            )

            # Unified partial gating message for all charts
            if single_chart_link and full_link:
                st.markdown(f"""
                <p style="font-size:14px; color:white; margin-top:4px;">
                    Only displaying 6 months of data. Get this chart with your full dataset for $5. 
                    <a href="{single_chart_link}" target="_blank" style="color:#87CEFA; text-decoration:underline;">
                        [Unlock Full Chart]
                    </a>
                    <br/>
                    Get all 9 charts with your full dataset for $20 (SAVE 55%). 
                    <a href="{full_link}" target="_blank" style="color:#87CEFA; text-decoration:underline;">
                        [Unlock All Charts]
                    </a>
                </p>
                """, unsafe_allow_html=True)

        st.markdown("---")

################################################################################
# (N) MAIN APPLICATION LOGIC
################################################################################

def main():
    # Make the SKU filter background dark green (#1b3929) and add border
    st.markdown("""
    <style>
    div[data-baseweb="select"] {
        background-color: #1b3929 !important;
        border: 2px solid #ffffff !important;  /* Added border */
        border-radius: 5px !important;
    }
    div[data-baseweb="select"] * {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Handle Stripe payment callbacks
    handle_payment()

    # If returning from Stripe with a successful payment, refresh data
    if st.session_state.get("show_refresh_spinner", False):
        with st.spinner("Payment successful! Just a second while we refresh your chart(s) with the full data set..."):
            # Re-download Order Data if not present
            if st.session_state["order_data_df"].empty and st.session_state["order_data_filename"]:
                file_bytes = download_file_from_s3(st.session_state["upload_id"], st.session_state["order_data_filename"])
                if file_bytes:
                    try:
                        if st.session_state["order_data_filename"].lower().endswith(".csv"):
                            raw_order = pd.read_csv(io.BytesIO(file_bytes))
                        else:
                            xls = pd.ExcelFile(io.BytesIO(file_bytes))
                            needed_cols = ["purchase-date","buyer-email","item-price","sku","amazon-order-id"]
                            found_sheet = None
                            for sheet in xls.sheet_names:
                                temp_df = xls.parse(sheet)
                                if all(col in temp_df.columns for col in needed_cols):
                                    raw_order = temp_df
                                    found_sheet = sheet
                                    break
                            if not found_sheet:
                                st.error(f"No sheet found with required columns: {needed_cols}")
                                st.stop()
                        st.session_state["order_data_df"] = preprocess_order_data(raw_order)
                        st.success("Order Data reloaded successfully!")
                    except Exception as ex:
                        st.error(f"Error re-downloading order data after payment: {ex}")

            # Re-download Marketing Spend if not present and filename exists
            if st.session_state["marketing_spend_df"].empty and st.session_state["marketing_spend_filename"]:
                file_bytes = download_file_from_s3(st.session_state["upload_id"], st.session_state["marketing_spend_filename"])
                if file_bytes:
                    try:
                        if st.session_state["marketing_spend_filename"].lower().endswith(".csv"):
                            raw_spend = pd.read_csv(io.BytesIO(file_bytes))
                        else:
                            ms_xl = pd.ExcelFile(io.BytesIO(file_bytes))
                            raw_spend = ms_xl.parse(ms_xl.sheet_names[0])
                        st.session_state["marketing_spend_df"] = preprocess_marketing_spend(raw_spend)
                        st.success("Marketing Spend reloaded successfully!")
                    except Exception as ex:
                        st.error(f"Error re-downloading marketing spend after payment: {ex}")

        st.session_state["show_refresh_spinner"] = False

    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.markdown("""
            <h2 style='color:white; margin-bottom:4px;'>
                Amazon Customer Analytics Engine
            </h2>
        """, unsafe_allow_html=True)

        # --- ADD IMAGE (fills whole space) ---
        st.markdown("""
            <a href="https://freeimage.host/i/2r7et14">
                <img src="https://iili.io/2r7et14.md.png" 
                     alt="2r7et14.md.png" 
                     border="0" 
                     style="width: 100%;" />
            </a>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown("<div style='padding-top:25%;'>", unsafe_allow_html=True)  # Increased padding-top for more space

        # Upload Order Data if not already uploaded
        if st.session_state["order_data_df"].empty:
            st.markdown("""
                <p style="font-size:16px; font-weight:bold; margin-bottom:4px;">
                    UPLOAD YOUR ORDER DATA (CSV OR XLSX)
                </p>
            """, unsafe_allow_html=True)
            order_data_file = st.file_uploader(
                "Order Data Upload",
                type=["csv", "xlsx"],
                label_visibility="collapsed"
            )
            st.markdown("""
                <p style="font-size:14px;">
                    <strong>Required Columns:</strong><br/>
                    <strong>1:</strong> purchase-date<br/>
                    <strong>2:</strong> buyer-email<br/>
                    <strong>3:</strong> item-price<br/>
                    <strong>4:</strong> sku<br/>
                    <strong>5:</strong> amazon-order-id<br/>
                    Extra columns are allowed; we only need the ones above.
                </p>
            """, unsafe_allow_html=True)
        else:
            order_data_file = None
            st.success("Order Data is already in session. Ready for analysis!")

        # Upload Marketing Spend if not already uploaded
        if st.session_state["marketing_spend_df"].empty:
            st.markdown("""
                <p style="font-size:16px; font-weight:bold; margin:12px 0 4px;">
                    UPLOAD YOUR MARKETING SPEND (CSV OR XLSX)
                </p>
            """, unsafe_allow_html=True)
            marketing_spend_file = st.file_uploader(
                "Marketing Spend Upload",
                type=["csv", "xlsx"],
                key="marketing_spend_file",
                label_visibility="collapsed"
            )
            st.markdown("""
                <p style="font-size:14px;">
                    <strong>Required Columns:</strong><br/>
                    <strong>1:</strong> Date<br/>
                    <strong>2:</strong> Marketing Spend (daily)
                </p>
            """, unsafe_allow_html=True)
        else:
            marketing_spend_file = None
            st.info("Marketing Spend is already in session.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Handle Order Data Upload
    if order_data_file is not None:
        with st.spinner("Parsing and uploading Order Data..."):
            try:
                file_bytes = order_data_file.getvalue()
                filename = order_data_file.name
                success_up = upload_file_to_s3(st.session_state["upload_id"], file_bytes, filename)
                if not success_up:
                    st.stop()
                st.session_state["order_data_filename"] = filename

                # Parse the uploaded file
                if filename.lower().endswith(".csv"):
                    raw_order = pd.read_csv(io.BytesIO(file_bytes))
                else:
                    xls = pd.ExcelFile(io.BytesIO(file_bytes))
                    needed_cols = ["purchase-date","buyer-email","item-price","sku","amazon-order-id"]
                    found_sheet = None
                    for sheet in xls.sheet_names:
                        temp_df = xls.parse(sheet)
                        if all(col in temp_df.columns for col in needed_cols):
                            raw_order = temp_df
                            found_sheet = sheet
                            break
                    if not found_sheet:
                        st.error(f"No sheet found with required columns: {needed_cols}")
                        st.stop()

                st.session_state["order_data_df"] = preprocess_order_data(raw_order)
                st.success("Order Data uploaded and parsed successfully!")
            except Exception as ex:
                st.error(f"Failed to parse Order Data: {ex}")
                st.stop()

    # Handle Marketing Spend Upload
    if marketing_spend_file is not None:
        with st.spinner("Parsing and uploading Marketing Spend..."):
            try:
                file_bytes = marketing_spend_file.getvalue()
                filename = marketing_spend_file.name
                success_up = upload_file_to_s3(st.session_state["upload_id"], file_bytes, filename)
                if not success_up:
                    st.stop()
                st.session_state["marketing_spend_filename"] = filename

                # Parse the uploaded file
                if filename.lower().endswith(".csv"):
                    raw_spend = pd.read_csv(io.BytesIO(file_bytes))
                else:
                    ms_xl = pd.ExcelFile(io.BytesIO(file_bytes))
                    raw_spend = ms_xl.parse(ms_xl.sheet_names[0])
                st.session_state["marketing_spend_df"] = preprocess_marketing_spend(raw_spend)
                st.success("Marketing Spend uploaded and parsed successfully!")
            except Exception as ex:
                st.error(f"Failed to parse Marketing Spend: {ex}")

    # Access the data
    order_data = st.session_state["order_data_df"]
    marketing_spend_df = st.session_state["marketing_spend_df"]

    if order_data.empty:
        st.info("Please upload your Order Data to proceed.")
        st.stop()

    # Preview of Order Data
    needed_cols = ["Date","Customer ID","Order Total","SKU","Order ID"]
    preview_cols = [c for c in needed_cols if c in order_data.columns]
    preview_df = order_data[preview_cols].copy()

    st.markdown("<p style='color:white;font-size:14px;'><strong>Preview of Order Data (Filtered):</strong></p>", unsafe_allow_html=True)
    st.dataframe(preview_df, height=300)

    # Removed the "Download CSV" button beneath Preview of Order Data (Filtered)

    # Extra spacing
    st.markdown("<br/><br/>", unsafe_allow_html=True)

    # Filter by SKU - now placed BELOW the preview with more spacing
    if "SKU" in order_data.columns:
        st.markdown("<div style='border: 2px solid #ffffff; padding: 10px; border-radius:5px;'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:14px; color:white;'><strong>Filter by SKU</strong></p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:12px; color:white;'>Filter your charts by specific SKU</p>", unsafe_allow_html=True)
        unique_skus = sorted(order_data["SKU"].unique())
        sku_options = ["All"] + unique_skus
        selected_sku = st.selectbox("", sku_options, index=0)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        selected_sku = None
        st.info("Add a column for SKU to your CSV to activate SKU filter.")

    # Filter data if a SKU is chosen
    if selected_sku and selected_sku != "All":
        order_data = order_data[order_data["SKU"] == selected_sku].copy()

    if order_data.empty:
        st.warning("No data found for the selected SKU. Please choose another SKU or re-upload.")
        st.stop()

    freq = "Monthly"

    # Compute Metrics
    with st.spinner("Computing metrics..."):
        try:
            avg_ltv = calculate_avg_ltv(order_data, freq, marketing_spend_df)
            ret_df = calculate_percent_retained(order_data, freq)
        except Exception as ex:
            st.error(f"An unexpected error occurred during metric calculations: {ex}")
            st.stop()

    # Render Charts with Partial Gating
    plot_charts_with_partial_gating(avg_ltv, ret_df, order_data)

################################################################################
# (O) ADDITIONAL HELPER CODE / EXAMPLES
################################################################################

class AdvancedDataTransformer:
    """
    A complex class for advanced transformations on the order data and marketing spend data.
    Includes methods for filling missing dates, removing outliers, and more.
    """
    def __init__(self, order_data: pd.DataFrame, spend_data: pd.DataFrame):
        self.order_data = order_data
        self.spend_data = spend_data
        self._debug_info = []

    def fill_missing_dates_with_zeros(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        if df.empty or date_col not in df.columns:
            return df
        min_d = df[date_col].min()
        max_d = df[date_col].max()
        if not isinstance(min_d, datetime) or not isinstance(max_d, datetime):
            return df
        date_range = pd.date_range(start=min_d, end=max_d, freq='D')
        df = df.set_index(date_col).reindex(date_range).fillna(0).reset_index()
        df.rename(columns={"index": date_col}, inplace=True)
        return df

    def remove_outliers(self, df: pd.DataFrame, col: str, z_threshold=3.0) -> pd.DataFrame:
        if df.empty or col not in df.columns:
            return df
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val == 0 or np.isnan(std_val):
            return df
        z_scores = (df[col] - mean_val) / std_val
        mask = np.abs(z_scores) <= z_threshold
        removed = len(df) - mask.sum()
        self._debug_info.append(f"remove_outliers: removed {removed} rows with z > {z_threshold}")
        return df[mask].copy()

    def transform_order_data(self) -> pd.DataFrame:
        df = self.fill_missing_dates_with_zeros(self.order_data, "Date")
        df = self.remove_outliers(df, "Order Total", z_threshold=4.0)
        return df

    def transform_spend_data(self) -> pd.DataFrame:
        df = self.fill_missing_dates_with_zeros(self.spend_data, "Date")
        df = self.remove_outliers(df, "Marketing Spend", z_threshold=5.0)
        return df

    def debug_info(self) -> List[str]:
        return self._debug_info

def advanced_example_usage_of_transformer():
    if st.session_state["order_data_df"].empty:
        st.write("No order data for advanced transformer example.")
        return
    if st.session_state["marketing_spend_df"].empty:
        st.write("No spend data for advanced transformer example.")
        return
    transformer = AdvancedDataTransformer(st.session_state["order_data_df"], st.session_state["marketing_spend_df"])
    transformed_order = transformer.transform_order_data()
    transformed_spend = transformer.transform_spend_data()
    st.write("Transformations complete. Debug info:", transformer.debug_info())
    st.dataframe(transformed_order.head(10))
    st.dataframe(transformed_spend.head(10))

def advanced_bfs_example(graph: Dict[Any, List[Any]], start: Any) -> List[Any]:
    visited = []
    queue = [start]
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            neighbors = graph.get(node, [])
            for n in neighbors:
                if n not in visited:
                    queue.append(n)
    return visited

def concurrency_simulation_example():
    lock = threading.Lock()
    data_list = []

    def worker(i: int):
        with lock:
            data_list.append(i)

    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    st.write("Concurrency simulation example complete:", data_list)

def repeated_expansion_block(number: int) -> int:
    if number < 0:
        raise ValueError("No factorial for negative numbers.")
    if number <= 1:
        return 1
    fact = 1
    for i in range(2, number + 1):
        fact *= i
    return fact

def repeated_expansion_usage():
    results = []
    for x in range(1, 11):
        val = repeated_expansion_block(x)
        results.append(val)
    st.write("Factorials 1..10:", results)

def repeated_expansion_block_v2(number: int) -> int:
    return math.factorial(number)

def repeated_expansion_usage_v2():
    st.write("Using repeated_expansion_block_v2 for factorials 1..5:")
    results = []
    for x in range(1, 6):
        val = repeated_expansion_block_v2(x)
        results.append(val)
    st.write(results)

def advanced_debug_logging_example(order_data: pd.DataFrame) -> None:
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("cohort_debugger")
    logger.debug("advanced_debug_logging_example invoked.")
    if order_data.empty:
        logger.debug("order_data is empty.")
    else:
        logger.debug(f"order_data has {len(order_data)} rows, columns: {list(order_data.columns)}")

def advanced_cli_args_example():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        st.write(f"Advanced CLI arg example, got arg: {arg}")

def advanced_bfs_v2(graph: Dict[Any, List[Any]], start: Any) -> List[Any]:
    visited = []
    queue = [(start, 0)]
    result = []
    while queue:
        node, dist = queue.pop(0)
        if node not in visited:
            visited.append(node)
            result.append((node, dist))
            neighbors = graph.get(node, [])
            for n in neighbors:
                if n not in visited:
                    queue.append((n, dist + 1))
    return result

def advanced_bfs_v3(graph: Dict[Any, List[Any]], start: Any) -> List[List[Any]]:
    visited = set()
    queue = [start]
    layering = []
    while queue:
        level_size = len(queue)
        level_nodes = []
        for _ in range(level_size):
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                level_nodes.append(node)
                neighbors = graph.get(node, [])
                for n in neighbors:
                    if n not in visited:
                        queue.append(n)
        if level_nodes:
            layering.append(level_nodes)
    return layering

def advanced_bfs_with_random_cutoffs(graph: Dict[Any, List[Any]], start: Any, cutoff: int) -> List[Any]:
    visited = []
    queue = [(start, 0)]
    while queue:
        node, depth = queue.pop(0)
        if node not in visited:
            visited.append(node)
            if depth < cutoff:
                neighbors = graph.get(node, [])
                for n in neighbors:
                    if n not in visited:
                        queue.append((n, depth + 1))
    return visited

def advanced_bfs_usage_with_cutoffs():
    st.write("advanced_bfs_usage_with_cutoffs example:")
    sample_graph = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["E", "F"],
        "D": [],
        "E": ["G"],
        "F": [],
        "G": []
    }
    visited_nodes = advanced_bfs_with_random_cutoffs(sample_graph, "A", 2)
    st.write("Visited up to cutoff=2:", visited_nodes)

def advanced_time_segmentations(df: pd.DataFrame, time_col: str, seg: str = "month") -> pd.DataFrame:
    if df.empty or time_col not in df.columns:
        return df
    df = df.copy()
    if seg == "month":
        df["Month"] = df[time_col].dt.to_period("M")
    elif seg == "week":
        df["Week"] = df[time_col].dt.to_period("W")
    elif seg == "year":
        df["Year"] = df[time_col].dt.year
    return df

def concurrency_simulation_example_v2():
    lock = threading.Lock()
    results = []

    def random_task(i: int):
        with lock:
            value = random.randint(1, 100) * i
            results.append((i, value))

    threads = []
    for i in range(5):
        t = threading.Thread(target=random_task, args=(i + 1,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    st.write("concurrency_simulation_example_v2 results:", results)

def giant_data_merge_example(dfs: List[pd.DataFrame], on_col: str = "Date") -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on=on_col, how="outer")
    merged.sort_values(on_col, inplace=True)
    return merged

def advanced_recursion_example(n: int) -> int:
    if n <= 0:
        return 0
    return n + advanced_recursion_example(n - 1)

def advanced_quicksort_example(lst: List[int]) -> List[int]:
    if len(lst) < 2:
        return lst
    pivot = lst[0]
    lesser = [x for x in lst[1:] if x <= pivot]
    greater = [x for x in lst[1:] if x > pivot]
    return advanced_quicksort_example(lesser) + [pivot] + advanced_quicksort_example(greater)

def advanced_mergesort_example(lst: List[int]) -> List[int]:
    if len(lst) <= 1:
        return lst
    mid = len(lst) // 2
    left = advanced_mergesort_example(lst[:mid])
    right = advanced_mergesort_example(lst[mid:])
    return merge_two_sorted_lists(left, right)

def merge_two_sorted_lists(left: List[int], right: List[int]) -> List[int]:
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def advanced_lru_cache_demo(n: int) -> int:
    @functools.lru_cache(None)
    def fib(x: int) -> int:
        if x < 2:
            return x
        return fib(x - 1) + fib(x - 2)
    return fib(n)

def advanced_lru_cache_usage():
    st.write("Fibonacci 0..10 with LRU Cache:")
    res = []
    for i in range(11):
        val = advanced_lru_cache_demo(i)
        res.append(val)
    st.write(res)

def advanced_dataclass_demo():
    from dataclasses import dataclass
    @dataclass
    class CustomerPurchase:
        date: datetime
        customer_id: str
        order_total: float

    sample = CustomerPurchase(datetime.now(), "CUST-1234", 99.99)
    st.write("Dataclass sample instance:", sample)

def repeated_expansion_block_v3(number: int) -> int:
    def tail_fact(n, acc=1):
        if n <= 1:
            return acc
        return tail_fact(n - 1, acc * n)
    return tail_fact(number, 1)

def repeated_expansion_usage_v3():
    st.write("Using repeated_expansion_block_v3 for factorials 1..5 (tail recursion):")
    results = []
    for x in range(1, 6):
        val = repeated_expansion_block_v3(x)
        results.append(val)
    st.write(results)

def repeated_expansion_block_v4(number: int) -> int:
    if number < 0:
        raise ValueError("Cannot compute factorial for negative numbers.")
    product = 1
    for i in range(number, 0, -1):
        product *= i
    return product

def repeated_expansion_usage_v4():
    st.write("Using repeated_expansion_block_v4 for factorials 6..10 (reverse approach):")
    results = []
    for x in range(6, 11):
        val = repeated_expansion_block_v4(x)
        results.append(val)
    st.write(results)

def factorial_bfs(graph: Dict[Any, List[Any]], start: Any) -> int:
    visited = []
    queue = [start]
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            neighbors = graph.get(node, [])
            for n in neighbors:
                if n not in visited:
                    queue.append(n)
    count = len(visited)
    return repeated_expansion_block(count)

################################################################################
# (P) RUNNING THE MAIN FUNCTION
################################################################################

if __name__ == "__main__":
    main()
