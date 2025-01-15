################################################################################
# (1) COHORT ANALYSIS TOOL - PARTIAL GATING WITH STRIPE & AWS S3
#     Complete codebase with recent adjustments:
#       - Title font slightly reduced
#       - Background color reverted to previous default (no #034546)
#       - Fixed extra “%%” in Retention chart
#       - Partial gating for multiple orders chart
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
import re
import zipfile

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
BUCKET_NAME = st.secrets["S3_BUCKET_NAME"]

################################################################################
# (C) ITEMS_FOR_SALE
################################################################################
ITEMS_FOR_SALE = [
    {"name": "Average Lifetime Value (LTV) by Cohort (Heatmap)", "price": 5},
    {"name": "Average Retention by Cohort (Heatmap)", "price": 5},
    {"name": "LTV:CAC Ratio by Cohort", "price": 5},
    {"name": "CAC & New Customers vs. Repeat Customers", "price": 5},
    {"name": "Customer Retention During First 5 Orders", "price": 5},
    {"name": "Time Between First and Second Purchase", "price": 5},
    {"name": "What Customers Buy After Their First Order", "price": 5},
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

if "order_data_filename" not in st.session_state:
    st.session_state["order_data_filename"] = None
if "marketing_spend_filename" not in st.session_state:
    st.session_state["marketing_spend_filename"] = None

if "order_data_df" not in st.session_state:
    st.session_state["order_data_df"] = pd.DataFrame()
if "marketing_spend_df" not in st.session_state:
    st.session_state["marketing_spend_df"] = pd.DataFrame()

if "upload_id" not in st.session_state:
    st.session_state["upload_id"] = str(uuid.uuid4())

# Store invalid date rows for debugging (optional)
if "invalid_date_rows_orders" not in st.session_state:
    st.session_state["invalid_date_rows_orders"] = pd.DataFrame()
if "invalid_date_rows_spend" not in st.session_state:
    st.session_state["invalid_date_rows_spend"] = pd.DataFrame()

# Store invalid SKU rows (optional debug)
if "invalid_sku_rows" not in st.session_state:
    st.session_state["invalid_sku_rows"] = pd.DataFrame()

################################################################################
# (E) STRIPE CHECKOUT CREATION & PAYMENT HANDLING
################################################################################

BASE_URL = "https://sellercentral.streamlit.app/"  # Replace with your actual URL

SUCCESS_URL_TEMPLATE = f"{BASE_URL}/?success=true&session_id={{CHECKOUT_SESSION_ID}}&upload_id="
CANCEL_URL_TEMPLATE  = f"{BASE_URL}/?canceled=true&session_id={{CHECKOUT_SESSION_ID}}&upload_id="

def create_checkout_session(
    item_list: List[Dict[str, Any]],
    upload_id: str,
    order_filename: Optional[str],
    spend_filename: Optional[str]
) -> Optional[str]:
    """
    Creates a Stripe Checkout Session. We also embed the user's current
    purchased_items into 'prev_purchases' so we can restore them after the next purchase.
    """
    success_url = SUCCESS_URL_TEMPLATE + upload_id
    cancel_url  = CANCEL_URL_TEMPLATE + upload_id

    # If user already purchased charts, let's pass them along so we don't lose them.
    prev_purchases_str = ",".join(st.session_state["purchased_items"])

    # Append them to success/cancel URLs
    success_url += f"&prev_purchases={prev_purchases_str}"
    cancel_url  += f"&prev_purchases={prev_purchases_str}"

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
                "unit_amount": it["price"] * 100,
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
    qp = st.query_params

    success = qp.get("success", ["false"])
    canceled = qp.get("canceled", ["false"])
    session_id = qp.get("session_id", [""])
    upload_id = qp.get("upload_id", [""])
    odf = qp.get("odf", [None])
    msf = qp.get("msf", [None])
    prev_purchases = qp.get("prev_purchases", [""])

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
    if isinstance(prev_purchases, list):
        prev_purchases = prev_purchases[0]

    # Merge the "prev_purchases" we had from before
    if prev_purchases:
        old_items = prev_purchases.split(",")
        for item_name in old_items:
            if item_name and item_name not in st.session_state["purchased_items"]:
                st.session_state["purchased_items"].append(item_name)

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
                purchased_names = [li.description.strip() for li in line_items.data if li.description]
                for item_name in purchased_names:
                    for it in ITEMS_FOR_SALE:
                        if it["name"].strip() == item_name:
                            if it["name"] not in st.session_state["purchased_items"]:
                                st.session_state["purchased_items"].append(it["name"])
        except Exception as e:
            st.error(f"Error retrieving Stripe checkout session: {e}")
    elif canceled == "true":
        st.warning("Payment canceled.")

################################################################################
# (F) DATA PREPROCESSING - ROBUST DATE-PARSE HELPERS
################################################################################

def robust_date_parse(df: pd.DataFrame, colname: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Attempts to parse df[colname] as datetime, ignoring timezone offsets.
    Returns (df_valid, df_invalid):
      - df_valid: rows with successful date parse
      - df_invalid: rows where date parse failed (NaT)
    """
    df[colname] = df[colname].astype(str).str.strip()
    df[colname] = pd.to_datetime(df[colname], errors="coerce", utc=True, infer_datetime_format=True)
    df[colname] = df[colname].dt.tz_localize(None)

    df_invalid = df[df[colname].isna()].copy()
    df_valid = df.dropna(subset=[colname]).copy()
    return df_valid, df_invalid

################################################################################
# (F) DATA PREPROCESSING FUNCTIONS
################################################################################
@st.cache_data
def preprocess_order_data(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["Purchase Date", "Buyer Email", "Item Price", "Merchant SKU", "Amazon Order Id"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Order Data: {missing}")

    df = df.dropna(subset=needed, how="any").copy()

    rename_map = {
        "Purchase Date": "Date",
        "Buyer Email": "Customer ID",
        "Item Price": "Order Total",
        "Merchant SKU": "SKU",
        "Amazon Order Id": "Order ID"
    }
    df.rename(columns=rename_map, inplace=True)

    valid_df, invalid_df = robust_date_parse(df, "Date")
    if len(invalid_df) > 0:
        st.session_state["invalid_date_rows_orders"] = invalid_df
    df = valid_df

    df["Order Total"] = pd.to_numeric(df["Order Total"], errors="coerce").round(2)
    df = df.dropna(subset=["Order Total"])

    df["Customer ID"] = df["Customer ID"].astype(str).str.strip()
    df = df[df["Customer ID"] != ""]

    df["SKU"] = (
        df["SKU"]
        .astype(str)
        .str.replace(r"[\ufeff]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    blank_skus = df[df["SKU"] == ""]
    if not blank_skus.empty:
        st.session_state["invalid_sku_rows"] = pd.concat([st.session_state["invalid_sku_rows"], blank_skus])
        df = df[df["SKU"] != ""]

    return df

@st.cache_data
def preprocess_marketing_spend(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["Date", "Marketing Spend"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame(columns=needed)

    valid_df, invalid_df = robust_date_parse(df, "Date")
    if len(invalid_df) > 0:
        st.session_state["invalid_date_rows_spend"] = invalid_df

    df = valid_df
    df["Date"] = df["Date"].dt.floor("D")
    df["Marketing Spend"] = pd.to_numeric(df["Marketing Spend"], errors="coerce").fillna(0)
    df = df.groupby("Date")["Marketing Spend"].sum().reset_index()
    return df

################################################################################
# (G) FREQUENCY / INDEX HELPERS
################################################################################

def unify_cohort_label_for_date(d: datetime, freq: str) -> str:
    return d.strftime("%Y-%m")

def date_to_period_index(row, freq: str) -> int:
    yd = row["Date"].year - row["Cohort Date"].year
    md = row["Date"].month - row["Cohort Date"].month
    return yd * 12 + md + 1

def pivot_index(row, freq: str = "Monthly") -> int:
    return date_to_period_index(row, freq)

@st.cache_data
def group_spend_by_freq(spend_df: pd.DataFrame, freq: str) -> Dict[str, float]:
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
        lambda row: (row["Marketing Spend"] / row["Cohort Size"]) if row["Cohort Size"] else np.nan,
        axis=1
    )
    for col in avg_ltv.columns:
        avg_ltv[col] = pd.to_numeric(avg_ltv[col], errors="coerce").round(2)
    avg_ltv = avg_ltv.reset_index()
    return avg_ltv

@st.cache_data
def calculate_percent_retained(order_df: pd.DataFrame, freq: str) -> pd.DataFrame:
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
    key = f"uploads/{upload_id}/{filename}"
    try:
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=file_bytes)
        return True
    except Exception as ex:
        st.error(f"Failed to upload file to S3: {ex}")
        return False

def download_file_from_s3(upload_id: str, filename: str) -> Optional[bytes]:
    key = f"uploads/{upload_id}/{filename}"
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return obj["Body"].read()
    except Exception as ex:
        st.error(f"Failed to download file from S3: {ex}")
        return None

################################################################################
# (J) PARTIAL GATING HELPERS
################################################################################

def limit_to_6_months_ltv(full_df: pd.DataFrame) -> pd.DataFrame:
    if full_df.empty or "Cohort" not in full_df.columns:
        return full_df
    month_cols = [c for c in full_df.columns if c.startswith("Month ")]
    keep_months = month_cols[:6]
    keep_cols = ["Cohort", "Marketing Spend", "Cohort Size", "CAC"] + keep_months
    keep_cols = [x for x in keep_cols if x in full_df.columns]
    return full_df[keep_cols].copy()

def limit_to_6_months_ret(full_df: pd.DataFrame) -> pd.DataFrame:
    if full_df.empty or "Cohort" not in full_df.columns:
        return full_df
    month_cols = [c for c in full_df.columns if c.startswith("Month ")]
    keep_months = month_cols[:6]
    keep_cols = ["Cohort"] + keep_months
    keep_cols = [x for x in keep_cols if x in full_df.columns]
    return full_df[keep_cols].copy()

################################################################################
# (K) CHART PLOTTING UTILS
################################################################################

def _add_session_note():
    st.markdown(
        "<p style='font-size:14px; color:white; margin-top:-1px;'>"
        "NOTE: CHARTS WILL NOT BE SAVED AFTER THIS SESSION. Download your CSV and a png of "
        "the chart so you have a record."
        "</p>",
        unsafe_allow_html=True
    )

def maybe_omit_cac(df: pd.DataFrame, sku_selected: str) -> pd.DataFrame:
    if sku_selected != "All" and "CAC" in df.columns:
        df = df.copy()
        df["CAC"] = np.nan
    return df

def _add_cac_omitted_note_if_needed(sku_selected: str, df: pd.DataFrame):
    if sku_selected != "All" and "CAC" in df.columns:
        st.markdown(
            "<p style='font-size:14px; color:white; margin-top:-10px;'>"
            "Note: CAC is omitted when filtering for a specific SKU. To view chart with CAC figure, set SKU filter to ALL."
            "</p>",
            unsafe_allow_html=True
        )

################################################################################
# (L) CHART PLOTTING FUNCTIONS
################################################################################

def plot_cohort_lifetime_value_line_chart(avg_ltv: pd.DataFrame, purchased: bool) -> bool:
    if avg_ltv.empty or "Cohort" not in avg_ltv.columns:
        st.write("No data available for the LTV line chart.")
        return False

    mon_cols = [c for c in avg_ltv.columns if c.startswith("Month ")]
    if not mon_cols:
        st.write("No month columns found in LTV data.")
        return False

    sorted_mon_cols = sorted(mon_cols, key=lambda x: int(x.split()[1]))

    fig = go.Figure()
    avg_ltv_sorted = avg_ltv.sort_values("Cohort")
    total_cohorts = len(avg_ltv_sorted)

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 128, 0),
        (128, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 128, 128),
        (128, 128, 0)
    ]

    for idx, (_, row) in enumerate(avg_ltv_sorted.iterrows()):
        cohort = row["Cohort"]
        vals = []
        xvals = []
        for mc in sorted_mon_cols:
            val = row[mc]
            if pd.isna(val):
                continue
            month_num = int(mc.split()[1])
            xvals.append(month_num)
            vals.append(val)

        if vals:
            opacity = 0.3 + 0.7 * (idx / total_cohorts)
            line_width = 1 + 2 * (idx / total_cohorts)
            color_idx = idx % len(colors)
            r, g, b = colors[color_idx]
            fig.add_trace(go.Scatter(
                x=xvals,
                y=vals,
                mode="lines+markers",
                line=dict(width=line_width, color=f'rgba({r},{g},{b},{opacity})'),
                name=str(cohort),
                hovertemplate=(
                    f"Cohort: {cohort}<br>"
                    "Month: %{x}<br>"
                    "LTV: %{y:.2f}"
                    "<extra></extra>"
                )
            ))

    data_cols = avg_ltv[sorted_mon_cols]
    means = {}
    for c in sorted_mon_cols:
        col_data = data_cols[c].dropna()
        means[c] = col_data.mean() if len(col_data) > 0 else np.nan

    x_means = []
    y_means = []
    for c in sorted_mon_cols:
        if not pd.isna(means[c]):
            x_means.append(int(c.split()[1]))
            y_means.append(means[c])

    if x_means and y_means:
        fig.add_trace(go.Scatter(
            x=x_means,
            y=y_means,
            mode="lines+markers",
            name="Average (All Cohorts)",
            line=dict(dash="dash", color="white", width=3),
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

    if purchased:
        csv_data = avg_ltv.to_csv(index=False)
        try:
            png_bytes = fig.to_image(format="png", width=1920, height=1080)
        except Exception as e:
            st.error(f"Failed to generate PNG: {e}")
            png_bytes = None

        st.download_button("Download CSV", data=csv_data, file_name="avg_ltv_line.csv", mime="text/csv")
        if png_bytes:
            st.download_button("Download PNG", data=png_bytes, file_name="avg_ltv_line.png", mime="image/png")
        _add_session_note()
        st.session_state["fig_avg_ltv_line"] = fig

    return True

def plot_ltv_heatmap(avg_ltv: pd.DataFrame, purchased: bool, sku_selected: str) -> bool:
    if avg_ltv.empty or "Cohort" not in avg_ltv.columns:
        st.write("No data available for the LTV Heatmap.")
        return False

    df = maybe_omit_cac(avg_ltv, sku_selected)

    if "Cohort Size" not in df.columns:
        st.write("No 'Cohort Size' column in LTV data.")
        return False

    month_cols = [col for col in df.columns if col.startswith("Month ")]
    reorder = ["Cohort", "Cohort Size", "CAC"] + month_cols
    reorder = [c for c in reorder if c in df.columns]
    df_heat = df[reorder].copy()
    df_heat.set_index("Cohort", inplace=True)

    z_vals = df_heat.values.astype(float)

    if "Cohort Size" in df_heat.columns:
        idx_size = df_heat.columns.get_loc("Cohort Size")
        z_vals[:, idx_size] = np.nan
    if "CAC" in df_heat.columns:
        idx_cac = df_heat.columns.get_loc("CAC")
        z_vals[:, idx_cac] = np.nan

    df_heat_for_text = df_heat.copy().fillna("")
    text_vals = df_heat_for_text.astype(str).values

    x_labels = df_heat.columns.tolist()
    y_labels = df_heat.index.tolist()

    valid_z = z_vals[~np.isnan(z_vals)]
    if len(valid_z) == 0:
        st.write("All LTV values are NaN. Nothing to display.")
        _add_cac_omitted_note_if_needed(sku_selected, df)
        return False

    zmin = 0
    zmax = np.nanmax(valid_z)
    row_count = len(y_labels)
    row_height = 40
    fig_height = max(row_count * row_height, 400)

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

    for i, row_name in enumerate(y_labels):
        if "Cohort Size" in df_heat.columns:
            idx_size = df_heat.columns.get_loc("Cohort Size")
            size_str = text_vals[i, idx_size]
            fig.add_annotation(
                x="Cohort Size",
                y=row_name,
                text=size_str,
                showarrow=False,
                font=dict(color="white"),
                xref="x",
                yref="y"
            )
        if "CAC" in df_heat.columns:
            idx_cac = df_heat.columns.get_loc("CAC")
            cac_str = text_vals[i, idx_cac]
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
    _add_cac_omitted_note_if_needed(sku_selected, df)

    if purchased:
        csv_data = avg_ltv.to_csv(index=False)
        try:
            png_bytes = fig.to_image(format="png", width=1920, height=1080)
        except Exception as e:
            st.error(f"Failed to generate PNG: {e}")
            png_bytes = None

        st.download_button("Download CSV", data=csv_data, file_name="avg_ltv_heatmap.csv", mime="text/csv")
        if png_bytes:
            st.download_button("Download PNG", data=png_bytes, file_name="avg_ltv_heatmap.png", mime="image/png")
        _add_session_note()
        st.session_state["fig_avg_ltv_heatmap"] = fig

    return True

def plot_retention_heatmap_with_size(ret_df: pd.DataFrame, avg_ltv: pd.DataFrame,
                                     purchased: bool, sku_selected: str) -> bool:
    if ret_df.empty or "Cohort" not in ret_df.columns:
        st.write("No data available for the Retention Heatmap.")
        return False
    if avg_ltv.empty or "Cohort" not in avg_ltv.columns or "Cohort Size" not in avg_ltv.columns:
        st.write("Insufficient data to merge 'Cohort Size' & 'CAC' for Retention Heatmap.")
        return False

    merged = pd.merge(ret_df, avg_ltv[["Cohort", "Cohort Size", "CAC"]], on="Cohort", how="left")
    if merged.empty:
        st.write("No data after merging Cohort Size & CAC into Retention Heatmap.")
        return False

    merged = maybe_omit_cac(merged, sku_selected)

    month_cols = [col for col in merged.columns if col.startswith("Month ")]
    reorder = ["Cohort", "Cohort Size", "CAC"] + month_cols
    reorder = [c for c in reorder if c in merged.columns]
    merged = merged[reorder].copy()
    merged.set_index("Cohort", inplace=True)

    z_vals = merged.values.astype(float)

    if "Month 1" in merged.columns:
        idx_m1 = merged.columns.get_loc("Month 1")
        z_vals[:, idx_m1] = np.nan
    if "Cohort Size" in merged.columns:
        idx_size = merged.columns.get_loc("Cohort Size")
        z_vals[:, idx_size] = np.nan
    if "CAC" in merged.columns:
        idx_cac = merged.columns.get_loc("CAC")
        z_vals[:, idx_cac] = np.nan

    merged_for_text = merged.copy().fillna("")
    text_vals = merged_for_text.astype(str).values
    x_labels = merged.columns.tolist()
    y_labels = merged.index.tolist()

    valid_z = z_vals[~np.isnan(z_vals)]
    if len(valid_z) == 0:
        st.write("No numeric data for retention heatmap.")
        _add_cac_omitted_note_if_needed(sku_selected, merged)
        return False

    zmin = 0
    zmax = np.nanmax(valid_z)
    row_count = len(y_labels)
    row_height = 40
    fig_height = max(row_count * row_height, 400)

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
        hovertemplate="Cohort: %{y}<br>%{x}: %{z:.2f}%<extra></extra>"
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
        if "CAC" in merged.columns:
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
    _add_cac_omitted_note_if_needed(sku_selected, merged)

    if purchased:
        csv_data = merged.reset_index().to_csv(index=False)
        try:
            png_bytes = fig.to_image(format="png", width=1920, height=1080)
        except Exception as e:
            st.error(f"Failed to generate PNG: {e}")
            png_bytes = None

        st.download_button("Download CSV", data=csv_data, file_name="retention_heatmap.csv", mime="text/csv")
        if png_bytes:
            st.download_button("Download PNG", data=png_bytes, file_name="retention_heatmap.png", mime="image/png")
        _add_session_note()
        st.session_state["fig_retention_heatmap"] = fig

    return True

def plot_percent_retained_line_chart(ret_df: pd.DataFrame, purchased: bool) -> bool:
    if ret_df.empty or "Cohort" not in ret_df.columns:
        st.write("No data available for the Retention line chart.")
        return False

    month_cols = [c for c in ret_df.columns if c.startswith("Month ")]
    if not month_cols:
        st.write("No month columns found in Retention data.")
        return False

    sorted_mon_cols = sorted(month_cols, key=lambda x: int(x.split()[1]))

    fig = go.Figure()
    ret_df_sorted = ret_df.sort_values("Cohort")
    total_cohorts = len(ret_df_sorted)

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 128, 0),
        (128, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 128, 128),
        (128, 128, 0)
    ]

    for idx, (_, row) in enumerate(ret_df_sorted.iterrows()):
        cohort = row["Cohort"]
        vals = []
        xvals = []
        for mc in sorted_mon_cols:
            val = row[mc]
            if pd.isna(val):
                continue
            month_num = int(mc.split()[1])
            xvals.append(month_num)
            vals.append(val)

        if vals:
            opacity = 0.3 + 0.7 * (idx / total_cohorts)
            line_width = 1 + 2 * (idx / total_cohorts)
            color_idx = idx % len(colors)
            r, g, b = colors[color_idx]
            fig.add_trace(go.Scatter(
                x=xvals,
                y=vals,
                mode="lines+markers",
                line=dict(width=line_width, color=f'rgba({r},{g},{b},{opacity})'),
                name=str(cohort),
                hovertemplate=(
                    f"Cohort: {cohort}<br>"
                    "Month: %{x}<br>"
                    "Retention: %{y:.2f}%"
                    "<extra></extra>"
                )
            ))

    col_averages = {}
    for c in sorted_mon_cols:
        col_vals = ret_df[c].dropna()
        col_averages[c] = col_vals.mean() if len(col_vals) > 0 else np.nan

    x_means = []
    y_means = []
    for c in sorted_mon_cols:
        val = col_averages[c]
        if not pd.isna(val):
            x_means.append(int(c.split()[1]))
            y_means.append(val)

    if x_means and y_means:
        fig.add_trace(go.Scatter(
            x=x_means,
            y=y_means,
            mode="lines+markers",
            name="Average (All Cohorts)",
            line=dict(dash="dash", color="white", width=3),
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

    if purchased:
        csv_data = ret_df.to_csv(index=False)
        try:
            png_bytes = fig.to_image(format="png", width=1920, height=1080)
        except Exception as e:
            st.error(f"Failed to generate PNG: {e}")
            png_bytes = None

        st.download_button("Download CSV", data=csv_data, file_name="retention_line.csv", mime="text/csv")
        if png_bytes:
            st.download_button("Download PNG", data=png_bytes, file_name="retention_line.png", mime="image/png")
        _add_session_note()
        st.session_state["fig_retention_line"] = fig

    return True

def plot_cohort_size_and_ltv_analysis_chart(avg_ltv: pd.DataFrame, purchased: bool) -> bool:
    if avg_ltv.empty or "Cohort" not in avg_ltv.columns:
        st.write("No data for Cohort Size vs. LTV chart.")
        return False
    if "Cohort Size" not in avg_ltv.columns:
        st.write("Missing 'Cohort Size' in LTV data.")
        return False

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=avg_ltv["Cohort"],
        y=avg_ltv["Cohort Size"],
        name="Cohort Size",
        yaxis="y1",
        marker=dict(color="lightblue"),
        hovertemplate="Cohort: %{x}<br>Cohort Size: %{y:,.0f}<extra></extra>"
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

    if purchased:
        csv_data = avg_ltv.to_csv(index=False)
        try:
            png_bytes = fig.to_image(format="png", width=1920, height=1080)
        except Exception as e:
            st.error(f"Failed to generate PNG: {e}")
            png_bytes = None

        st.download_button("Download CSV", data=csv_data, file_name="cohort_size_vs_ltv.csv", mime="text/csv")
        if png_bytes:
            st.download_button("Download PNG", data=png_bytes, file_name="cohort_size_vs_ltv.png", mime="image/png")
        _add_session_note()
        st.session_state["fig_cohort_size_vs_ltv"] = fig

    return True

def plot_ltv_cac_by_cohort(avg_ltv: pd.DataFrame, purchased: bool, sku_selected: str) -> bool:
    if avg_ltv.empty or "Cohort" not in avg_ltv.columns:
        st.write("No data for LTV:CAC chart.")
        return False
    if "CAC" not in avg_ltv.columns:
        st.write("Missing 'CAC' in LTV data.")
        return False
    mon_cols = [c for c in avg_ltv.columns if c.startswith("Month ")]
    if not mon_cols:
        st.write("No month columns found to compute LTV:CAC.")
        return False

    df = maybe_omit_cac(avg_ltv, sku_selected)

    fig = go.Figure()
    all_ratios = []
    trace_count = 0
    for _, row in df.iterrows():
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
            line=dict(width=2),
            name=cohort,
            hovertemplate=(
                f"Cohort: {cohort}<br>"
                "Month: %{x}<br>"
                "LTV:CAC Ratio: %{y:.2f}"
                "<extra></extra>"
            )
        ))
        all_ratios.extend(ratio_vals)
        trace_count += 1
    if trace_count == 0:
        st.write("If chart is not displaying, it's because you need to upload marketing spend and set SKU filter to 'All' for CAC to populate (CAC is currently not available at the individual SKU level).")
        _add_cac_omitted_note_if_needed(sku_selected, df)
        return False

    rmin = min(all_ratios)
    rmax = max(all_ratios)
    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=14),
        xaxis=dict(title="Months Since First Purchase", dtick=1),
        yaxis=dict(
            title="LTV:CAC Ratio",
            range=[rmin * 0.9 if rmin > 0 else rmin * 1.1, rmax * 1.1]
        ),
        legend=dict(x=1.02, y=1.0),
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)
    _add_cac_omitted_note_if_needed(sku_selected, df)

    if purchased:
        csv_data = avg_ltv.to_csv(index=False)
        try:
            png_bytes = fig.to_image(format="png", width=1920, height=1080)
        except Exception as e:
            st.error(f"Failed to generate PNG: {e}")
            png_bytes = None

        st.download_button("Download CSV", data=csv_data, file_name="ltv_cac_ratio.csv", mime="text/csv")
        if png_bytes:
            st.download_button("Download PNG", data=png_bytes, file_name="ltv_cac_ratio.png", mime="image/png")
        _add_session_note()
        st.session_state["fig_ltv_cac_ratio"] = fig

    return True

################################################################################
# (NEW) CHARTS REQUESTED
################################################################################

def plot_cac_new_vs_repeat_all_data(
    order_df: pd.DataFrame,
    spend_df: pd.DataFrame,
    sku_selected: str,
    purchased: bool = False
) -> bool:
    if order_df.empty:
        st.write("No data available for CAC & New vs. Repeat chart.")
        return False

    df = order_df.copy()
    if sku_selected != "All":
        df = df[df["SKU"] == sku_selected].copy()
        if df.empty:
            st.write("No data available for the selected SKU.")
            return False

    df["Month"] = df["Date"].dt.to_period("M")
    monthly_spend = spend_df.copy()
    if not monthly_spend.empty:
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

    fig.add_trace(go.Bar(
        x=merged["Month"],
        y=merged["NewCustomers"],
        name="New Customers",
        marker=dict(color="#386092"),
        yaxis="y",
        hovertemplate="Month: %{x}<br>New Customers: %{y:,}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=merged["Month"],
        y=merged["RepeatCustomers"],
        name="Repeat Customers",
        marker=dict(color="#9a3936"),
        yaxis="y",
        hovertemplate="Month: %{x}<br>Repeat Customers: %{y:,}<extra></extra>"
    ))

    if sku_selected == "All":
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
    else:
        fig.update_layout(
            barmode="group",
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            height=600,
            xaxis=dict(type="category", title="Month"),
            yaxis=dict(title="Number of Customers"),
            legend=dict(x=1.02, y=1.0)
        )

    st.plotly_chart(fig, use_container_width=True)

    if purchased:
        csv_data = merged.to_csv(index=False)
        try:
            png_bytes = fig.to_image(format="png", width=1920, height=1080)
        except Exception as e:
            st.error(f"Failed to generate PNG: {e}")
            png_bytes = None

        st.download_button("Download CSV", data=csv_data, file_name="new_vs_repeat_cac.csv", mime="text/csv")
        if png_bytes:
            st.download_button("Download PNG", data=png_bytes, file_name="new_vs_repeat_cac.png", mime="image/png")
        _add_session_note()
        st.session_state["fig_cac_new_repeat"] = fig

    return True

def plot_percent_multiple_orders(
    order_df: pd.DataFrame, 
    sku_selected: str, 
    purchased: bool = False
) -> bool:
    """
    Show what % of customers place multiple orders (1..5).
    If not purchased, only show up to 3 orders. If purchased, show up to 5.
    """
    if order_df.empty:
        st.write("No data for 'Customer Retention During First 5 Orders'.")
        return False

    df = order_df.copy()
    if sku_selected != "All":
        df = df[df["SKU"] == sku_selected].copy()
        if df.empty:
            st.write("No data available for the selected SKU.")
            return False

    order_counts = df.groupby("Customer ID")["Order ID"].nunique().reset_index()
    order_counts.rename(columns={"Order ID": "TotalOrders"}, inplace=True)
    counts = order_counts["TotalOrders"].value_counts().sort_index()

    # If user hasn't purchased, show up to 3 orders; if purchased, up to 5
    if purchased:
        max_orders = 5
    else:
        max_orders = 3

    all_range = list(range(1, max_orders + 1))
    counts = counts.reindex(all_range, fill_value=0)

    if order_counts.empty:
        st.write("No data found after SKU filtering.")
        return False

    results = []
    c1 = counts.get(1, 0)
    p1 = 100.0  # Everyone included has at least 1 order
    results.append((1, c1, p1))

    if max_orders >= 2:
        c2 = counts.get(2, 0)
        p2 = (c2 / c1 * 100) if c1 > 0 else 0
        results.append((2, c2, p2))

    if max_orders >= 3:
        c3 = counts.get(3, 0)
        p3 = (c3 / c2 * 100) if c2 > 0 else 0
        results.append((3, c3, p3))

    if purchased and max_orders >= 4:
        c4 = counts.get(4, 0)
        p4 = (c4 / c3 * 100) if c3 > 0 else 0
        results.append((4, c4, p4))

        c5 = counts.get(5, 0)
        p5 = (c5 / c4 * 100) if c4 > 0 else 0
        results.append((5, c5, p5))

    df_plot = pd.DataFrame(results, columns=["OrderN", "Count", "PctRelative"]).sort_values("OrderN", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot["Count"],
        y=df_plot["OrderN"].astype(str),
        orientation="h",
        marker=dict(color="#9a3936"),
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
        yaxis=dict(title="Number of Orders", tickmode="array", tickvals=all_range, autorange="reversed"),
        showlegend=False
    )

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

    if purchased:
        csv_data = df_plot.to_csv(index=False)
        try:
            png_bytes = fig.to_image(format="png", width=1920, height=1080)
        except Exception as e:
            st.error(f"Failed to generate PNG: {e}")
            png_bytes = None

        st.download_button("Download CSV", data=csv_data, file_name="percent_multiple_orders.csv", mime="text/csv")
        if png_bytes:
            st.download_button("Download PNG", data=png_bytes, file_name="percent_multiple_orders.png", mime="image/png")
        _add_session_note()
        st.session_state["fig_percent_multiple_orders"] = fig

    return True

def plot_time_between_first_and_second_purchase(
    order_df: pd.DataFrame, 
    sku_selected: str, 
    purchased: bool = False
) -> bool:
    if order_df.empty:
        st.write("No data for 'Time Between First and Second Purchase'.")
        return False

    df = order_df.copy()
    if sku_selected != "All":
        df = df[df["SKU"] == sku_selected].copy()
        if df.empty:
            st.write("No data available for the selected SKU.")
            return False

    df.sort_values(["Customer ID", "Date"], inplace=True)
    df["OrderIndex"] = df.groupby("Customer ID").cumcount() + 1
    second_orders_df = df[df["OrderIndex"] == 2].copy()

    if second_orders_df.empty:
        st.write("No customers have a 2nd purchase in this dataset.")
        return False

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

    if merged.empty:
        st.write("No valid pairs of first+second purchase found.")
        return False

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

    if purchased:
        csv_data = df_plot.to_csv(index=False)
        try:
            png_bytes = fig.to_image(format="png", width=1920, height=1080)
        except Exception as e:
            st.error(f"Failed to generate PNG: {e}")
            png_bytes = None

        st.download_button("Download CSV", data=csv_data, file_name="time_between_first_second.csv", mime="text/csv")
        if png_bytes:
            st.download_button("Download PNG", data=png_bytes, file_name="time_between_first_second.png", mime="image/png")
        _add_session_note()
        st.session_state["fig_time_between_purchases"] = fig

    return True

################################################################################
# (NEW) CROSS-SELL TABLE
################################################################################

def plot_sku_cross_sell_table(purchased: bool = False, sku_selected: str = "All") -> bool:
    if st.session_state["order_data_df"].empty or "SKU" not in st.session_state["order_data_df"].columns:
        st.write("No data available or missing 'SKU' column for Cross-Sell Table.")
        return False

    df_unfiltered = st.session_state["order_data_df"].copy()

    if sku_selected != "All":
        df_firstpurchase = df_unfiltered[df_unfiltered["SKU"] == sku_selected].copy()
    else:
        df_firstpurchase = df_unfiltered.copy()

    if df_firstpurchase.empty:
        st.write("No data for the selected SKU.")
        return False

    df_firstpurchase.sort_values(["Customer ID", "Date"], inplace=True)
    df_temp = df_firstpurchase[["Customer ID", "Date", "SKU"]].copy()
    df2 = df_unfiltered.rename(columns={"Customer ID": "CustomerID2", "Date": "Date2", "SKU": "SKU2"})

    merged = df_temp.merge(df2, left_on="Customer ID", right_on="CustomerID2", how="inner")
    subseq = merged[merged["Date2"] > merged["Date"]]
    if subseq.empty:
        st.write("No subsequent purchases found in the dataset.")
        return False

    counts = subseq.groupby(["SKU", "SKU2"]).size().reset_index(name="count")
    total_counts = counts.groupby("SKU")["count"].sum().reset_index(name="total")
    counts = counts.merge(total_counts, on="SKU", how="left")
    counts["pct"] = (counts["count"] / counts["total"] * 100).round(2)
    counts["rank"] = counts.groupby("SKU")["count"].rank(method="first", ascending=False)
    top_3 = counts[counts["rank"] <= 3].copy()

    def top_3_formatter(g):
        g2 = g.sort_values("count", ascending=False).head(3)
        ret = {}
        for i in range(3):
            if i < len(g2):
                r = g2.iloc[i]
                if i == 0:
                    ret["Most Commonly Purchased in Next Order"] = f"{r['SKU2']} ({r['pct']}%)"
                elif i == 1:
                    ret["2nd Most Commonly Purchased in Next Order"] = f"{r['SKU2']} ({r['pct']}%)"
                else:
                    ret["3rd Most Commonly Purchased in Next Order"] = f"{r['SKU2']} ({r['pct']}%)"
            else:
                if i == 0:
                    ret["Most Commonly Purchased in Next Order"] = ""
                elif i == 1:
                    ret["2nd Most Commonly Purchased in Next Order"] = ""
                else:
                    ret["3rd Most Commonly Purchased in Next Order"] = ""
        return pd.Series(ret)

    final_df = top_3.groupby("SKU").apply(top_3_formatter).reset_index()
    final_df.rename(columns={"SKU": "First Purchase SKU"}, inplace=True)

    st.dataframe(final_df, use_container_width=True)

    if purchased:
        csv_data = final_df.to_csv(index=False)
        st.download_button("Download CSV", data=csv_data, file_name="most_common_cross_sell_skus.csv", mime="text/csv")
        _add_session_note()
        st.session_state["df_cross_sell_table"] = final_df

    return True

################################################################################
# (M) RENDER CHARTS WITH PARTIAL GATING
################################################################################

def plot_charts_with_partial_gating(
    avg_ltv_full: pd.DataFrame,
    ret_df_full: pd.DataFrame,
    order_data: pd.DataFrame,
    sku_selected: str,
    freq: str,
    marketing_spend_df: pd.DataFrame
):
    avg_ltv_limited = limit_to_6_months_ltv(avg_ltv_full)
    ret_df_limited  = limit_to_6_months_ret(ret_df_full)

    full_unlock_purchased = ("Unlock All Charts" in st.session_state["purchased_items"])

    if sku_selected != "All":
        sku_order_data = order_data[order_data["SKU"] == sku_selected].copy()
        avg_ltv_full_sku = calculate_avg_ltv(sku_order_data, freq, marketing_spend_df)
        ret_df_full_sku  = calculate_percent_retained(sku_order_data, freq)
        avg_ltv_limited_sku = limit_to_6_months_ltv(avg_ltv_full_sku)
        ret_df_limited_sku  = limit_to_6_months_ret(ret_df_full_sku)
    else:
        avg_ltv_full_sku    = avg_ltv_full
        ret_df_full_sku     = ret_df_full
        avg_ltv_limited_sku = avg_ltv_limited
        ret_df_limited_sku  = ret_df_limited

    chart_subtitles = {
        "Average Lifetime Value (LTV) by Cohort (Heatmap)": "The average lifetime value of customers in each cohort by month after first purchase.",
        "Average Retention by Cohort (Heatmap)": "The percentage of customers in each cohort making a purchase by month after first purchase. Month 1 is excluded as it's 100% by default.",
        "LTV:CAC Ratio by Cohort": "Customer Lifetime Value / Customer Acquisition Cost. Think of this as your cumulative ROAS. Most businesses aim for at least a 3x on this metric.",
        "CAC & New Customers vs. Repeat Customers": "New vs. repeat customers, plus CAC for the month (Note: CAC is only shown if sku filter is set to All).",
        "Customer Retention During First 5 Orders": "Measure of customer stickiness over time. Most businesses see their retention increase as customers make more orders.",
        "Time Between First and Second Purchase": "Distribution of timing of second orders from your customers.",
        "What Customers Buy After Their First Order": "See what customers are most likely to purchase next based on their first purchase sku.",
        "Cohort Size vs. Average Lifetime Value (LTV) by Cohort": "Is customer quality changing as you scale customer acquisition?",
        "Average Lifetime Value (LTV) by Cohort (Line)": "See how customer lifetime value is evolving across cohorts",
        "Average Retention by Cohort (Line)": "See how customer retention is evolving across cohorts"
    }

    for item in ITEMS_FOR_SALE:
        if item["name"] == "Unlock All Charts":
            continue

        st.write(f"### {item['name']}")

        if item["name"] in chart_subtitles:
            subtitle = chart_subtitles[item["name"]]
            st.markdown(f"<p style='color:white; font-size:16px;'>{subtitle}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:white; font-size:16px;'>subtitle placeholder text</p>", unsafe_allow_html=True)

        has_item = (item["name"] in st.session_state["purchased_items"])
        purchased = (full_unlock_purchased or has_item)

        if purchased:
            df_ltv = avg_ltv_full_sku
            df_ret = ret_df_full_sku
        else:
            df_ltv = avg_ltv_limited_sku
            df_ret = ret_df_limited_sku

        chart_shown = False

        if item["name"] == "Average Lifetime Value (LTV) by Cohort (Heatmap)":
            chart_shown = plot_ltv_heatmap(df_ltv, purchased, sku_selected)

        elif item["name"] == "Average Retention by Cohort (Heatmap)":
            chart_shown = plot_retention_heatmap_with_size(df_ret, df_ltv, purchased, sku_selected)

        elif item["name"] == "LTV:CAC Ratio by Cohort":
            chart_shown = plot_ltv_cac_by_cohort(df_ltv, purchased, sku_selected)

        elif item["name"] == "CAC & New Customers vs. Repeat Customers":
            if purchased:
                chart_shown = plot_cac_new_vs_repeat_all_data(
                    order_data, st.session_state["marketing_spend_df"], sku_selected, purchased=True
                )
            else:
                tmp_orders = order_data.copy()
                tmp_orders["Month"] = tmp_orders["Date"].dt.to_period("M")
                unique_months = sorted(tmp_orders["Month"].unique())
                keep_months = unique_months[:6]
                tmp_orders = tmp_orders[tmp_orders["Month"].isin(keep_months)]
                tmp_spend = st.session_state["marketing_spend_df"].copy()
                if not tmp_spend.empty:
                    tmp_spend["Month"] = tmp_spend["Date"].dt.to_period("M")
                    tmp_spend = tmp_spend[tmp_spend["Month"].isin(keep_months)]
                chart_shown = plot_cac_new_vs_repeat_all_data(
                    tmp_orders, tmp_spend, sku_selected, purchased=False
                )

        elif item["name"] == "Customer Retention During First 5 Orders":
            if purchased:
                chart_shown = plot_percent_multiple_orders(order_data, sku_selected, purchased=True)
            else:
                tmp = order_data.copy()
                chart_shown = plot_percent_multiple_orders(tmp, sku_selected, purchased=False)

        elif item["name"] == "Time Between First and Second Purchase":
            if purchased:
                chart_shown = plot_time_between_first_and_second_purchase(order_data, sku_selected, purchased=True)
            else:
                tmp = order_data.copy()
                if sku_selected != "All":
                    tmp = tmp[tmp["SKU"] == sku_selected].copy()
                tmp.sort_values(["Customer ID", "Date"], inplace=True)
                tmp["OrderIndex"] = tmp.groupby("Customer ID").cumcount() + 1
                df_first = tmp[tmp["OrderIndex"] == 1][["Customer ID", "Date"]].rename(columns={"Date": "FirstPurchaseDate"})
                df_second = tmp[tmp["OrderIndex"] == 2][["Customer ID", "Date"]].rename(columns={"Date": "SecondPurchaseDate"})
                merged = pd.merge(df_second, df_first, on="Customer ID", how="inner")
                merged["DiffDays"] = (merged["SecondPurchaseDate"] - merged["FirstPurchaseDate"]).dt.days
                allowed_customers = merged[merged["DiffDays"] <= 30]["Customer ID"].unique()
                tmp = tmp[tmp["Customer ID"].isin(allowed_customers)]
                chart_shown = plot_time_between_first_and_second_purchase(tmp, sku_selected, purchased=False)

        elif item["name"] == "What Customers Buy After Their First Order":
            chart_shown = plot_sku_cross_sell_table(purchased, sku_selected)

        elif item["name"] == "Cohort Size vs. Average Lifetime Value (LTV) by Cohort":
            chart_shown = plot_cohort_size_and_ltv_analysis_chart(df_ltv, purchased)

        elif item["name"] == "Average Lifetime Value (LTV) by Cohort (Line)":
            chart_shown = plot_cohort_lifetime_value_line_chart(df_ltv, purchased)

        elif item["name"] == "Average Retention by Cohort (Line)":
            chart_shown = plot_percent_retained_line_chart(df_ret, purchased)

        else:
            pass

        if chart_shown and not purchased and item["name"] != "What Customers Buy After Their First Order":
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

            if item["name"] == "Time Between First and Second Purchase":
                partial_text = f"""
                <p style="font-size:14px; color:white; margin-top:4px;">
                    Only displaying first 30 days after purchase for the selected SKU. 
                    Get this chart with your full dataset for $5. 
                    <a href="{single_chart_link}" target="_blank" style="color:#87CEFA; text-decoration:underline;">
                        [Unlock Full Chart]
                    </a>
                    <br/>
                    Get all 10 items (charts + table) with your full dataset for $20 (SAVE 60%). 
                    <a href="{full_link}" target="_blank" style="color:#87CEFA; text-decoration:underline;">
                        [Unlock All Charts]
                    </a>
                </p>
                """
            else:
                partial_text = f"""
                <p style="font-size:14px; color:white; margin-top:4px;">
                    Only displaying 6 months of data (or partial subset). Get this chart with your full dataset for $5. 
                    <a href="{single_chart_link}" target="_blank" style="color:#87CEFA; text-decoration:underline;">
                        [Unlock Full Chart]
                    </a>
                    <br/>
                    Get all 10 items (charts + table) with your full dataset for $20 (SAVE 60%). 
                    <a href="{full_link}" target="_blank" style="color:#87CEFA; text-decoration:underline;">
                        [Unlock All Charts]
                    </a>
                </p>
                """

            if single_chart_link and full_link:
                st.markdown(partial_text, unsafe_allow_html=True)

        elif chart_shown and not purchased and item["name"] == "What Customers Buy After Their First Order":
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
            partial_text = f"""
            <p style="font-size:14px; color:white; margin-top:4px;">
                Only displaying cross-sell results for 5 SKUs. Get the full table for $5. 
                <a href="{single_chart_link}" target="_blank" style="color:#87CEFA; text-decoration:underline;">
                    [Unlock Full Table]
                </a>
                <br/>
                Get all 10 items (charts + table) with your full dataset for $20 (SAVE 60%+). 
                <a href="{full_link}" target="_blank" style="color:#87CEFA; text-decoration:underline;">
                    [Unlock All Charts]
                </a>
            </p>
            """
            if single_chart_link and full_link:
                st.markdown(partial_text, unsafe_allow_html=True)

        st.markdown("---")

################################################################################
# (N) MAIN APPLICATION LOGIC
################################################################################

def main():
    st.markdown(f"""
    <style>
    /* Title with smaller font */
    h3 {{
        font-size:24px;
        color: white;
        margin-bottom: 4px;
    }}
    div[data-baseweb="select"] {{
        background-color: #1b3929 !important;
        border: 2px solid #ffffff !important;
        border-radius: 5px !important;
    }}
    div[data-baseweb="select"] * {{
        color: white !important;
    }}
    div.stDownloadButton > button[data-testid="stDownloadButton"] {{
        background-color: #339966 !important;
        color: white !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    handle_payment()

    if st.session_state.get("show_refresh_spinner", False):
        with st.spinner("Payment successful! Just a second while we refresh your chart(s) with the full data set..."):
            if st.session_state["order_data_df"].empty and st.session_state["order_data_filename"]:
                file_bytes = download_file_from_s3(
                    st.session_state["upload_id"],
                    st.session_state["order_data_filename"]
                )
                if file_bytes:
                    try:
                        if st.session_state["order_data_filename"].lower().endswith(".csv"):
                            try:
                                raw_order = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig", low_memory=False)
                            except UnicodeDecodeError:
                                raw_order = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
                        else:
                            xls = pd.ExcelFile(io.BytesIO(file_bytes))
                            needed_cols = [
                                "Purchase Date",
                                "Buyer Email",
                                "Item Price",
                                "Merchant SKU",
                                "Amazon Order Id"
                            ]
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

            if st.session_state["marketing_spend_df"].empty and st.session_state["marketing_spend_filename"]:
                file_bytes = download_file_from_s3(
                    st.session_state["upload_id"],
                    st.session_state["marketing_spend_filename"]
                )
                if file_bytes:
                    try:
                        if st.session_state["marketing_spend_filename"].lower().endswith(".csv"):
                            try:
                                raw_spend = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig", low_memory=False)
                            except UnicodeDecodeError:
                                raw_spend = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
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
        # Title with smaller font
        st.markdown("""
            <h3>
                Amazon Customer Analytics Tool | cohortanalysis.ai 
            </h3>
        """, unsafe_allow_html=True)

        st.markdown("""
            <a href="https://freeimage.host/i/26AUQup"><img src="https://iili.io/26AUQup.md.png" alt="26AUQup.md.png" border="0" 
                     style="width: 100%;" />
            </a>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown("<div style='padding-top:25%;'>", unsafe_allow_html=True)

        if st.session_state["order_data_df"].empty:
            st.markdown(
                "<p style='font-size:16px; font-weight:bold; margin-bottom:4px;'>"
                "UPLOAD YOUR ORDER DATA (CSV OR XLSX)</p>",
                unsafe_allow_html=True
            )
            order_data_file = st.file_uploader(
                "Order Data",
                type=["csv", "xlsx"],
                label_visibility="visible",
            )
            st.markdown("""
                <p style="font-size:14px;">
                    <strong>Required Columns (make sure they're named this way):</strong><br/>
                    <strong>1:</strong> Purchase Date<br/>
                    <strong>2:</strong> Buyer Email<br/>
                    <strong>3:</strong> Item Price<br/>
                    <strong>4:</strong> Merchant SKU<br/>
                    <strong>5:</strong> Amazon Order Id<br/>
                </p>
            """, unsafe_allow_html=True)
        else:
            order_data_file = None
            st.success("Order Data is already in session. Ready for analysis!")

        if st.session_state["marketing_spend_df"].empty:
            st.markdown(
                "<p style='font-size:16px; font-weight:bold; margin:12px 0 4px;'>"
                "UPLOAD YOUR MARKETING SPEND (CSV OR XLSX)</p>",
                unsafe_allow_html=True
            )
            marketing_spend_file = st.file_uploader(
                "Marketing Spend",
                type=["csv", "xlsx"],
                label_visibility="visible",
                key="marketing_spend_file",
            )
            st.markdown("""
                <p style="font-size:14px;">
                    <strong>Required Columns:</strong><br/>
                    <strong>1:</strong> Date<br/>
                    <strong>2:</strong> Marketing Spend
                </p>
            """, unsafe_allow_html=True)
        else:
            marketing_spend_file = None
            st.info("Marketing Spend is already in session.")

        st.markdown("</div>", unsafe_allow_html=True)

    if order_data_file is not None:
        with st.spinner("Parsing and uploading Order Data..."):
            try:
                file_bytes = order_data_file.getvalue()
                filename = order_data_file.name
                success_up = upload_file_to_s3(st.session_state["upload_id"], file_bytes, filename)
                if not success_up:
                    st.stop()
                st.session_state["order_data_filename"] = filename

                if filename.lower().endswith(".csv"):
                    try:
                        raw_order = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig", low_memory=False)
                    except UnicodeDecodeError:
                        raw_order = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
                else:
                    xls = pd.ExcelFile(io.BytesIO(file_bytes))
                    needed_cols = [
                        "Purchase Date",
                        "Buyer Email",
                        "Item Price",
                        "Merchant SKU",
                        "Amazon Order Id"
                    ]
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

    if marketing_spend_file is not None:
        with st.spinner("Parsing and uploading Marketing Spend..."):
            try:
                file_bytes = marketing_spend_file.getvalue()
                filename = marketing_spend_file.name
                success_up = upload_file_to_s3(st.session_state["upload_id"], file_bytes, filename)
                if not success_up:
                    st.stop()
                st.session_state["marketing_spend_filename"] = filename

                if filename.lower().endswith(".csv"):
                    try:
                        raw_spend = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig", low_memory=False)
                    except UnicodeDecodeError:
                        raw_spend = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
                else:
                    ms_xl = pd.ExcelFile(io.BytesIO(file_bytes))
                    raw_spend = ms_xl.parse(ms_xl.sheet_names[0])

                st.session_state["marketing_spend_df"] = preprocess_marketing_spend(raw_spend)
                st.success("Marketing Spend uploaded and parsed successfully!")
            except Exception as ex:
                st.error(f"Failed to parse Marketing Spend: {ex}")

    purchased_any = len(st.session_state["purchased_items"]) > 0

    if not purchased_any:
        order_data = st.session_state["order_data_df"]
        if not order_data.empty:
            needed_cols = ["Date","Customer ID","Order Total","SKU","Order ID"]
            preview_cols = [c for c in needed_cols if c in order_data.columns]
            preview_df = order_data[preview_cols].copy()

            st.markdown("<p style='color:white;font-size:14px;'><strong>Preview of Uploaded Data:</strong></p>", unsafe_allow_html=True)
            st.dataframe(preview_df, height=300)
            st.markdown("<br/><br/>", unsafe_allow_html=True)

            if not st.session_state["invalid_date_rows_orders"].empty:
                st.warning("Some order rows had invalid or unparseable dates and were dropped. See below:")
                st.dataframe(st.session_state["invalid_date_rows_orders"].head(10))

            if not st.session_state["invalid_date_rows_spend"].empty:
                st.warning("Some marketing-spend rows had invalid or unparseable dates and were dropped. See below:")
                st.dataframe(st.session_state["invalid_date_rows_spend"].head(10))

            if not st.session_state["invalid_sku_rows"].empty:
                st.warning("Some order rows had a blank or invalid SKU after cleaning and were dropped. See below:")
                st.dataframe(st.session_state["invalid_sku_rows"].head(10))

    order_data = st.session_state["order_data_df"]
    marketing_spend_df = st.session_state["marketing_spend_df"]

    if order_data.empty:
        st.info("Please upload your Order Data to proceed.")
        st.stop()

    purchased_something = (
        "Unlock All Charts" in st.session_state["purchased_items"] or
        any(item["name"] in st.session_state["purchased_items"] for item in ITEMS_FOR_SALE if item["name"] != "Unlock All Charts")
    )
    if purchased_something:
        def create_zip():
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for item in st.session_state["purchased_items"]:
                    sanitized_name = re.sub(r'[^\w\s-]', '', item).strip().replace(' ', '_')
                    csv_filename = f"{sanitized_name}.csv"
                    png_filename = f"{sanitized_name}.png"

                    df_content = pd.DataFrame()
                    fig = None
                    if item == "What Customers Buy After Their First Order":
                        df_content = st.session_state.get("df_cross_sell_table", pd.DataFrame())
                        fig = None
                    elif item == "Average Lifetime Value (LTV) by Cohort (Heatmap)":
                        df_content = st.session_state.get("avg_ltv_full", pd.DataFrame())
                        fig = st.session_state.get("fig_avg_ltv_heatmap")
                    elif item == "Average Retention by Cohort (Heatmap)":
                        df_content = st.session_state.get("ret_df_full", pd.DataFrame())
                        fig = st.session_state.get("fig_retention_heatmap")
                    elif item == "Average Lifetime Value (LTV) by Cohort (Line)":
                        df_content = st.session_state.get("avg_ltv_full", pd.DataFrame())
                        fig = st.session_state.get("fig_avg_ltv_line")
                    elif item == "Average Retention by Cohort (Line)":
                        df_content = st.session_state.get("ret_df_full", pd.DataFrame())
                        fig = st.session_state.get("fig_retention_line")
                    elif item == "Cohort Size vs. Average Lifetime Value (LTV) by Cohort":
                        df_content = st.session_state.get("avg_ltv_full", pd.DataFrame())
                        fig = st.session_state.get("fig_cohort_size_vs_ltv")
                    elif item == "LTV:CAC Ratio by Cohort":
                        df_content = st.session_state.get("avg_ltv_full", pd.DataFrame())
                        fig = st.session_state.get("fig_ltv_cac_ratio")
                    elif item == "CAC & New Customers vs. Repeat Customers":
                        df_content = st.session_state.get("order_data_df", pd.DataFrame())
                        fig = st.session_state.get("fig_cac_new_repeat")
                    elif item == "Percent of Customers Making Multiple Orders":
                        df_content = st.session_state.get("order_data_df", pd.DataFrame())
                        fig = st.session_state.get("fig_percent_multiple_orders")
                    elif item == "Time Between First and Second Purchase":
                        df_content = st.session_state.get("order_data_df", pd.DataFrame())
                        fig = st.session_state.get("fig_time_between_purchases")
                    else:
                        df_content = pd.DataFrame()
                        fig = None

                    csv_bytes = b""
                    if not df_content.empty:
                        csv_bytes = df_content.to_csv(index=False).encode()

                    if len(csv_bytes) > 0:
                        zip_file.writestr(csv_filename, csv_bytes)

                    if fig is not None:
                        try:
                            fig_bytes = fig.to_image(format="png", width=1920, height=1080)
                            zip_file.writestr(png_filename, fig_bytes)
                        except Exception as e:
                            zip_file.writestr(png_filename, f"Failed to generate PNG: {e}".encode())

            zip_buffer.seek(0)
            return zip_buffer

        zip_data = create_zip()
        st.download_button(
            label="Download CSVs and PNGs",
            data=zip_data,
            file_name="purchased_charts.zip",
            mime="application/zip"
        )

    freq = "Monthly"
    with st.spinner("Computing metrics..."):
        try:
            avg_ltv = calculate_avg_ltv(order_data, freq, marketing_spend_df)
            ret_df = calculate_percent_retained(order_data, freq)
            st.session_state["avg_ltv_full"] = avg_ltv
            st.session_state["ret_df_full"] = ret_df
        except Exception as ex:
            st.error(f"An unexpected error occurred during metric calculations: {ex}")
            st.stop()

    if "SKU" in order_data.columns:
        st.write("### FILTER CHARTS BY SKU")
        st.markdown(
            "<p style='font-size:12px; color:white;'>"
            "Note: CAC is omitted when filtering for a specific SKU. To view chart with CAC figure, set SKU filter to ALL."
            "</p>",
            unsafe_allow_html=True
        )
        unique_skus = sorted(order_data["SKU"].unique())
        sku_options = ["All"] + unique_skus
        selected_sku = st.selectbox("Choose a SKU:", sku_options, index=0, label_visibility="visible")
    else:
        selected_sku = "All"
        st.info("Add a column for SKU to your CSV to activate SKU filter.")

    plot_charts_with_partial_gating(
        avg_ltv_full=st.session_state["avg_ltv_full"],
        ret_df_full=st.session_state["ret_df_full"],
        order_data=order_data,
        sku_selected=selected_sku,
        freq=freq,
        marketing_spend_df=marketing_spend_df,
    )

if __name__ == "__main__":
    main()
