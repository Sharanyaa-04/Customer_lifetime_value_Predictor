# app.py
"""
CLV Predictor — robust coefficient handling + safe download fallback
- Fixes: coef shape mismatch (handles intercept-in-coef cases)
- Fixes: download works even if xlsxwriter not installed (falls back to CSV)
Place model.pkl (optional) and Online Retail.xlsx (optional) in the same folder.
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from io import BytesIO
from pathlib import Path

st.set_page_config(page_title="CLV Predictor (Fixed)", layout="wide")

PROJECT_DIR = Path.cwd()
MODEL_PATH = PROJECT_DIR / "model.pkl"
DATASET_PATH = PROJECT_DIR / "Online Retail.xlsx"

# ---------- helpers ----------
def try_load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def df_to_bytes_and_name(df, excel_name="data.xlsx", csv_name="data.csv"):
    """
    Try to create an Excel bytes object using xlsxwriter/openpyxl.
    If xlsxwriter not available, fallback to CSV bytes.
    Returns (bytes_data, filename, mime)
    """
    buf = BytesIO()
    try:
        # try xlsxwriter first
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        buf.seek(0)
        return buf.read(), excel_name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    except Exception:
        # fallback CSV
        buf = BytesIO()
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        buf.write(csv_bytes)
        buf.seek(0)
        return buf.read(), csv_name, "text/csv"

# ---- feature engineering helper (minimal) ----
def compute_customer_features_from_df(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    def find_col(cols, keywords):
        for k in keywords:
            for c in cols:
                if k.lower() in c.lower():
                    return c
        return None

    invoice_col = find_col(df.columns, ['invoice no','invoiceno','invoice number','invoice'])
    date_col = find_col(df.columns, ['invoice date','date','invoicedate'])
    cust_col = find_col(df.columns, ['customerid','customer id','customer'])
    qty_col = find_col(df.columns, ['quantity','qty'])
    unitprice_col = find_col(df.columns, ['unitprice','unit price','price'])
    stock_col = find_col(df.columns, ['stockcode','stock code','stock_code','stock'])

    if invoice_col: df.rename(columns={invoice_col: 'InvoiceNo'}, inplace=True)
    if date_col: df.rename(columns={date_col: 'InvoiceDate'}, inplace=True)
    if cust_col: df.rename(columns={cust_col: 'CustomerID'}, inplace=True)
    if qty_col: df.rename(columns={qty_col: 'Quantity'}, inplace=True)
    if unitprice_col: df.rename(columns={unitprice_col: 'UnitPrice'}, inplace=True)
    if stock_col: df.rename(columns={stock_col: 'StockCode'}, inplace=True)

    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

    if 'TotalPrice' not in df.columns:
        if 'UnitPrice' in df.columns and 'Quantity' in df.columns:
            df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce').fillna(0)
            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
            df['TotalPrice'] = df['UnitPrice'] * df['Quantity']
        else:
            for c in df.columns:
                if 'amount' in c.lower() or 'total' in c.lower():
                    df.rename(columns={c:'TotalPrice'}, inplace=True)
                    break
    df['TotalPrice'] = pd.to_numeric(df.get('TotalPrice', pd.Series([0]*len(df))), errors='coerce').fillna(0)

    if 'CustomerID' not in df.columns:
        df['CustomerID'] = df.index.astype(str)
    df['CustomerID'] = df['CustomerID'].astype(str)

    if 'InvoiceDate' in df.columns and df['InvoiceDate'].notna().any():
        snapshot = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    else:
        snapshot = pd.Timestamp.today()

    grp = df.groupby('CustomerID')
    frequency = grp['InvoiceNo'].nunique() if 'InvoiceNo' in df.columns else grp.size()
    last_purchase = grp['InvoiceDate'].max() if 'InvoiceDate' in df.columns else pd.Series(pd.NaT, index=frequency.index)
    recency = (snapshot - last_purchase).dt.days.fillna(0)
    monetary = grp['TotalPrice'].sum().fillna(0)

    customer = pd.DataFrame({
        'CustomerID': monetary.index,
        'Recency': recency.values,
        'Frequency': frequency.values,
        'Monetary': monetary.values
    })
    customer['AvgOrderValue'] = (customer['Monetary'] / customer['Frequency'].replace(0, np.nan)).fillna(0)
    customer['OrderAmountStd'] = grp['TotalPrice'].std().reindex(customer['CustomerID']).fillna(0).values
    if 'StockCode' in df.columns:
        customer['UniqueProducts'] = grp['StockCode'].nunique().reindex(customer['CustomerID']).fillna(0).values
    else:
        customer['UniqueProducts'] = 0
    if 'InvoiceDate' in df.columns:
        df_sorted = df.sort_values(['CustomerID','InvoiceDate'])
        df_sorted['Prev'] = df_sorted.groupby('CustomerID')['InvoiceDate'].shift(1)
        df_sorted['DaysBetween'] = (df_sorted['InvoiceDate'] - df_sorted['Prev']).dt.days
        stats = df_sorted.groupby('CustomerID')['DaysBetween'].agg(['mean','std']).reindex(customer['CustomerID']).fillna(0)
        customer['MeanDaysBetween'] = stats['mean'].values
        customer['StdDaysBetween'] = stats['std'].values
    else:
        customer['MeanDaysBetween'] = 0
        customer['StdDaysBetween'] = 0

    customer['CLV'] = customer['Monetary']
    return customer

# ---- Determine coefficients ----
intercept = None
coefs = None
trained_customer_df = None
loaded_from = None

# 1) try model.pkl
if MODEL_PATH.exists():
    obj = try_load_pickle(MODEL_PATH)
    if obj is not None:
        if isinstance(obj, tuple) and len(obj) >= 1:
            mdl = obj[0]
        else:
            mdl = obj
        # attempt to extract coef and intercept safely
        if hasattr(mdl, 'coef_'):
            candidate = np.atleast_1d(mdl.coef_)
            # some models store coef_ as shape (n,) or (1,n) — flatten
            candidate = candidate.flatten()
            # if intercept exists separately read it
            candidate_intercept = getattr(mdl, 'intercept_', None)
            # handle case where coef_ includes intercept (length 7)
            if candidate.size == 7 and (candidate_intercept is None or np.isnan(candidate_intercept).all() if isinstance(candidate_intercept, np.ndarray) else candidate_intercept is None):
                intercept = float(candidate[0])
                coefs = candidate[1:].astype(float).tolist()
            else:
                # normal case: coef length should be 6
                coefs = candidate.astype(float).tolist()
                if candidate_intercept is not None:
                    try:
                        intercept = float(candidate_intercept)
                    except Exception:
                        intercept = None
            loaded_from = f"model.pkl ({type(mdl).__name__})"

# 2) try to train on dataset
if (coefs is None or intercept is None) and DATASET_PATH.exists():
    try:
        df = pd.read_excel(DATASET_PATH, engine='openpyxl')
        cust = compute_customer_features_from_df(df)
        cust = cust[cust['Frequency'] > 0].copy()
        cust = cust.replace([np.inf, -np.inf], np.nan).dropna(subset=['Recency','Frequency','AvgOrderValue','OrderAmountStd','UniqueProducts','MeanDaysBetween','CLV'])
        if cust.shape[0] >= 10:
            from sklearn.linear_model import LinearRegression
            X = cust[['Recency','Frequency','AvgOrderValue','OrderAmountStd','UniqueProducts','MeanDaysBetween']].values
            y = cust['CLV'].values
            lr = LinearRegression().fit(X, y)
            intercept = lr.intercept_
            coefs = np.atleast_1d(lr.coef_).astype(float).tolist()
            trained_customer_df = cust
            loaded_from = "trained on Online Retail.xlsx"
    except Exception:
        pass

# 3) fallback defaults
if coefs is None or intercept is None:
    intercept = 50.0
    coefs = [-0.6, 25.0, 0.8, 0.3, 5.0, -0.2]
    loaded_from = "fallback defaults"

# if coefs length is 7 (intercept included), split just in case
if hasattr(coefs, '__len__') and len(coefs) == 7:
    intercept = coefs[0]
    coefs = coefs[1:]

# final validation: ensure coefs length == 6
if not hasattr(coefs, '__len__') or len(coefs) != 6:
    intercept = 50.0
    coefs = [-0.6, 25.0, 0.8, 0.3, 5.0, -0.2]
    loaded_from = "fallback defaults (shape corrected)"

# convert coefs to numpy array for stable math
coefs = np.array(coefs, dtype=float)
intercept = float(intercept)

# ---------- UI ----------
st.title("Customer Lifetime Value (CLV) Predictor")
st.markdown("**What is CLV?**")
st.markdown("CLV estimates expected revenue from a customer and helps prioritize retention, personalize offers and allocate marketing budget. This CLV score estimates the expected short-term revenue from the customer based on their purchase frequency, spending pattern, and recency. The feature contributions explain what drives the value, and the distribution comparison shows how this customer ranks among others. Use this prediction as a guide for segmentation, targeting, and retention strategy.")
st.divider()

# Input fields
st.header("Enter customer features")
left, right = st.columns(2)
with left:
    recency = st.number_input("Recency (days since last purchase)", min_value=0.0, value=30.0)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0.0, value=3.0)
    avg_order_value = st.number_input("Average Order Value (₹)", min_value=0.0, value=120.0)
with right:
    order_std = st.number_input("Order amount standard deviation", min_value=0.0, value=10.0)
    unique_products = st.number_input("Unique products bought", min_value=0.0, value=4.0)
    mean_days_between = st.number_input("Mean days between purchases", min_value=0.0, value=12.0)

st.markdown("**Model coefficients (order):** [Recency, Frequency, AvgOrderValue, OrderAmountStd, UniqueProducts, MeanDaysBetween]")
st.write("Intercept:", intercept)
st.write("Coefficients:", coefs.tolist())

# prediction function without forcing Python float types
def predict_clv_safe(r,f,aov,std,up,mg):
    arr = np.array([r,f,aov,std,up,mg])
    if coefs.shape[0] != arr.shape[0]:
        raise ValueError(f"Coefficient length {coefs.shape[0]} doesn't match input length {arr.shape[0]}")
    return intercept + np.dot(coefs, arr)

# button
if st.button("Predict CLV"):
    try:
        pred = predict_clv_safe(recency, frequency, avg_order_value, order_std, unique_products, mean_days_between)
        # display: convert to Python scalar if numpy scalar for nice formatting
        try:
            display_val = pred.item() if hasattr(pred, "item") else pred
        except Exception:
            display_val = pred
        st.metric("Predicted CLV (₹)", f"{display_val:,.2f}")
    except Exception as e:
        st.error("Prediction failed: " + str(e))
        pred = None

    if pred is not None:
        contrib = {
            'Intercept': intercept,
            'Recency': coefs[0] * recency,
            'Frequency': coefs[1] * frequency,
            'AvgOrderValue': coefs[2] * avg_order_value,
            'OrderAmountStd': coefs[3] * order_std,
            'UniqueProducts': coefs[4] * unique_products,
            'MeanDaysBetween': coefs[5] * mean_days_between
        }
        contrib_df = pd.DataFrame.from_dict(contrib, orient='index', columns=['Contribution']).reset_index().rename(columns={'index':'Feature'})
        fig, ax = plt.subplots(figsize=(8,3.5))
        colors = ['#2ca02c' if v>=0 else '#d62728' for v in contrib_df['Contribution']]
        ax.barh(contrib_df['Feature'], contrib_df['Contribution'], color=colors)
        ax.axvline(0, color='black', linewidth=0.6)
        ax.set_xlabel("Contribution to CLV (₹)")
        st.subheader("Feature contributions")
        st.pyplot(fig)
        st.table(contrib_df.set_index('Feature').round(2))

        # distribution
        try:
            if trained_customer_df is None and DATASET_PATH.exists():
                df = pd.read_excel(DATASET_PATH, engine='openpyxl')
                trained_customer_df = compute_customer_features_from_df(df)
            if trained_customer_df is not None:
                trained_customer_df['PredCLV'] = trained_customer_df.apply(lambda row: predict_clv_safe(row['Recency'], row['Frequency'], row['AvgOrderValue'], row['OrderAmountStd'], row['UniqueProducts'], row['MeanDaysBetween']), axis=1)
                fig2, ax2 = plt.subplots(figsize=(8,3))
                ax2.hist(trained_customer_df['PredCLV'], bins=30, alpha=0.9)
                ax2.axvline(pred, color='red', linestyle='--', linewidth=2, label='Current customer')
                ax2.set_xlabel("Predicted CLV (₹)")
                ax2.set_ylabel("Count")
                ax2.legend()
                st.subheader("Where this customer sits vs dataset customers")
                st.pyplot(fig2)
                percentile = (trained_customer_df['PredCLV'] < pred).mean() * 100
                st.caption(f"This customer is at approximately the {percentile:.1f} percentile vs dataset customers.")
            else:
                # synthetic fallback
                np.random.seed(42)
                sample_size = 500
                sample_freq = np.random.poisson(3.0, sample_size)
                sample_aov = np.random.normal(100, 30, sample_size).clip(1)
                sample_recency = np.random.exponential(40, sample_size)
                sample_std = np.random.normal(12, 5, sample_size).clip(0)
                sample_up = np.random.poisson(4.0, sample_size)
                sample_gap = np.random.exponential(15, sample_size)
                sample_clv = [predict_clv_safe(r,f,aov,s,up,mg) for r,f,aov,s,up,mg in zip(sample_recency, sample_freq, sample_aov, sample_std, sample_up, sample_gap)]
                fig2, ax2 = plt.subplots(figsize=(8,3))
                ax2.hist(sample_clv, bins=30, alpha=0.9)
                ax2.axvline(pred, color='red', linestyle='--', linewidth=2, label='Current customer')
                ax2.set_xlabel("Predicted CLV (₹)")
                ax2.set_ylabel("Count")
                ax2.legend()
                st.subheader("Where this customer sits vs a synthetic sample")
                st.pyplot(fig2)
                percentile = (np.array(sample_clv) < pred).mean() * 100
                st.caption(f"This customer is at approximately the {percentile:.1f} percentile vs the synthetic sample.")
        except Exception as e:
            st.info("Failed to compute distribution: " + str(e))

        # download
        out_df = pd.DataFrame([{
            'Recency': recency,
            'Frequency': frequency,
            'AvgOrderValue': avg_order_value,
            'OrderAmountStd': order_std,
            'UniqueProducts': unique_products,
            'MeanDaysBetween': mean_days_between,
            'PredictedCLV': pred
        }])
        bytes_data, fname, mime = df_to_bytes_and_name(out_df, excel_name="clv_prediction.xlsx", csv_name="clv_prediction.csv")
        st.download_button("Download prediction", data=bytes_data, file_name=fname, mime=mime)

else:
    # live preview (safe)
    try:
        preview = predict_clv_safe(recency, frequency, avg_order_value, order_std, unique_products, mean_days_between)
        preview_val = preview.item() if hasattr(preview, "item") else preview
        st.subheader("Live preview")
        st.write(f"Predicted CLV (with current inputs): ₹ {preview_val:,.2f}")
    except Exception as e:
        st.write("Preview unavailable: " + str(e))
    st.markdown("- Tip: Increase Frequency and AvgOrderValue to increase CLV. Higher Recency usually reduces short-term CLV.")
