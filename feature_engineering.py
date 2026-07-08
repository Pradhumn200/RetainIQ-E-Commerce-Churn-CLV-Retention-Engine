import os
import sqlite3
import pandas as pd
import numpy as np
import json

def main():
    # Resolve paths dynamically relative to script location
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "data", "processed", "olist.db")
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    
    print("Connecting to database and loading data...")
    conn = sqlite3.connect(db_path)
    
    # Load customer orders fact view
    orders = pd.read_sql_query("SELECT * FROM customer_order_fact", conn)
    # Load payments for voucher/discount analysis
    payments = pd.read_sql_query("SELECT order_id, payment_type FROM olist_order_payments", conn)
    # Load RFM output
    rfm = pd.read_csv(os.path.join(processed_dir, "rfm_table.csv"))
    # Load pre-calculated purchase gaps from SQLite (computed using SQL LAG window function)
    gaps_df = pd.read_sql_query("SELECT customer_unique_id, gap FROM customer_purchase_gaps", conn)
    
    conn.close()
    
    # Convert dates to datetime
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    rfm['last_order_date'] = pd.to_datetime(rfm['last_order_date'])
    
    print("Calculating repurchase gap distribution from SQL LAG table...")
    # Filter out same-day orders (gap > 0) to get typical purchase cycle
    repeat_gaps = gaps_df[gaps_df['gap'] > 0]['gap']
    
    if len(repeat_gaps) > 0:
        median_gap = repeat_gaps.median()
        mean_gap_val = repeat_gaps.mean()
        print(f"Found {len(repeat_gaps)} repurchase gaps in database.")
        print(f"Median repurchase gap: {median_gap:.1f} days")
        print(f"Mean repurchase gap: {mean_gap_val:.1f} days")
    else:
        median_gap = 45.0  # fallback
        print("Warning: No repeat purchase gaps found. Using fallback median gap of 45 days.")
        
    # Define churn threshold N (2x median gap)
    churn_threshold = int(2 * median_gap)
    # Ensure a reasonable minimum churn window (e.g. 60 days) to prevent over-labeling
    churn_threshold = max(churn_threshold, 60)
    print(f"Data-driven Churn Threshold (N = 2x Median Gap): {churn_threshold} days")
    
    # Churn label: 1 if recency > churn_threshold else 0
    rfm['churn'] = (rfm['recency'] > churn_threshold).astype(int)
    print(f"Churn rate in customer base: {rfm['churn'].mean() * 100:.2f}%")
    
    # Calculate snapshot date (max order date + 1 day)
    snapshot_date = orders['order_date'].max() + pd.Timedelta(days=1)
    print(f"Snapshot Date: {snapshot_date.strftime('%Y-%m-%d')}")
    
    # Save metadata to JSON to prevent hardcoding in modeling & A/B testing
    metadata = {
        "churn_threshold": int(churn_threshold),
        "snapshot_date": snapshot_date.strftime('%Y-%m-%d')
    }
    metadata_path = os.path.join(processed_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved run metadata to {metadata_path}")
    
    # Define Windows
    cutoff_recent = snapshot_date - pd.Timedelta(days=90)
    cutoff_prior = snapshot_date - pd.Timedelta(days=180)
    
    print("Building customer-level trend features...")
    # Identify orders in recent (last 3m) and prior (3m-6m) windows
    orders['in_recent'] = (orders['order_date'] >= cutoff_recent) & (orders['order_date'] < snapshot_date)
    orders['in_prior'] = (orders['order_date'] >= cutoff_prior) & (orders['order_date'] < cutoff_recent)
    
    # Group by customer and compute window metrics
    # 1. Purchase frequency in windows
    freq_recent = orders[orders['in_recent']].groupby('customer_unique_id')['order_id'].count()
    freq_prior = orders[orders['in_prior']].groupby('customer_unique_id')['order_id'].count()
    
    # 2. Average Order Value (AOV) in windows
    aov_recent = orders[orders['in_recent']].groupby('customer_unique_id')['order_value'].mean()
    aov_prior = orders[orders['in_prior']].groupby('customer_unique_id')['order_value'].mean()
    
    # 3. Category diversity in windows
    cat_recent = orders[orders['in_recent']].groupby('customer_unique_id')['category'].nunique()
    cat_prior = orders[orders['in_prior']].groupby('customer_unique_id')['category'].nunique()
    
    # 4. Discount / Voucher dependency
    voucher_orders = set(payments[payments['payment_type'] == 'voucher']['order_id'].unique())
    orders['used_voucher'] = orders['order_id'].apply(lambda x: 1 if x in voucher_orders else 0)
    
    voucher_counts = orders.groupby('customer_unique_id')['used_voucher'].sum()
    total_counts = orders.groupby('customer_unique_id')['order_id'].count()
    discount_dep = voucher_counts / total_counts
    
    # 5. Customer specific average gap from SQLite gap table
    customer_avg_gaps = gaps_df.groupby('customer_unique_id')['gap'].mean()
    # Impute missing gaps (customers with 1 purchase) with the global median gap
    customer_avg_gaps = customer_avg_gaps.reindex(rfm['customer_unique_id']).fillna(median_gap)
    # Avoid zero-division by enforcing a minimum gap of 1 day
    customer_avg_gaps = customer_avg_gaps.clip(lower=1.0)
    
    # Reindex metrics to match RFM customer list
    customer_ids = rfm['customer_unique_id']
    
    freq_recent = freq_recent.reindex(customer_ids, fill_value=0)
    freq_prior = freq_prior.reindex(customer_ids, fill_value=0)
    
    aov_recent = aov_recent.reindex(customer_ids, fill_value=0.0)
    aov_prior = aov_prior.reindex(customer_ids, fill_value=0.0)
    
    cat_recent = cat_recent.reindex(customer_ids, fill_value=0)
    cat_prior = cat_prior.reindex(customer_ids, fill_value=0)
    
    discount_dep = discount_dep.reindex(customer_ids, fill_value=0.0)
    
    # Build Trends (use +1 smoothing to stabilize ratio metrics)
    purchase_freq_trend = (freq_recent + 1) / (freq_prior + 1)
    aov_trend = (aov_recent + 1.0) / (aov_prior + 1.0)
    category_diversity_trend = (cat_recent + 1) / (cat_prior + 1)
    
    # Recency ratio
    recency_ratio = rfm['recency'] / customer_avg_gaps.values
    
    # Combine everything into a feature DataFrame
    features_df = pd.DataFrame({
        'customer_unique_id': rfm['customer_unique_id'],
        'recency': rfm['recency'],
        'frequency': rfm['frequency'],
        'monetary': rfm['monetary'],
        'purchase_freq_trend': purchase_freq_trend.values,
        'recency_ratio': recency_ratio.values,
        'discount_dependency': discount_dep.values,
        'category_diversity_trend': category_diversity_trend.values,
        'aov_trend': aov_trend.values,
        'churn': rfm['churn']
    })
    
    # Print summary statistics
    print("\nFeature Summary:")
    print(features_df.describe().T[['mean', 'std', 'min', 'max']])
    
    # Save the feature table
    output_path = os.path.join(processed_dir, "customer_features.csv")
    features_df.to_csv(output_path, index=False)
    print(f"\nFeature table saved successfully to {output_path} ({len(features_df)} rows).")

if __name__ == "__main__":
    main()
