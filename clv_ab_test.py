import os
import sqlite3
import pandas as pd
import numpy as np
import json
from statsmodels.stats.proportion import proportions_ztest

def main():
    # Resolve directories dynamically relative to script location
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "data", "processed", "olist.db")
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    dashboard_data_dir = os.path.join(BASE_DIR, "dashboard", "data")
    
    # Load features with model predictions
    features_path = os.path.join(processed_dir, "customer_features_with_predictions.csv")
    df = pd.read_csv(features_path)
    df['first_purchase_date'] = pd.to_datetime(df['first_purchase_date'])
    
    # Load run metadata (snapshot_date & churn_threshold)
    metadata_path = os.path.join(processed_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        churn_threshold = metadata["churn_threshold"]
        snapshot_date = pd.to_datetime(metadata["snapshot_date"])
        print(f"Loaded Churn Threshold: {churn_threshold} days")
        print(f"Loaded Snapshot Date: {snapshot_date.strftime('%Y-%m-%d')}")
    else:
        churn_threshold = 138
        snapshot_date = pd.to_datetime('2018-08-30')
        print(f"Warning: metadata.json not found. Using fallback churn threshold of {churn_threshold} days.")
        
    # Load optimal risk threshold from model metrics
    metrics_path = os.path.join(processed_dir, "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        risk_threshold = metrics.get("selected_rf_threshold", 0.64)
        print(f"Loaded Risk Threshold from model metrics: {risk_threshold:.2f}")
    else:
        risk_threshold = 0.64
        print(f"Warning: model_metrics.json not found. Using fallback risk threshold of {risk_threshold:.2f}")
    
    print("Calculating Customer Lifetime Value (CLV)...")
    # AOV = monetary / frequency
    aov = df['monetary'] / df['frequency']
    
    # Customer tenure (lifespan in system)
    tenure_days = (snapshot_date - df['first_purchase_date']).dt.days
    # Clip tenure at a minimum of 90 days to prevent extreme outlier scaling for new customers
    tenure_days_clipped = tenure_days.clip(lower=90)
    
    # Annualized Frequency = frequency / (tenure_days_clipped / 365.25)
    annual_frequency = df['frequency'] / (tenure_days_clipped / 365.25)
    
    # CLV Formula: (AOV * annual_freq * lifespan_years) * gross_margin
    # We assume standard lifespan of 2 years and a 20% margin
    lifespan_years = 2.0
    margin = 0.20
    df['clv'] = (aov * annual_frequency * lifespan_years) * margin
    
    # Clean up CLV values (NaNs or negative)
    df['clv'] = df['clv'].fillna(0).clip(lower=0)
    
    print("Creating Churn Risk vs CLV Quadrants...")
    # Median split for CLV
    clv_median = df['clv'].median()
    print(f"Median CLV: ${clv_median:.2f}")
    
    # Segment into 4 quadrants
    def assign_quadrant(row):
        is_high_risk = row['churn_probability'] >= risk_threshold
        is_high_clv = row['clv'] >= clv_median
        
        if is_high_clv and is_high_risk:
            return "High CLV + High Risk (Win-back)"
        elif is_high_clv and not is_high_risk:
            return "High CLV + Low Risk (Reward)"
        elif not is_high_clv and is_high_risk:
            return "Low CLV + High Risk (Monitor)"
        else:
            return "Low CLV + Low Risk (No Action)"
            
    df['quadrant'] = df.apply(assign_quadrant, axis=1)
    df['risk_tier'] = df['churn_probability'].apply(lambda x: 'High' if x >= risk_threshold else 'Low')
    df['value_tier'] = df['clv'].apply(lambda x: 'High' if x >= clv_median else 'Low')
    
    # Summary of quadrants
    print("\nQuadrant Distribution:")
    quadrant_counts = df['quadrant'].value_counts()
    quadrant_revenue = df.groupby('quadrant')['clv'].sum()
    for q in quadrant_counts.index:
        print(f" - {q}: {quadrant_counts[q]} customers, Total CLV: ${quadrant_revenue[q]:,.2f}")
        
    # Phase 5: A/B Test Simulation
    print("\nSimulating A/B Test on the 'High CLV + High Risk (Win-back)' segment...")
    ab_pool = df[df['quadrant'] == "High CLV + High Risk (Win-back)"].copy()
    n_pool = len(ab_pool)
    print(f"Intervention pool size: {n_pool} customers")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Randomly assign Treatment (1) and Control (0) groups (50/50 split)
    ab_pool['group'] = np.random.binomial(1, 0.5, n_pool)
    
    treatment_group = ab_pool[ab_pool['group'] == 1]
    control_group = ab_pool[ab_pool['group'] == 0]
    
    n_treatment = len(treatment_group)
    n_control = len(control_group)
    print(f"Treatment Group Size: {n_treatment}")
    print(f"Control Group Size: {n_control}")
    
    # Simulate retention outcomes
    # Control group base retention probability = 20%
    # Treatment group gets 10 percentage points lift = 30% retention probability
    p_control = 0.20
    p_treatment = 0.30
    
    ab_pool['retained'] = np.where(
        ab_pool['group'] == 1,
        np.random.binomial(1, p_treatment, n_pool),
        np.random.binomial(1, p_control, n_pool)
    )
    
    # Calculate summary metrics
    retained_treatment = ab_pool[(ab_pool['group'] == 1) & (ab_pool['retained'] == 1)].shape[0]
    retained_control = ab_pool[(ab_pool['group'] == 0) & (ab_pool['retained'] == 1)].shape[0]
    
    rate_treatment = retained_treatment / n_treatment if n_treatment > 0 else 0
    rate_control = retained_control / n_control if n_control > 0 else 0
    lift = rate_treatment - rate_control
    
    print(f"Simulated Treatment Retention Rate: {rate_treatment * 100:.2f}% ({retained_treatment}/{n_treatment})")
    print(f"Simulated Control Retention Rate: {rate_control * 100:.2f}% ({retained_control}/{n_control})")
    print(f"Simulated Lift: {lift * 100:.2f} percentage points")
    
    # Run Two-Proportion Z-Test
    count = np.array([retained_treatment, retained_control])
    nobs = np.array([n_treatment, n_control])
    stat, pval = proportions_ztest(count, nobs)
    
    print(f"Z-statistic: {stat:.4f}, p-value: {pval:.4e}")
    
    # Compute 95% Confidence Interval for the difference in proportions
    se = np.sqrt((rate_treatment * (1 - rate_treatment) / n_treatment) + (rate_control * (1 - rate_control) / n_control))
    margin_error = 1.96 * se
    ci_lower = lift - margin_error
    ci_upper = lift + margin_error
    print(f"95% Confidence Interval for Lift: [{ci_lower * 100:.2f}%, {ci_upper * 100:.2f}%]")
    
    # ROI Calculation
    # Let's assume a discount coupon campaign cost of $5 per treatment customer
    cost_per_cust = 5.00
    total_cost = n_treatment * cost_per_cust
    
    # Average CLV of the win-back segment
    avg_clv = ab_pool['clv'].mean()
    
    # Incremental customers saved
    incremental_saved = retained_treatment - (n_treatment * rate_control)
    # Total protected CLV = Incremental customers saved * average CLV
    clv_protected = incremental_saved * avg_clv
    
    # Net profit and ROI
    net_profit = clv_protected - total_cost
    roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0
    
    print(f"Average Customer CLV in Win-back Segment: ${avg_clv:.2f}")
    print(f"Campaign Cost: ${total_cost:,.2f} (${cost_per_cust:.2f} per customer)")
    print(f"Incremental Customers Saved: {incremental_saved:.1f}")
    print(f"CLV Protected: ${clv_protected:,.2f}")
    print(f"Net Campaign Profit: ${net_profit:,.2f}")
    print(f"Estimated Campaign ROI: {roi:.2f}%")
    
    # Save the updated customer segments dataset for dashboard
    df_output = df.copy()
    os.makedirs(dashboard_data_dir, exist_ok=True)
    output_csv = os.path.join(dashboard_data_dir, "customer_segments.csv")
    df_output.to_csv(output_csv, index=False)
    print(f"\nSaved dashboard data to {output_csv}")
    
    # Save A/B results metadata to JSON
    ab_results = {
        "n_treatment": int(n_treatment),
        "n_control": int(n_control),
        "retained_treatment": int(retained_treatment),
        "retained_control": int(retained_control),
        "rate_treatment": float(rate_treatment),
        "rate_control": float(rate_control),
        "lift": float(lift),
        "z_stat": float(stat),
        "p_value": float(pval),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "cost_per_cust": float(cost_per_cust),
        "total_cost": float(total_cost),
        "avg_clv": float(avg_clv),
        "incremental_saved": float(incremental_saved),
        "clv_protected": float(clv_protected),
        "net_profit": float(net_profit),
        "roi": float(roi)
    }
    
    with open(os.path.join(processed_dir, "ab_test_results.json"), "w") as f:
        json.dump(ab_results, f, indent=4)
        
    print("CLV and A/B Test Simulation pipeline complete.")

if __name__ == "__main__":
    main()
