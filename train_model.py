import os
import sqlite3
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, classification_report

def main():
    # Resolve directories dynamically relative to script location
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "data", "processed", "olist.db")
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    features_path = os.path.join(processed_dir, "customer_features.csv")
    
    print("Loading engineered features...")
    df_features = pd.read_csv(features_path)
    
    # Retrieve customer first purchase dates from database for time-based split
    print("Retrieving customer cohort dates for time-based split...")
    conn = sqlite3.connect(db_path)
    cohort_dates = pd.read_sql_query(
        "SELECT customer_unique_id, MIN(order_date) AS first_purchase_date FROM customer_order_fact GROUP BY customer_unique_id", 
        conn
    )
    conn.close()
    
    # Merge first purchase date into features
    df_features = df_features.merge(cohort_dates, on='customer_unique_id', how='left')
    df_features['first_purchase_date'] = pd.to_datetime(df_features['first_purchase_date'])
    
    # Load run metadata (snapshot_date & churn_threshold)
    metadata_path = os.path.join(processed_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        churn_threshold = metadata["churn_threshold"]
        snapshot_date = pd.to_datetime(metadata["snapshot_date"])
        print(f"Loaded Churn Threshold from metadata: {churn_threshold} days")
        print(f"Loaded Snapshot Date from metadata: {snapshot_date.strftime('%Y-%m-%d')}")
    else:
        churn_threshold = 138
        snapshot_date = pd.to_datetime('2018-08-30')
        print(f"Warning: metadata.json not found. Using fallback churn threshold of {churn_threshold} days.")
        
    mature_date = snapshot_date - pd.Timedelta(days=churn_threshold)
    print(f"Maturity Date Cutoff (Snapshot - {churn_threshold} days): {mature_date.strftime('%Y-%m-%d')}")
    
    # Train/Test Split on mature cohorts:
    # Train: joined before 2018-03-01
    # Test (Holdout): joined between 2018-03-01 and 2018-04-14 (most recent mature cohort)
    split_date = pd.to_datetime('2018-03-01')
    
    train_df = df_features[df_features['first_purchase_date'] < split_date]
    test_df = df_features[(df_features['first_purchase_date'] >= split_date) & (df_features['first_purchase_date'] <= mature_date)]
    
    print(f"Train set: {len(train_df)} customers (cohorts before March 2018)")
    print(f"Test set (holdout): {len(test_df)} customers (cohorts between March 1, 2018 and {mature_date.strftime('%Y-%m-%d')})")
    
    # Check class distribution
    print(f"Train Churn Rate: {train_df['churn'].mean() * 100:.2f}%")
    print(f"Test Churn Rate: {test_df['churn'].mean() * 100:.2f}%")
    
    # Define features and target
    # CRITICAL: Exclude 'recency' and 'recency_ratio' because they are directly derived from the target definition
    # (target = recency > threshold). Including them would cause 100% target leakage.
    feature_cols = [
        'frequency',
        'monetary',
        'purchase_freq_trend',
        'discount_dependency',
        'category_diversity_trend',
        'aov_trend'
    ]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df['churn'].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df['churn'].copy()
    
    # Handle any NaNs/Infs
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Train Logistic Regression (Interpretable baseline, handling class imbalance)
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr.fit(X_train_scaled, y_train)
    
    # Coefficients
    lr_coefs = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': lr.coef_[0]
    }).sort_values(by='Coefficient', key=abs, ascending=False)
    print("Logistic Regression Coefficients:")
    print(lr_coefs.to_string(index=False))
    
    # 2. Train Random Forest (Comparison model, handling class imbalance)
    print("\nTraining Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced')
    rf.fit(X_train_scaled, y_train)
    
    # Feature Importances
    rf_importances = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("Random Forest Feature Importances:")
    print(rf_importances.to_string(index=False))
    
    # 3. Evaluation
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
    
    lr_auc = roc_auc_score(y_test, lr_probs)
    rf_auc = roc_auc_score(y_test, rf_probs)
    
    print(f"\nLogistic Regression Test AUC-ROC: {lr_auc:.4f}")
    print(f"Random Forest Test AUC-ROC: {rf_auc:.4f}")
    
    # Optimize Threshold for Recall (since missing a churner is costlier than a false alarm)
    # We want a recall of at least 80% while maximizing precision.
    precisions, recalls, thresholds = precision_recall_curve(y_test, rf_probs)
    suitable_indices = np.where(recalls >= 0.80)[0]
    best_idx = suitable_indices[-1] if len(suitable_indices) > 0 else 0
    rf_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    # Ensure threshold is reasonable (e.g. between 0.3 and 0.7)
    rf_thresh = max(min(rf_thresh, 0.7), 0.3)
    print(f"Selected Decision Threshold for Random Forest: {rf_thresh:.2f} (to maintain high Recall)")
    
    # Predict using selected thresholds
    lr_preds = (lr_probs >= 0.5).astype(int)
    rf_preds = (rf_probs >= rf_thresh).astype(int)
    
    # Classification Reports
    print("\nLogistic Regression Classification Report (Threshold = 0.5):")
    print(classification_report(y_test, lr_preds))
    
    print(f"\nRandom Forest Classification Report (Threshold = {rf_thresh:.2f}):")
    print(classification_report(y_test, rf_preds))
    
    # Confusion Matrices
    lr_cm = confusion_matrix(y_test, lr_preds)
    rf_cm = confusion_matrix(y_test, rf_preds)
    
    # Save predictions and probabilities for CLV & A/B simulation
    all_features_scaled = scaler.transform(df_features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0))
    df_features['churn_probability'] = rf.predict_proba(all_features_scaled)[:, 1]
    df_features['churn_prediction'] = (df_features['churn_probability'] >= rf_thresh).astype(int)
    
    # Save the updated customer segments table back (with predictions)
    df_features.to_csv(os.path.join(processed_dir, "customer_features_with_predictions.csv"), index=False)
    print("\nPredictions appended to features table.")
    
    # Save metrics JSON for the comparison report and dashboard
    metrics = {
        "lr_auc": lr_auc,
        "rf_auc": rf_auc,
        "selected_rf_threshold": float(rf_thresh),
        "lr_coefs": lr_coefs.to_dict(orient='records'),
        "rf_importances": rf_importances.to_dict(orient='records'),
        "lr_confusion_matrix": lr_cm.tolist(),
        "rf_confusion_matrix": rf_cm.tolist(),
        "test_size": len(y_test),
        "train_size": len(y_train)
    }
    
    with open(os.path.join(processed_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Model training pipeline execution complete.")

if __name__ == "__main__":
    main()
