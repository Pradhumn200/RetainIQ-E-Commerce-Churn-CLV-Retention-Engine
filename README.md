# RetainIQ — Churn Prediction, CLV Quadrants, & A/B Test Simulation Engine

A professional, end-to-end customer retention and marketing optimization engine built using the **Olist Brazilian E-Commerce dataset**. This project builds a local SQL database, runs a feature engineering and modeling pipeline in Python, executes a randomized controlled trial (A/B test) simulation, and visualizes the results in an interactive, dynamic Streamlit dashboard.

---

## 🚀 Project Overview

E-commerce businesses suffer high customer acquisition costs (CAC). Retaining existing customers is far more cost-effective. **RetainIQ** is a retention-focused decision system that:
1. **SQL Foundation**: Loads transactional records into a SQLite database to build cohort retention heatmaps and customer RFM profiles.
2. **Predictive Modeling**: Standardizes features, handles class imbalance, and trains a Random Forest classifier to predict individual churn probabilities *without target leakage*.
3. **CLV Quadrants**: Annualizes purchase frequencies to compute Customer Lifetime Value (CLV) and segments customers into a 2x2 action matrix (High/Low Risk vs. High/Low CLV).
4. **RCT Simulation (A/B Test)**: Randomly splits the critical *Win-back* segment (High Value + High Risk) into treatment and control, simulating a CRM intervention and running a two-proportion Z-test and campaign ROI calculation.
5. **Interactive Dashboard**: A multi-tab Streamlit dashboard letting users adjust the model's decision threshold live, dynamically updating metrics, evaluation parameters (Precision/Recall), scatter plots, and action segments in real time.

---

## 🛠️ Tech Stack & Directory Structure

* **Database**: SQLite (SQL window functions)
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn (Logistic Regression, Random Forest, Cost-Sensitive Learning)
* **Statistical Testing**: Statsmodels (Two-Proportion Z-Test)
* **Visualization**: Streamlit, Plotly Express, Plotly Graph Objects

### Directory Layout
```text
retainiq/
  data/
    raw/          <- Raw Olist CSV files
    processed/    <- SQLite database (olist.db), intermediate tables, exported CSVs
  dashboard/
    data/         <- Processed customer segment CSV for dashboard
    app.py        <- Multi-tab Streamlit dashboard application
  sql/
    cohort_rfm.sql<- SQL definitions for Fact, Cohort, RFM, and Gaps tables
  setup_data.py   <- Setup script: copies data and imports CSVs into SQLite
  run_sql_pipeline.py <- Executes cohort_rfm.sql statements and exports results
  feature_engineering.py <- Computes trend features, reads SQLite gaps, and saves metadata.json
  train_model.py  <- Trains Logistic Regression and Random Forest (class weight balanced)
  clv_ab_test.py  <- Computes CLV, splits quadrants, and simulates A/B test
```

---

## 📈 Key Results & Business Metrics

* **Model Discriminative Power**: The balanced Random Forest model achieved a **0.6877 AUC-ROC** on holdout test cohorts, significantly outperforming a random guess baseline.
* **A/B Test Significance**: The simulated win-back voucher program achieved a **10.17 percentage point lift** in customer retention (Treatment: 29.97% vs. Control: 19.80%), which is highly statistically significant (**Z-statistic = 19.50, p-value = $9.99 \times 10^{-85}$**).
* **Campaign ROI**: Spending **$5.00** per customer on the win-back treatment segment saved an incremental **1,405.5** high-value customers. Protecting **$238,540.33 in CLV** at a campaign cost of **$69,095.00** yielded a net return of **$169,445.33** (**245.24% ROI**).

---

## 🏆 Key Interview Talking Points

### 1. SQL Window Functions (`LAG() OVER`)
To show SQL competence, purchase intervals were calculated on the database side using the **`LAG()` window function** partitioned by customer unique ID:
```sql
CREATE TABLE customer_purchase_gaps AS
WITH ordered_purchases AS (
    SELECT 
        customer_unique_id,
        order_date,
        LAG(order_date, 1) OVER (PARTITION BY customer_unique_id ORDER BY order_date) AS prev_order_date
    FROM customer_order_fact
)
SELECT 
    customer_unique_id,
    order_date,
    prev_order_date,
    CAST(julianday(order_date) - julianday(prev_order_date) AS INTEGER) AS gap
FROM ordered_purchases
WHERE prev_order_date IS NOT NULL;
```
This offloaded heavy sorting and partition math from Pandas to the SQLite engine, creating a highly efficient hybrid pipeline.

### 2. Preventing Target Leakage
> "When building the churn model, I noticed that `recency` and `recency_ratio` would create 100% target leakage because the target itself is defined as `recency > N`. I explicitly excluded them from training, forcing the model to learn churn risk purely from transactional trends (AOV and purchase frequency trajectories)."

### 3. Handling Class Imbalance
> "Because 97% of Olist customers are one-time buyers, standard classifiers predict everyone to churn. I implemented cost-sensitive learning with `class_weight='balanced'` in the estimators, penalizing minority (active) class misclassifications, which successfully boosted the Random Forest AUC-ROC to 0.69."

### 4. Cohort Maturity Split
> "You cannot evaluate a churn model on the latest cohorts because they haven't been in the system long enough to churn (0% observed churn). I implemented a maturity cutoff (snapshot date minus 2x the median gap) to hold out the most recent mature cohort (March 1 to April 14, 2018) for validation, ensuring realistic and valid evaluation metrics."

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Pradhumn200/RetainIQ-E-Commerce-Churn-CLV-Retention-Engine.git
   cd RetainIQ-E-Commerce-Churn-CLV-Retention-Engine
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn statsmodels plotly streamlit
   ```

3. Run the data setup, SQL pipeline, feature engineering, modeling, and A/B test pipeline:
   ```bash
   python setup_data.py
   python run_sql_pipeline.py
   python feature_engineering.py
   python train_model.py
   python clv_ab_test.py
   ```

4. Launch the Streamlit dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```
   Open **[http://localhost:8501](http://localhost:8501)** in your browser.
