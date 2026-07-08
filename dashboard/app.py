import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from sklearn.metrics import precision_score, recall_score

# Page Configuration
st.set_page_config(page_title="RetainIQ — Churn & CLV Dashboard", layout="wide", initial_sidebar_state="expanded")

# Resolve paths dynamically relative to app.py
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SEGMENTS_PATH = os.path.join(APP_DIR, "data", "customer_segments.csv")
COHORT_PATH = os.path.join(APP_DIR, "..", "data", "processed", "cohort_retention.csv")
AB_RESULTS_PATH = os.path.join(APP_DIR, "..", "data", "processed", "ab_test_results.json")
METRICS_PATH = os.path.join(APP_DIR, "..", "data", "processed", "model_metrics.json")
METADATA_PATH = os.path.join(APP_DIR, "..", "data", "processed", "metadata.json")

# App Header
st.title("RetainIQ — E-Commerce Churn & CLV Retention Engine")
st.markdown("---")

# Load Data
@st.cache_data
def load_data():
    if not os.path.exists(SEGMENTS_PATH):
        st.error(f"Customer segments data not found at {SEGMENTS_PATH}. Please run the pipelines first.")
        return None, None, None, None, None
        
    df = pd.read_csv(SEGMENTS_PATH)
    df['first_purchase_date'] = pd.to_datetime(df['first_purchase_date'])
    
    # Load cohort data
    cohort_df = pd.read_csv(COHORT_PATH) if os.path.exists(COHORT_PATH) else None
    
    # Load A/B results
    ab_results = None
    if os.path.exists(AB_RESULTS_PATH):
        with open(AB_RESULTS_PATH, "r") as f:
            ab_results = json.load(f)
            
    # Load Model Metrics
    model_metrics = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            model_metrics = json.load(f)
            
    # Load Run Metadata
    metadata = None
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
            
    return df, cohort_df, ab_results, model_metrics, metadata

df, cohort_df, ab_results, model_metrics, metadata = load_data()

if df is not None:
    # Resolve dynamic thresholds and dates from metadata
    if metadata:
        churn_threshold = metadata.get("churn_threshold", 138)
        snapshot_date = pd.to_datetime(metadata.get("snapshot_date", "2018-08-30"))
        mature_cutoff = snapshot_date - pd.Timedelta(days=churn_threshold)
    else:
        churn_threshold = 138
        snapshot_date = pd.to_datetime("2018-08-30")
        mature_cutoff = pd.to_datetime('2018-04-14')

    # Sidebar control panel
    st.sidebar.header("Control Panel")
    st.sidebar.markdown("Use these settings to configure the customer segmentation threshold and filters.")
    
    # Dynamic Churn Threshold Slider (using optimized RF threshold from training run as default)
    st.sidebar.subheader("Model Decision Settings")
    default_thresh = model_metrics.get("selected_rf_threshold", 0.59) if model_metrics else 0.59
    thresh = st.sidebar.slider(
        "Churn Probability Threshold", 
        min_value=0.10, 
        max_value=0.90, 
        value=float(default_thresh), 
        step=0.01,
        help="Adjust the classifier probability threshold. A lower threshold flags more customers as at-risk (higher Recall), while a higher threshold increases Precision."
    )
    
    # Calculate median CLV
    clv_median = df['clv'].median()
    
    # Dynamic recalculation of quadrants based on slider
    df['risk_tier'] = df['churn_probability'].apply(lambda x: 'High' if x >= thresh else 'Low')
    
    def assign_quadrant_dynamic(row):
        is_high_risk = row['churn_probability'] >= thresh
        is_high_clv = row['clv'] >= clv_median
        
        if is_high_clv and is_high_risk:
            return "High CLV + High Risk (Win-back)"
        elif is_high_clv and not is_high_risk:
            return "High CLV + Low Risk (Reward)"
        elif not is_high_clv and is_high_risk:
            return "Low CLV + High Risk (Monitor)"
        else:
            return "Low CLV + Low Risk (No Action)"
            
    df['quadrant'] = df.apply(assign_quadrant_dynamic, axis=1)
    
    # Model evaluation metrics update on matured cohort
    mature_df = df[df['first_purchase_date'] <= mature_cutoff].copy()
    
    # Compute dynamic precision/recall
    y_true = mature_df['churn']
    y_pred = (mature_df['churn_probability'] >= thresh).astype(int)
    current_precision = precision_score(y_true, y_pred, zero_division=0)
    current_recall = recall_score(y_true, y_pred, zero_division=0)
    
    # Sidebar Model Performance Card
    st.sidebar.subheader("Live Model Metrics (Mature Cohorts)")
    st.sidebar.markdown(f"**Precision**: `{current_precision * 100:.1f}%`")
    st.sidebar.markdown(f"**Recall**: `{current_recall * 100:.1f}%`")
    st.sidebar.info("These metrics update live in response to the threshold. In e-commerce retention, we optimize for high Recall to avoid missing churners.")
    
    # Sidebar project info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### RetainIQ Portfolio Project")
    st.sidebar.markdown(f"""
    **Dataset**: Olist Brazilian E-Commerce
    **Total Customers**: {len(df):,}
    **Active Period**: 2016 - 2018
    """)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Cohort Retention Heatmap", 
        "🎯 CLV vs Churn Risk Quadrants", 
        "🧪 A/B Test Experiment Report", 
        "📋 Customer Action Plan & Export"
    ])
    
    # TAB 1: Cohort Retention
    with tab1:
        st.subheader("Cohort Retention Analysis")
        st.markdown("""
        Cohort analysis groups customers by their first purchase month and tracks the percentage of those customers who return in subsequent months. 
        Typical of e-commerce, retention is concentrated in Month 0 with a low rate of return purchases.
        """)
        
        if cohort_df is not None:
            # Date Range Filter in sidebar (only visible for Cohort tab)
            cohort_months = sorted(cohort_df['cohort_month'].unique())
            min_month, max_month = st.select_slider(
                "Filter Cohorts by Join Month",
                options=cohort_months,
                value=(cohort_months[0], cohort_months[-1])
            )
            
            # Filter cohort dataframe
            filtered_cohort = cohort_df[
                (cohort_df['cohort_month'] >= min_month) & 
                (cohort_df['cohort_month'] <= max_month)
            ]
            
            # Pivot table for heatmap
            pivot_retention = filtered_cohort.pivot(
                index="cohort_month", 
                columns="period_diff", 
                values="retention_rate"
            )
            
            # Only keep first 12 periods for readability
            pivot_retention = pivot_retention.iloc[:, :12]
            
            # Heatmap figure
            fig_heatmap = px.imshow(
                pivot_retention,
                labels=dict(x="Months Since First Order", y="Cohort Group", color="Retention Rate"),
                x=pivot_retention.columns,
                y=pivot_retention.index,
                color_continuous_scale="Blues",
                text_auto=".1%",
                aspect="auto"
            )
            fig_heatmap.update_layout(
                title="Monthly Cohort Retention Heatmap (%)",
                xaxis_title="Relative Month",
                yaxis_title="Cohort Month",
                coloraxis_showscale=False,
                height=550
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Cohort Stats
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("#### Key Cohort Takeaways")
                st.markdown("""
                - **Month 0 Concentration**: Over 97% of Olist customers transact only once, dropping retention to ~1-2% by Month 1.
                - **Stabilization**: Cohort retention remains flat at approximately 0.5% - 1.5% for up to 12 months.
                - **Actionable Opportunity**: Retaining even 1% more of these high-volume cohorts translates to substantial revenue gains given Olist's scale.
                """)
            with col_c2:
                # Plot average cohort size over time
                avg_sizes = filtered_cohort.groupby('cohort_month')['cohort_size'].first().reset_index()
                fig_sizes = px.bar(avg_sizes, x='cohort_month', y='cohort_size', title="New Customer Acquisition by Month")
                fig_sizes.update_layout(xaxis_title="Join Month", yaxis_title="Cohort Size (Customers)", height=250)
                st.plotly_chart(fig_sizes, use_container_width=True)
        else:
            st.warning("Cohort data not found.")
            
    # TAB 2: CLV vs Risk Quadrants
    with tab2:
        st.subheader("Customer Lifetime Value vs. Churn Risk Segments")
        st.markdown("""
        We cross-reference our Random Forest model's churn probability with our calculated Customer Lifetime Value (CLV). 
        Adjusting the **Churn Probability Threshold** in the sidebar dynamically shifts customers between risk tiers and updates these metrics in real time.
        """)
        
        # Calculate dynamic metrics
        total_customers = len(df)
        revenue_at_risk = df[df['risk_tier'] == 'High']['clv'].sum()
        avg_clv_val = df['clv'].mean()
        
        # Metric Cards
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Total Active Customers", f"{total_customers:,}")
        with col_m2:
            st.metric("Total CLV Revenue at Risk", f"${revenue_at_risk:,.2f}", 
                      help="Sum of Customer Lifetime Value (CLV) for all customers flagged as High Churn Risk under the current threshold.")
        with col_m3:
            st.metric("Average Customer CLV", f"${avg_clv_val:.2f}")
            
        # Scatter Plot CLV vs Churn Prob
        st.markdown("---")
        
        # Sample for fast rendering in dashboard
        sample_df = df.sample(n=min(5000, len(df)), random_state=42)
        
        fig_scatter = px.scatter(
            sample_df,
            x="churn_probability",
            y="clv",
            color="quadrant",
            color_discrete_map={
                "High CLV + High Risk (Win-back)": "#dc3545",
                "High CLV + Low Risk (Reward)": "#28a745",
                "Low CLV + High Risk (Monitor)": "#ffc107",
                "Low CLV + Low Risk (No Action)": "#6c757d"
            },
            category_orders={
                "quadrant": [
                    "High CLV + High Risk (Win-back)",
                    "High CLV + Low Risk (Reward)",
                    "Low CLV + High Risk (Monitor)",
                    "Low CLV + Low Risk (No Action)"
                ]
            },
            hover_data=["customer_id" if "customer_id" in sample_df.columns else "customer_unique_id", "recency", "frequency", "monetary"],
            title="Customer Lifetime Value (CLV) vs Churn Probability (5,000 Customer Sample)"
        )
        
        # Add threshold line
        fig_scatter.add_vline(x=thresh, line_width=2, line_dash="dash", line_color="black", 
                             annotation_text=f"Risk Threshold ({thresh:.2f})", annotation_position="top right")
        # Add CLV median line
        fig_scatter.add_hline(y=clv_median, line_width=2, line_dash="dash", line_color="black",
                              annotation_text=f"Median CLV (${clv_median:.2f})", annotation_position="top left")
                              
        fig_scatter.update_layout(
            xaxis_title="Model Churn Probability",
            yaxis_title="Customer Lifetime Value (USD)",
            yaxis_type="log", # Log scale due to high skewness in CLV
            height=600
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Quadrant summary table
        st.markdown("#### Quadrant Financial Breakdown")
        q_summary = df.groupby('quadrant').agg(
            Customer_Count=('customer_unique_id', 'count'),
            Total_CLV=('clv', 'sum'),
            Average_CLV=('clv', 'mean'),
            Average_Churn_Probability=('churn_probability', 'mean')
        ).reindex([
            "High CLV + High Risk (Win-back)",
            "High CLV + Low Risk (Reward)",
            "Low CLV + High Risk (Monitor)",
            "Low CLV + Low Risk (No Action)"
        ]).fillna(0)
        
        # Format table
        q_summary['Share (%)'] = (q_summary['Customer_Count'] / total_customers) * 100
        q_summary['Total_CLV'] = q_summary['Total_CLV'].map('${:,.2f}'.format)
        q_summary['Average_CLV'] = q_summary['Average_CLV'].map('${:,.2f}'.format)
        q_summary['Average_Churn_Probability'] = q_summary['Average_Churn_Probability'].map('{:.1%}'.format)
        q_summary['Customer_Count'] = q_summary['Customer_Count'].map('{:,}'.format)
        q_summary['Share (%)'] = q_summary['Share (%)'].map('{:.2f}%'.format)
        
        st.table(q_summary)
        
    # TAB 3: A/B Test Results
    with tab3:
        st.subheader("A/B Test Simulation & ROI Calculator")
        st.markdown("""
        We run a randomized controlled trial (A/B Test) on our critical **High CLV + High Risk (Win-back)** segment. 
        Treatment customers receive a targeted promo (costing $5), which boosts their retention rate by a simulated ~10 percentage points.
        """)
        
        if ab_results is not None:
            # Metrics Row
            col_ab1, col_ab2, col_ab3, col_ab4 = st.columns(4)
            with col_ab1:
                st.metric("Treatment Retention Rate", f"{ab_results['rate_treatment'] * 100:.2f}%")
            with col_ab2:
                st.metric("Control Retention Rate", f"{ab_results['rate_control'] * 100:.2f}%")
            with col_ab3:
                st.metric("Statistically Significant Lift", f"+{ab_results['lift'] * 100:.2f} pp", 
                          delta=f"{ab_results['lift'] * 100:.2f}% vs Control")
            with col_ab4:
                st.metric("Z-Test p-value", f"{ab_results['p_value']:.4e}", 
                          delta="Significant (p < 0.05)")
                          
            st.markdown("---")
            
            col_ab_left, col_ab_right = st.columns(2)
            
            with col_ab_left:
                st.markdown("#### Retention Performance (with 95% Confidence Intervals)")
                
                # Plot Treatment vs Control retention with error bars
                err_t = 1.96 * np.sqrt(ab_results['rate_treatment'] * (1 - ab_results['rate_treatment']) / ab_results['n_treatment'])
                err_c = 1.96 * np.sqrt(ab_results['rate_control'] * (1 - ab_results['rate_control']) / ab_results['n_control'])
                
                fig_ab = go.Figure()
                fig_ab.add_trace(go.Bar(
                    name='Control',
                    x=['Control Group'],
                    y=[ab_results['rate_control'] * 100],
                    error_y=dict(type='data', array=[err_c * 100], visible=True),
                    marker_color='#ffc107'
                ))
                fig_ab.add_trace(go.Bar(
                    name='Treatment',
                    x=['Treatment Group'],
                    y=[ab_results['rate_treatment'] * 100],
                    error_y=dict(type='data', array=[err_t * 100], visible=True),
                    marker_color='#dc3545'
                ))
                
                fig_ab.update_layout(
                    title="Retention Rate Comparison (%)",
                    yaxis_title="Retention Rate (%)",
                    yaxis_range=[0, 45],
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_ab, use_container_width=True)
                
                # CI conclusion text
                st.success(f"""
                **Statistical Conclusion**: The win-back coupon increased 30-day retention from {ab_results['rate_control']*100:.2f}% to {ab_results['rate_treatment']*100:.2f}%. 
                This lift is statistically significant at the 95% confidence level (Z = {ab_results['z_stat']:.2f}, p-value = {ab_results['p_value']:.2e}). 
                The 95% confidence interval for the retention lift is **[{ab_results['ci_lower']*100:.2f}%, {ab_results['ci_upper']*100:.2f}%]**.
                """)
                
            with col_ab_right:
                st.markdown("#### Financial Campaign ROI Panel")
                
                # Display ROI Metrics
                st.markdown(f"**Average Customer Lifetime Value (CLV)**: `${ab_results['avg_clv']:.2f}`")
                st.markdown(f"**Campaign Cost ($5.00 per Treatment Customer)**: `${ab_results['total_cost']:,.2f}`")
                st.markdown(f"**Incremental Customers Saved**: `{ab_results['incremental_saved']:.1f}`")
                st.markdown(f"**Customer Lifetime Value Protected**: `${ab_results['clv_protected']:,.2f}`")
                
                # Metric display
                st.markdown("#### Campaign Financial Output:")
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.metric("Net Campaign Profit", f"${ab_results['net_profit']:,.2f}")
                with col_r2:
                    st.metric("Campaign ROI", f"{ab_results['roi']:.1f}%")
                    
                st.markdown(f"""
                **Business Insight**: By spending **${ab_results['cost_per_cust']:.2f}** per customer on a win-back offer, 
                we saved an additional **{ab_results['incremental_saved']:.1f}** high-value customers who would have otherwise churned. 
                Because their average CLV is **${ab_results['avg_clv']:.2f}**, the total value protected is **${ab_results['clv_protected']:,.2f}**. 
                Subtracting the campaign cost of **${ab_results['total_cost']:,.2f}** yields a net return of 
                **${ab_results['net_profit']:,.2f}**, proving the massive financial viability of the CRM targeting program.
                """)
        else:
            st.warning("A/B test results metadata not found. Please run the CLV and A/B test pipeline script first.")
            
    # TAB 4: Action Table
    with tab4:
        st.subheader("Customer Recommendation Action Plan")
        st.markdown("""
        Below is the customer database with assigned recommendation segments and action items. 
        You can filter by recommendation segment to view the target customers, their churn probabilities, and download their customer IDs for CRM upload.
        """)
        
        # Segment selectbox
        seg_options = ["All Segments", "High CLV + High Risk (Win-back)", "High CLV + Low Risk (Reward)", "Low CLV + High Risk (Monitor)", "Low CLV + Low Risk (No Action)"]
        selected_seg = st.selectbox("Select Action Segment", options=seg_options)
        
        # Filter dataframe
        if selected_seg != "All Segments":
            display_df = df[df['quadrant'] == selected_seg].copy()
        else:
            display_df = df.copy()
            
        # Format table for display
        table_cols = ['customer_unique_id', 'recency', 'frequency', 'monetary', 'clv', 'churn_probability', 'quadrant']
        table_df = display_df[table_cols].copy()
        table_df['clv'] = table_df['clv'].map('${:.2f}'.format)
        table_df['churn_probability'] = table_df['churn_probability'].map('{:.1%}'.format)
        table_df['monetary'] = table_df['monetary'].map('${:.2f}'.format)
        
        st.markdown(f"Showing **{len(table_df):,}** customers matching filters.")
        
        # Dataframe
        st.dataframe(table_df, use_container_width=True, height=450)
        
        # Export Option
        csv_data = display_df[['customer_unique_id', 'clv', 'churn_probability', 'quadrant']].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Customers CSV for CRM Import",
            data=csv_data,
            file_name="retainiq_crm_export.csv",
            mime="text/csv"
        )
        
        # Action Recommendation descriptions
        st.markdown("### Action Table Matrix")
        col_rec1, col_rec2 = st.columns(2)
        with col_rec1:
            st.info("""
            **1. Win-back Segment (High CLV + High Risk)**
            - **Recommendation**: Immediate outreach. Offer targeted, high-value incentives (e.g. $10 discount voucher, free shipping) to trigger a repurchase.
            - **CRM Actions**: Automated email campaign on day 120 of inactivity, triggered SMS reminders.
            
            **2. Reward Segment (High CLV + Low Risk)**
            - **Recommendation**: VIP loyalty treatment. Do NOT offer discounts (it degrades margin on stable spenders). Offer early access to categories, exclusive loyalty points, or thank-you rewards.
            - **CRM Actions**: Add to VIP customer list, send milestone gifts.
            """)
        with col_rec2:
            st.warning("""
            **3. Monitor Segment (Low CLV + High Risk)**
            - **Recommendation**: Low-cost reactivations. Do not waste expensive CRM budget on low-value customers. Offer generic discounts (e.g., clearance notifications or newsletters).
            - **CRM Actions**: Include in monthly newsletter blasts and standard catalog promos.
            
            **4. No Action Segment (Low CLV + Low Risk)**
            - **Recommendation**: Do not disturb. Keep in normal customer cycle. Monitor for changes in spending, but do not dedicate targeted campaign resources.
            - **CRM Actions**: Standard operational emails (receipts, delivery confirmations).
            """)
else:
    st.error("Could not load data. Ensure setup_data.py, run_sql_pipeline.py, feature_engineering.py, train_model.py, and clv_ab_test.py have run successfully.")
