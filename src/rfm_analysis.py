"""
Module: rfm_analysis.py
Project: Customer Churn Prediction & Risk Analytics Using E-Commerce Data
Organization: IBM SkillsBuild Internship

Description:
This module creates customer-level aggregated profiles from transaction logs,
computes Recency, Frequency, Monetary (RFM) metrics, applies a justified
churn definition based on the observation window, generates RFM & churn
visualizations, and saves the customer summary dataset.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def build_customer_level_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction records to ensure exactly ONE row per unique Customer_ID.
    
    Calculates:
    - Order_Count (Frequency)
    - Total_Revenue (Monetary)
    - Avg_Order_Value (Total_Revenue / Order_Count)
    - First_Purchase (earliest order date)
    - Most_Recent_Purchase (latest order date)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned transaction dataframe.
        
    Returns:
    --------
    pd.DataFrame:
        Customer-level dataframe (1 row per Customer_ID).
    """
    df = df.copy()
    if not np.issubdtype(df['Order_Date'].dtype, np.datetime64):
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        
    cust_df = df.groupby('Customer_ID').agg(
        Order_Count=('Order_ID', 'nunique'),
        Total_Revenue=('Revenue', 'sum'),
        Avg_Order_Value=('Revenue', 'mean'),
        First_Purchase=('Order_Date', 'min'),
        Most_Recent_Purchase=('Order_Date', 'max')
    ).reset_index()
    
    # Validation check: Ensure one Customer_ID = one row
    assert cust_df['Customer_ID'].is_unique, "Customer_ID is not unique in customer summary!"
    assert len(cust_df) == df['Customer_ID'].nunique(), "Customer count mismatch between transactions and summary!"
    
    print(f"[INFO] Customer aggregation verified: Exactly {len(cust_df)} unique customer records created.")
    return cust_df


def calculate_rfm(cust_df: pd.DataFrame, reference_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary (RFM) metrics.
    
    Definitions:
    - Recency: Days elapsed since customer's most recent purchase date relative to the reference date.
    - Frequency: Total count of distinct orders placed by the customer.
    - Monetary: Total revenue amount spent across all transactions.
    - Avg_Order_Value: Average spend per order.
    
    Parameters:
    -----------
    cust_df : pd.DataFrame
        Customer-level aggregated dataframe.
    reference_date : pd.Timestamp, optional
        Date anchor for recency computation. Defaults to maximum purchase date in dataset.
        
    Returns:
    --------
    pd.DataFrame:
        Customer dataframe enriched with RFM columns.
    """
    cust_df = cust_df.copy()
    if reference_date is None:
        reference_date = cust_df['Most_Recent_Purchase'].max()
        
    print(f"[INFO] Recency calculated relative to reference date: {reference_date.strftime('%Y-%m-%d')}")
    
    cust_df['Recency'] = (reference_date - cust_df['Most_Recent_Purchase']).dt.days
    cust_df['Frequency'] = cust_df['Order_Count']
    cust_df['Monetary'] = cust_df['Total_Revenue']
    
    return cust_df


def define_churn_status(cust_df: pd.DataFrame, threshold_days: int = 180) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply business churn definition based on customer inactivity duration.
    
    Justification:
    The dataset spans 341 days (January 1, 2026 to December 8, 2026).
    In an e-commerce context, a customer who has made zero purchases for 180 days
    (approx. 6 months, representing > 52% of the active transaction window) has
    effectively lapsed or churned.
    
    Definition:
    - Churn_Status = 1 (Churned) if Recency >= 180 days
    - Churn_Status = 0 (Not Churned / Active) if Recency < 180 days
    
    Parameters:
    -----------
    cust_df : pd.DataFrame
        Customer dataframe with 'Recency' column.
    threshold_days : int
        Inactivity threshold in days. Default is 180.
        
    Returns:
    --------
    Tuple[pd.DataFrame, dict]:
        Dataframe with 'Churn_Status' and dictionary of churn statistics.
    """
    cust_df = cust_df.copy()
    cust_df['Churn_Status'] = (cust_df['Recency'] >= threshold_days).astype(int)
    
    churn_counts = cust_df['Churn_Status'].value_counts()
    churned_count = int(churn_counts.get(1, 0))
    active_count = int(churn_counts.get(0, 0))
    total_customers = len(cust_df)
    churn_rate = (churned_count / total_customers) * 100
    
    churn_stats = {
        "churn_threshold_days": threshold_days,
        "total_customers": total_customers,
        "churned_customers": churned_count,
        "active_customers": active_count,
        "churn_percentage": float(churn_rate),
        "active_percentage": 100.0 - float(churn_rate)
    }
    
    print("\n" + "=" * 60)
    print(" CHURN DEFINITION & DISTRIBUTION ")
    print("=" * 60)
    print(f"Inactivity Threshold:       >= {threshold_days} days without purchase")
    print(f"Total Customer Cohort:      {total_customers:,}")
    print(f"Active Customers (0):       {active_count:,} ({churn_stats['active_percentage']:.2f}%)")
    print(f"Churned Customers (1):      {churned_count:,} ({churn_stats['churn_percentage']:.2f}%)")
    print("=" * 60)
    
    return cust_df, churn_stats


def generate_rfm_figures(cust_df: pd.DataFrame, output_dir: str | Path) -> None:
    """
    Generate and save RFM distribution and Churn status visualizations.
    
    Parameters:
    -----------
    cust_df : pd.DataFrame
        Customer dataframe with RFM and Churn_Status.
    output_dir : str or Path
        Target directory for figure exports.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------
    # Figure 8: RFM Multi-Panel Distributions
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel 1: Recency
    sns.histplot(cust_df['Recency'], bins=30, kde=True, color='#2563eb', ax=axes[0])
    axes[0].axvline(180, color='red', linestyle='--', linewidth=2, label='Churn Threshold (180d)')
    axes[0].set_title("Recency Distribution (Days)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Days Since Last Purchase", fontsize=11)
    axes[0].set_ylabel("Customer Count", fontsize=11)
    axes[0].legend(loc='upper right')
    
    # Panel 2: Frequency
    sns.histplot(cust_df['Frequency'], bins=25, color='#16a34a', ax=axes[1])
    axes[1].set_title("Frequency Distribution (Order Count)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Total Orders per Customer", fontsize=11)
    axes[1].set_ylabel("Customer Count", fontsize=11)
    
    # Panel 3: Monetary
    sns.histplot(cust_df['Monetary'], bins=30, kde=True, color='#9333ea', ax=axes[2])
    axes[2].set_title("Monetary Distribution (Total Spend)", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Total Spend (INR)", fontsize=11)
    axes[2].set_ylabel("Customer Count", fontsize=11)
    axes[2].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"₹{x*1e-3:.0f}K" if x < 1e6 else f"₹{x*1e-6:.1f}M"))
    
    plt.suptitle("8. Customer RFM Feature Distributions", fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig8_path = out_path / "08_rfm_distributions.png"
    plt.savefig(fig8_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------
    # Figure 9: Churn Status Distribution (Bar + Donut Chart)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    counts = cust_df['Churn_Status'].value_counts()
    labels = ['Not Churned (Active)', 'Churned (Inactive >=180d)']
    vals = [counts.get(0, 0), counts.get(1, 0)]
    palette = ['#22c55e', '#ef4444']
    
    # Bar Chart
    bars = ax1.bar(labels, vals, color=palette, edgecolor='black', width=0.5)
    ax1.set_title("Customer Churn Count", fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylabel("Number of Customers", fontsize=11)
    for b in bars:
        h = b.get_height()
        ax1.annotate(f"{h:,}\n({h/len(cust_df):.1%})", (b.get_x() + b.get_width() / 2., h / 2),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # Donut Chart
    wedges, texts, autotexts = ax2.pie(
        vals,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=palette,
        wedgeprops=dict(width=0.45, edgecolor='black')
    )
    plt.setp(autotexts, size=11, weight='bold', color='white')
    ax2.set_title("Customer Churn Proportion", fontsize=13, fontweight='bold', pad=10)
    
    plt.suptitle("9. Customer Churn Status Overview (Inactivity >= 180 Days)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig9_path = out_path / "09_churn_status_distribution.png"
    plt.savefig(fig9_path, dpi=300)
    plt.close()
    
    print("[SUCCESS] Exported Figure 08 (RFM Distributions) and Figure 09 (Churn Status Distribution).")


def run_rfm_pipeline(clean_df: pd.DataFrame, output_dir: str | Path, processed_dir: str | Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute full customer aggregation, RFM calculation, churn labeling, and visualization.
    
    Parameters:
    -----------
    clean_df : pd.DataFrame
        Cleaned transaction dataset.
    output_dir : str or Path
        Directory for figure exports.
    processed_dir : str or Path
        Directory for customer summary CSV export.
        
    Returns:
    --------
    Tuple[pd.DataFrame, dict]:
        Customer RFM dataframe and churn summary dictionary.
    """
    cust_df = build_customer_level_data(clean_df)
    cust_df = calculate_rfm(cust_df)
    cust_df, churn_stats = define_churn_status(cust_df, threshold_days=180)
    
    generate_rfm_figures(cust_df, output_dir)
    
    # Export customer summary CSV
    summary_path = Path(processed_dir) / "customer_rfm_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cust_df.to_csv(summary_path, index=False)
    print(f"[SUCCESS] Customer RFM summary dataset saved to: {summary_path.resolve()}")
    
    return cust_df, churn_stats


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    clean_csv = project_root / "data" / "processed" / "cleaned_ecommerce_dataset.csv"
    fig_dir = project_root / "outputs" / "figures"
    proc_dir = project_root / "data" / "processed"
    
    if not clean_csv.exists():
        print("[ERROR] Cleaned dataset not found. Run data_cleaning.py first.")
    else:
        df = pd.read_csv(clean_csv)
        run_rfm_pipeline(df, fig_dir, proc_dir)
