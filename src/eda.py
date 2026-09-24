"""
Module: eda.py
Project: Customer Churn Prediction & Risk Analytics Using E-Commerce Data
Organization: IBM SkillsBuild Internship

Description:
This module performs comprehensive Exploratory Data Analysis (EDA) on the cleaned
e-commerce transaction dataset. It generates 7 publication-ready figures,
calculates descriptive statistics, and outputs data-driven business interpretations.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning)

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Configure plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8


def run_eda(df: pd.DataFrame, output_dir: str | Path) -> Dict[str, Any]:
    """
    Generate all 7 required exploratory visualizations and return summary statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned transaction dataframe.
    output_dir : str or Path
        Directory to store output figure image files.
        
    Returns:
    --------
    dict:
        Descriptive statistics and interpretations for EDA.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    df = df.copy()
    if not np.issubdtype(df['Order_Date'].dtype, np.datetime64):
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        
    eda_stats = {}
    
    # -------------------------------------------------------------
    # 1. Revenue Distribution (Histogram + KDE)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['Revenue'], bins=50, kde=True, color='#1f77b4', edgecolor='black', ax=ax)
    ax.set_title("1. Transaction Revenue Distribution", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Order Revenue (INR)", fontsize=12)
    ax.set_ylabel("Transaction Count", fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"₹{x*1e-3:.0f}K" if x < 1e6 else f"₹{x*1e-6:.1f}M"))
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig1_path = out_path / "01_revenue_distribution.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    
    eda_stats['revenue_mean'] = float(df['Revenue'].mean())
    eda_stats['revenue_median'] = float(df['Revenue'].median())
    eda_stats['revenue_max'] = float(df['Revenue'].max())
    eda_stats['revenue_min'] = float(df['Revenue'].min())
    
    # -------------------------------------------------------------
    # 2. Monthly Revenue Trend
    # -------------------------------------------------------------
    df['YearMonth'] = df['Order_Date'].dt.to_period('M')
    monthly_rev = df.groupby('YearMonth')['Revenue'].sum().reset_index()
    monthly_rev['YearMonth_Str'] = monthly_rev['YearMonth'].astype(str)
    
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.lineplot(data=monthly_rev, x='YearMonth_Str', y='Revenue', marker='o', color='#2ca02c', linewidth=2.5, ax=ax)
    ax.set_title("2. Monthly Total Revenue Trend (2026)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Month (Year-Month)", fontsize=12)
    ax.set_ylabel("Total Revenue (INR)", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, p: f"₹{y*1e-6:.1f}M"))
    plt.xticks(rotation=45)
    
    # Annotate peak and lowest months
    peak_idx = monthly_rev['Revenue'].idxmax()
    low_idx = monthly_rev['Revenue'].idxmin()
    ax.annotate(f"Peak: ₹{monthly_rev.loc[peak_idx, 'Revenue']*1e-6:.2f}M",
                (peak_idx, monthly_rev.loc[peak_idx, 'Revenue']),
                textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color='#1b5e20')
    
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig2_path = out_path / "02_monthly_revenue_trend.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    
    eda_stats['monthly_revenue_peak'] = (monthly_rev.loc[peak_idx, 'YearMonth_Str'], float(monthly_rev.loc[peak_idx, 'Revenue']))
    eda_stats['monthly_revenue_low'] = (monthly_rev.loc[low_idx, 'YearMonth_Str'], float(monthly_rev.loc[low_idx, 'Revenue']))
    
    # -------------------------------------------------------------
    # 3. Monthly Order Trend
    # -------------------------------------------------------------
    monthly_orders = df.groupby('YearMonth')['Order_ID'].nunique().reset_index()
    monthly_orders['YearMonth_Str'] = monthly_orders['YearMonth'].astype(str)
    
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=monthly_orders, x='YearMonth_Str', y='Order_ID', color='#3b82f6', ax=ax)
    ax.set_title("3. Monthly Order Volume Trend (2026)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Month (Year-Month)", fontsize=12)
    ax.set_ylabel("Number of Orders", fontsize=12)
    plt.xticks(rotation=45)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    fig3_path = out_path / "03_order_trend.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # 4. Customer Order Frequency Distribution
    # -------------------------------------------------------------
    cust_orders = df.groupby('Customer_ID')['Order_ID'].nunique().reset_index()
    cust_orders.rename(columns={'Order_ID': 'Order_Frequency'}, inplace=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=cust_orders, x='Order_Frequency', color='#ff7f0e', ax=ax)
    ax.set_title("4. Customer Order Frequency Distribution", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Number of Orders Placed per Customer", fontsize=12)
    ax.set_ylabel("Customer Count", fontsize=12)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    fig4_path = out_path / "04_customer_order_frequency.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    
    single_order_count = int((cust_orders['Order_Frequency'] == 1).sum())
    eda_stats['single_order_customers'] = single_order_count
    eda_stats['single_order_pct'] = (single_order_count / len(cust_orders)) * 100
    
    # -------------------------------------------------------------
    # 5. Customer Revenue Distribution
    # -------------------------------------------------------------
    cust_rev = df.groupby('Customer_ID')['Revenue'].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=cust_rev['Revenue'], color='#a855f7', ax=ax)
    ax.set_title("5. Customer Total Lifetime Revenue Distribution (Boxplot)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Total Lifetime Revenue (INR)", fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"₹{x*1e-3:.0f}K" if x < 1e6 else f"₹{x*1e-6:.1f}M"))
    plt.tight_layout()
    fig5_path = out_path / "05_customer_revenue_distribution.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    
    eda_stats['cust_rev_mean'] = float(cust_rev['Revenue'].mean())
    eda_stats['cust_rev_median'] = float(cust_rev['Revenue'].median())
    eda_stats['cust_rev_max'] = float(cust_rev['Revenue'].max())
    
    # -------------------------------------------------------------
    # 6. Top 10 Customers by Revenue
    # -------------------------------------------------------------
    top10_cust = cust_rev.sort_values(by='Revenue', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=top10_cust, x='Revenue', y='Customer_ID', hue='Customer_ID', palette='magma', legend=False, ax=ax)
    ax.set_title("6. Top 10 Customers by Total Revenue", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Total Spend (INR)", fontsize=12)
    ax.set_ylabel("Customer ID", fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"₹{x*1e-6:.2f}M"))
    for p in ax.patches:
        w = p.get_width()
        ax.annotate(f" ₹{w*1e-6:.2f}M", (w, p.get_y() + p.get_height() / 2.),
                    va='center', fontsize=9, fontweight='semibold')
    ax.grid(True, linestyle='--', alpha=0.5, axis='x')
    plt.tight_layout()
    fig6_path = out_path / "06_top_customers_revenue.png"
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    
    top10_total = float(top10_cust['Revenue'].sum())
    eda_stats['top10_revenue_sum'] = top10_total
    eda_stats['top10_revenue_share_pct'] = (top10_total / df['Revenue'].sum()) * 100
    
    # -------------------------------------------------------------
    # 7. Average Order Value (AOV) Distribution
    # -------------------------------------------------------------
    cust_aov = df.groupby('Customer_ID')['Revenue'].mean().reset_index()
    cust_aov.rename(columns={'Revenue': 'Avg_Order_Value'}, inplace=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(cust_aov['Avg_Order_Value'], bins=40, kde=True, color='#0ea5e9', edgecolor='black', ax=ax)
    ax.set_title("7. Customer Average Order Value (AOV) Distribution", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Average Order Value (INR)", fontsize=12)
    ax.set_ylabel("Customer Count", fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"₹{x*1e-3:.0f}K" if x < 1e6 else f"₹{x*1e-6:.1f}M"))
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig7_path = out_path / "07_aov_distribution.png"
    plt.savefig(fig7_path, dpi=300)
    plt.close()
    
    eda_stats['aov_mean'] = float(cust_aov['Avg_Order_Value'].mean())
    eda_stats['aov_median'] = float(cust_aov['Avg_Order_Value'].median())
    eda_stats['aov_max'] = float(cust_aov['Avg_Order_Value'].max())
    
    print("\n" + "=" * 60)
    print(" EXPLORATORY DATA ANALYSIS SUMMARY ")
    print("=" * 60)
    print(f"Transaction Revenue Mean:      INR {eda_stats['revenue_mean']:,.2f}")
    print(f"Transaction Revenue Median:    INR {eda_stats['revenue_median']:,.2f}")
    print(f"Peak Revenue Month:            {eda_stats['monthly_revenue_peak'][0]} (INR {eda_stats['monthly_revenue_peak'][1]:,.2f})")
    print(f"Single-Order Customers:        {eda_stats['single_order_customers']} ({eda_stats['single_order_pct']:.1f}% of customer base)")
    print(f"Top 10 Customers Revenue:      INR {eda_stats['top10_revenue_sum']:,.2f} ({eda_stats['top10_revenue_share_pct']:.1f}% of total revenue)")
    print(f"Average Order Value (Mean):    INR {eda_stats['aov_mean']:,.2f}")
    print(f"Average Order Value (Median):  INR {eda_stats['aov_median']:,.2f}")
    print("All 7 EDA figures exported successfully to outputs/figures/")
    print("=" * 60)
    
    return eda_stats


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    clean_csv = project_root / "data" / "processed" / "cleaned_ecommerce_dataset.csv"
    fig_dir = project_root / "outputs" / "figures"
    
    if not clean_csv.exists():
        print("[ERROR] Cleaned dataset not found. Run data_cleaning.py first.")
    else:
        df = pd.read_csv(clean_csv)
        run_eda(df, fig_dir)
