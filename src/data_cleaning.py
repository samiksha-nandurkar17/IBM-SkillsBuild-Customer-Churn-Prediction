"""
Module: data_cleaning.py
Project: Customer Churn Prediction & Risk Analytics Using E-Commerce Data
Organization: IBM SkillsBuild Internship

Description:
This module handles data ingestion, structural validation, anomaly detection,
cleaning of transactional e-commerce data, and persistence of the cleaned dataset.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load raw transaction data from an Excel (.xlsx/.xls) or CSV file.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the raw dataset.
        
    Returns:
    --------
    pd.DataFrame:
        Raw dataframe.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {path.resolve()}")
    
    print(f"[INFO] Ingesting raw dataset from: {path.name}")
    if path.suffix.lower() in ['.xlsx', '.xls']:
        # Inspect sheet names
        excel_file = pd.ExcelFile(path)
        sheet = 'Raw_Data' if 'Raw_Data' in excel_file.sheet_names else excel_file.sheet_names[0]
        df = pd.read_excel(path, sheet_name=sheet)
        print(f"[INFO] Loaded sheet '{sheet}' from Excel.")
    elif path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
        print("[INFO] Loaded CSV dataset.")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Expected .xlsx, .xls, or .csv.")
        
    return df


def inspect_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform a comprehensive structural inspection of the raw dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input raw dataframe.
        
    Returns:
    --------
    dict:
        Dictionary containing inspection metrics.
    """
    metrics = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_orders": int(df['Order_ID'].nunique()) if 'Order_ID' in df.columns else None,
        "unique_customers": int(df['Customer_ID'].nunique(dropna=True)) if 'Customer_ID' in df.columns else None,
    }
    
    print("=" * 60)
    print(" RAW DATASET INSPECTION SUMMARY ")
    print("=" * 60)
    print(f"Total Rows:             {metrics['num_rows']:,}")
    print(f"Total Columns:          {metrics['num_cols']}")
    print(f"Duplicate Rows:         {metrics['duplicate_rows']:,}")
    print(f"Unique Orders:          {metrics['unique_orders']}")
    print(f"Unique Customers:       {metrics['unique_customers']}")
    print("\nMissing Values per Column:")
    for col, null_count in metrics["missing_values"].items():
        pct = (null_count / metrics['num_rows']) * 100
        print(f" - {col:<15}: {null_count:>5} missing ({pct:>5.1f}%)")
    print("=" * 60)
    
    return metrics


def clean_currency_value(value: Any) -> float:
    """
    Parse and clean currency values containing symbols like '₹', '$', 'INR', commas, or whitespace.
    
    Parameters:
    -----------
    value : Any
        Raw revenue value.
        
    Returns:
    --------
    float:
        Clean positive float value or np.nan.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value) if value > 0 else np.nan
    
    text = str(value).strip()
    # Remove non-numeric characters except decimal points and negative signs
    clean_str = re.sub(r'[^\d.-]', '', text)
    try:
        val = float(clean_str)
        return val if val > 0 else np.nan
    except ValueError:
        return np.nan


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute end-to-end data cleaning pipeline:
    1. Deduplicate identical rows and duplicate Order IDs.
    2. Validate and drop missing Customer IDs (cannot predict customer churn without identity).
    3. Parse and standardize Order_Date into datetime; discard invalid calendar dates (e.g. Feb 31).
    4. Validate Quantity to ensure numeric and positive values (> 0).
    5. Cleanse Revenue field by stripping currency symbols and enforcing positive monetary amounts.
    6. Ensure numeric Profit and clean categorical text fields.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe.
        
    Returns:
    --------
    Tuple[pd.DataFrame, dict]:
        Cleaned dataframe and a dictionary of before/after cleaning metrics.
    """
    stats = {}
    stats["initial_rows"] = len(df)
    stats["initial_cols"] = len(df.columns)
    
    # 1. Deduplication
    df_clean = df.drop_duplicates().copy()
    stats["rows_after_dedup"] = len(df_clean)
    stats["duplicate_records_removed"] = stats["initial_rows"] - stats["rows_after_dedup"]
    
    # 2. Customer_ID validation
    missing_cust = df_clean['Customer_ID'].isna().sum()
    df_clean = df_clean.dropna(subset=['Customer_ID']).copy()
    stats["missing_customer_ids_removed"] = int(missing_cust)
    
    # 3. Order_Date parsing and calendar validation
    df_clean['Order_Date'] = pd.to_datetime(df_clean['Order_Date'], errors='coerce')
    invalid_dates = df_clean['Order_Date'].isna().sum()
    df_clean = df_clean.dropna(subset=['Order_Date']).copy()
    stats["invalid_dates_removed"] = int(invalid_dates)
    
    # 4. Quantity cleaning (positive integers)
    df_clean['Quantity'] = pd.to_numeric(df_clean['Quantity'], errors='coerce')
    invalid_qty = (df_clean['Quantity'].isna()) | (df_clean['Quantity'] <= 0)
    stats["invalid_or_negative_qty_removed"] = int(invalid_qty.sum())
    df_clean = df_clean[df_clean['Quantity'] > 0].copy()
    
    # 5. Revenue cleaning
    df_clean['Revenue'] = df_clean['Revenue'].apply(clean_currency_value)
    invalid_rev = df_clean['Revenue'].isna().sum()
    stats["invalid_or_missing_revenue_removed"] = int(invalid_rev)
    df_clean = df_clean.dropna(subset=['Revenue']).copy()
    
    # 6. Profit cleaning
    df_clean['Profit'] = pd.to_numeric(df_clean['Profit'], errors='coerce')
    
    # 7. Text field standardization
    for col in ['Product', 'Category', 'Region']:
        if col in df_clean.columns:
            df_clean[col] = (
                df_clean[col]
                .astype(str)
                .str.strip()
                .replace({'nan': np.nan, 'None': np.nan, '': np.nan})
            )
    
    stats["final_rows"] = len(df_clean)
    stats["final_cols"] = len(df_clean.columns)
    stats["final_unique_customers"] = int(df_clean['Customer_ID'].nunique())
    stats["final_unique_orders"] = int(df_clean['Order_ID'].nunique())
    stats["min_date"] = df_clean['Order_Date'].min().strftime('%Y-%m-%d')
    stats["max_date"] = df_clean['Order_Date'].max().strftime('%Y-%m-%d')
    stats["date_span_days"] = (df_clean['Order_Date'].max() - df_clean['Order_Date'].min()).days
    stats["total_revenue"] = float(df_clean['Revenue'].sum())
    
    print("\n" + "=" * 60)
    print(" DATA CLEANING PIPELINE RESULTS ")
    print("=" * 60)
    print(f"Raw Records:                   {stats['initial_rows']:,}")
    print(f"Duplicate Rows Removed:        {stats['duplicate_records_removed']:,}")
    print(f"Missing Customer IDs Dropped:  {stats['missing_customer_ids_removed']:,}")
    print(f"Invalid Dates Dropped:         {stats['invalid_dates_removed']:,}")
    print(f"Negative/Invalid Qty Dropped:  {stats['invalid_or_negative_qty_removed']:,}")
    print(f"Invalid Revenue Dropped:       {stats['invalid_or_missing_revenue_removed']:,}")
    print(f"Final Cleaned Records:         {stats['final_rows']:,}")
    print(f"Cleaned Unique Customers:      {stats['final_unique_customers']:,}")
    print(f"Cleaned Unique Orders:         {stats['final_unique_orders']:,}")
    print(f"Transaction Date Span:         {stats['min_date']} to {stats['max_date']} ({stats['date_span_days']} days)")
    print(f"Total Validated Revenue:       INR {stats['total_revenue']:,.2f}")
    print("=" * 60)
    
    return df_clean, stats


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save cleaned dataframe to CSV format.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe.
    output_path : str or Path
        Target CSV file path.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"[SUCCESS] Cleaned data successfully saved to: {out_file.resolve()}")


if __name__ == "__main__":
    # Test script standalone
    project_root = Path(__file__).resolve().parent.parent
    raw_csv = project_root / "data" / "raw" / "ecommerce_raw_dataset.xlsx"
    processed_csv = project_root / "data" / "processed" / "cleaned_ecommerce_dataset.csv"
    
    raw_df = load_raw_data(raw_csv)
    inspect_raw_data(raw_df)
    clean_df, clean_stats = clean_dataset(raw_df)
    save_processed_data(clean_df, processed_csv)
