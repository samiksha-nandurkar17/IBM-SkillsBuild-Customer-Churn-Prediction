"""
================================================================================
IBM SKILLSBUILD FINAL PROJECT: AUTOMATED END-TO-END PIPELINE RUNNER
Title: Customer Churn Prediction & Risk Analytics Using E-Commerce Data
Track: Data Analytics with AI (Masterclass 3: AI Predicts What Happens Next)
================================================================================

This master script coordinates the entire analytics and machine learning workflow:
1. Workspace verification and dataset ingestion
2. Robust data cleaning and validation
3. Exploratory Data Analysis (EDA) and visualization generation
4. Customer-level aggregation and RFM feature engineering
5. Date-window validated churn definition
6. Target leakage audit and mitigation
7. Scikit-learn Pipeline construction (StandardScaler + Logistic Regression)
8. Model training, evaluation, and diagnostic curves
9. Churn probability scoring via predict_proba()
10. Calibrated risk tier classification (High, Medium, Low)
11. Final customer risk table formatting and CSV exports
12. Data-driven business insights and targeted retention recommendations
"""

import sys
import time
from pathlib import Path
import pandas as pd

# Add src to system path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import load_raw_data, inspect_raw_data, clean_dataset, save_processed_data
from src.eda import run_eda
from src.rfm_analysis import run_rfm_pipeline
from src.churn_model import run_churn_modeling_pipeline
from src.risk_analysis import run_risk_analysis_pipeline

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def print_banner(step_num: int, title: str) -> None:
    """Print standard styled progress banner."""
    print("\n" + "=" * 75)
    print(f" STEP {step_num:02d} | {title.upper()}")
    print("=" * 75)


def run_pipeline():
    start_time = time.time()
    
    print("*" * 75)
    print(" IBM SKILLSBUILD FINAL PROJECT PIPELINE EXECUTION ")
    print(" Customer Churn Prediction & Risk Analytics Using E-Commerce Data ")
    print("*" * 75)
    
    # Define directories
    raw_data_file = PROJECT_ROOT / "data" / "raw" / "ecommerce_raw_dataset.xlsx"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    models_dir = PROJECT_ROOT / "outputs" / "models"
    predictions_dir = PROJECT_ROOT / "outputs" / "predictions"
    reports_dir = PROJECT_ROOT / "reports"
    
    # -------------------------------------------------------------
    # Step 1: Load Raw Dataset
    # -------------------------------------------------------------
    print_banner(1, "Data Ingestion & Verification")
    if not raw_data_file.exists():
        raise FileNotFoundError(f"Missing raw dataset at {raw_data_file.resolve()}. Please place the dataset in data/raw/.")
    raw_df = load_raw_data(raw_data_file)
    
    # -------------------------------------------------------------
    # Step 2: Inspect Raw Data
    # -------------------------------------------------------------
    print_banner(2, "Raw Dataset Inspection")
    inspect_raw_data(raw_df)
    
    # -------------------------------------------------------------
    # Step 3: Clean Dataset
    # -------------------------------------------------------------
    print_banner(3, "Data Cleaning Pipeline")
    clean_df, clean_stats = clean_dataset(raw_df)
    cleaned_csv_path = processed_dir / "cleaned_ecommerce_dataset.csv"
    save_processed_data(clean_df, cleaned_csv_path)
    
    # -------------------------------------------------------------
    # Step 4: Exploratory Data Analysis
    # -------------------------------------------------------------
    print_banner(4, "Exploratory Data Analysis (EDA)")
    eda_stats = run_eda(clean_df, figures_dir)
    
    # -------------------------------------------------------------
    # Step 5: Customer-Level Aggregation & RFM Analysis
    # -------------------------------------------------------------
    print_banner(5, "Customer-Level RFM & Churn Definition")
    cust_rfm_df, churn_stats = run_rfm_pipeline(clean_df, figures_dir, processed_dir)
    
    # -------------------------------------------------------------
    # Step 6: Churn Modeling & Evaluation
    # -------------------------------------------------------------
    print_banner(6, "Target Leakage Audit & Logistic Regression Modeling")
    scored_cust_df, pipeline_model, model_metrics = run_churn_modeling_pipeline(
        cust_rfm_df, figures_dir, models_dir
    )
    
    # -------------------------------------------------------------
    # Step 7: Risk Stratification & Business Recommendations
    # -------------------------------------------------------------
    print_banner(7, "Customer Risk Analytics & Strategic Insights")
    risk_table, top20, insights, recommendations = run_risk_analysis_pipeline(
        scored_cust_df, figures_dir, predictions_dir, PROJECT_ROOT
    )
    
    # -------------------------------------------------------------
    # Verification & Summary
    # -------------------------------------------------------------
    print_banner(8, "Pipeline Verification & Final Checklist")
    exported_root_csv = PROJECT_ROOT / "customer_churn_risk_predictions.csv"
    exported_pred_csv = predictions_dir / "customer_churn_risk_predictions.csv"
    saved_model_file = models_dir / "logistic_regression_churn_model.joblib"
    
    assert exported_root_csv.exists() and exported_root_csv.stat().st_size > 0, "Root CSV export failed!"
    assert exported_pred_csv.exists() and exported_pred_csv.stat().st_size > 0, "Predictions CSV export failed!"
    assert saved_model_file.exists(), "Model persistence failed!"
    
    fig_count = len(list(figures_dir.glob("*.png")))
    assert fig_count >= 12, f"Expected at least 12 figures, found {fig_count}."
    
    elapsed = time.time() - start_time
    print(f"\n[ALL CHECKS PASSED] Pipeline executed successfully in {elapsed:.2f} seconds!")
    print(f" - Cleaned Transactions:    {clean_stats['final_rows']:,}")
    print(f" - Unique Customer Cohort:  {len(risk_table):,}")
    print(f" - Model Test Accuracy:     {model_metrics['accuracy']*100:.2f}%")
    print(f" - Model Test Precision:    {model_metrics['precision']*100:.2f}%")
    print(f" - Model Test Recall:       {model_metrics['recall']*100:.2f}%")
    print(f" - Total Figures Generated: {fig_count}")
    print(f" - Model Artifact:          {saved_model_file.name}")
    print(f" - Risk Table Output:       {exported_root_csv.name}")
    print("=" * 75)
    print("\nProject execution is 100% complete and submission-ready.\n")


if __name__ == "__main__":
    run_pipeline()
