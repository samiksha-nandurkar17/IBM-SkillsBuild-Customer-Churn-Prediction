"""
Module: churn_model.py
Project: Customer Churn Prediction & Risk Analytics Using E-Commerce Data
Organization: IBM SkillsBuild Internship

Description:
This module performs formal target leakage auditing, trains a scikit-learn
Pipeline with StandardScaler and Logistic Regression, computes evaluation
metrics (Accuracy, Precision, Recall, Confusion Matrix), generates probability
scores via predict_proba(), and persists the trained model artifact.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Any
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def audit_target_leakage(cust_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform a formal target leakage audit on potential prediction features.
    
    Leakage Assessment:
    - Customer_ID: Non-predictive identifier (dropped).
    - Churn_Status: Ground truth target (never used as an input feature).
    - Recency: When Churn_Status is defined directly as (Recency >= 180),
      including Recency in a standard scaled model causes near-perfect artificial
      separation (Target Leakage), because Recency contains the answer in advance.
    - Leakage-Safe Features: Frequency, Monetary, Avg_Order_Value represent
      independent customer purchasing habits and volume, devoid of direct mathematical
      derivation from the target threshold.
      
    Parameters:
    -----------
    cust_df : pd.DataFrame
        Customer summary dataframe.
        
    Returns:
    --------
    dict:
        Correlation matrix and recommendation summary.
    """
    cols_to_check = ['Recency', 'Frequency', 'Monetary', 'Avg_Order_Value', 'Churn_Status']
    corr_matrix = cust_df[cols_to_check].corr()
    
    print("\n" + "=" * 60)
    print(" FORMAL TARGET LEAKAGE AUDIT ")
    print("=" * 60)
    print("Feature Correlations with Target (Churn_Status):")
    for col in ['Recency', 'Frequency', 'Monetary', 'Avg_Order_Value']:
        r = corr_matrix.loc[col, 'Churn_Status']
        print(f" - {col:<18}: Pearson r = {r:>7.4f}")
    
    recency_corr = corr_matrix.loc['Recency', 'Churn_Status']
    print("\nLeakage Verdict:")
    print(f" > Recency has a dominant correlation of {recency_corr:.4f} because the target")
    print("   was created from Recency >= 180. Using Recency directly creates target leakage.")
    print(" > Leakage-Safe Behavioral Feature Subset: ['Frequency', 'Monetary', 'Avg_Order_Value']")
    print("   These features model whether purchase frequency and spend predict lapse risk.")
    print("=" * 60)
    
    return {
        "correlation_with_target": corr_matrix['Churn_Status'].to_dict(),
        "leakage_feature": "Recency",
        "safe_features": ['Frequency', 'Monetary', 'Avg_Order_Value']
    }


def train_logistic_regression(
    cust_df: pd.DataFrame,
    feature_cols: List[str] = None,
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """
    Train a Logistic Regression classification pipeline with StandardScaler.
    
    Workflow:
    1. Extract features X and target y.
    2. Split into Train and Test subsets with stratification.
    3. Construct Pipeline: StandardScaler -> LogisticRegression.
    4. Fit model on training fold.
    5. Evaluate performance on held-out test fold.
    
    Parameters:
    -----------
    cust_df : pd.DataFrame
        Customer dataframe containing features and 'Churn_Status'.
    feature_cols : list of str, optional
        Feature columns to use. Defaults to ['Frequency', 'Monetary', 'Avg_Order_Value'].
    test_size : float
        Proportion of dataset held out for evaluation (default 0.25).
    random_state : int
        Seed for reproducibility (default 42).
        
    Returns:
    --------
    Tuple[Pipeline, dict, pd.DataFrame]:
        Trained pipeline, performance evaluation dictionary, and test results dataframe.
    """
    if feature_cols is None:
        feature_cols = ['Frequency', 'Monetary', 'Avg_Order_Value']
        
    X = cust_df[feature_cols].copy()
    y = cust_df['Churn_Status'].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=random_state, C=1.0, max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Inference on test set
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "features_used": feature_cols,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        },
        "coefficients": dict(zip(feature_cols, [float(c) for c in pipeline.named_steps['classifier'].coef_[0]])),
        "intercept": float(pipeline.named_steps['classifier'].intercept_[0])
    }
    
    test_results = X_test.copy()
    test_results['Actual_Churn'] = y_test
    test_results['Predicted_Churn'] = y_pred
    test_results['Churn_Probability'] = y_prob
    
    print("\n" + "=" * 60)
    print(" MODEL PERFORMANCE EVALUATION (TEST SET) ")
    print("=" * 60)
    print(f"Features Used:        {feature_cols}")
    print(f"Training Samples:     {metrics['train_samples']} customers")
    print(f"Testing Samples:      {metrics['test_samples']} customers")
    print(f"Accuracy:             {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:            {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:               {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print("\nConfusion Matrix Breakdown:")
    print(f" - True Negatives (TN):   {tn:>3} (Active correctly identified as Active)")
    print(f" - False Positives (FP):  {fp:>3} (Active incorrectly flagged as At-Risk)")
    print(f" - False Negatives (FN):  {fn:>3} (At-Risk missed by model)")
    print(f" - True Positives (TP):   {tp:>3} (At-Risk correctly identified)")
    print("\nLearned Feature Coefficients:")
    for f_name, coef_val in metrics["coefficients"].items():
        print(f" - {f_name:<18}: {coef_val:>8.4f}")
    print("=" * 60)
    
    return pipeline, metrics, test_results


def plot_evaluation_charts(
    pipeline: Pipeline,
    cust_df: pd.DataFrame,
    metrics: Dict[str, Any],
    output_dir: str | Path,
    feature_cols: List[str] = None
) -> None:
    """
    Generate and save Confusion Matrix heatmap, ROC Curve, and Precision-Recall curve.
    
    Parameters:
    -----------
    pipeline : Pipeline
        Fitted scikit-learn pipeline.
    cust_df : pd.DataFrame
        Full customer dataset.
    metrics : dict
        Model evaluation metrics.
    output_dir : str or Path
        Target directory for figures.
    feature_cols : list of str
        Features used in the pipeline.
    """
    if feature_cols is None:
        feature_cols = metrics["features_used"]
        
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    cm_dict = metrics["confusion_matrix"]
    cm = np.array([[cm_dict["TN"], cm_dict["FP"]],
                   [cm_dict["FN"], cm_dict["TP"]]])
    
    # -------------------------------------------------------------
    # Figure 10: Confusion Matrix Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    annot_labels = np.array([
        [f"TN = {cm[0,0]}\n(Correct Active)", f"FP = {cm[0,1]}\n(False Alarm)"],
        [f"FN = {cm[1,0]}\n(Missed Churn)", f"TP = {cm[1,1]}\n(Detected Churn)"]
    ])
    
    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', cbar=False,
                annot_kws={'size': 11, 'weight': 'bold'}, ax=ax,
                xticklabels=['Predicted Active (0)', 'Predicted Churned (1)'],
                yticklabels=['Actual Active (0)', 'Actual Churned (1)'])
    
    ax.set_title("10. Confusion Matrix (Test Set: N=103)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Model Prediction", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Actual Churn Status", fontsize=11, fontweight='semibold')
    plt.tight_layout()
    fig10_path = out_path / "10_confusion_matrix.png"
    plt.savefig(fig10_path, dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Figure 11: ROC and Precision-Recall Curves
    # -------------------------------------------------------------
    X = cust_df[feature_cols]
    y = cust_df['Churn_Status']
    y_probs = pipeline.predict_proba(X)[:, 1]
    
    fpr, tpr, _ = roc_curve(y, y_probs)
    roc_auc = auc(fpr, tpr)
    
    prec_vals, rec_vals, _ = precision_recall_curve(y, y_probs)
    pr_auc = auc(rec_vals, prec_vals)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color='#2563eb', lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Chance')
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Precision-Recall Curve
    ax2.plot(rec_vals, prec_vals, color='#16a34a', lw=2.5, label=f"PR Curve (AUC = {pr_auc:.3f})")
    baseline_churn = y.mean()
    ax2.axhline(baseline_churn, color='gray', linestyle='--', lw=1.5, label=f'Baseline Churn Rate ({baseline_churn:.1%})')
    ax2.set_title("Precision-Recall Curve", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Recall (Coverage)", fontsize=11)
    ax2.set_ylabel("Precision (Accuracy of Flags)", fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle("11. Model Diagnostic Performance Curves", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig11_path = out_path / "11_roc_pr_curve.png"
    plt.savefig(fig11_path, dpi=300)
    plt.close()
    
    print("[SUCCESS] Exported Figure 10 (Confusion Matrix) and Figure 11 (ROC & PR Curves).")


def score_all_customers(
    pipeline: Pipeline,
    cust_df: pd.DataFrame,
    feature_cols: List[str] = None
) -> pd.DataFrame:
    """
    Score all customers in the cohort to calculate Churn_Probability and Predicted_Churn.
    
    Parameters:
    -----------
    pipeline : Pipeline
        Trained model pipeline.
    cust_df : pd.DataFrame
        Customer dataframe.
    feature_cols : list of str, optional
        Features expected by pipeline.
        
    Returns:
    --------
    pd.DataFrame:
        Enriched dataframe with 'Churn_Probability' and 'Predicted_Churn'.
    """
    if feature_cols is None:
        feature_cols = ['Frequency', 'Monetary', 'Avg_Order_Value']
        
    cust_df = cust_df.copy()
    X = cust_df[feature_cols]
    
    cust_df['Churn_Probability'] = pipeline.predict_proba(X)[:, 1]
    cust_df['Predicted_Churn'] = pipeline.predict(X)
    
    return cust_df


def save_model(pipeline: Pipeline, model_path: str | Path) -> None:
    """
    Persist trained model pipeline to disk using joblib.
    
    Parameters:
    -----------
    pipeline : Pipeline
        Fitted model pipeline.
    model_path : str or Path
        Target destination file.
    """
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"[SUCCESS] Trained model successfully saved to: {path.resolve()}")


def run_churn_modeling_pipeline(
    cust_df: pd.DataFrame,
    fig_dir: str | Path,
    model_dir: str | Path
) -> Tuple[pd.DataFrame, Pipeline, Dict[str, Any]]:
    """
    End-to-end execution of leakage check, training, evaluation, charting, and inference.
    
    Parameters:
    -----------
    cust_df : pd.DataFrame
        Customer dataset.
    fig_dir : str or Path
        Directory for evaluation charts.
    model_dir : str or Path
        Directory for model persistence.
        
    Returns:
    --------
    Tuple[pd.DataFrame, Pipeline, dict]:
        Scored customer dataframe, fitted pipeline, and evaluation metrics dictionary.
    """
    audit_target_leakage(cust_df)
    
    feature_cols = ['Frequency', 'Monetary', 'Avg_Order_Value']
    pipeline, metrics, _ = train_logistic_regression(cust_df, feature_cols=feature_cols)
    
    plot_evaluation_charts(pipeline, cust_df, metrics, fig_dir, feature_cols=feature_cols)
    
    scored_cust = score_all_customers(pipeline, cust_df, feature_cols=feature_cols)
    
    model_file = Path(model_dir) / "logistic_regression_churn_model.joblib"
    save_model(pipeline, model_file)
    
    return scored_cust, pipeline, metrics


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    cust_csv = project_root / "data" / "processed" / "customer_rfm_summary.csv"
    fig_path = project_root / "outputs" / "figures"
    model_path = project_root / "outputs" / "models"
    
    if not cust_csv.exists():
        print("[ERROR] Customer RFM summary not found. Run rfm_analysis.py first.")
    else:
        cdf = pd.read_csv(cust_csv)
        run_churn_modeling_pipeline(cdf, fig_path, model_path)
