# Customer Churn Prediction & Risk Analytics Using E-Commerce Data

### IBM SkillsBuild Internship in Data Analytics with AI — Final Project
**Curriculum Alignment:** Masterclass 3: *"AI Predicts What Happens Next"*  
**Author / Intern:** Samiksha Rajendra Nandurkar  
**Submission Date:** September 2026  
**License:** MIT License  

---

## 📌 Executive Summary & Overview

Customer retention is among the most vital drivers of profitability in e-commerce. Industry benchmarks consistently demonstrate that acquiring a new customer costs **5 to 7 times more** than retaining an existing account. Furthermore, a modest 5% increase in customer retention can boost corporate profits by 25% to 95%.

This repository contains a **complete, runnable, error-checked, and submission-ready** data analytics and machine learning solution. Developed using real transaction logs from the **IBM SkillsBuild Data Analytics with AI** curriculum, this project transitions enterprise decision-making from *reactive customer recovery* to *proactive predictive intervention*.

The project cleans and audits dirty transactional logs, aggregates records into unified customer profiles (1 customer = 1 row), computes **Recency, Frequency, and Monetary (RFM)** behavioral metrics, audits target leakage, and trains a regularized **Logistic Regression** pipeline with feature scaling. The model achieves **80.58% test accuracy** and calculates individual churn probabilities via `predict_proba()`, stratifying the customer base into **High, Medium, and Low Risk** tiers.

---

## 🎯 Business Problem & Project Objectives

### The Business Challenge
In modern non-contractual e-commerce:
1. **Silent Attrition:** Customers do not submit cancellation requests; they simply diminish purchase frequency and lapse into dormancy.
2. **Margin Erosion:** Untargeted blanket discount campaigns waste marketing capital on accounts that would have purchased anyway.
3. **High-Value Account Vulnerability:** Losing top spenders causes severe top-line revenue shocks.

### Core Objectives
- **Data Cleansing:** Remove duplicate records, corrupted currency strings, negative return quantities, and invalid calendar dates.
- **Exploratory Data Analysis (EDA):** Discover revenue skew, seasonality, order volume trends, and spend concentration.
- **RFM Behavioral Engineering:** Construct customer-level profiles and synthesize RFM features anchored against the latest dataset transaction date.
- **Empirical Churn Definition:** Establish a defensible inactivity cutoff (180 days) based on the actual 341-day observation period.
- **Target Leakage Audit:** Conduct a formal evaluation to eliminate circular dependencies between features and target labels.
- **Supervised ML Classification:** Train a `StandardScaler` + `LogisticRegression` pipeline with stratified cross-validation.
- **Risk Stratification:** Classify accounts into actionable operational tiers and output the top at-risk accounts.
- **Actionable Retention Strategy:** Formulate targeted, ROI-positive retention campaigns mapped to specific risk cohorts.

---

## 💻 Technology Stack

| Component | Library / Framework | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.12+ | Core programming and pipeline automation |
| **Data Manipulation** | Pandas, NumPy | Data ingestion, vectorization, cleaning, and customer aggregation |
| **Data Visualization** | Matplotlib, Seaborn | 12 high-resolution analytical figures and diagnostic heatmaps |
| **Machine Learning** | Scikit-learn | Pipeline assembly, StandardScaler, LogisticRegression, evaluation metrics |
| **Model Persistence** | Joblib | Serialization and storage of trained model pipelines |
| **Reporting & Presentation**| ReportLab, Python-PPTX | Automated generation of submission PDF report and 15-slide PowerPoint deck |
| **Interactive Analytics** | Jupyter Notebook | 26-section step-by-step notebook compliant with Colab and local VS Code |

---

## 📂 Project Directory Structure

```text
IBM_SkillsBuild_Customer_Churn_Project/
│
├── data/
│   ├── raw/
│   │   └── ecommerce_raw_dataset.xlsx          <- Original raw Excel transaction dataset (2,000 rows)
│   └── processed/
│       ├── cleaned_ecommerce_dataset.csv       <- Cleaned transaction records (1,838 rows)
│       └── customer_rfm_summary.csv            <- Customer-level aggregated dataset (409 rows)
│
├── notebooks/
│   └── IBM_SkillsBuild_Customer_Churn_Prediction.ipynb <- Complete 26-section runnable Jupyter notebook
│
├── src/
│   ├── __init__.py                             <- Package initialization
│   ├── data_cleaning.py                        <- Ingestion, validation, and data cleaning pipeline
│   ├── eda.py                                  <- Exploratory Data Analysis & figure generation
│   ├── rfm_analysis.py                         <- Customer aggregation, RFM features & churn labeling
│   ├── churn_model.py                          <- Leakage audit, Logistic Regression pipeline & evaluation
│   └── risk_analysis.py                        <- Risk stratification, top 20 extraction & strategy matrix
│
├── outputs/
│   ├── figures/                                <- 12 High-resolution (300 DPI) publication-ready figures
│   │   ├── 01_revenue_distribution.png
│   │   ├── 02_monthly_revenue_trend.png
│   │   ├── 03_order_trend.png
│   │   ├── 04_customer_order_frequency.png
│   │   ├── 05_customer_revenue_distribution.png
│   │   ├── 06_top_customers_revenue.png
│   │   ├── 07_aov_distribution.png
│   │   ├── 08_rfm_distributions.png
│   │   ├── 09_churn_status_distribution.png
│   │   ├── 10_confusion_matrix.png
│   │   ├── 11_roc_pr_curve.png
│   │   └── 12_risk_level_distribution.png
│   ├── predictions/
│   │   └── customer_churn_risk_predictions.csv <- Customer churn probabilities and risk levels
│   └── models/
│       └── logistic_regression_churn_model.joblib <- Trained scikit-learn model pipeline
│
├── reports/
│   ├── IBM_SkillsBuild_Customer_Churn_Project_Report.pdf <- Submission PDF technical report
│   └── IBM_SkillsBuild_Final_Project_Presentation.pptx  <- 15-Slide executive PowerPoint deck
│
├── customer_churn_risk_predictions.csv          <- Root-level export of risk table for convenience
├── run_project.py                              <- Automated master execution script
├── requirements.txt                            <- Python dependency specifications
├── .gitignore                                  <- Version control exclusions
└── README.md                                   <- Comprehensive project documentation
```

---

## 🔍 Data Hygiene & Cleaning Pipeline

The raw dataset contained **2,000 transaction rows** with several real-world data quality defects:
1. **Duplicates:** 23 exact duplicate order rows that artificially inflated transactional metrics.
2. **Missing Customer IDs:** 26 records missing identity (churn prediction requires unique customer identity).
3. **Invalid Calendar Dates:** 24 impossible calendar dates (e.g. `31-02-2026`, `31-04-2026`).
4. **Negative Quantities:** 62 records containing negative return numbers (`-3.0`) or null values.
5. **Formatted Revenue Strings:** 27 records containing currency formatting (`₹50000`, `374814 INR`, commas).

### Cleaning Results Summary
- **Raw Transaction Records:** 2,000
- **Total Defective Records Dropped:** 162
- **Final Cleaned Transactions:** 1,838
- **Cleaned Unique Customers:** 409
- **Cleaned Unique Orders:** 1,838
- **Date Span:** January 1, 2026 to December 8, 2026 (341 days)
- **Total Validated Net Revenue:** **INR 88,712,147.00**

---

## 📊 Exploratory Data Analysis (EDA) Highlights

1. **Revenue Distribution:** Right-skewed; median order spend is **INR 18,634**, while high-value commercial orders reach **INR 2.68M**.
2. **Seasonality:** Strong sales peak occurred in **April 2026 (INR 14.10M)**, followed by stable baseline demand.
3. **Single-Order Drop-off:** **89 customers (21.8%)** made only one purchase throughout the year.
4. **Revenue Concentration:** The top 10 customers generated **INR 14.83M (16.7% of total revenue)**, highlighting critical VIP churn risk.

---

## 📈 Customer RFM Modeling & Churn Definition

### One Customer = One Row
The unit of analysis is aggregated by `Customer_ID` into exactly **409 distinct accounts**.

### RFM Feature Definitions
- **Recency ($R$):** Days elapsed since the customer's most recent purchase relative to the reference date (`2026-12-08`).
  - *Mean:* 125.61 days | *Median:* 124.00 days | *Range:* 0 to 340 days.
- **Frequency ($F$):** Number of distinct orders placed.
  - *Mean:* 4.49 orders | *Median:* 3.00 orders | *Max:* 43 orders.
- **Monetary ($M$):** Total lifetime spend across all purchases.
  - *Mean:* INR 216,900.11 | *Median:* INR 97,523.00 | *Max:* INR 3,100,033.00.
- **Average Order Value ($AOV$):** Mean spend per order ($M / F$).
  - *Mean:* INR 51,504.49 | *Median:* INR 30,926.18 | *Max:* INR 775,008.25.

### Empirical Churn Definition
In accordance with IBM SkillsBuild Masterclass 3 guidelines:
- **Observation Span:** 341 days.
- **Inactivity Threshold:** $\ge 180 \text{ days}$ (half a year of complete purchase inactivity, covering > 52% of the trading period).
- **Active Accounts (0):** 310 customers (75.79%)
- **Churned Accounts (1):** 99 customers (24.21%)

---

## 🛡️ Target Leakage Audit & Machine Learning Pipeline

### Target Leakage Audit
If `Recency` is used as a predictor when `Churn_Status` was created directly from `Recency >= 180`, the model learns the exact threshold boundary, resulting in **100% artificial accuracy** (target leakage). 
To ensure a valid, generalizable predictive model:
- **Excluded Features:** `Customer_ID` (identifier), `Churn_Status` (target), `Recency` (target derivation source).
- **Leakage-Safe Predictor Matrix:** $X = [\text{Frequency, Monetary, Avg\_Order\_Value}]$.

### Machine Learning Architecture
- **Pipeline:** `StandardScaler` (z-score feature normalization) $\rightarrow$ `LogisticRegression(C=1.0, max_iter=1000)`.
- **Partition:** 75% Train (306 customers) / 25% Test (103 customers) with stratification.

### Test Set Performance Evaluation (N=103 Customers)
- **Accuracy:** **80.58%**
- **Precision:** **64.71%** (11 true churners caught out of 17 flagged; only 6 false alarms)
- **Recall:** **44.00%** (Identifies 44% of true churners using strictly behavioral features)
- **ROC AUC:** **0.781** (Solid diagnostic separation over 0.50 random baseline)

#### Test Confusion Matrix Breakdown
| | Predicted Active (0) | Predicted Churned (1) |
| :--- | :---: | :---: |
| **Actual Active (0)** | **72 (True Negative)** | **6 (False Positive)** |
| **Actual Churned (1)** | **14 (False Negative)** | **11 (True Positive)** |

### Learned Logistic Regression Coefficients
- **Frequency ($\beta = -2.7064$):** Dominant protective factor; repeat purchase volume strongly depresses churn risk.
- **Monetary ($\beta = -0.5701$):** Secondary protective factor; higher cumulative spend reduces churn probability.
- **Average Order Value ($\beta = +0.0096$):** Negligible impact; single basket size does not guarantee retention.

---

## 🎯 Calibrated Customer Risk Stratification

The baseline churn incidence is **24.21%**, with the highest model-predicted probability reaching **51.76%**. Applying an uncalibrated static 70% threshold would capture zero high-risk accounts. 

We calibrated operational risk tiers to reflect the empirical distribution:
- **High Risk ($P \ge 40\%$):** **86 customers (21.0%)** — top vulnerable accounts requiring immediate intervention.
- **Medium Risk ($20\% \le P < 40\%$):** **144 customers (35.2%)** — moderate risk hovering around baseline; habit formation.
- **Low Risk ($P < 20\%$):** **179 customers (43.8%)** — highly engaged, loyal brand champions.

### Top 10 Highest-Risk Customers
| Customer ID | Recency | Frequency | Monetary | Avg Order Value | Churn Probability | Risk Level |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **C182** | 323 days | 1 order | ₹831 | ₹831 | **51.76%** | High Risk |
| **C416** | 31 days | 1 order | ₹1,206 | ₹1,206 | **51.75%** | High Risk |
| **C295** | 175 days | 1 order | ₹1,909 | ₹1,909 | **51.72%** | High Risk |
| **C156** | 306 days | 1 order | ₹2,075 | ₹2,075 | **51.71%** | High Risk |
| **C320** | 260 days | 1 order | ₹2,128 | ₹2,128 | **51.71%** | High Risk |
| **C466** | 329 days | 1 order | ₹2,130 | ₹2,130 | **51.71%** | High Risk |
| **C387** | 127 days | 1 order | ₹2,206 | ₹2,206 | **51.71%** | High Risk |
| **C283** | 285 days | 1 order | ₹2,712 | ₹2,712 | **51.68%** | High Risk |
| **C340** | 32 days | 1 order | ₹3,454 | ₹3,454 | **51.65%** | High Risk |
| **C200** | 312 days | 1 order | ₹3,668 | ₹3,668 | **51.65%** | High Risk |

---

## 💡 Strategic Retention Action Plan

| Strategy | Target Customer Segment | Recommended Action | Data-Driven Rationale | Monitored KPI |
| :--- | :--- | :--- | :--- | :--- |
| **First-Order Onboarding Sequence** | High-Risk Single Buyers (Frequency = 1, e.g. C182, C416) | Automated 14-day post-purchase welcome sequence with 15% discount on 2nd purchase. | Over 85% of high-risk customers made only 1 order. Converting to 2 orders directly cuts churn risk. | 30-Day Repeat Rate |
| **Lapsed-Customer Re-engagement** | Dormant Accounts (Recency 90–150 Days) | Dynamic 'We Miss You' SMS/Email workflows with personalized category recommendations. | Recency drifts past 90 days before reaching 180-day permanent churn; timely outreach rescues accounts. | Win-Back Conversion Rate |
| **VIP High-Value At-Risk Protection** | High Spenders with Elevated Risk (Monetary > ₹100K) | Dedicated account manager check-ins, VIP concierge support, and feedback surveys. | Top 10 customers drive 16.7% of total revenue. Losing a high-spend customer causes severe top-line harm. | Net Revenue Retention (NRR) |
| **Habit Formation & Replenishment** | Medium-Risk Accounts (Frequency 2–3, Risk 20%–40%) | Tiered loyalty point multipliers and predictive replenishment reminders. | Reinforces repeat ordering cadence and raises customer switching barriers. | Quarterly Order Frequency |
| **Advocate Appreciation Circle** | Low-Risk Champions (Frequency ≥ 5, Risk < 20%) | VIP insider pre-access to new product drops and referral bonuses. | Retains brand affinity and leverages organic advocacy as a low-cost acquisition engine. | Net Promoter Score (NPS) |

---

## 🚀 How to Run the Project

### Prerequisites
- Python 3.12 or higher installed
- Git installed (optional)

### 1. Clone or Open Project
```bash
cd "D:\My Projects\IBM_SkillsBuild_Customer_Churn_Project"
```

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Execute the Full Automated Pipeline
Run the master script to clean the data, generate all 12 figures, train the model, and export predictions:
```bash
python run_project.py
```

### 4. Interactive Jupyter Notebook
Launch Jupyter Notebook to inspect the 26 step-by-step masterclass sections:
```bash
jupyter notebook notebooks/IBM_SkillsBuild_Customer_Churn_Prediction.ipynb
```
*(Also 100% compatible with Google Colab: upload the notebook and `data/raw/ecommerce_raw_dataset.xlsx`).*

---

## 📋 Final Submission Quality Checklist

- [x] **Workspace Isolation:** Project created in `IBM_SkillsBuild_Customer_Churn_Project/` without touching external folders.
- [x] **Zero Fabricated Data:** All 409 customer metrics, model scores, and figures derive strictly from the actual dataset.
- [x] **Data Hygiene:** 162 defective records removed; clean dataset of 1,838 orders persisted.
- [x] **Customer Aggregation:** Verified 1 customer = 1 row (409 unique accounts).
- [x] **RFM Metrics:** Recency, Frequency, Monetary, and AOV engineered and analyzed.
- [x] **Empirical Churn Definition:** Inactivity threshold $\ge 180 \text{ days}$ justified (24.21% churn rate).
- [x] **Target Leakage Audit:** Audited and resolved; valid behavioral predictor set isolated.
- [x] **Machine Learning Pipeline:** `StandardScaler` + `LogisticRegression` pipeline trained with reproducible seed.
- [x] **Model Evaluation:** Exact Accuracy (80.58%), Precision (64.71%), Recall (44.00%), and Confusion Matrix generated.
- [x] **Calibrated Risk Stratification:** High (21.0%), Medium (35.2%), and Low (43.8%) tiers established.
- [x] **Top 20 At-Risk Table:** Sorted descending by probability and exported to CSV.
- [x] **Modular Python Modules:** Reusable code across `data_cleaning.py`, `eda.py`, `rfm_analysis.py`, `churn_model.py`, `risk_analysis.py`.
- [x] **Automated Runner:** `run_project.py` executes end-to-end in < 15 seconds.
- [x] **Jupyter Notebook:** 26-section notebook created at `notebooks/IBM_SkillsBuild_Customer_Churn_Prediction.ipynb`.
- [x] **PDF Technical Report:** Formal ReportLab document created at `reports/IBM_SkillsBuild_Customer_Churn_Project_Report.pdf` (2.22 MB).
- [x] **PowerPoint Presentation:** 15-Slide executive deck created at `reports/IBM_SkillsBuild_Final_Project_Presentation.pptx` (1.54 MB).

---

## 🎓 Academic & Internship Certification

This project was built for the **AICTE - IBM SkillsBuild Data Analytics with AI Internship** under Masterclass 3: *"AI Predicts What Happens Next"*. All statistical analyses, code architectures, figures, and strategic recommendations represent authentic, original student work based on empirical data science best practices.
