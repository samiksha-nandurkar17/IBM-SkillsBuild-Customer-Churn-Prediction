
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# PAGE CONFIG
st.set_page_config(
    page_title="ChurnIQ | AI Analytics",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "models" / "logistic_regression_churn_model.joblib"

# CUSTOM DESIGN
st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background: #111a30;
}
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #ffffff;
}
.subtitle {
    color: #94a3b8;
    font-size: 16px;
}
.hero {
    background: linear-gradient(120deg,#182b50,#25204b);
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #34456a;
    margin-bottom: 25px;
}
.metric-card {
    background: #151f35;
    border: 1px solid #293957;
    padding: 20px;
    border-radius: 14px;
    min-height: 115px;
}
.metric-label {
    color: #94a3b8;
    font-size: 14px;
}
.metric-value {
    color: #ffffff;
    font-size: 27px;
    font-weight: bold;
}
.section-title {
    color: #e2e8f0;
    font-size: 23px;
    font-weight: 700;
}
div.stButton > button {
    background: linear-gradient(90deg,#6366f1,#8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-weight: bold;
    width: 100%;
}
div.stButton > button:hover {
    background: #4f46e5;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Model not found. Run python run_project.py first.")
    st.stop()

model = load_model()


# SIDEBAR
with st.sidebar:
    st.markdown("## 📊 ChurnIQ")
    st.caption("AI-Powered Customer Analytics")
    st.divider()

    st.markdown("### 👤 Customer Information")

    frequency = st.number_input(
        "Purchase Frequency",
        min_value=0,
        value=3
    )

    monetary = st.number_input(
        "Total Spending (₹)",
        min_value=0.0,
        value=10000.0,
        step=500.0
    )

    avg_order_value = st.number_input(
        "Average Order Value (₹)",
        min_value=0.0,
        value=3000.0,
        step=100.0
    )

    predict = st.button("🔮 Predict Churn Risk")

    st.divider()
    st.caption("Developed by Samiksha Nandurkar")


# HEADER
st.markdown("""
<div class="hero">
    <div class="main-title">Customer Churn Analytics</div>
    <div class="subtitle">
        Intelligent Machine Learning Dashboard for Customer
        Retention & Risk Prediction
    </div>
</div>
""", unsafe_allow_html=True)


# TOP METRICS
st.markdown('<div class="section-title">📈 Model Performance</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

metrics = [
    ("Test Accuracy", "80.58%"),
    ("Precision", "64.71%"),
    ("Recall", "44.00%"),
    ("Customers Analysed", "409")
]

for col, (label, value) in zip([c1, c2, c3, c4], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.caption("Model metrics are from the original held-out test evaluation.")
st.divider()


# TABS
tab1, tab2, tab3 = st.tabs([
    "🔮 Customer Prediction",
    "📊 Model Insights",
    "ℹ️ About Project"
])


# PREDICTION TAB
with tab1:
    st.markdown("### Customer Churn Prediction")
    st.write("Enter customer information in the sidebar and click Predict.")

    if predict:

        input_data = pd.DataFrame([{
            "Frequency": frequency,
            "Monetary": monetary,
            "Avg_Order_Value": avg_order_value
        }])

        try:
            probabilities = model.predict_proba(input_data)[0]
            churn_index = list(model.classes_).index(1)

            probability = float(probabilities[churn_index])
            prediction = int(model.predict(input_data)[0])

            st.markdown("#### Prediction Result")

            left, right = st.columns(2)

            with left:
                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )
                st.progress(probability)

            with right:
                if prediction == 1:
                    st.error("⚠️ Model Prediction: Churned / At Risk")
                else:
                    st.success("✅ Model Prediction: Active")

            st.divider()

            st.markdown("### 💡 Business Recommendation")

            if prediction == 1:
                st.warning("""
                **Retention actions recommended**

                - Offer personalized discounts.
                - Introduce loyalty rewards.
                - Send targeted engagement campaigns.
                - Follow up with inactive customers.
                """)
            else:
                st.success("""
                **Customer engagement recommendations**

                - Maintain a positive customer experience.
                - Offer loyalty benefits.
                - Encourage repeat purchases.
                - Monitor future purchasing behaviour.
                """)

            st.markdown("### Customer Input Summary")
            st.dataframe(input_data, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    else:
        st.info("👈 Enter customer details in the sidebar and click Predict Churn Risk.")


# MODEL INSIGHTS TAB
with tab2:
    st.markdown("### Machine Learning Model Insights")

    a, b = st.columns(2)

    with a:
        st.markdown("#### Algorithms & Pipeline")
        st.write("""
        - Logistic Regression
        - StandardScaler
        - Scikit-learn Pipeline
        - Probability Prediction
        """)

    with b:
        st.markdown("#### Model Features")
        st.write("""
        - Purchase Frequency
        - Total Monetary Value
        - Average Order Value
        """)

    st.markdown("#### Evaluation Metrics")
    st.write("""
    The model was evaluated on 103 held-out customer records.
    Its recall is 44%, so some actual churn customers may be missed.
    Further model improvement is recommended.
    """)


# ABOUT TAB
with tab3:
    st.markdown("### About This Project")

    st.write("""
    ChurnIQ is a machine learning-based customer churn prediction
    application built using Python and Scikit-learn.

    The application analyses customer purchasing behaviour and
    estimates the probability of customer churn.
    """)

    st.markdown("#### Technologies Used")

    st.write("""
    Python | Pandas | NumPy | Scikit-learn |
    Logistic Regression | Streamlit | Joblib
    """)

    st.markdown("#### Project Workflow")

    st.write("""
    1. Data Collection
    2. Data Cleaning
    3. Exploratory Data Analysis
    4. RFM Feature Engineering
    5. Machine Learning Model Training
    6. Churn Probability Prediction
    7. Customer Retention Recommendations
    """)

st.divider()

st.markdown(
    "<center>Made with ❤️ by Samiksha Nandurkar | AI/ML Project</center>",
    unsafe_allow_html=True
)
