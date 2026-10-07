import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.title {
    font-size: 42px;
    font-weight: 800;
}

.subtitle {
    font-size: 18px;
    color: #94a3b8;
    margin-bottom: 30px;
}

.result-card {
    padding: 30px;
    border-radius: 16px;
    text-align: center;
    margin-top: 20px;
}

.fraud-card {
    background-color: #450a0a;
    border: 1px solid #ef4444;
}

.safe-card {
    background-color: #052e16;
    border: 1px solid #22c55e;
}

.result-title {
    font-size: 32px;
    font-weight: 800;
}

.score {
    font-size: 24px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_fraud_model():
    return load_model("best_model.h5")


try:
    model = load_fraud_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()


# =========================================================
# LOAD SCALERS
# =========================================================

@st.cache_resource
def load_scalers():

    return joblib.load("scalers.joblib")


try:
    scalers = load_scalers()

    time_scaler = scalers["time_scaler"]
    amount_scaler = scalers["amount_scaler"]

except Exception as e:

    st.error("Unable to load preprocessing scalers.")

    st.exception(e)

    st.stop()


# =========================================================
# LOAD DEMO DATASET
# =========================================================

@st.cache_data
def load_demo_dataset():

    return pd.read_csv("demo_transactions.csv")


try:

    data = load_demo_dataset()

except Exception as e:

    st.error("Unable to load demo_transactions.csv")

    st.exception(e)

    st.stop()


# =========================================================
# PREPROCESS FUNCTION
# =========================================================

def prepare_features(df):

    processed = df.copy()

    # Use the scalers fitted on the original full dataset
    scaled_time = time_scaler.transform(
        processed[["Time"]]
    )

    scaled_amount = amount_scaler.transform(
        processed[["Amount"]]
    )

    processed["Scaled Time"] = scaled_time
    processed["Scaled Amount"] = scaled_amount

    # IMPORTANT:
    # This is the same order used by the original CNN:
    #
    # Scaled Time
    # Scaled Amount
    # V1 ... V28

    feature_columns = (
        ["Scaled Time", "Scaled Amount"] +
        [f"V{i}" for i in range(1, 29)]
    )

    X = processed[
        feature_columns
    ].values.astype(
        np.float32
    )

    return X.reshape(
        X.shape[0],
        X.shape[1],
        1
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">💳 Credit Card Fraud Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'CNN-powered credit card transaction analysis'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📊 Demo Dataset")

    st.metric(
        "Transactions",
        f"{len(data):,}"
    )

    fraud_count = int(data["Class"].sum())

    legitimate_count = (
        len(data) - fraud_count
    )

    st.metric(
        "Fraud Transactions",
        f"{fraud_count:,}"
    )

    st.metric(
        "Legitimate Transactions",
        f"{legitimate_count:,}"
    )

    st.divider()

    st.header("🧠 Model")

    st.write("Architecture: CNN / Conv1D")

    st.write("Input Features: 30")

    st.write("Time + Amount + V1–V28")

    st.divider()

    st.caption(
        "Demonstration using anonymized Kaggle transaction data."
    )


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🔍 Single Transaction",
        "📂 Batch Prediction",
        "📚 About the Model"
    ]
)


# =========================================================
# SINGLE TRANSACTION
# =========================================================

with tab1:

    st.subheader("🔍 Analyze a Transaction")

    st.write(
        "Select a transaction from the demonstration "
        "dataset and run the trained CNN model."
    )

    transaction_number = st.number_input(
        "Transaction Number",
        min_value=1,
        max_value=len(data),
        value=1,
        step=1
    )

    index = transaction_number - 1

    transaction = data.iloc[index]

    st.markdown("### Transaction Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Time",
            f"{transaction['Time']:.2f}"
        )

    with col2:

        st.metric(
            "Amount",
            f"${transaction['Amount']:.2f}"
        )

    with col3:

        actual_label = (
            "FRAUD"
            if transaction["Class"] == 1
            else "LEGITIMATE"
        )

        st.metric(
            "Dataset Label",
            actual_label
        )

    with st.expander("View Transaction Features"):

        feature_data = transaction.drop(
            labels=["Class"]
        ).to_frame(
            name="Value"
        )

        st.dataframe(
            feature_data,
            use_container_width=True
        )

    st.divider()

    analyze = st.button(
        "🔍 Analyze Transaction",
        type="primary",
        use_container_width=True
    )

    if analyze:

        transaction_df = pd.DataFrame(
            [transaction]
        )

        transaction_input = prepare_features(
            transaction_df
        )

        prediction = model.predict(
            transaction_input,
            verbose=0
        )[0][0]

        score = float(prediction)

        is_fraud = score >= 0.5

        st.subheader("🤖 Model Prediction")

        if is_fraud:

            st.markdown(
                f"""
                <div class="result-card fraud-card">
                    <div class="result-title">
                        🚨 FRAUD DETECTED
                    </div>
                    <div class="score">
                        Model Score: {score * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-card safe-card">
                    <div class="result-title">
                        ✅ LEGITIMATE TRANSACTION
                    </div>
                    <div class="score">
                        Model Score: {score * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Model Score",
                f"{score * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Classification",
                "FRAUD" if is_fraud else "LEGITIMATE"
            )

        st.divider()

        if is_fraud == (transaction["Class"] == 1):

            st.success(
                "The model prediction matches the dataset label."
            )

        else:

            st.warning(
                "The model prediction differs from the dataset label."
            )


# =========================================================
# BATCH PREDICTION
# =========================================================

with tab2:

    st.subheader("📂 Batch Prediction")

    st.write(
        "Upload a CSV containing Time, Amount and V1–V28."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_data = pd.read_csv(
                uploaded_file
            )

            required_columns = (
                ["Time"] +
                [f"V{i}" for i in range(1, 29)] +
                ["Amount"]
            )

            missing = [
                column
                for column in required_columns
                if column not in uploaded_data.columns
            ]

            if missing:

                st.error("Missing required columns:")

                st.code(
                    ", ".join(missing)
                )

                st.stop()

            st.success(
                f"Loaded {len(uploaded_data):,} transactions."
            )

            batch_X = prepare_features(
                uploaded_data
            )

            with st.spinner(
                "Analyzing transactions..."
            ):

                probabilities = model.predict(
                    batch_X,
                    verbose=0
                ).flatten()

            predictions = (
                probabilities >= 0.5
            ).astype(int)

            results = uploaded_data.copy()

            results["Model Score"] = (
                probabilities * 100
            ).round(2)

            results["Prediction"] = np.where(
                predictions == 1,
                "FRAUD",
                "LEGITIMATE"
            )

            total = len(results)

            fraud = int(
                predictions.sum()
            )

            legitimate = total - fraud

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Transactions",
                    f"{total:,}"
                )

            with col2:
                st.metric(
                    "Fraud Detected",
                    f"{fraud:,}"
                )

            with col3:
                st.metric(
                    "Legitimate",
                    f"{legitimate:,}"
                )

            st.dataframe(
                results,
                use_container_width=True,
                height=500
            )

            csv_output = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Results",
                data=csv_output,
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(
                "Error processing the uploaded CSV."
            )

            st.exception(e)


# =========================================================
# ABOUT
# =========================================================

with tab3:

    st.subheader("📚 About the System")

    st.markdown("""
    ### Dataset

    The system uses an anonymized Kaggle credit card
    transaction dataset.

    ### Features

    The CNN receives 30 features:

    - Scaled Time
    - Scaled Amount
    - V1 through V28

    ### Model

    The saved neural network contains:

    1. Conv1D
    2. Conv1D
    3. Conv1D
    4. Flatten
    5. Dense sigmoid output

    ### Classification

    A score of 0.5 or greater is classified as fraudulent.

    Scores below 0.5 are classified as legitimate.

    ### Important

    This is a dataset-based fraud detection demonstration.
    It is not connected to a bank or payment network and
    does not process real banking transactions.
    """)

    st.info(
        "The model score is the sigmoid output of the "
        "trained neural network and is not a calibrated "
        "real-world fraud probability."
    )