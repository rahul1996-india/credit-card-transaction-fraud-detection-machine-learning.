import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image

image1 = Image.open("Fraud1.jpg")
image2 = Image.open("fraud3.jpg")

#load models
model = joblib.load("best_model_RFC.pkl")
label_encoders = joblib.load("label_encoder_RFC.pkl")
scaler= joblib.load("RFC_scaler.pkl")
#======================================================================================
#add background
st.markdown("""
<style>
.block-container {
    background: rgba(220, 252, 231, 0.85); /* light green */
    padding: 2rem;
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

#================================================================================

#title and sub title with images
st.markdown(
    """
    <h1 style='text-align:center;font-size:48px;font-weight:800;color:#16a34a;'>
        CREDITCARD TRANSACTION FRAUD DETECTION
    </h1>
    <p style='text-align:center;font-size:24px;color:#15803d;margin-top:-10px;'>
        Provide Transaction Details To Predict Potential Fraud
    </p>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)
with col1:
    st.image(image1, width=300)
with col2:
    st.image(image2, width=300)
#user input

merchant_category = st.selectbox("Merchant Category",["Grocery" ,"Electronics", "Fuel", "unknown", "Restaurant", "Travel"])
transaction_type = st.selectbox("Transaction Type",["Online", "POS", "ATM"])
transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0, step=100.0)
account_balance = st.number_input("Account Balance", min_value=-26000.0, value=50000.0, step=100.0)
transaction_hour = st.number_input("Transaction Hour", min_value=0, max_value=23,value=12, step=1)
is_international_ui = st.selectbox("Is International?", ["Yes","No"])
is_international = 1 if is_international_ui == "Yes" else 0
device_type = st.selectbox("Device Type", ["Mobile", "Web", "Card", "unknown"])
transaction_channel = st.selectbox("Transaction Channel", ["Online", "POS", "ATM"])
txn_count_last_24h = st.number_input("Txn Count(last 24h)", min_value= 0, value=5, step=1)


#label encoding
merchant_category_encoded = label_encoders["merchant_category"].transform([merchant_category])[0]
transaction_type_encoded = label_encoders["transaction_type"].transform([transaction_type])[0]
device_type_encoded = label_encoders["device_type"].transform([device_type])[0]
transaction_channel_encoded = label_encoders["transaction_channel"].transform([transaction_channel])[0]


# --- Predict ---
if st.button("Predict Fraud"):
    input_data = [[
        merchant_category_encoded,
        transaction_type_encoded,
        transaction_amount,
        account_balance,
        transaction_hour,
        is_international,
        device_type_encoded,
        transaction_channel_encoded,
        txn_count_last_24h,
       ]]

    input_scaled = scaler.transform(input_data)
    proba = model.predict_proba(input_scaled)[0]

    # --- Prediction Result (Probability-based) ---
    fraud_prob = proba[1]
    THRESHOLD = 0.10

    if fraud_prob >= THRESHOLD:
        st.markdown(
            "<h3 style='color:red'>🚨 Fraudulent Transaction Detected</h3>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h3 style='color:green'>✅ Transaction is Legitimate</h3>",
            unsafe_allow_html=True
        )

    # --- Confidence ---
    st.subheader("Prediction Confidence")
    st.progress(int(max(proba) * 100))

    st.write({
        "Legitimate Probability": round(proba[0], 2),
        "Fraud Probability": round(proba[1], 2)
    })

    # --- Probability Bar Chart ---
    proba_df = pd.DataFrame({
        "Class": ["Legitimate", "Fraud"],
        "Probability": [proba[0] * 100, proba[1] * 100]
    })

    st.subheader("Prediction Probability Distribution")
    st.bar_chart(proba_df.set_index("Class"))

    # --- Risk Assessment ---
    if fraud_prob > 0.50:
        risk = "High Risk"
        color = "red"
    elif fraud_prob > 0.20:
        risk = "Medium Risk"
        color = "orange"
    else:
        risk = "Low Risk"
        color = "green"

    st.subheader("Risk Assessment")
    st.markdown(
        f"<h3 style='color:{color}'>⚠️ {risk}</h3>",
        unsafe_allow_html=True
    )


