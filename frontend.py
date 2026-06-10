import streamlit as st
import requests

# Change this to your Render URL later
API_URL = "https://credit-card-fraud-detection-api-1-yba7.onrender.com/batch_predict"

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Credit Card Fraud Detection")

st.write("""
Upload a CSV file containing transaction data.
The model will analyze the transactions and return a CSV
containing fraud probabilities and predictions.
""")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    st.success(f"Selected file: {uploaded_file.name}")

    if st.button("Predict"):

        with st.spinner("Running fraud detection..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "text/csv"
                )
            }

            response = requests.post(
                API_URL,
                files=files
            )

        if response.status_code == 200:

            st.success("Prediction completed successfully!")

            st.download_button(
                label="📥 Download Predictions",
                data=response.content,
                file_name="predictions.csv",
                mime="text/csv"
            )

        else:

            st.error("Prediction failed")

            try:
                st.json(response.json())
            except:
                st.write(response.text)