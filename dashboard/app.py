
import os
from pathlib import Path

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------
# Configuration
# ------------------------------------------------

API_URL = os.getenv(
    "GEOVISION_API_URL",
    "http://127.0.0.1:8000"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="GeoVision ML Dashboard",
    page_icon="🛰️",
    layout="wide"
)


# ------------------------------------------------
# Helper functions
# ------------------------------------------------

def get_api_health():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException:
        return None


def get_retraining_status():
    try:
        response = requests.get(
            f"{API_URL}/retrain/status",
            timeout=5
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException:
        return None

def get_metrics():

    try:
        response = requests.get(
            f"{API_URL}/metrics",
            timeout=5
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException:
        return None

# ------------------------------------------------
# Sidebar
# ------------------------------------------------

st.sidebar.title("🛰️ GeoVision ML")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Image Prediction",
        "Upload Training Data",
        "Model Retraining",
        "Model Performance",
        "Prediction History"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Satellite Image Classification Pipeline")


# ------------------------------------------------
# Overview
# ------------------------------------------------

if page == "Overview":

    st.title("GeoVision ML Dashboard")

    st.write(
        "Monitor your satellite image classification "
        "pipeline and interact with the prediction API."
    )

    health = get_api_health()

    col1, col2, col3 = st.columns(3)

    with col1:
        if health:
            st.success("API Online")
        else:
            st.error("API Offline")

    with col2:
        if health:
            st.metric(
                "Loaded Model",
                health.get("model_name", "Unknown")
            )
        else:
            st.metric("Loaded Model", "Unavailable")

    with col3:
        if health:
            st.metric(
                "Model Loaded",
                "Yes" if health.get("model_loaded") else "No"
            )
        else:
            st.metric("Model Loaded", "Unknown")

    st.subheader("Available Land-use Classes")

    try:
        response = requests.get(
            f"{API_URL}/",
            timeout=5
        )
        response.raise_for_status()

        classes = response.json().get(
            "available_classes", []
        )

        st.write(", ".join(classes))

    except requests.RequestException:
        st.warning("Could not retrieve classification classes.")

    st.subheader("Retraining Status")

    status = get_retraining_status()

    if status:
        st.info(
            f"Current status: {status.get('status', 'Unknown')}"
        )
    else:
        st.warning("Retraining status is unavailable.")

    st.caption(
        "Use the navigation menu to access predictions, "
        "training uploads, retraining, and performance."
    )

    st.divider()

    st.subheader("Production Monitoring")
    
    metrics = get_metrics()
    
    if metrics:
    
        col1, col2, col3 = st.columns(3)
    
        with col1:
            st.metric(
                "Total API Requests",
                metrics["total_requests"]
            )
    
        with col2:
            st.metric(
                "Total Predictions",
                metrics["total_predictions"]
            )
    
        with col3:
            st.metric(
                "Failed Requests",
                metrics["failed_requests"]
            )
    
        col4, col5, col6 = st.columns(3)
    
        with col4:
            st.metric(
                "API Uptime",
                f'{metrics["uptime_seconds"] / 3600:.2f} hours'
            )
    
        with col5:
            st.metric(
                "Average Request Time",
                f'{metrics["average_request_time_ms"]:.2f} ms'
            )
    
        with col6:
            st.metric(
                "Average Prediction Time",
                f'{metrics["average_prediction_time_ms"]:.2f} ms'
            )
    
    else:
        st.warning("Monitoring metrics are unavailable.")

# ------------------------------------------------
# Image Prediction
# ------------------------------------------------

elif page == "Image Prediction":

    st.title("Satellite Image Prediction")

    st.write(
        "Upload a satellite image to predict its land-use category."
    )

    uploaded_image = st.file_uploader(
        "Choose a satellite image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        st.image(
            uploaded_image,
            caption="Uploaded Satellite Image",
            width=350
        )

        if st.button("Predict Image", type="primary"):

            try:
                uploaded_image.seek(0)

                files = {
                    "file": (
                        uploaded_image.name,
                        uploaded_image.getvalue(),
                        uploaded_image.type
                    )
                }

                with st.spinner("Classifying image..."):

                    response = requests.post(
                        f"{API_URL}/predict",
                        files=files,
                        timeout=120
                    )

                response.raise_for_status()
                result = response.json()

                st.success("Prediction completed!")

                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Predicted Class",
                        result["predicted_class"]
                    )

                with col2:
                    st.metric(
                        "Confidence",
                        f"{result['confidence'] * 100:.2f}%"
                    )

                st.subheader("Class Probabilities")

                probabilities = pd.Series(
                    result["probabilities"]
                ).sort_values(ascending=True)

                st.bar_chart(probabilities)

            except requests.RequestException as error:
                st.error(f"API request failed: {error}")

            except (KeyError, ValueError) as error:
                st.error(f"Unexpected prediction response: {error}")


# ------------------------------------------------
# Upload Training Data
# ------------------------------------------------

elif page == "Upload Training Data":

    st.title("Upload New Training Data")

    st.write(
        "Upload a ZIP archive containing labelled satellite images."
    )

    st.info(
        "The archive must contain folders named after the "
        "10 EuroSAT classes."
    )

    uploaded_zip = st.file_uploader(
        "Choose a ZIP file",
        type=["zip"]
    )

    if uploaded_zip:

        st.write("Selected file:", uploaded_zip.name)
        st.write(
            f"File size: {uploaded_zip.size / (1024 * 1024):.2f} MB"
        )

        if st.button("Upload Dataset", type="primary"):

            try:
                files = {
                    "file": (
                        uploaded_zip.name,
                        uploaded_zip.getvalue(),
                        "application/zip"
                    )
                }

                with st.spinner("Uploading dataset..."):

                    response = requests.post(
                        f"{API_URL}/upload-data",
                        files=files,
                        timeout=180
                    )

                response.raise_for_status()

                st.success("Dataset uploaded successfully!")
                st.json(response.json())

            except requests.RequestException as error:
                st.error(f"Upload failed: {error}")


# ------------------------------------------------
# Model Retraining
# ------------------------------------------------

elif page == "Model Retraining":

    st.title("Model Retraining")

    st.write(
        "Start retraining using the newly uploaded labelled images."
    )

    status = get_retraining_status()

    if status:

        current_status = status.get("status", "unknown")

        if current_status == "running":
            st.warning("Model retraining is currently running.")

        elif current_status == "completed":
            st.success("Last retraining completed successfully.")

            if status.get("result"):
                st.json(status["result"])

        elif current_status == "failed":
            st.error("Last retraining failed.")
            st.write(status.get("error"))

        else:
            st.info(f"Retraining status: {current_status}")

    else:
        st.warning("Could not retrieve retraining status.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Retraining", type="primary"):

            try:
                response = requests.post(
                    f"{API_URL}/retrain",
                    timeout=10
                )

                response.raise_for_status()

                st.success("Retraining has been started.")
                st.json(response.json())

            except requests.RequestException as error:
                st.error(f"Could not start retraining: {error}")

    with col2:
        if st.button("Refresh Status"):
            st.rerun()


# ------------------------------------------------
# Model Performance
# ------------------------------------------------

elif page == "Model Performance":

    st.title("Model Performance Comparison")

    st.write(
        "Compare the CNN and MobileNet models using "
        "your evaluation results."
    )

    results_file = (
        PROJECT_ROOT / "results" / "model_comparison.csv"
    )

    if results_file.exists():

        results = pd.read_csv(results_file)

        st.subheader("Evaluation Results")
        st.dataframe(
            results,
            use_container_width=True
        )

        numeric_columns = results.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:

            selected_metric = st.selectbox(
                "Select evaluation metric",
                numeric_columns
            )

            if "model" in results.columns:

                chart_data = results.set_index("model")[
                    selected_metric
                ]

                st.bar_chart(chart_data)

    else:
        st.info(
            "No model comparison results found. "
            "Complete model evaluation first and save "
            "results/model_comparison.csv."
        )

    st.subheader("Model Information")

    health = get_api_health()

    if health:
        st.write("Currently loaded model:", health.get("model_name"))
        st.write("Model loaded:", health.get("model_loaded"))
    else:
        st.warning("API is unavailable.")

elif page == "Prediction History":

    st.title("Prediction History")

    try:
        log_file = (
            PROJECT_ROOT /
            "results" /
            "prediction_logs.jsonl"
        )

        if log_file.exists():
        
            df = pd.read_json(
                log_file,
                lines=True
            )

            if not df.empty:
            
                st.subheader("Recent Predictions")

                st.dataframe(
                    df.sort_values(
                        "timestamp",
                        ascending=False
                    ),
                    use_container_width=True
                )

                st.subheader("Predictions by Class")

                class_counts = df[
                    "predicted_class"
                ].value_counts()

                st.bar_chart(class_counts)

                st.subheader("Prediction Confidence")

                st.line_chart(
                    df["confidence"]
                )

            else:
                st.info("No predictions have been recorded.")

        else:
            st.info("Prediction history is currently empty.")

    except Exception as error:
        st.error(f"Could not load prediction history: {error}")