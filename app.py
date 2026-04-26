import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import prediction function (NEW)
from src.predict import predict_aqi

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(page_title="AQI Prediction System", layout="wide")

# -----------------------------------
# Load Dataset (Cached for performance)
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/city_day.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Replace missing values
    df.replace("", np.nan, inplace=True)
    df.replace(-200, np.nan, inplace=True)

    # Convert numeric columns
    numeric_cols = df.columns[2:]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Drop rows where AQI missing
    df = df.dropna(subset=["AQI"])

    return df

df = load_data()

# -----------------------------------
# Title
# -----------------------------------
st.title("🌍 Intelligent AQI Prediction & Analytics System")
st.write("Predict AQI and analyze pollution trends using Machine Learning")

st.divider()

# -----------------------------------
# Tabs
# -----------------------------------
tab1, tab2 = st.tabs(["🔮 AQI Prediction", "📊 Data Dashboard"])

# =====================================================
# 🔮 TAB 1: AQI Prediction
# =====================================================
with tab1:

    st.subheader("Enter Pollutant Values")

    col1, col2, col3 = st.columns(3)

    with col1:
        pm25 = st.number_input("PM2.5", min_value=0.0)
        pm10 = st.number_input("PM10", min_value=0.0)

    with col2:
        no = st.number_input("NO", min_value=0.0)
        no2 = st.number_input("NO2", min_value=0.0)

    with col3:
        nox = st.number_input("NOx", min_value=0.0)
        nh3 = st.number_input("NH3", min_value=0.0)

    if st.button("Predict AQI"):

        try:
            aqi = predict_aqi(pm25, pm10, no, no2, nox, nh3)

            st.success(f"Predicted AQI: {aqi:.2f}")

            st.divider()
            st.subheader("🌡 AQI Category & Health Advice")

            if aqi <= 50:
                st.success("Good – Safe air quality.")
            elif aqi <= 100:
                st.info("Satisfactory – Minor discomfort to sensitive people.")
            elif aqi <= 200:
                st.warning("Moderate – Breathing discomfort possible.")
            elif aqi <= 300:
                st.warning("Poor – Avoid outdoor activities.")
            elif aqi <= 400:
                st.error("Very Poor – Health effects likely.")
            else:
                st.error("Severe – Serious health impacts.")

            st.divider()
            st.subheader("🛡 Personal Precautions")

            st.write("✔ Wear N95 mask")
            st.write("✔ Avoid outdoor exercise")
            st.write("✔ Use air purifier")
            st.write("✔ Keep windows closed during high pollution")

        except Exception as e:
            st.error(f"Error in prediction: {e}")

# =====================================================
# 📊 TAB 2: DATA DASHBOARD
# =====================================================
with tab2:

    st.header("📊 Dataset Analytics Dashboard")

    col1, col2 = st.columns(2)

    # AQI Distribution
    with col1:
        st.subheader("AQI Distribution")
        fig1, ax1 = plt.subplots()
        ax1.hist(df["AQI"], bins=30)
        ax1.set_xlabel("AQI")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)

    # PM2.5 Distribution
    with col2:
        st.subheader("PM2.5 Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(df["PM2.5"].dropna(), bins=30)
        ax2.set_xlabel("PM2.5")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)