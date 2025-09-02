import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import math

# Title
st.title("📈 Ethereum Price Prediction Dashboard (ARIMA Model)")

# Sidebar controls
st.sidebar.header("User Controls")
steps = st.sidebar.slider("Forecast days:", 1, 60, 7)
p = st.sidebar.number_input("AR (p)", 0, 10, 5)
d = st.sidebar.number_input("Differencing (d)", 0, 2, 1)
q = st.sidebar.number_input("MA (q)", 0, 10, 0)

# File uploader
uploaded_file = st.file_uploader("Upload Ethereum Price Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    # Assume 'Date' & 'Close'
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🔮 Model", "📈 Forecast", "⬇️ Download"])

    with tab1:
        st.subheader("Ethereum Close Price History")
        fig = px.line(df, x=df.index, y="Close", title="Ethereum Close Price")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("ARIMA Model Training")
        model = ARIMA(df['Close'], order=(p, d, q))
        model_fit = model.fit()
        st.text(model_fit.summary())

        # Calculate metrics (train vs fitted)
        fitted_values = model_fit.fittedvalues
        mse = mean_squared_error(df['Close'][1:], fitted_values[1:])
        rmse = math.sqrt(mse)
        st.metric("RMSE", f"{rmse:.2f}")

        fig, ax = plt.subplots()
        df['Close'].plot(ax=ax, label='Actual')
        fitted_values.plot(ax=ax, label='Fitted', alpha=0.7)
        plt.legend()
        st.pyplot(fig)

    with tab3:
        st.subheader(f"Next {steps} Days Forecast")
        forecast = model_fit.forecast(steps=steps)
        st.line_chart(forecast)

        fig, ax = plt.subplots()
        df['Close'].plot(ax=ax, label='Historical')
        forecast.plot(ax=ax, label='Forecast')
        plt.legend()
        st.pyplot(fig)

    with tab4:
        st.subheader("Download Predictions")
        forecast_df = pd.DataFrame({"Date": pd.date_range(start=df.index[-1], periods=steps+1, freq="D")[1:], 
                                    "Forecast": forecast})
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")
